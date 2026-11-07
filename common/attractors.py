"""
Error-correcting attractor fields on a 2D sampling surface, shared by the toy (vis_plateaus/) and GPT-2
(vis_plateaus_lm/) pipelines.

Gradient-based fields. With coordinates c = (u, v) on the surface, x(c) the ambient source activation and
ln P(k | x) the output log-probabilities, the in-surface Jacobian J = d ln P / dc [V, 2] is computed with two
forward-mode JVPs. Directions use the surface metric G = T^T T (T = dx/dc), so every arrow is the steepest-ascent
direction in the ambient geometry restricted to the surface (G = I on a flat plane with orthonormal coordinates):
    - per anchor a: the gradient of ln P(y_a | x), y_a = anchor a's argmax class (signed; flows into a's plateau)
    - max-norm row: the class k whose ln P(k | x) changes fastest along the surface, and the gradient of ln P(k | x)
    - top singular direction: the unit direction c (c^T G c = 1) maximizing ||J c|| (sign-free), with sigma = ||J c||
Rows of the log-probability Jacobian are used (not logits), so the fields are invariant to the shared logit shift
that softmax ignores.

Block-based trajectories. The residual stream at every layer from the source on is expressed in anchor-relative
(affine) coordinates of the plane through the three anchors' activations at the same layer: anchor 0 at (0, 0),
anchor 1 at (1, 0), anchor 2 at (0, 1). A point that converges onto an anchor's trajectory moves onto that anchor.
"""

import torch
from typing import Callable, Dict


def jvp_columns(f: Callable, coords: torch.Tensor):
    """
    f: [B, 2] -> [B, m], with output b depending only on input b. Returns (f(coords) [B, m], df/dc [B, m, 2]) from two
    forward-mode JVPs (one per coordinate direction).
    """
    columns = []
    out = None
    for i in range(2):
        tangent = torch.zeros_like(coords)
        tangent[:, i] = 1
        out, column = torch.func.jvp(f, (coords,), (tangent,))
        columns.append(column)
    return out, torch.stack(columns, dim=-1)


def surface_metric(to_ambient: Callable, coords: torch.Tensor) -> torch.Tensor:
    """Metric G = T^T T [B, 2, 2] of the surface chart c -> x(c), T = dx/dc [B, d, 2]."""
    _, T = jvp_columns(to_ambient, coords)
    T = T.double()
    return T.transpose(1, 2) @ T


def pooled_gradient_fields(logits: torch.Tensor, logit_jacobian: torch.Tensor, metric: torch.Tensor,
                           anchor_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Args:
        logits: [B, V] at the points
        logit_jacobian: [B, V, 2] d logits / dc
        metric: [B, 2, 2] surface metric G
        anchor_labels: [K] argmax class of each anchor
    Returns (vectors are in surface coordinates):
        'anchor_vectors' [K, B, 2], 'anchor_norms' [K, B]: G^-1 grad ln P(y_a) and its metric norm
        'maxnorm_vectors' [B, 2], 'maxnorm_norms' [B], 'maxnorm_tokens' [B]: the fastest-changing class and its gradient
        'singular_vectors' [B, 2] (unit in the metric), 'singular_sigma' [B]
    """
    logits, J_z, G = logits.double(), logit_jacobian.double(), metric.double()
    p = torch.softmax(logits, dim=-1)
    # Rows of d ln P / dc: d ln P_k = d z_k - sum_j P_j d z_j
    J = J_z - torch.einsum('bv,bvi->bi', p, J_z).unsqueeze(1)  # [B, V, 2]
    G_inv = torch.linalg.inv(G)

    rows = J[:, anchor_labels, :]  # [B, K, 2] covectors
    anchor_vectors = torch.einsum('bij,bkj->bki', G_inv, rows)
    anchor_norms = torch.einsum('bki,bki->bk', rows, anchor_vectors).clamp_min(0).sqrt()

    row_norms2 = torch.einsum('bvi,bij,bvj->bv', J, G_inv, J)  # [B, V]
    maxnorm_tokens = row_norms2.argmax(dim=-1)
    max_rows = J[torch.arange(J.shape[0], device=J.device), maxnorm_tokens]  # [B, 2]
    maxnorm_vectors = torch.einsum('bij,bj->bi', G_inv, max_rows)
    maxnorm_norms = row_norms2.gather(1, maxnorm_tokens.unsqueeze(1))[:, 0].clamp_min(0).sqrt()

    # max ||J c||^2 s.t. c^T G c = 1: with G = L L^T, eigenvectors w of L^-1 (J^T J) L^-T give c = L^-T w
    L_inv = torch.linalg.inv(torch.linalg.cholesky(G))
    A = J.transpose(1, 2) @ J
    eigvals, eigvecs = torch.linalg.eigh(L_inv @ A @ L_inv.transpose(1, 2))
    singular_vectors = torch.einsum('bji,bj->bi', L_inv, eigvecs[..., -1])
    singular_sigma = eigvals[..., -1].clamp_min(0).sqrt()

    return {
        'anchor_vectors': anchor_vectors.permute(1, 0, 2),
        'anchor_norms': anchor_norms.T,
        'maxnorm_vectors': maxnorm_vectors,
        'maxnorm_norms': maxnorm_norms,
        'maxnorm_tokens': maxnorm_tokens,
        'singular_vectors': singular_vectors,
        'singular_sigma': singular_sigma,
    }


def check_logit_rows(f: Callable, coords: torch.Tensor, labels: torch.Tensor, logit_jacobian: torch.Tensor,
                     rtol: float = 1e-3):
    """
    Validate the forward-mode Jacobian rows d z_k / dc [B, V, 2] against reverse-mode autograd for the classes in labels.
    Logit rows are compared (not log-probability rows): centering by the softmax cancels most of the logit gradient,
    which leaves float32 log-probability rows with ~1e-2 relative noise in both modes.
    f: coords [B, 2] -> logits [B, V]
    """
    coords = coords.detach().clone().requires_grad_(True)
    logits = f(coords)
    for label in labels.tolist():
        (expected,) = torch.autograd.grad(logits[:, label].sum(), coords, retain_graph=True)
        got = logit_jacobian[:, label, :].to(expected.dtype)
        err = ((got - expected).norm(dim=1) / expected.norm(dim=1).clamp_min(1e-12)).max().item()
        assert err < rtol, f"Forward-mode Jacobian row of class {label} disagrees with autograd: rel err {err:.2e}"


class AnchorFrame:
    """Affine frame of the plane through three anchor activations [3, d] at one layer."""

    def __init__(self, anchors: torch.Tensor):
        anchors = anchors.double()
        self.anchors = anchors
        self.origin = anchors[0]
        self.edges = torch.stack([anchors[1] - anchors[0], anchors[2] - anchors[0]], dim=1)  # [d, 2]
        self.pinv = torch.linalg.pinv(self.edges)  # [2, d]
        self.scale = torch.pdist(anchors).mean()  # Mean distance between the anchors

    def coords(self, points: torch.Tensor):
        """
        points [B, d] -> (affine coordinates [B, 2] of the orthogonal projection onto the plane,
        off-plane distance [B] in units of the mean anchor distance)
        """
        rel = points.double() - self.origin
        ab = rel @ self.pinv.T
        off = (rel - ab @ self.edges.T).norm(dim=1) / self.scale
        return ab, off

    def nearest_anchor_distance(self, points: torch.Tensor) -> torch.Tensor:
        """Distance [B] to the nearest of the three anchors, in units of the mean anchor distance."""
        return torch.cdist(points.double(), self.anchors).min(dim=1).values / self.scale


def trajectory_coords(resids, anchor_resids) -> Dict[str, torch.Tensor]:
    """
    Args:
        resids: list over layers of [B, d] activations of the points
        anchor_resids: list over the same layers of [3, d] activations of the frame anchors
    Returns:
        'coords' [n_layers, B, 2] affine coordinates, 'off_plane' [n_layers, B], 'nearest_dist' [n_layers, B]
    """
    coords, off_plane, nearest = [], [], []
    for points, anchors in zip(resids, anchor_resids):
        frame = AnchorFrame(anchors)
        ab, off = frame.coords(points)
        coords.append(ab)
        off_plane.append(off)
        nearest.append(frame.nearest_anchor_distance(points))
    return {'coords': torch.stack(coords), 'off_plane': torch.stack(off_plane), 'nearest_dist': torch.stack(nearest)}
