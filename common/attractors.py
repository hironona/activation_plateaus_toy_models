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

Boundary zooms. One window per pair of anchors with different argmax classes a, b (all pairs, or cyclically adjacent
anchors), on the a|b decision boundary of the coarse grid, is resampled on a finer local grid, where the gradient
fields above are recomputed; the window shows the gradients of ln P(a) and of ln P(b) separately.
"""

import numpy as np
import torch
from typing import Callable, Dict, List, Optional

from surfaces import grid_coords


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


def anchor_class_pairs(anchor_labels, mode: str = 'all') -> List[tuple]:
    """
    Pairs (i, j) of anchors whose argmax classes differ, one per class pair:
        'all':      every pair of anchors (the 3 pairs of GPT-2's three sentences)
        'adjacent': consecutive anchors in their order, cyclically (A|B, B|C, C|D, D|A on the ring)
    """
    labels = [int(x) for x in anchor_labels]
    K = len(labels)
    if mode == 'all':
        candidates = [(i, j) for i in range(K) for j in range(i + 1, K)]
    elif mode == 'adjacent':
        candidates = [(k, (k + 1) % K) for k in range(K if K > 2 else 1)]
    else:
        raise ValueError(f"Unknown pair mode {mode!r}; use 'all' or 'adjacent'")
    pairs, seen = [], set()
    for i, j in candidates:
        key = tuple(sorted((labels[i], labels[j])))
        if labels[i] != labels[j] and key not in seen:
            seen.add(key)
            pairs.append((i, j))
    return pairs


def boundary_edges(axis_u: np.ndarray, axis_v: np.ndarray, labels: np.ndarray, strength: np.ndarray):
    """
    Edges between 4-neighbor grid points with different argmax classes.
    labels, strength [R_v, R_u] (grid point (r, c) = (axis_u[c], axis_v[r])).
    Returns pairs [E, 2] (sorted class ids), midpoints [E, 2] in (u, v), strength [E] (mean of the two points).
    """
    pairs, points, values = [], [], []
    uu, vv = np.meshgrid(axis_u, axis_v)
    for sl_a, sl_b in [((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))]:
        la, lb = labels[sl_a], labels[sl_b]
        mask = la != lb
        pairs.append(np.stack([np.minimum(la, lb)[mask], np.maximum(la, lb)[mask]], axis=1))
        points.append(np.stack([(uu[sl_a] + uu[sl_b])[mask] / 2, (vv[sl_a] + vv[sl_b])[mask] / 2], axis=1))
        values.append(((strength[sl_a] + strength[sl_b]) / 2)[mask])
    return np.concatenate(pairs), np.concatenate(points), np.concatenate(values)


def region_rim(axis_u: np.ndarray, axis_v: np.ndarray, labels: np.ndarray, label: int) -> np.ndarray:
    """(u, v) [M, 2] of the grid points of class label with a 4-neighbor of another class (or on the grid edge)."""
    inside = labels == label
    padded = np.pad(inside, 1, constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    rows, cols = np.nonzero(inside & ~interior)
    return np.stack([axis_u[cols], axis_v[rows]], axis=1)


def boundary_windows(axis_u: np.ndarray, axis_v: np.ndarray, labels: np.ndarray, strength: np.ndarray,
                     anchor_pairs: List[tuple], anchor_labels, half_width: float,
                     center_mode: str = 'median') -> List[Dict]:
    """
    One square zoom window per anchor pair (i, j) (from anchor_class_pairs), on the decision boundary between the two
    anchors' argmax classes a = y_i and b = y_j in the coarse grid, centered on one edge of that boundary:
        'median': the edge nearest the coordinate-wise median of the boundary's edges (its middle; away from the
                  junctions where several boundaries meet)
        'peak':   the edge of largest strength (e.g. the max-norm gradient magnitude), often such a junction
    When the two regions share no boundary on the grid (other classes lie between them), the window is centered on
    the midpoint of their closest points instead ('shared_boundary' False).
    half_width is a fraction of the coarse grid's larger span; windows are shifted inward to stay inside the grid.
    Returns dicts with 'anchors' (i, j), 'pair' (a, b), 'shared_boundary', 'center' (u, v), 'bounds' (u0, u1, v0, v1).
    """
    edge_pairs, points, values = boundary_edges(axis_u, axis_v, labels, strength)
    half = float(half_width * max(axis_u[-1] - axis_u[0], axis_v[-1] - axis_v[0]))
    windows = []
    for i, j in anchor_pairs:
        a, b = int(anchor_labels[i]), int(anchor_labels[j])
        if not ((labels == a).any() and (labels == b).any()):
            print(f"  No zoom window for anchors {i}|{j}: class {a if not (labels == a).any() else b} is not on the grid")
            continue
        on_pair = np.flatnonzero((edge_pairs == (min(a, b), max(a, b))).all(axis=1))
        shared = len(on_pair) > 0
        if not shared:
            rim_a, rim_b = region_rim(axis_u, axis_v, labels, a), region_rim(axis_u, axis_v, labels, b)
            dist = np.linalg.norm(rim_a[:, None, :] - rim_b[None, :, :], axis=-1)
            k_a, k_b = np.unravel_index(np.argmin(dist), dist.shape)
            center = (rim_a[k_a] + rim_b[k_b]) / 2
        elif center_mode == 'peak':
            center = points[on_pair[np.argmax(values[on_pair])]]
        elif center_mode == 'median':
            median = np.median(points[on_pair], axis=0)
            center = points[on_pair[np.argmin(np.linalg.norm(points[on_pair] - median, axis=1))]]
        else:
            raise ValueError(f"Unknown center_mode {center_mode!r}; use 'median' or 'peak'")
        u = float(np.clip(center[0], axis_u[0] + half, axis_u[-1] - half))
        v = float(np.clip(center[1], axis_v[0] + half, axis_v[-1] - half))
        windows.append({'anchors': (int(i), int(j)), 'pair': (a, b), 'shared_boundary': bool(shared),
                        'center': (u, v), 'bounds': (u - half, u + half, v - half, v + half)})
    return windows


def window_gradient_fields(logits_fn: Callable, to_ambient: Callable, coords: torch.Tensor, anchor_labels: torch.Tensor,
                           batch_size: int) -> Dict[str, torch.Tensor]:
    """
    Gradient fields of pooled_gradient_fields and the argmax class ('labels' [B]) on the points of one zoom window,
    in batches; tensors on the CPU. logits_fn: coords [B, 2] -> logits [B, V]; to_ambient: coords -> activations.
    """
    keys = ['anchor_vectors', 'anchor_norms', 'maxnorm_vectors', 'maxnorm_norms', 'maxnorm_tokens', 'singular_vectors',
            'singular_sigma', 'labels']
    results = {key: [] for key in keys}
    for start in range(0, coords.shape[0], batch_size):
        c = coords[start:start + batch_size]
        logits, logit_jac = jvp_columns(logits_fn, c)
        logits, logit_jac = logits.detach(), logit_jac.detach()
        metric = surface_metric(to_ambient, c)
        fields = pooled_gradient_fields(logits, logit_jac, metric, anchor_labels)
        fields['labels'] = logits.argmax(dim=-1)
        for key in keys:
            value = fields[key]
            results[key].append(value.float().cpu() if value.is_floating_point() else value.cpu())
        del logits, logit_jac, fields
        if c.is_cuda:
            torch.cuda.empty_cache()
    point_dim = {'anchor_vectors': 1, 'anchor_norms': 1}
    return {key: torch.cat(vals, dim=point_dim.get(key, 0)) for key, vals in results.items()}


def fill_zoom_windows(logits_fn: Callable, to_ambient: Callable, windows: List[Dict], resolution: int,
                      anchor_labels: torch.Tensor, batch_size: int, device=None) -> List[Dict]:
    """
    Recompute the gradient fields of every window (from boundary_windows) on a resolution x resolution grid over its
    bounds. Adds 'axis_u', 'axis_v' [r], the fields of window_gradient_fields reshaped to the grid ([r, r, ...]) and,
    for the window's two anchors i, j, 'pair_vectors' [2, r, r, 2] / 'pair_norms' [2, r, r]: the gradients of
    ln P(y_i) and ln P(y_j) (rows i and j of the anchor fields), as numpy arrays.
    """
    for w in windows:
        u0, u1, v0, v1 = w['bounds']
        axis_u = torch.linspace(u0, u1, resolution, dtype=torch.float64)
        axis_v = torch.linspace(v0, v1, resolution, dtype=torch.float64)
        coords = grid_coords(axis_u, axis_v)
        if device is not None:
            coords = coords.to(device)
        fields = window_gradient_fields(logits_fn, to_ambient, coords, anchor_labels, batch_size)
        w['axis_u'], w['axis_v'] = axis_u.numpy(), axis_v.numpy()
        for key, value in fields.items():
            value = value.numpy()
            if key.startswith('anchor_'):
                w[key] = value.reshape(value.shape[0], resolution, resolution, *value.shape[2:])
            else:
                w[key] = value.reshape(resolution, resolution, *value.shape[1:])
        anchors = list(w['anchors'])
        w['pair_vectors'], w['pair_norms'] = w['anchor_vectors'][anchors], w['anchor_norms'][anchors]
    return windows


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
