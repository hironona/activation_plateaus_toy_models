#!/usr/bin/env python3
"""
Shared utilities for the GPT-2 plane visualization pipeline (no model loading here).
"""

import os
import re
import yaml
import torch
from typing import Dict, List, Tuple


def load_config(config_path: str = "./vis_plateaus_lm/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_anchor_words(token_pairs: List[List[str]]) -> List[str]:
    """Unique words across token pairs, in order of appearance. Must be exactly three to span a plane."""
    words = []
    for pair in token_pairs:
        for word in pair:
            if word not in words:
                words.append(word)
    assert len(words) == 3, f"Expected 3 unique words across token_pairs to span a plane, got {words}"
    return words


def resid_hook_name(layer_idx: int) -> str:
    """
    Hook name of the residual stream in the output space of layer layer_idx.
    -1: embedding space (embed + pos_embed), non-negative k: output of block k.
    """
    if layer_idx == -1:
        return 'blocks.0.hook_resid_pre'
    assert layer_idx >= 0, f"Invalid layer index: {layer_idx}"
    return f'blocks.{layer_idx}.hook_resid_post'


def build_plane_basis(anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthonormal basis of the affine plane through three points (Gram-Schmidt).
    The origin is anchors[0] and the first axis points to anchors[1].
    Args:
        anchors: [3, d]
    Returns:
        origin: [d], basis: [2, d]
    """
    anchors = anchors.double()
    origin = anchors[0]
    v1 = anchors[1] - origin
    v2 = anchors[2] - origin
    e1 = v1 / torch.norm(v1)
    w2 = v2 - (v2 @ e1) * e1
    assert torch.norm(w2) > 1e-6 * torch.norm(v2), "The three anchor activations are (nearly) collinear"
    e2 = w2 / torch.norm(w2)
    return origin, torch.stack([e1, e2])


def project_to_plane(points: torch.Tensor, origin: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection of points onto the plane, in plane coordinates.
    Args:
        points: [n, d], origin: [d], basis: [2, d]
    Returns:
        [n, 2]
    """
    return (points.double() - origin) @ basis.T


def plane_to_ambient(coords: torch.Tensor, origin: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Map plane coordinates [n, 2] back to the ambient space [n, d]."""
    return origin + coords.double() @ basis


def make_plane_grid(anchor_coords: torch.Tensor, resolution: int, margin: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Square, evenly spaced grid covering the anchors' bounding box plus a margin.
    Args:
        anchor_coords: [3, 2] plane coordinates of the anchors
        resolution: points per axis
        margin: padding on each side as a fraction of the bounding box's larger side
    Returns:
        axis_u: [resolution], axis_v: [resolution]. Grid point (r, c) is (axis_u[c], axis_v[r]).
    """
    lo = anchor_coords.min(dim=0).values
    hi = anchor_coords.max(dim=0).values
    center = (lo + hi) / 2
    half_side = (hi - lo).max() * (0.5 + margin)
    axis_u = torch.linspace(center[0] - half_side, center[0] + half_side, resolution, dtype=torch.float64)
    axis_v = torch.linspace(center[1] - half_side, center[1] + half_side, resolution, dtype=torch.float64)
    return axis_u, axis_v


def grid_coords(axis_u: torch.Tensor, axis_v: torch.Tensor) -> torch.Tensor:
    """Flatten the grid into [resolution_v * resolution_u, 2] in row-major (v, u) order."""
    vv, uu = torch.meshgrid(axis_v, axis_u, indexing='ij')
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)


SURFACES = ('flat', 'spherical')


class FlatSurface:
    """Affine plane through the three anchors. Coordinates are Euclidean distances in the plane."""

    def __init__(self, anchors: torch.Tensor):
        self.origin, self.basis = build_plane_basis(anchors)

    def to_ambient(self, coords: torch.Tensor) -> torch.Tensor:
        """[n, 2] -> [n, d]"""
        return plane_to_ambient(coords, self.origin, self.basis)

    def to_coords(self, points: torch.Tensor) -> torch.Tensor:
        """[n, d] -> [n, 2] (orthogonal projection onto the plane)"""
        return project_to_plane(points, self.origin, self.basis)

    def validate_grid(self, coords: torch.Tensor):
        pass

    def params(self) -> Dict[str, torch.Tensor]:
        return {'origin': self.origin.cpu(), 'basis': self.basis.cpu()}


class SphericalSurface:
    """
    Curved surface through the three anchors, generalizing vis_plots/utils.py:slerp_rescale from a path to a surface:
    the direction moves along the unit sphere and the norm is interpolated linearly.

    Directions d_k = a_k / ||a_k|| lie on the unit 2-sphere of span(d_0, d_1, d_2). That sphere is charted by the log
    map at d_0 (azimuthal equidistant chart), so a coordinate w = (u, v) is a tangent vector whose length is the
    geodesic angle from d_0 (radians). The norm r(w) is affine in w, fitted so r = ||a_k|| at each anchor. A point is
        x(w) = r(w) * exp_{d_0}(w),   exp_{d_0}(w) = cos|w| d_0 + sin|w| (u e_1 + v e_2) / |w|.
    Along the ray w = t * log_{d_0}(d_k), this is exactly slerp_rescale(a_0, a_k, t).
    """

    def __init__(self, anchors: torch.Tensor):
        anchors = anchors.double()
        norms = torch.norm(anchors, dim=1)
        directions = anchors / norms.unsqueeze(1)
        self.direction_origin = directions[0]

        # Orthonormal basis of the tangent plane at d_0 within span(d_0, d_1, d_2); e_1 points toward d_1
        t1 = directions[1] - (directions[1] @ self.direction_origin) * self.direction_origin
        t2 = directions[2] - (directions[2] @ self.direction_origin) * self.direction_origin
        assert torch.norm(t1) > 1e-6 and torch.norm(t2) > 1e-6, "An anchor is (anti)parallel to the first anchor"
        e1 = t1 / torch.norm(t1)
        w2 = t2 - (t2 @ e1) * e1
        assert torch.norm(w2) > 1e-6 * torch.norm(t2), "The three anchor directions lie on one great circle"
        self.basis = torch.stack([e1, w2 / torch.norm(w2)])

        # Norm is affine in the chart: r(w) = norm_origin + w @ norm_gradient, with r = ||a_k|| at each anchor
        self.norm_origin = norms[0]
        anchor_coords = self._log_map(directions)
        self.norm_gradient = torch.linalg.solve(anchor_coords[1:], norms[1:] - norms[0])

    def _log_map(self, directions: torch.Tensor) -> torch.Tensor:
        """Unit vectors [n, d] -> chart coordinates [n, 2]"""
        cos_angle = (directions @ self.direction_origin).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        tangent = (directions - cos_angle.unsqueeze(1) * self.direction_origin) @ self.basis.T  # [n, 2], length sin(angle)
        tangent_norm = torch.norm(tangent, dim=1, keepdim=True)
        return torch.where(tangent_norm > 1e-12, tangent / tangent_norm.clamp(min=1e-12) * angle.unsqueeze(1), torch.zeros_like(tangent))

    def radius(self, coords: torch.Tensor) -> torch.Tensor:
        """[n, 2] -> [n] interpolated norm"""
        return self.norm_origin + coords.double() @ self.norm_gradient

    def to_ambient(self, coords: torch.Tensor) -> torch.Tensor:
        """[n, 2] -> [n, d]"""
        coords = coords.double()
        angle = torch.norm(coords, dim=1, keepdim=True)
        # sin|w| / |w| without dividing by zero at the origin (torch.sinc(x) = sin(pi x) / (pi x))
        direction = torch.cos(angle) * self.direction_origin + torch.sinc(angle / torch.pi) * (coords @ self.basis)
        return self.radius(coords).unsqueeze(1) * direction

    def to_coords(self, points: torch.Tensor) -> torch.Tensor:
        """[n, d] -> [n, 2] (exact for points on the surface; others are projected onto span(d_0, d_1, d_2) first)"""
        points = points.double()
        return self._log_map(points / torch.norm(points, dim=1, keepdim=True))

    def validate_grid(self, coords: torch.Tensor):
        """The chart is one-to-one only within angle pi of d_0, and the interpolated norm must stay positive."""
        max_angle = torch.norm(coords.double(), dim=1).max().item()
        assert max_angle < torch.pi, f"Grid reaches angle {max_angle:.3f} >= pi from the first anchor; reduce margin"
        min_radius = self.radius(coords).min().item()
        assert min_radius > 0, f"Interpolated norm reaches {min_radius:.3f} <= 0 on the grid; reduce margin"

    def params(self) -> Dict[str, torch.Tensor]:
        return {'direction_origin': self.direction_origin.cpu(), 'basis': self.basis.cpu(),
                'norm_origin': self.norm_origin.cpu(), 'norm_gradient': self.norm_gradient.cpu()}


def build_surface(anchors: torch.Tensor, surface: str):
    """Sampling surface through the three anchors [3, d]: 'flat' (affine plane) or 'spherical' (slerp-like)."""
    if surface == 'flat':
        return FlatSurface(anchors)
    if surface == 'spherical':
        return SphericalSurface(anchors)
    raise ValueError(f"Unknown surface: {surface}. Options: {SURFACES}")


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def construct_filepath(config: Dict, source_layer_idx: int) -> str:
    """Path of the saved activations/metrics for one source layer. Flat keeps the original (suffix-free) name."""
    words = get_anchor_words(config['token_pairs'])
    surface_suffix = "" if config.get('surface', 'flat') == 'flat' else f"_{config['surface']}"
    filename = f"{_slug(config['prefix'])}__{'_'.join(_slug(w) for w in words)}__layer{source_layer_idx}_res{config['grid_resolution']}{surface_suffix}.pt"
    return os.path.join("./activations", config['model_name'], "lm_plane", filename)


def construct_plot_dir(config: Dict) -> str:
    return os.path.join("./plots", config['model_name'], "lm_plane")
