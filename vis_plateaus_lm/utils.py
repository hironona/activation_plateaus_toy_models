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


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def construct_filepath(config: Dict, source_layer_idx: int) -> str:
    """Path of the saved activations/metrics for one source layer."""
    words = get_anchor_words(config['token_pairs'])
    filename = f"{_slug(config['prefix'])}__{'_'.join(_slug(w) for w in words)}__layer{source_layer_idx}_res{config['grid_resolution']}.pt"
    return os.path.join("./activations", config['model_name'], "lm_plane", filename)


def construct_plot_dir(config: Dict) -> str:
    return os.path.join("./plots", config['model_name'], "lm_plane")
