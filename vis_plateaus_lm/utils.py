#!/usr/bin/env python3
"""
Shared utilities for the GPT-2 plane visualization pipeline (no model loading here).
"""

import os
import re
import sys
import yaml
import torch
from typing import Dict, List, Tuple

sys.path.append('./common')
# Surface geometry lives in common/surfaces.py (also used by the toy pipeline); re-exported here
from surfaces import (build_plane_basis, project_to_plane, plane_to_ambient, make_plane_grid, grid_coords, SURFACES,
                      FlatSurface, SphericalSurface, build_surface)


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


def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def construct_filepath(config: Dict, source_layer_idx: int, kind: str = "") -> str:
    """
    Path of the saved activations/metrics for one source layer. Flat keeps the original (suffix-free) name.
    kind: "" for compute_plane.py's output, or a tag such as "divergence" for other scripts' outputs on the same grid.
    """
    words = get_anchor_words(config['token_pairs'])
    surface_suffix = "" if config.get('surface', 'flat') == 'flat' else f"_{config['surface']}"
    kind_suffix = f"_{kind}" if kind else ""
    filename = f"{_slug(config['prefix'])}__{'_'.join(_slug(w) for w in words)}__layer{source_layer_idx}_res{config['grid_resolution']}{surface_suffix}{kind_suffix}.pt"
    return os.path.join("./activations", config['model_name'], "lm_plane", filename)


def construct_plot_dir(config: Dict) -> str:
    return os.path.join("./plots", config['model_name'], "lm_plane")
