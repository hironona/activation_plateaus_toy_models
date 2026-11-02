#!/usr/bin/env python3
"""
Shared helpers for the plateau visualizations of the toy ResNet.
"""

import yaml
from typing import Dict, Union

import sys
sys.path.append('./common')
from checkpoints import find_seed_checkpoints, load_model_from_checkpoint, plot_dir

# Named layers accepted for source_layer_idx / target_layer_idx in the configs
LAYER_NAME_TO_IDX = {
    'input': -2,
    'embed': -1,
    'embedding': -1,
    'logits': 'logits',
}


def load_config(config_path: str) -> Dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_layer_idx(raw_value) -> Union[int, str]:
    """-2 / 'input', -1 / 'embed', k (output of block k), or 'logits'."""
    try:
        return int(raw_value)
    except (ValueError, TypeError):
        return LAYER_NAME_TO_IDX[raw_value]
