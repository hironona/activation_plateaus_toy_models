#!/usr/bin/env python3
"""
Step Sizes Analysis Script

This script generates step size plots (L2 norm of differences between consecutive steps),
averaged over seed checkpoints.
"""

import argparse
import torch
import sys
from typing import Dict
sys.path.append('./vis_plots')
from utils import load_config, load_activations, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir, layer_label

config = load_config()
N_STEPS = config['n_steps']


def compute_step_sizes(activations: Dict[str, torch.Tensor], hook_name: str) -> Dict[str, torch.Tensor]:
    """Compute step sizes (L2 norm of differences between consecutive steps) for each layer."""
    step_sizes = {}

    for key, layer_activations in activations.items():
        if key.startswith('layer') and key.endswith(f'_{hook_name}'):
            step_diffs = layer_activations[1:] - layer_activations[:-1]  # [n_steps-1, hidden_dim]
            step_norms = torch.norm(step_diffs, p=2, dim=1)  # [n_steps-1]

            layer_idx = int(key.split('_')[0].replace('layer', ''))
            step_sizes[f"Layer {layer_idx}"] = step_norms
    return step_sizes


def compute_step_sizes_for_model(checkpoint_path, pairs_ids, layer_to_interpolate, n_steps):
    """Compute step sizes plot_data for a single model checkpoint."""
    plot_data = {}
    for pair_ids in pairs_ids:
        activations = load_activations(checkpoint_path, layer_to_interpolate, pair_ids, n_steps)
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        plot_data[pair_name] = compute_step_sizes(activations, 'resid_post')
    return plot_data


def main():
    parser = argparse.ArgumentParser(description='Plot step sizes along the interpolation path')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)
    layer_to_interpolate = config['layer_to_interpolate']

    all_seeds_data = [
        compute_step_sizes_for_model(ckpt, pairs_ids, layer_to_interpolate, N_STEPS) for ckpt in checkpoints
    ]
    mean_data, std_data = aggregate_metric_data(all_seeds_data)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_data,
        suptitle=f"Step size, interpolated at {layer_label(layer_to_interpolate)}",
        ylabel="Step size (L2)",
        output_path=f"{output_dir}/step_sizes_resid_post_layer{layer_to_interpolate}_interpolation.png",
        n_steps=N_STEPS - 1,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        std_dict=std_data,
        y_floor=0,
        n_seeds=len(checkpoints),
    )


if __name__ == "__main__":
    main()
