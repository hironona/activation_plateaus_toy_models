#!/usr/bin/env python3
"""
Logits Relative Distances Analysis Script

Generates the logits relative distance plot, averaged over seed checkpoints.
"""

import argparse
import torch
import sys
sys.path.append('./vis_plots')
from utils import load_activations, load_config, compute_relative_distances, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir

config = load_config()
N_STEPS = config['n_steps']


def compute_logits_data(checkpoint_path, pairs_ids, layer_to_interpolate, n_steps):
    """Compute logits relative distances plot_data for a single model checkpoint."""
    plot_data = {}
    for pair_ids in pairs_ids:
        logits = load_activations(checkpoint_path, layer_to_interpolate, pair_ids, n_steps)['logits']
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        plot_data[pair_name] = torch.tensor(compute_relative_distances(logits))
    return plot_data


def main():
    parser = argparse.ArgumentParser(description='Plot relative distances of the logits along the interpolation path')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)
    layer_to_interpolate = config['layer_to_interpolate']

    all_seeds_data = [compute_logits_data(ckpt, pairs_ids, layer_to_interpolate, N_STEPS) for ckpt in checkpoints]
    mean_data, std_data = aggregate_metric_data(all_seeds_data)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_data,
        suptitle=f"Logits Relative Distances, Interpolated at Layer {layer_to_interpolate}",
        ylabel="Relative Distance to A (0) vs B (1)",
        output_path=f"{output_dir}/relative_distances_logits_layer{layer_to_interpolate}_interpolation.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=False,
        std_dict=std_data
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
