#!/usr/bin/env python3
"""
Spline Analysis Script

This script computes Hamming distances of spline codes normalized by step size, averaged over seed checkpoints.
For each block and step, compute Hamming distance between consecutive spline codes (mlp_out > 0),
normalized by the L2 step size in resid_post space.
"""

import argparse
import sys
import torch
from tqdm import tqdm

sys.path.append('./vis_plots')
from utils import load_activations, load_config, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir

config = load_config()
N_STEPS = config['n_steps']


def compute_normalized_hamming_distances(mlp_out_activations, resid_post_activations):
    """Compute Hamming distances of spline codes normalized by resid_post step sizes."""
    spline_codes = [(mlp_out_activations[step] > 0).float() for step in range(mlp_out_activations.shape[0])]
    normalized_distances = []
    for step in range(len(spline_codes) - 1):
        hamming_dist = torch.sum(spline_codes[step] != spline_codes[step + 1]).item()
        step_size = torch.norm(resid_post_activations[step + 1] - resid_post_activations[step]).item()
        normalized_distances.append(hamming_dist / step_size if step_size > 0 else 0.0)
    return normalized_distances


def compute_spline_plot_data(checkpoint_path, pairs_ids, n_steps):
    """Compute spline hamming distance plot_data for a single model checkpoint.

    The spline code of block i is the sign pattern of its MLP output (mlp_out > 0), and the step size is measured
    on the block's resid_post. Uses the activations interpolated at layer 0.
    """
    all_activations = [load_activations(checkpoint_path, 0, pair_ids, n_steps) for pair_ids in pairs_ids]
    n_layers = len([k for k in all_activations[0].keys() if k.startswith('layer') and k.endswith('_mlp_out')])

    plot_data = {}
    for pair_idx, activations in enumerate(all_activations):
        layer_data = {}
        for layer_idx in tqdm(range(n_layers), desc=f"Pair {pair_idx + 1}/{len(all_activations)}"):
            distances = compute_normalized_hamming_distances(
                activations[f'layer{layer_idx}_mlp_out'], activations[f'layer{layer_idx}_resid_post']
            )
            if distances:
                layer_data[f"Layer {layer_idx}"] = torch.tensor(distances)

        pair_name = f"{pairs_ids[pair_idx][0]}_{pairs_ids[pair_idx][1]}"
        plot_data[pair_name] = layer_data
    return plot_data


def main():
    parser = argparse.ArgumentParser(description='Plot normalized Hamming distances of spline codes')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)

    all_seeds_data = [compute_spline_plot_data(ckpt, pairs_ids, N_STEPS) for ckpt in checkpoints]
    mean_data, std_data = aggregate_metric_data(all_seeds_data)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_data,
        suptitle="Normalized Hamming Distances of Spline Codes",
        ylabel="Normalized Hamming Distance",
        output_path=f"{output_dir}/spline_hamming_distances.png",
        n_steps=N_STEPS - 1,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=False,
        std_dict=std_data,
        y_floor=0
    )

if __name__ == "__main__":
    main()
