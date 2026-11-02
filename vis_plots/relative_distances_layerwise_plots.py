#!/usr/bin/env python3
"""
Layerwise Relative Distances Analysis Script

Generates three types of relative distance plots:
1. Interpolate in layer 0, record in all subsequent layers
2. Interpolate in each layer, record in the last layer only
3. Interpolate in layer i, record in layer i+N for various N values
"""

import argparse
import torch
import sys
sys.path.append('./vis_plots')
from utils import load_activations, load_config, get_n_layers, compute_relative_distances, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir

config = load_config()
N_STEPS = config['n_steps']


def variant1_compute(pairs_ids, checkpoint_path, layer_to_interpolate):
    """Variant 1: Compute data - Interpolate in layer 0, record in all subsequent layers."""
    plot_data = {}
    for pair_ids in pairs_ids:
        activations = load_activations(checkpoint_path, layer_to_interpolate, pair_ids, N_STEPS)
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_dict = {}
        for key, layer_activations in activations.items():
            if key.startswith('layer') and key.endswith('_resid_post'):
                layer_idx = int(key.split('_')[0].replace('layer', ''))
                layer_dict[f"Layer {layer_idx}"] = torch.tensor(compute_relative_distances(layer_activations))
        plot_data[pair_name] = layer_dict
    return plot_data


def variant1_plot(plot_data, pairs_ids, output_dir, layer_to_interpolate, std_dict=None):
    """Variant 1: Plot - Interpolate in layer 0, record in all subsequent layers."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Relative Distances (Interpolate in Layer {layer_to_interpolate})",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layerwise_layer{layer_to_interpolate}_interpolation.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=True,
        std_dict=std_dict
    )


def variant2_compute(n_layers, pairs_ids, checkpoint_path):
    """Variant 2: Compute data - Interpolate in each layer, record in last layer only."""
    last_layer_key = f'layer{n_layers - 1}_resid_post'
    plot_data = {}
    for pair_idx, pair_ids in enumerate(pairs_ids):
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_dict = {}
        for interpolation_layer in range(n_layers - 1):
            activations = load_activations(checkpoint_path, interpolation_layer, pair_ids, N_STEPS)
            distances = compute_relative_distances(activations[last_layer_key])
            layer_dict[f"Layer {interpolation_layer}"] = torch.tensor(distances)
        plot_data[pair_name] = layer_dict
    return plot_data


def variant2_plot(plot_data, pairs_ids, output_dir, std_dict=None):
    """Variant 2: Plot - Interpolate in each layer, record in last layer only."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Relative Distances (Record in Last Layer)",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layerwise_last_layer_recording.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=True,
        std_dict=std_dict
    )


def variant3_compute(n_layers, pairs_ids, checkpoint_path, N):
    """Variant 3: Compute data - Interpolate in layer i, record in layer i+N."""
    plot_data = {}
    for pair_ids in pairs_ids:
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_dict = {}
        for interpolation_layer in range(n_layers - N):
            activations = load_activations(checkpoint_path, interpolation_layer, pair_ids, N_STEPS)
            target_layer_key = f'layer{interpolation_layer + N}_resid_post'
            distances = compute_relative_distances(activations[target_layer_key])
            layer_dict[f"Layer {interpolation_layer}"] = torch.tensor(distances)
        plot_data[pair_name] = layer_dict
    return plot_data


def variant3_plot(plot_data, pairs_ids, output_dir, N, std_dict=None):
    """Variant 3: Plot - Interpolate in layer i, record in layer i+N."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Relative Distances (N={N})",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layerwise_N{N}.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=True,
        std_dict=std_dict
    )


def single_layer_compute(pairs_ids, checkpoint_path, layer_to_interpolate, target_layer):
    """Compute relative distances for a single target layer only.

    Reuses variant1_compute and filters to target_layer, returning single-tensor format.
    """
    full_data = variant1_compute(pairs_ids, checkpoint_path, layer_to_interpolate)
    target_key = f"Layer {target_layer}"
    filtered = {}
    for pair_name, layer_dict in full_data.items():
        if target_key not in layer_dict:
            raise KeyError(f"Layer {target_layer} not found in activations. Available: {list(layer_dict.keys())}")
        filtered[pair_name] = layer_dict[target_key]
    return filtered


def single_layer_plot(plot_data, pairs_ids, output_dir, layer_to_interpolate, target_layer, std_dict=None):
    """Plot relative distances for a single target layer."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Relative Distances — Layer {target_layer} (Interpolate in Layer {layer_to_interpolate})",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layer{target_layer}_interpolate{layer_to_interpolate}.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        std_dict=std_dict
    )


def main():
    parser = argparse.ArgumentParser(description='Plot layerwise relative distances along the interpolation path')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    parser.add_argument('--interpolate_only_first_layer', action='store_true', help='Only plot variant 1 (activations recorded with --interpolate_only_first_layer)')
    parser.add_argument('--single_layer', type=int, default=None, help='Plot relative distances for a single target layer only')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)
    layer_to_interpolate = config['layer_to_interpolate']
    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)

    if args.single_layer is not None:
        print(f"\nSingle layer mode: Layer {args.single_layer}, interpolate in layer {layer_to_interpolate}")
        all_sl = [single_layer_compute(pairs_ids, ckpt, layer_to_interpolate, args.single_layer) for ckpt in checkpoints]
        mean_sl, std_sl = aggregate_metric_data(all_sl)
        single_layer_plot(mean_sl, pairs_ids, output_dir, layer_to_interpolate, args.single_layer, std_dict=std_sl)
        print("\n=== Complete ===")
        return

    # Variant 1
    print(f"\nVariant 1: Interpolate in layer {layer_to_interpolate}, record in all layers")
    all_v1 = [variant1_compute(pairs_ids, ckpt, layer_to_interpolate) for ckpt in checkpoints]
    mean_v1, std_v1 = aggregate_metric_data(all_v1)
    variant1_plot(mean_v1, pairs_ids, output_dir, layer_to_interpolate, std_dict=std_v1)

    if not args.interpolate_only_first_layer:
        n_layers = get_n_layers(load_activations(checkpoints[0], layer_to_interpolate, pairs_ids[0], N_STEPS))

        # Variant 2
        print("\nVariant 2: Interpolate in each layer, record in last layer")
        all_v2 = [variant2_compute(n_layers, pairs_ids, ckpt) for ckpt in checkpoints]
        mean_v2, std_v2 = aggregate_metric_data(all_v2)
        variant2_plot(mean_v2, pairs_ids, output_dir, std_dict=std_v2)

        # Variant 3
        print("\nVariant 3: Interpolate in layer i, record in layer i+N")
        for N in [n for n in [1, 4, 8, 16, 24] if n < n_layers]:
            print(f"  Processing N={N}...")
            all_v3 = [variant3_compute(n_layers, pairs_ids, ckpt, N) for ckpt in checkpoints]
            mean_v3, std_v3 = aggregate_metric_data(all_v3)
            variant3_plot(mean_v3, pairs_ids, output_dir, N, std_dict=std_v3)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
