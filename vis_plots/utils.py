#!/usr/bin/env python3
"""
Utility functions for the interpolation analyses of the toy ResNet.
"""

import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from typing import List, Dict, Union

import sys
sys.path.append('./common')
from checkpoints import find_seed_checkpoints, artifact_subdir, plot_dir
from plot_style import layer_label, point_label, seeds_label


def linear_interpolation(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Simple linear interpolation without norm rescaling."""
    return (1 - t) * v0 + t * v1


def format_pair_ids_for_subdirectory(pairs_ids: List[List[str]]) -> str:
    """Format multiple pairs into a subdirectory path.

    Args:
        pairs_ids: List of pairs, each pair is a list of 2 strings.

    Returns:
        String like "pairs/a-b_c-d" (subdirectory-safe)
    """
    pair_strs = []
    for pair in pairs_ids:
        a, b = pair[0], pair[1]
        # Make safe for directory names
        a_clean = str(a).replace(" ", "_").replace(".", "_").replace("/", "_")
        b_clean = str(b).replace(" ", "_").replace(".", "_").replace("/", "_")
        pair_strs.append(f"{a_clean}-{b_clean}")
    return "pairs/" + "_".join(pair_strs)


def construct_filepath(checkpoint_path: str, interpolation_layer: int, pair_ids: List[str], n_steps: int) -> str:
    """Activation file for one (checkpoint, pair, interpolation layer), mirroring the checkpoint layout."""
    filename = f"interpolate_layer{interpolation_layer}_[{'_'.join(pair_ids)}]_{n_steps}steps.pt"
    return os.path.join("activations", artifact_subdir(checkpoint_path), filename)


def load_activations(checkpoint_path: str, interpolation_layer: int, pair_ids: List[str], n_steps: int) -> Dict:
    filepath = construct_filepath(checkpoint_path, interpolation_layer, pair_ids, n_steps)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Activations file not found: {filepath}")
    return torch.load(filepath, map_location='cpu')


def load_config(config_path: str = "./vis_plots/config.yaml") -> Dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_pairs(config: Dict) -> List[List[List[float]]]:
    return config['class_spiral']['pairs']


def get_pairs_ids(config: Dict) -> List[List[str]]:
    """String ids of each input pair, used in file names and plot titles."""
    return [[f'{point}' for point in pair] for pair in get_pairs(config)]


def get_checkpoint_dir(config: Dict, override: str = None) -> str:
    return override or config['checkpoint_dir']


def get_seed_checkpoints(config: Dict, override: str = None) -> List[str]:
    """Final-epoch checkpoint of every seed in the configured run directory."""
    checkpoints = find_seed_checkpoints(get_checkpoint_dir(config, override))
    print(f"{len(checkpoints)} seed checkpoint(s) | Steps: {config['n_steps']}")
    return checkpoints


def pairs_plot_dir(checkpoint_dir: str, pairs_ids: List[List[str]]) -> str:
    out = os.path.join(plot_dir(checkpoint_dir), format_pair_ids_for_subdirectory(pairs_ids))
    os.makedirs(out, exist_ok=True)
    return out


def get_n_layers(activations: Dict) -> int:
    """Get number of layers from activation dictionary keys."""
    max_layer = max([int(k.split('_')[0].replace('layer', ''))
                     for k in activations.keys() if k.startswith('layer')])
    return max_layer + 1


def compute_relative_distances(activations: torch.Tensor) -> List[float]:
    """Compute relative distances from endpoint A to endpoint B.

    Args:
        activations: Tensor of shape [n_steps, ...] where ... can be any dimensions

    Returns:
        List of relative distances for each step
    """
    # Flatten to [n_steps, -1]
    activations_flat = activations.view(activations.shape[0], -1)

    endpoint_a = activations_flat[0]
    endpoint_b = activations_flat[-1]

    relative_distances = []
    for step_idx in range(activations_flat.shape[0]):
        current = activations_flat[step_idx]
        dist_to_a = torch.norm(current - endpoint_a).item()
        dist_to_b = torch.norm(current - endpoint_b).item()
        if dist_to_a + dist_to_b == 0:
            result = 0.0  # Avoid division by zero; both endpoints are the same
        else:  
            result = dist_to_a / (dist_to_a + dist_to_b)
        relative_distances.append(result)

    return relative_distances


def aggregate_metric_data(
    all_seeds_data: List[Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]],
    confidence_level: float = 0.95
) -> tuple:
    """Aggregate plot_data dicts across seeds into mean and std dicts.

    Handles both multi-layer format {pair: {layer: tensor}} and
    single-line format {pair: tensor}.
    
        Args:
        all_seeds_data: List of plot_data dicts from different seeds, each with the same structure.
        confidence_level: Confidence level for standard deviation calculation (default: 0.95).

    Returns:
        (mean_dict, std_dict) with the same structure as each input dict.
    """
    mean_dict = {}
    std_dict = {}
    # Use first seed as reference for keys
    ref = all_seeds_data[0]
    for pair_key in ref:
        sample = ref[pair_key]
        if isinstance(sample, dict):
            # Multi-layer: {layer_key: tensor}
            mean_layers = {}
            std_layers = {}
            for layer_key in sample:
                stacked = torch.stack([seed[pair_key][layer_key] for seed in all_seeds_data])
                mean_layers[layer_key] = stacked.mean(dim=0)
                n = stacked.shape[0]
                stds = stacked.std(dim=0) if n > 1 else torch.zeros_like(stacked[0])
                std_layers[layer_key] = stds * confidence_level / np.sqrt(n)
            mean_dict[pair_key] = mean_layers
            std_dict[pair_key] = std_layers
        else:
            # Single tensor
            stacked = torch.stack([seed[pair_key] for seed in all_seeds_data])
            mean_dict[pair_key] = stacked.mean(dim=0)
            n = stacked.shape[0]
            stds = stacked.std(dim=0) if n > 1 else torch.zeros_like(stacked[0])
            std_dict[pair_key] = stds * confidence_level / np.sqrt(n)
    return mean_dict, std_dict


def generate_interpolation_results_plot(
    data_dict: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
    suptitle: str,
    ylabel: str,
    output_path: str,
    n_steps: int,
    pairs_ids: List[List[str]],
    alpha_range: List[float] = [0, 1],
    skip_interpolation_layer: bool = True,
    std_dict: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
    y_floor: float = None,
    n_seeds: int = None,
) -> None:
    """
    Generate standardized interpolation results plot with two subplots.

    Args:
        data_dict: Dictionary with token pair keys (e.g., "very_quite", "very_white") mapping to either:
                   - Dict with layer keys (for multi-layer plots)
                   - Single tensor (for single-line plots)
        suptitle: Overall title for the plot
        ylabel: Y-axis label
        output_path: Path to save the plot
        n_steps: Number of interpolation steps
        pairs_ids: List of input pairs (one subplot each)
        alpha_range: Range for alpha values (default: [0, 1])
        skip_interpolation_layer: Whether to skip layer 0 (interpolation layer) in multi-layer plots
        n_seeds: Number of seeds averaged over (appended to the title)
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    y_min, y_max = float('inf'), float('-inf')
    has_multi_layer_data = False

    for i, pair_ids in enumerate(pairs_ids):
        ax = axes[i]
        token_key = f"{pair_ids[0]}_{pair_ids[1]}"
        data = data_dict[token_key]

        is_multi_layer = isinstance(data, dict)

        if is_multi_layer:
            # Filter layers, skipping layer 0 if requested
            valid_layers = [(k, v) for k, v in data.items()
                           if not (skip_interpolation_layer and 'Layer 0' in k)]

            has_multi_layer_data = True

            # Include every 5th layer in legend
            legend_indices = set(range(0, len(valid_layers), 5))

            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_layers)))

            for j, (layer_key, layer_data) in enumerate(valid_layers):
                # Extract layer number from key (e.g., "Layer 5" -> 5)
                label = None
                if j in legend_indices:
                    parts = layer_key.split()
                    is_single_layer = len(parts) == 2 and parts[0] == 'Layer' and parts[1].lstrip('-').isdigit()
                    label = layer_label(parts[1]) if is_single_layer else layer_key

                ax.plot(alphas, layer_data, alpha=0.8, color=colors[j], linewidth=1, label=label)
                if std_dict is not None:
                    std_data_layer = std_dict[token_key][layer_key]
                    lower = layer_data - std_data_layer
                    upper = layer_data + std_data_layer
                    if y_floor is not None:
                        lower = torch.clamp(lower, min=y_floor)
                    ax.fill_between(alphas, lower, upper,
                                    alpha=0.15, color=colors[j])
                    y_min = min(y_min, lower.min().item())
                    y_max = max(y_max, upper.max().item())
                else:
                    y_min = min(y_min, layer_data.min().item())
                    y_max = max(y_max, layer_data.max().item())
        else:
            # Single-line plot
            ax.plot(alphas, data, color='#2C2C2C', linewidth=2, alpha=0.9)
            if std_dict is not None:
                std_data_single = std_dict[token_key]
                lower = data - std_data_single
                upper = data + std_data_single
                if y_floor is not None:
                    lower = torch.clamp(lower, min=y_floor)
                ax.fill_between(alphas, lower, upper,
                                alpha=0.15, color='#2C2C2C')
                y_min = min(y_min, lower.min().item())
                y_max = max(y_max, upper.max().item())
            else:
                y_min = min(y_min, data.min().item())
                y_max = max(y_max, data.max().item())

        # Reference lines and labels
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        for x, name in [(0, 'A'), (1, 'B')]:
            ax.text(x, 1.01, name, ha='center', va='bottom', color='gray', transform=ax.get_xaxis_transform())

        # Formatting
        point_a, point_b = (point_label(ast.literal_eval(p)) for p in pair_ids)
        ax.set_title(f'A {point_a} → B {point_b}', pad=24)
        ax.set_xlabel('Interpolation α')
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    # Shared y-limits
    if not (np.isnan(y_min) or np.isnan(y_max) or np.isinf(y_min) or np.isinf(y_max) or y_min == float('inf') or y_max == float('-inf')):
        y_padding = (y_max - y_min) * 0.05
        for ax in axes:
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Legend for multi-layer plots
    if has_multi_layer_data:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc='upper left')

    if n_seeds is not None:
        suptitle += f' ({seeds_label(n_seeds)})'
    plt.suptitle(suptitle)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()

