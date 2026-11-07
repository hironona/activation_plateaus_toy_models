#!/usr/bin/env python3
"""
Plot the results of compute_plane.py (no model needed):
    1. Heat map of the Jacobian product norm (source layer -> logits) on the surface (plane or slerp-like sphere), with
       the approximate decision boundaries of the argmax next token and the three anchor activations.
    2. Categorical map of the argmax next token on the surface.

Run from the project root:
    python vis_plateaus_lm/plot_plane.py [--source_layer_idx 6] [--surface spherical] [--input path/to/file.pt]
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.append('./vis_plateaus_lm')
sys.path.append('./common')
from utils import load_config, construct_filepath, construct_plot_dir, SURFACES
from plot_style import (ANNOTATION_FONTSIZE, SEQUENTIAL_BLUE, CATEGORICAL, OTHER_COLOR, TEXT_PRIMARY,
                        TEXT_SECONDARY, SURFACE)
from plot_maps import pixel_extent, draw_decision_boundaries, draw_region_labels as _draw_region_labels


def _grid(values: torch.Tensor, resolution: int) -> np.ndarray:
    return values.reshape(resolution, resolution).numpy()


def draw_region_labels(ax, axis_u, axis_v, label_grid, label_strings, labels_to_annotate, min_fraction=0.01):
    """Annotate each large region with its argmax token (repr), at the pixel farthest from the region's boundary."""
    label_text = {label: repr(label_strings[label]) for label in labels_to_annotate}
    _draw_region_labels(ax, axis_u, axis_v, label_grid, label_text, labels_to_annotate, min_fraction)


def draw_anchors(ax, anchor_coords, words):
    """Mark the three last-token activations (the input tokens, not predictions)."""
    ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1], s=70, marker='o', facecolor=TEXT_PRIMARY, edgecolor=SURFACE, linewidth=2, zorder=5)
    for (u, v), word in zip(anchor_coords, words):
        ax.annotate(f"“… {word}”", (u, v), xytext=(7, 7), textcoords='offset points', fontsize=ANNOTATION_FONTSIZE + 1, fontweight='bold', color=TEXT_PRIMARY,
                    path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)], zorder=6)


def surface_name(data) -> str:
    return data.get('surface', 'flat')  # Files from before the surface option are flat


def surface_noun(data) -> str:
    return "plane" if surface_name(data) == 'flat' else "spherical surface"


def style_axes(ax, data):
    words = data['words']
    ax.set_aspect('equal')
    if surface_name(data) == 'flat':
        ax.set_xlabel(f"e₁  (toward “{words[1]}”)", color=TEXT_SECONDARY)
        ax.set_ylabel("e₂", color=TEXT_SECONDARY)
    else:
        # Azimuthal equidistant chart: distance from the origin is the geodesic angle from the first anchor's direction
        ax.set_xlabel(f"e₁  (toward “{words[1]}”) · geodesic angle, rad", color=TEXT_SECONDARY)
        ax.set_ylabel("e₂ · geodesic angle, rad", color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(OTHER_COLOR)


def suptitle(data) -> str:
    config = data['config']
    return (f"{config['model_name']} · layer {data['source_layer_idx']} ({data['hook_name']}) · "
            f"“{config['prefix']} ___” · {surface_name(data)} · origin at “{data['words'][0]}”")


def plot_jacobian_heatmap(data, save_path, log_scale=True, labels_to_annotate=()):
    resolution = data['axis_u'].shape[0]
    axis_u, axis_v = data['axis_u'].numpy(), data['axis_v'].numpy()
    norms = _grid(data['jacobian_norms'], resolution)
    label_grid = _grid(data['labels'], resolution)
    values = np.log10(norms) if log_scale else norms

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    im = ax.imshow(values, origin='lower', extent=pixel_extent(axis_u, axis_v), cmap=SEQUENTIAL_BLUE, interpolation='nearest')
    draw_decision_boundaries(ax, axis_u, axis_v, label_grid)
    draw_region_labels(ax, axis_u, axis_v, label_grid, data['label_strings'], labels_to_annotate)
    draw_anchors(ax, data['anchor_coords'].numpy(), data['words'])
    style_axes(ax, data)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label((r"$\log_{10}$ " if log_scale else "") + rf"$\|J\|_F$  (layer {data['source_layer_idx']} → logits)", color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title("Jacobian norm to logits, with argmax-token boundaries", color=TEXT_PRIMARY)
    fig.suptitle(suptitle(data), color=TEXT_SECONDARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_label_map(data, save_path, shown_labels):
    resolution = data['axis_u'].shape[0]
    axis_u, axis_v = data['axis_u'].numpy(), data['axis_v'].numpy()
    label_grid = _grid(data['labels'], resolution)

    # Class index per pixel: shown labels in fixed slot order, everything else -> "other" (last index)
    class_grid = np.full(label_grid.shape, len(shown_labels))
    for idx, label in enumerate(shown_labels):
        class_grid[label_grid == label] = idx
    has_other = (class_grid == len(shown_labels)).any()
    colors = CATEGORICAL[:len(shown_labels)] + [OTHER_COLOR]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    ax.imshow(class_grid, origin='lower', extent=pixel_extent(axis_u, axis_v),
              cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5, interpolation='nearest')
    draw_decision_boundaries(ax, axis_u, axis_v, label_grid, color=SURFACE, outline=SURFACE)
    draw_region_labels(ax, axis_u, axis_v, label_grid, data['label_strings'], shown_labels)
    draw_anchors(ax, data['anchor_coords'].numpy(), data['words'])
    style_axes(ax, data)

    n_total = label_grid.size
    handles = [Patch(facecolor=CATEGORICAL[idx], label=f"{data['label_strings'][label]!r}  ({(label_grid == label).sum() / n_total:.0%})")
               for idx, label in enumerate(shown_labels)]
    if has_other:
        n_other_tokens = len(np.setdiff1d(np.unique(label_grid), shown_labels))
        handles.append(Patch(facecolor=OTHER_COLOR, label=f"other ({n_other_tokens} token{'s' if n_other_tokens != 1 else ''}, {(class_grid == len(shown_labels)).sum() / n_total:.0%})"))
    ax.legend(handles=handles, title="argmax next token", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    ax.set_title(f"Argmax next token on the {surface_noun(data)}", color=TEXT_PRIMARY)
    fig.suptitle(suptitle(data), color=TEXT_SECONDARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot the Jacobian heat map and argmax regions on the sampling surface')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--surface', type=str, choices=SURFACES, help='Override surface from config (selects the .pt file)')
    parser.add_argument('--input', type=str, help='Path to a .pt file from compute_plane.py (overrides config lookup)')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source_layer_idx is not None:
        config['source_layer_idx'] = args.source_layer_idx
    if args.surface is not None:
        config['surface'] = args.surface
    input_path = args.input or construct_filepath(config, config['source_layer_idx'])
    data = torch.load(input_path, weights_only=True)
    print(f"Loaded: {input_path}")

    # Most frequent argmax tokens, shown individually, in fixed categorical slots ordered by area
    labels, counts = torch.unique(data['labels'], return_counts=True)
    shown_labels = labels[torch.argsort(counts, descending=True)][:config['max_label_classes']].tolist()

    stem = os.path.splitext(os.path.basename(input_path))[0]
    plot_dir = construct_plot_dir(data['config'])
    os.makedirs(plot_dir, exist_ok=True)

    heatmap_path = os.path.join(plot_dir, f"{stem}_jacobian_heatmap.png")
    plot_jacobian_heatmap(data, heatmap_path, log_scale=config['log_scale'], labels_to_annotate=shown_labels)
    print(f"Saved: {heatmap_path}")

    label_map_path = os.path.join(plot_dir, f"{stem}_argmax_labels.png")
    plot_label_map(data, label_map_path, shown_labels)
    print(f"Saved: {label_map_path}")


if __name__ == "__main__":
    main()
