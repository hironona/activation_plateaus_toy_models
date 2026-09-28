#!/usr/bin/env python3
"""
Plot the results of compute_plane.py (no model needed):
    1. Heat map of the Jacobian product norm (source layer -> logits) on the plane, with the approximate decision
       boundaries of the argmax next token and the three anchor activations.
    2. Categorical map of the argmax next token on the plane.

Run from the project root:
    python vis_plateaus_lm/plot_plane.py [--source_layer_idx 6] [--input path/to/file.pt]
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

sys.path.append('./vis_plateaus_lm')
from utils import load_config, construct_filepath, construct_plot_dir

# Sequential ramp (one hue, light -> dark) and categorical slots in fixed order
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list('seq_blue', [
    '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER_COLOR = '#c9c8c3'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
SURFACE = '#fcfcfb'


def _grid(values: torch.Tensor, resolution: int) -> np.ndarray:
    return values.reshape(resolution, resolution).numpy()


def pixel_extent(axis_u: np.ndarray, axis_v: np.ndarray):
    """imshow extent whose pixel centers sit on the grid points (so contours and fills line up)."""
    du, dv = axis_u[1] - axis_u[0], axis_v[1] - axis_v[0]
    return (axis_u[0] - du / 2, axis_u[-1] + du / 2, axis_v[0] - dv / 2, axis_v[-1] + dv / 2)


def draw_decision_boundaries(ax, axis_u, axis_v, label_grid, color='white', outline=TEXT_PRIMARY):
    """Boundaries between argmax regions: the 0.5 level set of each token's indicator on the grid."""
    for label in np.unique(label_grid):
        mask = (label_grid == label).astype(float)
        if mask.all():
            continue
        cs = ax.contour(axis_u, axis_v, mask, levels=[0.5], colors=color, linewidths=1.2)
        cs.set(path_effects=[pe.Stroke(linewidth=2.6, foreground=outline), pe.Normal()])


def draw_region_labels(ax, axis_u, axis_v, label_grid, label_strings, labels_to_annotate, min_fraction=0.01):
    """Annotate each large region with its argmax token, at the pixel farthest from the region's boundary."""
    for label in labels_to_annotate:
        mask = label_grid == label
        if mask.mean() < min_fraction:
            continue
        dist = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        r, c = np.unravel_index(np.argmax(dist), dist.shape)
        ax.text(axis_u[c], axis_v[r], repr(label_strings[label]), ha='center', va='center', fontsize=9, color=TEXT_PRIMARY,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=SURFACE, edgecolor='none', alpha=0.85))


def draw_anchors(ax, anchor_coords, words):
    """Mark the three last-token activations (the input tokens, not predictions)."""
    ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1], s=70, marker='o', facecolor=TEXT_PRIMARY, edgecolor=SURFACE, linewidth=2, zorder=5)
    for (u, v), word in zip(anchor_coords, words):
        ax.annotate(f"“… {word}”", (u, v), xytext=(7, 7), textcoords='offset points', fontsize=10, fontweight='bold', color=TEXT_PRIMARY,
                    path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)], zorder=6)


def style_axes(ax, words):
    ax.set_aspect('equal')
    ax.set_xlabel(f"e₁  (toward “{words[1]}”)", color=TEXT_SECONDARY)
    ax.set_ylabel("e₂", color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(OTHER_COLOR)


def suptitle(data) -> str:
    config = data['config']
    return (f"{config['model_name']} · layer {data['source_layer_idx']} ({data['hook_name']}) · "
            f"“{config['prefix']} ___” · origin at “{data['words'][0]}”")


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
    style_axes(ax, data['words'])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label((r"$\log_{10}$ " if log_scale else "") + rf"$\|J\|_F$  (layer {data['source_layer_idx']} → logits)", color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title("Jacobian product norm, with argmax next-token boundaries", color=TEXT_PRIMARY, fontsize=12)
    fig.suptitle(suptitle(data), color=TEXT_SECONDARY, fontsize=9)
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
    style_axes(ax, data['words'])

    n_total = label_grid.size
    handles = [Patch(facecolor=CATEGORICAL[idx], label=f"{data['label_strings'][label]!r}  ({(label_grid == label).sum() / n_total:.0%})")
               for idx, label in enumerate(shown_labels)]
    if has_other:
        n_other_tokens = len(np.setdiff1d(np.unique(label_grid), shown_labels))
        handles.append(Patch(facecolor=OTHER_COLOR, label=f"other ({n_other_tokens} tokens, {(class_grid == len(shown_labels)).sum() / n_total:.0%})"))
    ax.legend(handles=handles, title="argmax next token", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False,
              fontsize=9, title_fontsize=9, labelcolor=TEXT_PRIMARY)
    ax.set_title("Argmax next token on the plane", color=TEXT_PRIMARY, fontsize=12)
    fig.suptitle(suptitle(data), color=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot the Jacobian heat map and argmax regions on the plane')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--input', type=str, help='Path to a .pt file from compute_plane.py (overrides config lookup)')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source_layer_idx is not None:
        config['source_layer_idx'] = args.source_layer_idx
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
