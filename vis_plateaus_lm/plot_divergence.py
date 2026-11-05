#!/usr/bin/env python3
"""
Plot the results of compute_divergence.py (no model needed). One figure per metric, with one panel per reference
anchor a (sharing a color scale), each showing on the surface (plane or slerp-like sphere):
    1. KL divergence KL(P_a || P_x) between the next-token distributions at anchor a and at grid point x.
    2. Surprisal -ln P_a(y_x) of x's argmax next token y_x under anchor a's distribution.
    3. Surprisal -ln P_x(y_a) of anchor a's argmax next token y_a under x's distribution.
The argmax-token decision boundaries and the three anchor activations are overlaid as in plot_plane.py; the
reference anchor of each panel is circled.

Run from the project root:
    python vis_plateaus_lm/plot_divergence.py [--source_layer_idx 6] [--surface spherical] [--input path/to/file.pt]
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('./vis_plateaus_lm')
sys.path.append('./common')
from utils import load_config, construct_filepath, construct_plot_dir, SURFACES
from plot_plane import (_grid, pixel_extent, draw_decision_boundaries, draw_region_labels, draw_anchors, style_axes,
                        suptitle, surface_noun, SEQUENTIAL_BLUE, CATEGORICAL, TEXT_PRIMARY, TEXT_SECONDARY, SURFACE)

REFERENCE_COLOR = CATEGORICAL[1]  # Orange ring around the reference anchor, distinct from the blue ramp


def draw_reference_anchor(ax, anchor_coords, idx):
    u, v = anchor_coords[idx]
    ax.scatter([u], [v], s=320, marker='o', facecolor='none', edgecolor=REFERENCE_COLOR, linewidth=2.5, zorder=6)


def plot_anchor_panels(data, metric_grids, save_path, cbar_label, title, extend='neither', labels_to_annotate=()):
    """
    Args:
        metric_grids: [3, R, R] values to color by, one grid per reference anchor (already transformed, e.g. log10)
    """
    axis_u, axis_v = data['axis_u'].numpy(), data['axis_v'].numpy()
    label_grid = _grid(data['labels'], axis_u.shape[0])
    anchor_coords = data['anchor_coords'].numpy()
    words = data['words']
    vmin, vmax = np.nanmin(metric_grids), np.nanmax(metric_grids)

    fig, axes = plt.subplots(1, len(words), figsize=(7.5 * len(words), 7), facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for idx, ax in enumerate(axes):
        im = ax.imshow(metric_grids[idx], origin='lower', extent=pixel_extent(axis_u, axis_v), cmap=SEQUENTIAL_BLUE,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        draw_decision_boundaries(ax, axis_u, axis_v, label_grid)
        draw_region_labels(ax, axis_u, axis_v, label_grid, data['label_strings'], labels_to_annotate)
        draw_anchors(ax, anchor_coords, words)
        draw_reference_anchor(ax, anchor_coords, idx)
        style_axes(ax, data)
        if idx > 0:
            ax.set_ylabel("")
        anchor_token = data['label_strings'][data['anchor_labels'][idx].item()]
        ax.set_title(f"reference “… {words[idx]}”  (argmax {anchor_token!r})", color=TEXT_PRIMARY)

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8, pad=0.02, extend=extend)
    cbar.set_label(cbar_label, color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    fig.suptitle(f"{title}\n{suptitle(data)}", color=TEXT_SECONDARY)
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def kl_grids(data, log_scale: bool, floor: float):
    """[3, R, R] KL grids, log10 of the values clipped at floor if log_scale. Returns (grids, clipped anywhere)."""
    resolution = data['axis_u'].shape[0]
    kl = np.stack([_grid(row, resolution) for row in data['kl']])
    if not log_scale:
        return kl, False
    return np.log10(np.maximum(kl, floor)), bool((kl < floor).any())


def metric_grids(data, key: str):
    """[3, N] metric -> [3, R, R] grids"""
    resolution = data['axis_u'].shape[0]
    return np.stack([_grid(row, resolution) for row in data[key]])


def main():
    parser = argparse.ArgumentParser(description='Plot KL divergence and argmax surprisal w.r.t. each anchor on the sampling surface')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--surface', type=str, choices=SURFACES, help='Override surface from config (selects the .pt file)')
    parser.add_argument('--input', type=str, help='Path to a .pt file from compute_divergence.py (overrides config lookup)')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source_layer_idx is not None:
        config['source_layer_idx'] = args.source_layer_idx
    if args.surface is not None:
        config['surface'] = args.surface
    input_path = args.input or construct_filepath(config, config['source_layer_idx'], kind="divergence")
    data = torch.load(input_path, weights_only=True)
    print(f"Loaded: {input_path}")

    # Most frequent argmax tokens get region annotations, as in plot_plane.py
    labels, counts = torch.unique(data['labels'], return_counts=True)
    shown_labels = labels[torch.argsort(counts, descending=True)][:config['max_label_classes']].tolist()

    stem = os.path.splitext(os.path.basename(input_path))[0]
    plot_dir = construct_plot_dir(data['config'])
    os.makedirs(plot_dir, exist_ok=True)
    noun = surface_noun(data)

    log_scale = config['log_scale']
    floor = config['kl_floor']
    grids, clipped = kl_grids(data, log_scale, floor)
    kl_label = r"KL$(P_a \,\|\, P_x)$, nats"
    if log_scale:
        kl_label = r"$\log_{10}$ " + kl_label + (f"  (clipped at {floor:g})" if clipped else "")
    kl_path = os.path.join(plot_dir, f"{stem}_kl_heatmap.png")
    plot_anchor_panels(data, grids, kl_path, kl_label,
                       f"KL divergence from the reference anchor's next-token distribution on the {noun}",
                       extend='min' if clipped else 'neither', labels_to_annotate=shown_labels)
    print(f"Saved: {kl_path}")

    surprisal_path = os.path.join(plot_dir, f"{stem}_surprisal_heatmap.png")
    plot_anchor_panels(data, metric_grids(data, 'surprisal'), surprisal_path, r"$-\ln P_a(\hat{y}_x)$, nats",
                       rf"Surprisal of each point's argmax token $\hat{{y}}_x$ under the reference anchor on the {noun}",
                       labels_to_annotate=shown_labels)
    print(f"Saved: {surprisal_path}")

    anchor_token_path = os.path.join(plot_dir, f"{stem}_anchor_token_surprisal_heatmap.png")
    plot_anchor_panels(data, metric_grids(data, 'anchor_token_surprisal'), anchor_token_path, r"$-\ln P_x(\hat{y}_a)$, nats",
                       rf"Surprisal of the reference anchor's argmax token $\hat{{y}}_a$ at each point on the {noun}",
                       labels_to_annotate=shown_labels)
    print(f"Saved: {anchor_token_path}")


if __name__ == "__main__":
    main()
