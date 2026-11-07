#!/usr/bin/env python3
"""
Plot the results of compute_attractors.py (no model needed):
    1. Gradient of each anchor's argmax log-probability on the surface (one panel per reference anchor)
    2. Max-norm row of the log-probability Jacobian, and which token it belongs to
    3. Top singular direction of the log-probability Jacobian (sign-free segments)
    4. Residual-stream trajectories through the blocks in anchor-relative coordinates, over the basins of the final
       argmax token; one panel per block with its update; median convergence curves
The heat maps show magnitudes (log10) and the arrows only directions; the argmax-token boundaries and the anchors are
overlaid as in plot_plane.py.

Run from the project root:
    python vis_plateaus_lm/plot_attractors.py [--source_layer_idx 6] [--surface spherical] [--input path/to/file.pt]
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append('./vis_plateaus_lm')
sys.path.append('./common')
from utils import load_config, construct_filepath, construct_plot_dir, SURFACES
from plot_plane import suptitle, surface_name
from plot_style import layer_label
import attractor_plots as ap


def axis_labels(data):
    words = data['words']
    if surface_name(data) == 'flat':
        return f"e₁  (toward “{words[1]}”)", "e₂"
    return f"e₁  (toward “{words[1]}”) · geodesic angle, rad", "e₂ · geodesic angle, rad"


def main():
    parser = argparse.ArgumentParser(description='Plot gradient- and block-based attractor fields')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--surface', type=str, choices=SURFACES, help='Override surface from config (selects the .pt file)')
    parser.add_argument('--grid_resolution', type=int, help='Override grid_resolution from config (selects the .pt file)')
    parser.add_argument('--input', type=str, help='Path to a .pt file from compute_attractors.py (overrides config lookup)')
    args = parser.parse_args()

    config = load_config(args.config)
    for key in ('source_layer_idx', 'surface', 'grid_resolution'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    input_path = args.input or construct_filepath(config, config['source_layer_idx'], kind="attractors")
    data = torch.load(input_path, weights_only=True)
    print(f"Loaded: {input_path}")

    R = data['axis_u'].shape[0]
    labels = data['labels'].numpy()
    ids, counts = np.unique(labels, return_counts=True)
    shown = ids[np.argsort(-counts)][:config['max_label_classes']].tolist()
    # Non-ASCII tokens (often byte fragments of CJK characters among the max-norm rows) are escaped for the plot font
    label_text = {i: repr(s) if s.isascii() else ascii(s) for i, s in data['label_strings'].items()}
    xlabel, ylabel = axis_labels(data)
    subtitle = suptitle(data)

    fields = {
        'axis_u': data['axis_u'].numpy(), 'axis_v': data['axis_v'].numpy(),
        'xlabel': xlabel, 'ylabel': ylabel, 'subtitle': subtitle,
        'labels': labels.reshape(R, R), 'label_text': label_text, 'shown_labels': shown,
        'anchor_coords': data['anchor_coords'].numpy(),
        'anchor_names': [f"“… {w}”" for w in data['words']],
        'anchor_labels': data['anchor_labels'].tolist(),
        'anchor_vectors': data['anchor_vectors'].numpy().reshape(3, R, R, 2),
        'anchor_norms': data['anchor_norms'].numpy().reshape(3, R, R),
        'maxnorm_vectors': data['maxnorm_vectors'].numpy().reshape(R, R, 2),
        'maxnorm_norms': data['maxnorm_norms'].numpy().reshape(R, R),
        'maxnorm_tokens': data['maxnorm_tokens'].numpy().reshape(R, R),
        'singular_vectors': data['singular_vectors'].numpy().reshape(R, R, 2),
        'singular_sigma': data['singular_sigma'].numpy().reshape(R, R),
    }
    traj = {
        'layer_names': [layer_label(k) for k in data['layers']],
        'frame_names': [f"“… {w}”" for w in data['words']],
        'grid_shape': (R, R), 'subtitle': subtitle,
        'coords': data['coords'].numpy(), 'off_plane': data['off_plane'].numpy(),
        'nearest_dist': data['nearest_dist'].numpy(),
        'final_labels': labels, 'label_text': label_text, 'shown_labels': shown,
    }

    stem = os.path.splitext(os.path.basename(input_path))[0]
    plot_dir = construct_plot_dir(data['config'])
    os.makedirs(plot_dir, exist_ok=True)
    arrow_stride, traj_stride = config['arrow_stride'], config['trajectory_stride']

    ap.plot_anchor_gradients(fields, os.path.join(plot_dir, f"{stem}_grad_anchor_logprob.png"), arrow_stride)
    ap.plot_maxnorm_gradient(fields, os.path.join(plot_dir, f"{stem}_grad_maxnorm_row.png"), arrow_stride,
                             config['max_label_classes'])
    ap.plot_singular_direction(fields, os.path.join(plot_dir, f"{stem}_grad_top_singular.png"), arrow_stride)
    ap.plot_trajectories(traj, os.path.join(plot_dir, f"{stem}_trajectories.png"), traj_stride)
    ap.plot_block_updates(traj, os.path.join(plot_dir, f"{stem}_block_updates.png"), traj_stride)

    def quartiles(values):
        return np.median(values, axis=1), np.percentile(values, 25, axis=1), np.percentile(values, 75, axis=1)
    series = {
        "Distance to the nearest anchor (/ mean anchor distance)": [("median, IQR", *quartiles(traj['nearest_dist']))],
        "Distance from the anchors' plane (/ mean anchor distance)": [("median, IQR", *quartiles(traj['off_plane']))],
    }
    ap.plot_convergence(traj['layer_names'], series, os.path.join(plot_dir, f"{stem}_convergence.png"), subtitle)


if __name__ == "__main__":
    main()
