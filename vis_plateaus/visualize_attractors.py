#!/usr/bin/env python3
"""
Error-correcting attractor fields of the toy ResNet (toy counterpart of vis_plateaus_lm/compute_attractors.py +
plot_attractors.py; definitions in common/attractors.py):
    1. Gradient-based: the in-surface Jacobian of the class log-probabilities, pooled into the gradient of each anchor's
       argmax log-probability, the max-norm row and the top singular direction. The source surface is the 2D input
       plane (source_layer_idx -2) or the plane through the frame anchors' activations at a hidden layer.
       Boundary zooms: the fields recomputed on finer local grids in windows on the decision boundaries between
       adjacent anchors' classes (A|B, B|C, ...), with the gradients of both classes' log-probabilities.
       True-class gradient (input-plane source of class_ring_mixture runs): the gradient of the model's log-probability
       of the task's true argmax class at each point, colored by that class.
    2. Block-based: the residual stream of each grid point at every layer from the embedding (or the hidden source
       layer) on, in anchor-relative coordinates of the plane through the frame anchors' activations at that layer.
Maps are drawn per seed (map_seeds); the convergence curves are the mean over all seeds with 95% CIs.

Run from the project root:
    python vis_plateaus/visualize_attractors.py [--checkpoint_dir checkpoints/class_ring_mixture/noise-0.00] [--source_layer_idx -1]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('./vis_plateaus')
sys.path.append('./train')
sys.path.append('./common')
from utils import load_config, find_seed_checkpoints, load_model_from_checkpoint, plot_dir
from plot_style import layer_label, TEXT_SECONDARY
from surfaces import FlatSurface, make_plane_grid, grid_coords
from attractors import (jvp_columns, surface_metric, pooled_gradient_fields, check_logit_rows, trajectory_coords,
                        anchor_class_pairs, boundary_windows, fill_zoom_windows, class_gradient_field)
import attractor_plots as ap
from data import RingMixtureTask

CI_Z = 1.96  # 95% confidence interval of the seed mean


def toy_forward(model, x: torch.Tensor, source_layer_idx: int):
    """
    Forward x from the output space of source_layer_idx (-2 input, -1 embedding, k output of block k).
    Returns (residual stream at every layer from max(source, -1) to the last block, logits [B, C] with a single binary
    logit z expanded to [0, z] so that softmax gives the class probabilities).
    """
    if source_layer_idx == -2:
        x = model.input_layer(x)
    resids = [x]
    for k in range(max(source_layer_idx, -1) + 1, len(model.blocks)):
        x = model.blocks[k](x)
        resids.append(x)
    logits = model.output_layer(F.relu(model.final_norm(x)))
    if logits.shape[1] == 1:
        logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
    return resids, logits


def resolve_anchors(config, task_config):
    """Anchor input points [K, 2] and names."""
    if config['anchor_points'] is not None:
        points = np.asarray(config['anchor_points'], dtype=np.float64)
        names = config['anchor_names'] or [str(i) for i in range(len(points))]
        return points, list(names)
    if task_config.get('name') != 'class_ring_mixture':
        raise ValueError("Set anchor_points in the config for tasks other than class_ring_mixture")
    ring = RingMixtureTask(task_config['num_classes'], task_config.get('ring'))
    return ring.means, ring.names


def true_argmax_fn(task_config, source_layer_idx):
    """Input points [B, 2] -> the task's true argmax class [B] (input-plane source of class_ring_mixture runs only)."""
    if source_layer_idx != -2 or task_config.get('name') != 'class_ring_mixture':
        return None
    ring = RingMixtureTask(task_config['num_classes'], task_config.get('ring'))
    return lambda X: ring.true_probs(X).argmax(axis=1)


def compute_seed(model, config, anchor_points, anchor_names, source_layer_idx, true_argmax=None):
    """
    Gradient fields on the source surface and trajectories for one model. true_argmax (from true_argmax_fn) adds the
    true argmax class ('true_labels') and the model's gradient of its log-probability ('true_vectors', 'true_norms').
    """
    model = model.double()
    n_blocks = len(model.blocks)
    anchors_in = torch.from_numpy(anchor_points)
    frame_idx = [anchor_names.index(name) for name in config['frame_anchors']]
    assert len(frame_idx) == 3, "frame_anchors must name exactly three anchors"

    with torch.no_grad():
        anchor_resids, anchor_logits = toy_forward(model, anchors_in, -2)  # resids from the embedding (layer -1) on
        full = model(anchors_in)
        if full.shape[1] == 1:
            full = torch.cat([torch.zeros_like(full), full], dim=1)
        assert torch.allclose(anchor_logits, full, atol=1e-8), "toy_forward does not reproduce the model's forward pass"
    anchor_labels = anchor_logits.argmax(dim=-1)

    R = config['grid_resolution']
    if source_layer_idx == -2:
        axis = torch.linspace(-config['lim'], config['lim'], R, dtype=torch.float64)
        axis_u = axis_v = axis
        to_ambient = lambda c: c  # noqa: E731
        anchor_coords = anchors_in
    else:
        source_acts = anchor_resids[source_layer_idx + 1]  # [K, d] at the source layer
        surface = FlatSurface(source_acts[frame_idx])
        to_ambient = surface.to_ambient
        anchor_coords = surface.to_coords(source_acts)
        axis_u, axis_v = make_plane_grid(anchor_coords, R, config['margin'])
    coords = grid_coords(axis_u, axis_v)

    def logits_fn(c):
        return toy_forward(model, to_ambient(c), source_layer_idx)[1]

    keys = ['anchor_vectors', 'anchor_norms', 'maxnorm_vectors', 'maxnorm_norms', 'maxnorm_tokens', 'singular_vectors',
            'singular_sigma', 'labels', 'coords', 'off_plane', 'nearest_dist']
    if true_argmax is not None:
        keys += ['true_labels', 'true_vectors', 'true_norms']
    results = {key: [] for key in keys}
    traj_start = max(source_layer_idx, -1)
    frame_resids = [r[frame_idx] for r in anchor_resids[traj_start + 1:]]  # layers traj_start .. last block
    for start in range(0, coords.shape[0], config['batch_size']):
        c = coords[start:start + config['batch_size']]
        logits, logit_jac = jvp_columns(logits_fn, c)
        metric = surface_metric(to_ambient, c)
        fields = pooled_gradient_fields(logits.detach(), logit_jac.detach(), metric, anchor_labels)
        if start == 0:
            check_logit_rows(logits_fn, c[:8], torch.unique(torch.cat([anchor_labels, fields['maxnorm_tokens'][:8]])),
                             logit_jac[:8], rtol=1e-6)
        with torch.no_grad():
            resids, _ = toy_forward(model, to_ambient(c), source_layer_idx)
        fields.update(trajectory_coords(resids, frame_resids))
        fields['labels'] = logits.argmax(dim=-1)
        if true_argmax is not None:
            fields['true_labels'] = torch.from_numpy(true_argmax(to_ambient(c).numpy()))
            fields['true_vectors'], fields['true_norms'] = class_gradient_field(
                logits.detach(), logit_jac.detach(), metric, fields['true_labels'])
        for key in keys:
            results[key].append(fields[key].detach())
    point_dim = {'anchor_vectors': 1, 'anchor_norms': 1, 'coords': 1, 'off_plane': 1, 'nearest_dist': 1}
    results = {key: torch.cat(vals, dim=point_dim.get(key, 0)).numpy() for key, vals in results.items()}
    results.update({
        'axis_u': axis_u.numpy(), 'axis_v': axis_v.numpy(), 'anchor_coords': anchor_coords.numpy(),
        'anchor_labels': anchor_labels.tolist(), 'layers': list(range(traj_start, n_blocks)),
        'n_classes': logits.shape[1],
    })
    return results, (logits_fn, to_ambient)


def compute_zoom(res, chart, config, true_argmax=None):
    """Windows on the boundaries between anchor pairs' classes, with the gradient fields on finer local grids."""
    R = len(res['axis_u'])
    pairs = anchor_class_pairs(res['anchor_labels'], config['zoom_pairs'])
    windows = boundary_windows(res['axis_u'], res['axis_v'], res['labels'].reshape(R, R),
                               res['maxnorm_norms'].reshape(R, R), pairs, res['anchor_labels'],
                               config['zoom_half_width'], center_mode=config.get('zoom_center_mode', 'median'))
    logits_fn, to_ambient = chart
    return fill_zoom_windows(logits_fn, to_ambient, windows, config['zoom_resolution'],
                             torch.tensor(res['anchor_labels']), config['batch_size'], true_argmax=true_argmax)


def density_overlay(task_config, source_layer_idx):
    """Dotted input-density contours on the input plane of class_ring_mixture runs."""
    if source_layer_idx != -2 or task_config.get('name') != 'class_ring_mixture':
        return None
    ring = RingMixtureTask(task_config['num_classes'], task_config.get('ring'))

    def overlay(ax):
        # Called after the panel's limits are set to the grid
        axis_u = np.linspace(*ax.get_xlim(), 200)
        axis_v = np.linspace(*ax.get_ylim(), 200)
        uu, vv = np.meshgrid(axis_u, axis_v)
        density = ring.density(np.stack([uu.ravel(), vv.ravel()], axis=1)).reshape(uu.shape)
        ax.contour(axis_u, axis_v, density, levels=sorted(density.max() * np.array([0.1, 0.01])), colors=TEXT_SECONDARY,
                   linewidths=0.8, linestyles=':', alpha=0.8)
    return overlay


def plot_seed(res, config, anchor_names, source_layer_idx, overlay, out_dir, subtitle, windows=None):
    R = len(res['axis_u'])
    C = res['n_classes']
    label_text = {c: f"class {c}" for c in range(C)}
    shown = list(range(C))
    names = anchor_names
    if source_layer_idx == -2:
        xlabel, ylabel = "x₁", "x₂"
    else:
        f = config['frame_anchors']
        xlabel, ylabel = f"e₁  (toward {f[1]})", "e₂"
    K = len(names)
    fields = {
        'axis_u': res['axis_u'], 'axis_v': res['axis_v'], 'xlabel': xlabel, 'ylabel': ylabel, 'subtitle': subtitle,
        'labels': res['labels'].reshape(R, R), 'label_text': label_text, 'shown_labels': shown,
        'anchor_coords': res['anchor_coords'], 'anchor_names': names, 'anchor_labels': res['anchor_labels'],
        'anchor_vectors': res['anchor_vectors'].reshape(K, R, R, 2), 'anchor_norms': res['anchor_norms'].reshape(K, R, R),
        'maxnorm_vectors': res['maxnorm_vectors'].reshape(R, R, 2), 'maxnorm_norms': res['maxnorm_norms'].reshape(R, R),
        'maxnorm_tokens': res['maxnorm_tokens'].reshape(R, R),
        'maxnorm_shown_labels': shown,
        'singular_vectors': res['singular_vectors'].reshape(R, R, 2), 'singular_sigma': res['singular_sigma'].reshape(R, R),
        'overlay': overlay,
    }
    if 'true_labels' in res:
        fields.update({'true_labels': res['true_labels'].reshape(R, R), 'true_vectors': res['true_vectors'].reshape(R, R, 2),
                       'true_norms': res['true_norms'].reshape(R, R)})
    traj = {
        'layer_names': [layer_label(k) for k in res['layers']], 'frame_names': config['frame_anchors'],
        'grid_shape': (R, R), 'subtitle': subtitle, 'coords': res['coords'], 'off_plane': res['off_plane'],
        'nearest_dist': res['nearest_dist'], 'final_labels': res['labels'], 'label_text': label_text, 'shown_labels': shown,
    }
    os.makedirs(out_dir, exist_ok=True)
    stride, traj_stride = config['arrow_stride'], config['trajectory_stride']
    ap.plot_anchor_gradients(fields, os.path.join(out_dir, "grad_anchor_logprob.png"), stride)
    ap.plot_maxnorm_gradient(fields, os.path.join(out_dir, "grad_maxnorm_row.png"), stride, max_label_classes=C)
    ap.plot_singular_direction(fields, os.path.join(out_dir, "grad_top_singular.png"), stride)
    if 'true_labels' in fields:
        ap.plot_true_class_gradient(fields, os.path.join(out_dir, "grad_true_argmax.png"), stride)
    if windows:
        ap.plot_boundary_zoom(fields, windows, os.path.join(out_dir, "grad_boundary_zoom.png"),
                              config['zoom_arrow_stride'], max_label_classes=C)
    ap.plot_trajectories(traj, os.path.join(out_dir, "trajectories.png"), traj_stride)
    ap.plot_block_updates(traj, os.path.join(out_dir, "block_updates.png"), traj_stride)


def main():
    parser = argparse.ArgumentParser(description='Gradient- and block-based attractor fields of the toy ResNet')
    parser.add_argument('--config', type=str, default='./vis_plateaus/attractor_config.yaml')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir (run directory, seed directory or .pt file)')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx')
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    if args.source_layer_idx is not None:
        config['source_layer_idx'] = args.source_layer_idx
    source_layer_idx = config['source_layer_idx']
    checkpoints = find_seed_checkpoints(checkpoint_dir)
    map_seeds = range(len(checkpoints)) if config['map_seeds'] == 'all' else config['map_seeds']

    out_root = os.path.join(plot_dir(checkpoint_dir), "attractors", f"src-{source_layer_idx}")
    run_label = os.path.relpath(os.path.normpath(checkpoint_dir), 'checkpoints')
    per_seed_curves = {'nearest_dist': [], 'off_plane': []}
    layer_names = None
    for idx, ckpt in enumerate(checkpoints):
        model, _, full_config = load_model_from_checkpoint(ckpt)
        task_config = full_config['task']
        anchor_points, anchor_names = resolve_anchors(config, task_config)
        seed_name = os.path.basename(os.path.dirname(ckpt))
        print(f"Seed {idx + 1}/{len(checkpoints)}: {ckpt}")
        true_argmax = true_argmax_fn(task_config, source_layer_idx)
        res, chart = compute_seed(model, config, anchor_points, anchor_names, source_layer_idx, true_argmax)
        layer_names = [layer_label(k) for k in res['layers']]
        for key in per_seed_curves:
            per_seed_curves[key].append(np.median(res[key], axis=1))
        if idx in map_seeds:
            subtitle = f"{run_label} · {seed_name} · source: {layer_label(source_layer_idx)} · frame {'/'.join(config['frame_anchors'])}"
            windows = compute_zoom(res, chart, config, true_argmax) if config.get('zoom_pairs', 'none') != 'none' else None
            for w in windows or []:
                i, j = w['anchors']
                print(f"  zoom window {anchor_names[i]}|{anchor_names[j]} (classes {w['pair']}): center "
                      f"({w['center'][0]:.3f}, {w['center'][1]:.3f}){'' if w['shared_boundary'] else ', no shared boundary'}")
            plot_seed(res, config, anchor_names, source_layer_idx, density_overlay(task_config, source_layer_idx),
                      os.path.join(out_root, seed_name), subtitle, windows)

    n = len(checkpoints)
    series = {}
    for key, title in [('nearest_dist', "Distance to the nearest frame anchor (/ mean anchor distance)"),
                       ('off_plane', "Distance from the frame anchors' plane (/ mean anchor distance)")]:
        curves = np.stack(per_seed_curves[key])  # [S, n_layers]
        mean = curves.mean(axis=0)
        ci = curves.std(axis=0, ddof=1) * CI_Z / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        series[title] = [(f"median over the grid, mean of {n} seed{'s' if n != 1 else ''} (95% CI)", mean, mean - ci, mean + ci)]
        print(f"{key} per layer (mean of seed medians): {np.round(mean, 3).tolist()}")
    ap.plot_convergence(layer_names, series, os.path.join(out_root, "convergence.png"),
                        f"{run_label} · source: {layer_label(source_layer_idx)} · frame {'/'.join(config['frame_anchors'])}")


if __name__ == "__main__":
    main()
