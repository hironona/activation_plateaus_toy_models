#!/usr/bin/env python3
"""
Error-correcting attractor fields on the sampling surface (same grid as compute_plane.py) through three sentences'
last-token activations at a source layer of GPT-2 (see common/attractors.py for the definitions):
    1. Gradient-based: the in-surface Jacobian of the next-token log-probabilities (two forward-mode JVPs per point),
       pooled into the gradient of each anchor's argmax log-probability, the max-norm row and the top singular
       direction.
    2. Block-based: the last-token residual stream at every layer from the source on (whole-block updates: attention +
       MLP), in anchor-relative coordinates of the plane through the three sentences' activations at that layer.
Saves a .pt file (suffix "_attractors") that plot_attractors.py reads.

Run from the project root (GPU recommended):
    python vis_plateaus_lm/compute_attractors.py [--source_layer_idx 6] [--surface spherical] [--grid_resolution 40]
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.append('./vis_plateaus_lm')
sys.path.append('./common')
from utils import load_config, get_anchor_words, resid_hook_name, construct_filepath, grid_coords, SURFACES
from compute_metrics import load_model, forward_from_layer
from compute_plane import tokenize_sentences, collect_anchor_activations, sample_surface_grid
from attractors import jvp_columns, surface_metric, pooled_gradient_fields, check_logit_rows, trajectory_coords


def compute_attractors(model, tokens: torch.Tensor, words, config, source_layer_idx: int, device: str) -> dict:
    n_layers = model.cfg.n_layers
    assert -1 <= source_layer_idx < n_layers, f"source_layer_idx must be in [-1, {n_layers - 1}], got {source_layer_idx}"

    anchors, contexts, anchor_logits = collect_anchor_activations(model, tokens, source_layer_idx)
    anchor_labels = anchor_logits.argmax(dim=-1)  # [3]
    grid = sample_surface_grid(anchors, words, config, device)
    surface = grid['surface']
    coords = grid_coords(grid['axis_u'], grid['axis_v']).to(device)  # [N, 2] float64, row-major (v, u)

    def logits_fn(c):
        return forward_from_layer(model, surface.to_ambient(c).float(), source_layer_idx, contexts)['logits']

    # Frame anchors: the three sentences' last-token activations at every layer from the source on
    with torch.no_grad():
        anchor_resids = forward_from_layer(model, anchors, source_layer_idx, contexts)['resids']

    keys = ['anchor_vectors', 'anchor_norms', 'maxnorm_vectors', 'maxnorm_norms', 'maxnorm_tokens', 'singular_vectors',
            'singular_sigma', 'labels', 'coords', 'off_plane', 'nearest_dist']
    results = {key: [] for key in keys}
    batch_size = config['attractor_batch_size']
    for start in tqdm(range(0, coords.shape[0], batch_size), desc="Attractors", leave=False):
        c = coords[start:start + batch_size]
        logits, logit_jac = jvp_columns(logits_fn, c)
        metric = surface_metric(surface.to_ambient, c)
        fields = pooled_gradient_fields(logits.detach(), logit_jac.detach(), metric, anchor_labels)
        if start == 0:
            check_logit_rows(logits_fn, c[:4], torch.cat([anchor_labels, fields['maxnorm_tokens'][:4]]), logit_jac[:4])
        with torch.no_grad():
            out = forward_from_layer(model, surface.to_ambient(c).float(), source_layer_idx, contexts)
        traj = trajectory_coords(out['resids'], anchor_resids)
        fields.update(traj)
        fields['labels'] = logits.argmax(dim=-1)
        for key in keys:
            value = fields[key]
            results[key].append(value.float().cpu() if value.is_floating_point() else value.cpu())
        del logits, logit_jac, fields, out
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Point axis: dim 1 for the per-anchor and per-layer arrays, dim 0 otherwise
    point_dim = {'anchor_vectors': 1, 'anchor_norms': 1, 'coords': 1, 'off_plane': 1, 'nearest_dist': 1}
    results = {key: torch.cat(vals, dim=point_dim.get(key, 0)) for key, vals in results.items()}

    # Anchors in their own frames sit at (0, 0), (1, 0), (0, 1)
    anchor_traj = trajectory_coords(anchor_resids, anchor_resids)
    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, device=anchor_traj['coords'].device)
    frame_err = (anchor_traj['coords'] - expected).abs().max().item()
    assert frame_err < 1e-6, f"Anchors are not at their frame positions: {frame_err}"

    label_ids = torch.unique(torch.cat([results['labels'], results['maxnorm_tokens'], anchor_labels.cpu()])).tolist()
    label_strings = {i: model.to_string([i]) for i in label_ids}
    layers = list(range(source_layer_idx, n_layers))
    print(f"Anchor argmax tokens: {[label_strings[i] for i in anchor_labels.tolist()]}")
    print("Median distance to the nearest anchor / off-plane distance per layer (in mean anchor distances):")
    for k, layer in enumerate(layers):
        print(f"  layer {layer:>2}: {results['nearest_dist'][k].median():.3f} / {results['off_plane'][k].median():.3f}")

    return {
        'config': config,
        'words': words,
        'tokens': tokens.cpu(),
        'source_layer_idx': source_layer_idx,
        'hook_name': resid_hook_name(source_layer_idx),
        'n_layers': n_layers,
        'surface': grid['surface_name'],
        **surface.params(),
        'axis_u': grid['axis_u'],  # [R]
        'axis_v': grid['axis_v'],  # [R]; grid point (r, c) = (axis_u[c], axis_v[r]), flattened row-major
        'anchor_coords': grid['anchor_coords'].cpu(),  # [3, 2]
        'anchor_labels': anchor_labels.cpu(),  # [3]
        'label_strings': label_strings,
        'layers': layers,  # Layer index of each trajectory step (-1 = embedding, k = output of block k)
        **results,
        # labels [N]; anchor_vectors [3, N, 2]; anchor_norms [3, N]; maxnorm_vectors [N, 2]; maxnorm_norms [N];
        # maxnorm_tokens [N]; singular_vectors [N, 2]; singular_sigma [N]; coords [n_layers, N, 2];
        # off_plane [n_layers, N]; nearest_dist [n_layers, N]
    }


def main():
    parser = argparse.ArgumentParser(description='Gradient- and block-based attractor fields on the sampling surface')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--surface', type=str, choices=SURFACES, help='Override surface from config')
    parser.add_argument('--grid_resolution', type=int, help='Override grid_resolution from config')
    args = parser.parse_args()

    config = load_config(args.config)
    for key in ('source_layer_idx', 'surface', 'grid_resolution'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    source_layer_idx = config['source_layer_idx']
    words = get_anchor_words(config['token_pairs'])

    model = load_model(config['model_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Same device choice as the loader
    model = model.to(device)
    model.eval()
    print(f"Model: {config['model_name']} | Source layer: {source_layer_idx} ({resid_hook_name(source_layer_idx)}) | Surface: {config.get('surface', 'flat')} | Device: {device}")

    tokens = tokenize_sentences(model, config['prefix'], words, config['prepend_bos']).to(device)
    save_dict = compute_attractors(model, tokens, words, config, source_layer_idx, device)

    save_path = construct_filepath(config, source_layer_idx, kind="attractors")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
