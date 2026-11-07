#!/usr/bin/env python3
"""
Boundary zooms of the attractor fields (see common/attractors.py): one square window per pair of the three sentences,
on the decision boundary of compute_attractors.py's grid between their argmax tokens, where the gradient fields
(including the gradients of both tokens' log-probabilities) are recomputed on a finer local grid. Reads the "_attractors" .pt file (for the boundaries and the surface) and saves
a "_attractors_zoom" .pt file that plot_attractors.py draws when it exists.

Run from the project root after compute_attractors.py (GPU recommended):
    python vis_plateaus_lm/compute_attractor_zoom.py [--source_layer_idx 6] [--surface spherical] [--grid_resolution 40]
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.append('./vis_plateaus_lm')
sys.path.append('./common')
from utils import load_config, get_anchor_words, construct_filepath, SURFACES
from compute_metrics import load_model, forward_from_layer
from compute_plane import tokenize_sentences, collect_anchor_activations, sample_surface_grid
from attractors import anchor_class_pairs, boundary_windows, fill_zoom_windows


def compute_zoom(model, tokens: torch.Tensor, words, config, coarse: dict, device: str) -> dict:
    source_layer_idx = coarse['source_layer_idx']
    anchors, contexts, anchor_logits = collect_anchor_activations(model, tokens, source_layer_idx)
    anchor_labels = anchor_logits.argmax(dim=-1)
    assert torch.equal(anchor_labels.cpu(), coarse['anchor_labels']), "Anchor argmax tokens differ from the coarse run"
    grid = sample_surface_grid(anchors, words, config, device)
    surface = grid['surface']
    for key in ('axis_u', 'axis_v'):
        err = (grid[key].double() - coarse[key].double()).abs().max().item()
        assert err < 1e-5, f"Surface grid {key} differs from the coarse run by {err}: same config?"

    def logits_fn(c):
        return forward_from_layer(model, surface.to_ambient(c).float(), source_layer_idx, contexts)['logits']

    R = coarse['axis_u'].shape[0]
    labels = anchor_labels.tolist()
    windows = boundary_windows(coarse['axis_u'].numpy(), coarse['axis_v'].numpy(), coarse['labels'].numpy().reshape(R, R),
                               coarse['maxnorm_norms'].numpy().reshape(R, R), anchor_class_pairs(labels, 'all'), labels,
                               config['zoom_half_width'], center_mode=config.get('zoom_center_mode', 'median'))
    for w in tqdm(windows, desc="Zoom windows", leave=False):
        fill_zoom_windows(logits_fn, surface.to_ambient, [w], config['zoom_resolution'], anchor_labels,
                          config['attractor_batch_size'], device=device)
        (i, j), (a, b) = w['anchors'], w['pair']
        print(f"  {words[i]} {model.to_string([a])!r} | {words[j]} {model.to_string([b])!r}: center "
              f"({w['center'][0]:.3f}, {w['center'][1]:.3f}){'' if w['shared_boundary'] else ', no shared boundary'}")

    ids = set()
    for w in windows:
        ids.update(w['labels'].ravel().tolist(), w['maxnorm_tokens'].ravel().tolist(), w['pair'])
    label_strings = {i: model.to_string([i]) for i in sorted(ids)}
    # numpy -> tensors for torch.load(weights_only=True)
    windows = [{key: torch.from_numpy(value) if hasattr(value, 'dtype') else value for key, value in w.items()}
               for w in windows]
    return {'config': config, 'source_layer_idx': source_layer_idx, 'windows': windows, 'label_strings': label_strings}


def main():
    parser = argparse.ArgumentParser(description='Attractor fields on finer grids around the decision boundaries')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--surface', type=str, choices=SURFACES, help='Override surface from config')
    parser.add_argument('--grid_resolution', type=int, help='Override grid_resolution from config (of the coarse grid)')
    args = parser.parse_args()

    config = load_config(args.config)
    for key in ('source_layer_idx', 'surface', 'grid_resolution'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    source_layer_idx = config['source_layer_idx']
    coarse_path = construct_filepath(config, source_layer_idx, kind="attractors")
    coarse = torch.load(coarse_path, weights_only=True)
    print(f"Loaded: {coarse_path}")
    words = get_anchor_words(config['token_pairs'])

    model = load_model(config['model_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Same device choice as the loader
    model = model.to(device)
    model.eval()

    tokens = tokenize_sentences(model, config['prefix'], words, config['prepend_bos']).to(device)
    save_dict = compute_zoom(model, tokens, words, config, coarse, device)

    save_path = construct_filepath(config, source_layer_idx, kind="attractors_zoom")
    torch.save(save_dict, save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
