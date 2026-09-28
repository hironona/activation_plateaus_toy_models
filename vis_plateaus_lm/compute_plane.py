#!/usr/bin/env python3
"""
Sample last-token activations on the plane spanned by three sentences' last-token activations in GPT-2's residual
stream (output space of a source layer), label each point with its argmax next token, and compute the Frobenius
norm of the layerwise Jacobian product from the source layer to the logits. Saves everything to a .pt file that
plot_plane.py reads (no model needed for plotting).

Run from the project root (GPU recommended):
    python vis_plateaus_lm/compute_plane.py [--source_layer_idx 6]
"""

import os
import sys
import argparse
import importlib.util
import torch
from tqdm import tqdm

sys.path.append('./vis_plateaus_lm')
from utils import (load_config, get_anchor_words, resid_hook_name, build_plane_basis, project_to_plane,
                   plane_to_ambient, make_plane_grid, grid_coords, construct_filepath)
from compute_metrics import (forward_from_layer, unembed_gram_factor, jacobian_norm_layerwise_prod_to_logits,
                             jacobian_norm_direct_to_logits)

# vis_plots/utils.py has the model loaders; import it under a distinct name to avoid clashing with ./utils.py
_spec = importlib.util.spec_from_file_location("vis_plots_utils", "./vis_plots/utils.py")
vis_plots_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vis_plots_utils)


def tokenize_sentences(model, prefix: str, words, prepend_bos: bool) -> torch.Tensor:
    """Tokens of prefix + ' ' + word for each word. Returns [n_words, seq_len]."""
    prefix_tokens = model.to_tokens(prefix, prepend_bos=prepend_bos)  # [1, n_ctx]
    rows = []
    for word in words:
        word_tokens = model.to_tokens(" " + word, prepend_bos=False)
        assert word_tokens.shape[1] == 1, f"' {word}' is not a single token: {model.to_str_tokens(word_tokens)}"
        rows.append(torch.cat([prefix_tokens, word_tokens], dim=1))
    return torch.cat(rows, dim=0)


def collect_anchor_activations(model, tokens: torch.Tensor, source_layer_idx: int):
    """
    Run the three sentences and collect (a) the last-token activations in the output space of the source layer and
    (b) the shared context activations at every layer from the source layer on.
    Returns:
        anchors: [3, d], contexts: {k: [n_ctx, d]}, logits: [3, d_vocab] (last position)
    """
    n_layers = model.cfg.n_layers
    hook_names = {k: resid_hook_name(k) for k in range(source_layer_idx, n_layers)}
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_names.values())

    anchors = cache[hook_names[source_layer_idx]][:, -1, :].clone()
    contexts = {}
    for k in range(source_layer_idx, n_layers - 1):
        ctx = cache[hook_names[k]][:, :-1, :]
        # The context positions cannot see the last token (causal mask), so they must agree across sentences
        assert torch.allclose(ctx, ctx[:1].expand_as(ctx), atol=1e-4), f"Context activations differ across sentences at layer {k}"
        contexts[k] = ctx[0].clone()

    # Validate the last-token forward pass against the full model
    with torch.no_grad():
        out = forward_from_layer(model, anchors, source_layer_idx, contexts)
    for step, k in enumerate(range(source_layer_idx, n_layers)):
        recorded = cache[hook_names[k]][:, -1, :]
        diff = (out['resids'][step] - recorded).abs().max().item()
        assert diff < 1e-3 * max(1.0, recorded.abs().max().item()), f"Forward validation failed at layer {k}: max diff {diff}"
    diff = (out['logits'] - logits[:, -1, :]).abs().max().item()
    assert diff < 1e-3 * max(1.0, logits.abs().max().item()), f"Forward validation failed at logits: max diff {diff}"

    return anchors, contexts, logits[:, -1, :]


def compute_plane(model, tokens: torch.Tensor, words, config, source_layer_idx: int, device: str) -> dict:
    """
    Plane sampling, argmax labeling and Jacobian norms for three tokenized sentences [3, seq_len].
    Returns the dict saved by main().
    """
    n_layers = model.cfg.n_layers
    assert -1 <= source_layer_idx < n_layers, f"source_layer_idx must be in [-1, {n_layers - 1}], got {source_layer_idx}"

    # 1. Last-token activations of the three sentences in the output space of the source layer
    anchors, contexts, anchor_logits_full = collect_anchor_activations(model, tokens, source_layer_idx)

    # 2. Plane through the three activations, and a grid of samples on it
    origin, basis = build_plane_basis(anchors)
    anchor_coords = project_to_plane(anchors, origin, basis)
    reconstruction_error = (plane_to_ambient(anchor_coords, origin, basis) - anchors.double()).norm(dim=1).max().item()
    assert reconstruction_error < 1e-6 * anchors.norm(dim=1).max().item(), f"Anchors are not on the plane: {reconstruction_error}"

    axis_u, axis_v = make_plane_grid(anchor_coords.cpu(), config['grid_resolution'], config['margin'])
    coords = grid_coords(axis_u, axis_v).to(device)  # [N, 2]
    samples = plane_to_ambient(coords, origin, basis).float()  # [N, d]
    print(f"Anchor distances: |{words[0]}-{words[1]}| = {anchor_coords[1].norm():.2f}, |{words[0]}-{words[2]}| = {anchor_coords[2].norm():.2f}")
    print(f"Grid: {config['grid_resolution']}x{config['grid_resolution']} = {samples.shape[0]} points, side {axis_u[-1] - axis_u[0]:.2f}")

    # 3. Label each sample with its argmax next token and compute the Jacobian product norm
    gram_factor = unembed_gram_factor(model).to(device)
    chunk_size = config['jacobian_chunk_size']

    def compute(points):
        results = {'metric_values': [], 'layer_norms': [], 'cumulative_norms': [], 'labels': [], 'label_probs': [], 'logit_margins': []}
        for start in tqdm(range(0, points.shape[0], config['batch_size']), desc="Jacobians", leave=False):
            out = jacobian_norm_layerwise_prod_to_logits(model, points[start:start + config['batch_size']], source_layer_idx, contexts, gram_factor, chunk_size)
            probs = torch.softmax(out['logits'].float(), dim=-1)
            top2 = out['logits'].topk(2, dim=-1)
            results['metric_values'].append(out['metric_values'].cpu())
            results['layer_norms'].append(out['layer_norms'].cpu())
            results['cumulative_norms'].append(out['cumulative_norms'].cpu())
            results['labels'].append(top2.indices[:, 0].cpu())
            results['label_probs'].append(probs.gather(1, top2.indices[:, :1])[:, 0].cpu())
            results['logit_margins'].append((top2.values[:, 0] - top2.values[:, 1]).cpu())
            del out, probs, top2
            if device == 'cuda':
                torch.cuda.empty_cache()
        return {key: torch.cat(vals, dim=-1) for key, vals in results.items()}

    anchor_results = compute(anchors)
    assert torch.equal(anchor_results['labels'], anchor_logits_full.argmax(dim=-1).cpu()), "Anchor labels differ from the full model"
    direct = jacobian_norm_direct_to_logits(model, anchors, source_layer_idx, contexts, gram_factor, chunk_size).cpu()
    rel_diff = ((direct - anchor_results['metric_values']).abs() / direct).max().item()
    assert rel_diff < 1e-3, f"Layerwise Jacobian product disagrees with the direct Jacobian at the anchors: rel diff {rel_diff}"

    sample_results = compute(samples)

    label_ids = torch.unique(torch.cat([sample_results['labels'], anchor_results['labels']])).tolist()
    label_strings = {i: model.to_string([i]) for i in label_ids}
    for word, label, norm in zip(words, anchor_results['labels'].tolist(), anchor_results['metric_values'].tolist()):
        print(f"  '... {word}' -> {label_strings[label]!r} | ||J||_F = {norm:.3e}")
    print(f"Unique argmax tokens on the grid: {len(torch.unique(sample_results['labels']))}")

    return {
        'config': config,
        'words': words,
        'tokens': tokens.cpu(),
        'source_layer_idx': source_layer_idx,
        'hook_name': resid_hook_name(source_layer_idx),
        'n_layers': n_layers,
        # Plane
        'origin': origin.cpu(),  # [d]
        'basis': basis.cpu(),  # [2, d]
        'axis_u': axis_u,  # [R]
        'axis_v': axis_v,  # [R]; grid point (r, c) = (axis_u[c], axis_v[r]), flattened row-major
        # Anchors (the three sentences)
        'anchor_activations': anchors.cpu(),  # [3, d]
        'anchor_coords': anchor_coords.cpu(),  # [3, 2]
        'anchor_labels': anchor_results['labels'],  # [3]
        'anchor_label_probs': anchor_results['label_probs'],
        'anchor_jacobian_norms': anchor_results['metric_values'],
        # Samples
        'sample_activations': samples.cpu(),  # [N, d]
        'sample_coords': project_to_plane(samples, origin, basis).cpu(),  # [N, 2]
        'labels': sample_results['labels'],  # [N] argmax next-token ids
        'label_probs': sample_results['label_probs'],  # [N] probability of the argmax token
        'logit_margins': sample_results['logit_margins'],  # [N] top-1 minus top-2 logit
        'label_strings': label_strings,  # {token id: decoded string}
        'jacobian_norms': sample_results['metric_values'],  # [N] ||J_{source -> logits}||_F
        'layer_jacobian_norms': sample_results['layer_norms'],  # [n_blocks, N] per-block ||J_k||_F
        'cumulative_jacobian_norms': sample_results['cumulative_norms'],  # [n_blocks, N] ||J_k ... J_{i+1}||_F
    }


def main():
    parser = argparse.ArgumentParser(description='Sample and label activations on the plane through three last-token activations')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source_layer_idx is not None:
        config['source_layer_idx'] = args.source_layer_idx
    source_layer_idx = config['source_layer_idx']
    words = get_anchor_words(config['token_pairs'])

    model = vis_plots_utils.load_model(config['model_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Same device choice as the loader
    model = model.to(device)
    model.eval()
    print(f"Model: {config['model_name']} | Source layer: {source_layer_idx} ({resid_hook_name(source_layer_idx)}) | Device: {device}")

    tokens = tokenize_sentences(model, config['prefix'], words, config['prepend_bos']).to(device)
    for row in tokens:
        print(f"  {model.to_str_tokens(row)}")

    save_dict = compute_plane(model, tokens, words, config, source_layer_idx, device)

    save_path = construct_filepath(config, source_layer_idx)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
