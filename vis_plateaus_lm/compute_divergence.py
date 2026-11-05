#!/usr/bin/env python3
"""
Compare the next-token distribution at each grid point on the sampling surface (same grid as compute_plane.py) with
the next-token distribution at each of the three anchor activations. With P_a the distribution at anchor a and P_x
the distribution at grid point x, both in nats:
    1. KL divergence KL(P_a || P_x) = sum_y P_a(y) (ln P_a(y) - ln P_x(y))
    2. Surprisal of x's argmax token under the anchor, -ln P_a(y_x) with y_x = argmax_y P_x(y)
    3. Surprisal of the anchor's argmax token at x, -ln P_x(y_a) with y_a = argmax_y P_a(y)
Forward passes only (no Jacobians). Saves to a separate .pt file (suffix "_divergence") that plot_divergence.py reads.

Run from the project root:
    python vis_plateaus_lm/compute_divergence.py [--source_layer_idx 6] [--surface spherical]
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.append('./vis_plateaus_lm')
from utils import load_config, get_anchor_words, resid_hook_name, construct_filepath, SURFACES
from compute_metrics import load_model, forward_from_layer
from compute_plane import tokenize_sentences, collect_anchor_activations, sample_surface_grid


def divergence_from_anchors(anchor_log_probs: torch.Tensor, logits: torch.Tensor) -> dict:
    """
    Args:
        anchor_log_probs: [3, d_vocab] float64 log next-token probabilities at the anchors
        logits: [B, d_vocab] next-token logits at the points
    Returns:
        Dict with 'kl' [3, B] KL(P_a || P_x), 'surprisal' [3, B] -ln P_a(argmax P_x),
        'anchor_token_surprisal' [3, B] -ln P_x(argmax P_a), 'labels' [B], 'label_probs' [B]
    """
    log_probs = torch.log_softmax(logits.double(), dim=-1)  # [B, V]
    anchor_probs = anchor_log_probs.exp()
    # KL(P_a || P_x) = -H(P_a) - CE(P_a, P_x), as one matmul over the vocabulary
    neg_entropy = (anchor_probs * anchor_log_probs).sum(dim=-1, keepdim=True)  # [3, 1]
    kl = neg_entropy - anchor_probs @ log_probs.T  # [3, B]
    labels = log_probs.argmax(dim=-1)  # [B]
    anchor_labels = anchor_log_probs.argmax(dim=-1)  # [3]
    return {
        'kl': kl,
        'surprisal': -anchor_log_probs[:, labels],  # [3, B]
        'anchor_token_surprisal': -log_probs[:, anchor_labels].T,  # [3, B]
        'labels': labels,
        'label_probs': log_probs.gather(1, labels.unsqueeze(1))[:, 0].exp(),
    }


def compute_divergence(model, tokens: torch.Tensor, words, config, source_layer_idx: int, device: str) -> dict:
    """KL divergence and argmax surprisal w.r.t. each anchor for three tokenized sentences [3, seq_len]."""
    n_layers = model.cfg.n_layers
    assert -1 <= source_layer_idx < n_layers, f"source_layer_idx must be in [-1, {n_layers - 1}], got {source_layer_idx}"

    # 1. Anchor activations and their next-token distributions (from the full model)
    anchors, contexts, anchor_logits = collect_anchor_activations(model, tokens, source_layer_idx)
    anchor_log_probs = torch.log_softmax(anchor_logits.double(), dim=-1)  # [3, V]

    # 2. Same surface and grid as compute_plane.py
    grid = sample_surface_grid(anchors, words, config, device)
    samples = grid['samples']

    # 3. Next-token distributions at the points, compared with each anchor
    def compute(points):
        results = {'kl': [], 'surprisal': [], 'anchor_token_surprisal': [], 'labels': [], 'label_probs': []}
        batch_size = config['divergence_batch_size']
        for start in tqdm(range(0, points.shape[0], batch_size), desc="Divergences", leave=False):
            with torch.no_grad():
                logits = forward_from_layer(model, points[start:start + batch_size], source_layer_idx, contexts)['logits']
            out = divergence_from_anchors(anchor_log_probs, logits)
            for key in results:
                results[key].append(out[key].cpu())
        return {key: torch.cat(vals, dim=-1) for key, vals in results.items()}

    anchor_results = compute(anchors)  # [3 (reference anchor), 3 (evaluated anchor)]
    self_kl = anchor_results['kl'].diagonal().abs().max().item()
    assert self_kl < 1e-4, f"KL of an anchor's distribution with itself is {self_kl}, not ~0"
    assert torch.equal(anchor_results['labels'], anchor_logits.argmax(dim=-1).cpu()), "Anchor labels differ from the full model"
    # Both surprisals reduce to -ln max P_a at the reference anchor itself
    self_diff = (anchor_results['surprisal'].diagonal() - anchor_results['anchor_token_surprisal'].diagonal()).abs().max().item()
    assert self_diff < 1e-4, f"The two surprisals disagree at the reference anchors: {self_diff}"

    sample_results = compute(samples)

    label_ids = torch.unique(torch.cat([sample_results['labels'], anchor_results['labels']])).tolist()
    label_strings = {i: model.to_string([i]) for i in label_ids}
    for a, word in enumerate(words):
        others = ", ".join(f"{words[b]}: KL {anchor_results['kl'][a, b]:.3f}, surprisal {anchor_results['surprisal'][a, b]:.3f}, "
                           f"anchor-token surprisal {anchor_results['anchor_token_surprisal'][a, b]:.3f}"
                           for b in range(len(words)) if b != a)
        print(f"  vs '... {word}' -> {label_strings[anchor_results['labels'][a].item()]!r} | {others}")
    print(f"KL range on the grid: [{sample_results['kl'].min():.3e}, {sample_results['kl'].max():.3e}] nats | "
          f"surprisal range: [{sample_results['surprisal'].min():.3e}, {sample_results['surprisal'].max():.3e}] nats | "
          f"anchor-token surprisal range: [{sample_results['anchor_token_surprisal'].min():.3e}, {sample_results['anchor_token_surprisal'].max():.3e}] nats")

    surface = grid['surface']
    return {
        'config': config,
        'words': words,
        'tokens': tokens.cpu(),
        'source_layer_idx': source_layer_idx,
        'hook_name': resid_hook_name(source_layer_idx),
        'n_layers': n_layers,
        # Surface parameters (see compute_plane.py)
        'surface': grid['surface_name'],
        **surface.params(),
        'axis_u': grid['axis_u'],  # [R]
        'axis_v': grid['axis_v'],  # [R]; grid point (r, c) = (axis_u[c], axis_v[r]), flattened row-major
        # Anchors (the three sentences)
        'anchor_coords': grid['anchor_coords'].cpu(),  # [3, 2]
        'anchor_labels': anchor_results['labels'],  # [3]
        'anchor_label_probs': anchor_results['label_probs'],  # [3]
        'anchor_kl': anchor_results['kl'],  # [3, 3] KL(P_a || P_b), a = reference anchor (row), b = evaluated anchor
        'anchor_surprisal': anchor_results['surprisal'],  # [3, 3] -ln P_a(argmax P_b)
        'anchor_anchor_token_surprisal': anchor_results['anchor_token_surprisal'],  # [3, 3] -ln P_b(argmax P_a)
        # Samples
        'labels': sample_results['labels'],  # [N] argmax next-token ids
        'label_probs': sample_results['label_probs'],  # [N] probability of the argmax token
        'label_strings': label_strings,  # {token id: decoded string}
        'kl': sample_results['kl'],  # [3, N] KL(P_a || P_x) in nats, one row per anchor a
        'surprisal': sample_results['surprisal'],  # [3, N] -ln P_a(argmax P_x) in nats, one row per anchor a
        'anchor_token_surprisal': sample_results['anchor_token_surprisal'],  # [3, N] -ln P_x(argmax P_a) in nats, one row per anchor a
    }


def main():
    parser = argparse.ArgumentParser(description='KL divergence and argmax surprisal w.r.t. the anchors on the sampling surface')
    parser.add_argument('--config', type=str, default='./vis_plateaus_lm/config.yaml', help='Path to config file')
    parser.add_argument('--source_layer_idx', type=int, help='Override source_layer_idx from config')
    parser.add_argument('--surface', type=str, choices=SURFACES, help='Override surface from config')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source_layer_idx is not None:
        config['source_layer_idx'] = args.source_layer_idx
    if args.surface is not None:
        config['surface'] = args.surface
    source_layer_idx = config['source_layer_idx']
    words = get_anchor_words(config['token_pairs'])

    model = load_model(config['model_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Same device choice as the loader
    model = model.to(device)
    model.eval()
    print(f"Model: {config['model_name']} | Source layer: {source_layer_idx} ({resid_hook_name(source_layer_idx)}) | Surface: {config.get('surface', 'flat')} | Device: {device}")

    tokens = tokenize_sentences(model, config['prefix'], words, config['prepend_bos']).to(device)
    for row in tokens:
        print(f"  {model.to_str_tokens(row)}")

    save_dict = compute_divergence(model, tokens, words, config, source_layer_idx, device)

    save_path = construct_filepath(config, source_layer_idx, kind="divergence")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
