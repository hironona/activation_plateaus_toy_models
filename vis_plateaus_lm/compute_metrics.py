#!/usr/bin/env python3
"""
Last-token forward passes and Jacobians for a HookedTransformer, starting from the residual stream of a source layer.

Only the last token position is perturbed. Because of the causal mask, the activations of the context positions
(the shared prefix) do not depend on the last token, so they are fixed at every layer. The Jacobian of the last
token's logits w.r.t. its activation at the source layer is therefore exactly the product of the per-layer
last-token Jacobians:
    J = W_U^T · J_ln_final · J_{L-1} · ... · J_{i+1}
"""

import torch
from typing import Callable, Dict, List


def block_last_token(model, block_idx: int, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply block block_idx to the sequence [context, x] and return the last position.
    Args:
        context: [n_ctx, d] residual stream of the context positions at the block's input
        x: [B, d] residual stream of the last position at the block's input
    Returns:
        [B, d] residual stream of the last position at the block's output
    """
    seq = torch.cat([context.unsqueeze(0).expand(x.shape[0], -1, -1), x.unsqueeze(1)], dim=1)
    return model.blocks[block_idx](seq)[:, -1, :]


def ln_final_last_token(model, x: torch.Tensor) -> torch.Tensor:
    """[B, d] -> [B, d]"""
    return model.ln_final(x.unsqueeze(1))[:, 0, :]


def forward_from_layer(model, x: torch.Tensor, source_layer_idx: int, contexts: Dict[int, torch.Tensor]) -> Dict:
    """
    Forward last-token activations from the output space of source_layer_idx to the logits.
    Args:
        x: [B, d] last-token activations in the output space of source_layer_idx
        contexts: {k: [n_ctx, d]} context activations in the output space of layer k, for k in [source_layer_idx, L-2]
    Returns:
        Dict with 'resids': list of [B, d] for layers source_layer_idx..L-1, and 'logits': [B, d_vocab]
    """
    n_layers = model.cfg.n_layers
    resids = [x]
    for k in range(source_layer_idx, n_layers - 1):
        x = block_last_token(model, k + 1, contexts[k], x)
        resids.append(x)
    logits = model.unembed(model.ln_final(x.unsqueeze(1)))[:, 0, :]
    return {'resids': resids, 'logits': logits}


def batched_jacobian(f: Callable, x: torch.Tensor, chunk_size: int = None) -> torch.Tensor:
    """
    Per-sample Jacobians of a function that treats samples independently.
    Args:
        f: [B, d_in] -> [B, d_out], with output b depending only on input b
        x: [B, d_in]
    Returns:
        [B, d_out, d_in]
    """
    # Summing over the batch is exact here because samples do not interact
    jac = torch.func.jacrev(lambda x_: f(x_).sum(0), chunk_size=chunk_size)(x)  # [d_out, B, d_in]
    return jac.permute(1, 0, 2)


def unembed_gram_factor(model) -> torch.Tensor:
    """
    C [d, d] with C^T C = W_U W_U^T, so ||W_U^T M||_F = ||C M||_F for any M [d, d].
    Avoids materializing the [d_vocab, d] unembedding Jacobian per point (b_U does not affect the Jacobian).
    """
    W_U = model.W_U.detach().double()  # [d, d_vocab]
    eigvals, eigvecs = torch.linalg.eigh(W_U @ W_U.T)
    return (eigvecs * eigvals.clamp(min=0).sqrt()).T


def compute_frobenius_norm(jacobians: torch.Tensor) -> torch.Tensor:
    """[..., d_out, d_in] -> [...]"""
    return torch.linalg.matrix_norm(jacobians, ord='fro')


def jacobian_norm_layerwise_prod_to_logits(model, x: torch.Tensor, source_layer_idx: int, contexts: Dict[int, torch.Tensor],
                                           gram_factor: torch.Tensor, chunk_size: int = None) -> Dict[str, torch.Tensor]:
    """
    Compute ||W_U^T · J_ln_final · J_{L-1} · ... · J_{i+1}||_F at each point, where each J is a last-token Jacobian.
    The Jacobians are multiplied as matrices first (in float64), then the Frobenius norm of the product is taken.
    Args:
        x: [B, d] last-token activations in the output space of source_layer_idx
        contexts: see forward_from_layer
        gram_factor: output of unembed_gram_factor
    Returns:
        Dict with
            'metric_values': [B] Frobenius norm of the full product (source layer -> logits)
            'layer_norms': [n_blocks, B] Frobenius norm of each block's Jacobian
            'cumulative_norms': [n_blocks, B] Frobenius norm of the product up to and including each block
            'logits': [B, d_vocab]
    """
    with torch.no_grad():
        out = forward_from_layer(model, x, source_layer_idx, contexts)

    product = None
    layer_norms, cumulative_norms = [], []
    for step, k in enumerate(range(source_layer_idx, model.cfg.n_layers - 1)):
        jac = batched_jacobian(lambda z: block_last_token(model, k + 1, contexts[k], z), out['resids'][step], chunk_size).detach().double()
        product = jac if product is None else torch.bmm(jac, product)
        layer_norms.append(compute_frobenius_norm(jac))
        cumulative_norms.append(compute_frobenius_norm(product))
        del jac

    jac_ln = batched_jacobian(lambda z: ln_final_last_token(model, z), out['resids'][-1], chunk_size).detach().double()
    product = jac_ln if product is None else torch.bmm(jac_ln, product)
    metric = compute_frobenius_norm(torch.matmul(gram_factor, product))

    n_points = x.shape[0]
    empty = torch.empty(0, n_points, dtype=torch.float64, device=x.device)
    return {
        'metric_values': metric,
        'layer_norms': torch.stack(layer_norms) if layer_norms else empty,
        'cumulative_norms': torch.stack(cumulative_norms) if cumulative_norms else empty,
        'logits': out['logits'],
    }


def jacobian_norm_direct_to_logits(model, x: torch.Tensor, source_layer_idx: int, contexts: Dict[int, torch.Tensor],
                                   gram_factor: torch.Tensor, chunk_size: int = None) -> torch.Tensor:
    """Same quantity as jacobian_norm_layerwise_prod_to_logits, by differentiating the composed map. For validation."""
    gram_factor = gram_factor.to(x.dtype)

    def forward(z):
        resid = forward_from_layer(model, z, source_layer_idx, contexts)['resids'][-1]
        return ln_final_last_token(model, resid) @ gram_factor.T

    jac = batched_jacobian(forward, x, chunk_size).detach().double()
    return compute_frobenius_norm(jac)
