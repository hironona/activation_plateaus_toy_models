#!/usr/bin/env python3
"""
Layerwise Residual Jacobian Analysis: Compute and plot ∂(layer_i+1 resid_post) / ∂(layer_i resid_post) for each layer.
"""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append('./vis_plots')
sys.path.append('./common')
from utils import load_config, load_activations, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir
from checkpoints import load_model_from_checkpoint

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_layerwise_toy(model, resid_post_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for single block: ∂(layer_i+1 resid_post) / ∂(layer_i resid_post).
    Toy ResNet: no sequence dimension.

    Args:
        resid_post_interpolated: [n_steps, hidden_dim]
        layer_idx: Which layer's resid_post is the input

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_interpolated.shape[0]
    jacobians = []

    def forward_single_layer(resid):
        activation = resid.unsqueeze(0)
        activation = model.blocks[layer_idx + 1](activation)
        return activation.squeeze(0)

    for step_idx in range(n_steps):
        resid = resid_post_interpolated[step_idx].to(device)
        jac = torch.func.jacrev(forward_single_layer)(resid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_single_layer(resid)
            recorded_out = activations[f'layer{layer_idx+1}_resid_post'][step_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Layerwise validation failed: layer {layer_idx}→{layer_idx+1}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_jacobian_embed_toy(model, input_data: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute Jacobian of the embedding layer (input_layer) for toy ResNet.

    Args:
        model: Toy ResNet model
        input_data: [n_steps, input_dim]
        device: target device

    Returns:
        Jacobian: [n_steps, hidden_dim, input_dim]
    """
    n_steps = input_data.shape[0]

    def forward_embed(x):
        return model.input_layer(x)

    jac_fn = torch.func.jacrev(forward_embed)
    jacobians = []
    for step_idx in range(n_steps):
        x = input_data[step_idx].to(device)
        jac = jac_fn(x)
        jacobians.append(jac.detach().cpu())
        del jac, x
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_jacobian_unembed_toy(model, last_block_data: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute Jacobian of the unembedding layers (final_norm + relu + output_layer) for toy ResNet.

    Args:
        model: Toy ResNet model
        last_block_data: [n_steps, hidden_dim]
        device: target device

    Returns:
        Jacobian: [n_steps, output_dim, hidden_dim]
    """
    n_steps = last_block_data.shape[0]

    def forward_unembed(resid):
        x = model.final_norm(resid.unsqueeze(0))
        x = F.relu(x)
        x = model.output_layer(x)
        return x.squeeze(0)

    jac_fn = torch.func.jacrev(forward_unembed)
    jacobians = []
    for step_idx in range(n_steps):
        x = last_block_data[step_idx].to(device)
        jac = jac_fn(x)
        jacobians.append(jac.detach().cpu())
        del jac, x
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_layerwise_data_for_model(checkpoint_path, pairs_ids, layer_to_interpolate, n_steps):
    """Compute layerwise jacobian norms and product norms for a single model checkpoint.

    Includes the embedding (Input→Embed) and unembedding (LastBlock→Logits) Jacobians. The product norm is:
        ||J_unembed · J_{n-1} · ... · J_0 · J_embed||_F

    Returns:
        (layerwise_norms_data, product_norms_data)
        layerwise_norms_data: {pair_key: {f"block {i} → {i+1}": tensor}} — multi-layer format
        product_norms_data: {pair_key: tensor} — single-line format
    """
    model, _, _ = load_model_from_checkpoint(checkpoint_path)
    n_layers = len(model.blocks)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    layerwise_norms_data = {}
    product_norms_data = {}

    for pair_ids in pairs_ids:
        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_norms = {}
        jacs_list = []

        activations = load_activations(checkpoint_path, layer_to_interpolate, pair_ids, n_steps)

        # --- Embedding Jacobian ---
        embed_jac = None
        input_data = activations.get('layer-2_resid_post')
        if input_data is not None:
            embed_jac = compute_jacobian_embed_toy(model, input_data, device)
            layer_norms["input → embed"] = torch.norm(embed_jac.view(embed_jac.shape[0], -1), dim=1)

        # --- Hidden layer Jacobians ---
        for layer_idx in tqdm(range(0, n_layers - 1), desc=f"Layerwise Jacobians {pair_key}", leave=False):
            resid_post = activations[f'layer{layer_idx}_resid_post']

            jacobians = compute_jacobian_layerwise_toy(model, resid_post, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            layer_norms[f"block {layer_idx} → {layer_idx+1}"] = norms
            jacs_list.append(jacobians)

            del resid_post
            torch.cuda.empty_cache()

        # --- Unembedding Jacobian ---
        unembed_jac = None
        last_block_data = activations.get(f'layer{n_layers - 1}_resid_post')
        if last_block_data is not None:
            unembed_jac = compute_jacobian_unembed_toy(model, last_block_data, device)
            layer_norms["last block → logits"] = torch.norm(unembed_jac.view(unembed_jac.shape[0], -1), dim=1)

        layerwise_norms_data[pair_key] = layer_norms

        # Compute product norms:
        # ||J_unembed · J_{n-1} · ... · J_0 · J_embed||_F
        product = jacs_list[0]
        for jac in jacs_list[1:]:
            product = torch.bmm(jac, product)

        # Include embedding in the product if available
        if embed_jac is not None:
            product = torch.bmm(product, embed_jac)

        # Include unembedding in the product if available
        if unembed_jac is not None:
            product = torch.bmm(unembed_jac, product)

        product_norm = torch.norm(product.view(product.shape[0], -1), dim=1)
        product_norms_data[pair_key] = product_norm

        del activations, jacs_list, product
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return layerwise_norms_data, product_norms_data


def main():
    parser = argparse.ArgumentParser(description='Plot layerwise Jacobian norms along the interpolation path')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)
    layer_to_interpolate = config['layer_to_interpolate']

    all_layerwise = []
    all_products = []
    for idx, ckpt in enumerate(checkpoints):
        print(f"  Seed {idx + 1}/{len(checkpoints)}: {ckpt}")
        lw, prod = compute_layerwise_data_for_model(ckpt, pairs_ids, layer_to_interpolate, N_STEPS)
        all_layerwise.append(lw)
        all_products.append(prod)

    mean_lw, std_lw = aggregate_metric_data(all_layerwise)
    mean_prod, std_prod = aggregate_metric_data(all_products)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_lw,
        suptitle="Jacobian norm per layer",
        ylabel="Frobenius norm",
        output_path=f"{output_dir}/jacobians_layerwise_norms.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        skip_interpolation_layer=False,
        std_dict=std_lw,
        y_floor=0,
        n_seeds=len(checkpoints),
    )

    generate_interpolation_results_plot(
        data_dict=mean_prod,
        suptitle="Jacobian norm, input → logits (layerwise product)",
        ylabel="Frobenius norm",
        output_path=f"{output_dir}/jacobians_layerwise_product.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        std_dict=std_prod,
        y_floor=0,
        n_seeds=len(checkpoints),
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
