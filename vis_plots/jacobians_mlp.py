#!/usr/bin/env python3
"""
MLP Jacobian Analysis: Compute and plot ∂(mlp_out) / ∂(resid_pre) for each residual block.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import argparse
sys.path.append('./vis_plots')
sys.path.append('./common')
from utils import load_config, load_activations, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir
from checkpoints import load_model_from_checkpoint

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_mlp_toy(model, resid_pre: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for MLP: ∂(mlp_out) / ∂(resid_pre) (toy ResNet).
    The MLP is the entire non-residual path: ln1 → relu → fc1 → dropout → ln2 → relu → fc2.

    Args:
        resid_pre: [n_steps, hidden_dim]
        layer_idx: Which block to compute jacobian for

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    def forward_mlp(resid):
        block = model.blocks[layer_idx]
        out = block.ln1(resid.unsqueeze(0))
        out = F.relu(out)
        out = block.fc1(out)
        out = block.dropout(out)
        out = block.ln2(out)
        out = F.relu(out)
        out = block.fc2(out)
        return out.squeeze(0)

    n_steps = resid_pre.shape[0]
    jacobians = []

    for step_idx in range(n_steps):
        resid = resid_pre[step_idx].to(device)
        jac = torch.func.jacrev(forward_mlp)(resid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_mlp(resid)
            recorded_out = activations[f'layer{layer_idx}_mlp_out'][step_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"MLP validation failed: layer {layer_idx}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_mlp_data_for_model(checkpoint_path, pairs_ids, layer_to_interpolate, n_steps):
    """Compute MLP jacobian norms and product norms for a single model checkpoint.

    Returns:
        (layerwise_norms_data, product_norms_data)
        layerwise_norms_data: {pair_key: {f"Layer {i}": tensor}} — multi-layer format
        product_norms_data: {pair_key: tensor} — single-line format
    """
    model, _, _ = load_model_from_checkpoint(checkpoint_path)
    layer_range = range(0, len(model.blocks))

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

        for layer_idx in tqdm(layer_range, desc=f"MLP Jacobians {pair_key}", leave=False):
            # Block i reads the resid_post of block i-1 (the embedding for block 0)
            resid_input = activations[f'layer{layer_idx-1}_resid_post']
            jacobians = compute_jacobian_mlp_toy(model, resid_input, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            layer_norms[f"Layer {layer_idx}"] = norms
            jacs_list.append(jacobians)

            del jacobians
            torch.cuda.empty_cache()

        layerwise_norms_data[pair_key] = layer_norms

        # Compute product norms
        product = jacs_list[0]
        for jac in jacs_list[1:]:
            product = torch.bmm(jac, product)
        product_norms_data[pair_key] = torch.norm(product.view(product.shape[0], -1), dim=1)

        del activations, jacs_list, product
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return layerwise_norms_data, product_norms_data


def main():
    parser = argparse.ArgumentParser(description='Plot MLP Jacobian norms along the interpolation path')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)
    layer_to_interpolate = config['layer_to_interpolate']

    all_layerwise = []
    all_products = []
    for idx, ckpt in enumerate(checkpoints):
        print(f"  Seed {idx + 1}/{len(checkpoints)}: {ckpt}")
        lw, prod = compute_mlp_data_for_model(ckpt, pairs_ids, layer_to_interpolate, N_STEPS)
        all_layerwise.append(lw)
        all_products.append(prod)

    mean_lw, std_lw = aggregate_metric_data(all_layerwise)
    mean_prod, std_prod = aggregate_metric_data(all_products)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_lw,
        suptitle="Jacobian Norms: MLP (∂mlp_out / ∂resid_pre)",
        ylabel="Frobenius Norm",
        output_path=f"{output_dir}/jacobians_mlp_norms.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        skip_interpolation_layer=False,
        std_dict=std_lw,
        y_floor=0
    )

    generate_interpolation_results_plot(
        data_dict=mean_prod,
        suptitle="Jacobian Products: MLP (Full Chain)",
        ylabel="Frobenius Norm",
        output_path=f"{output_dir}/jacobians_mlp_product.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        std_dict=std_prod,
        y_floor=0
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
