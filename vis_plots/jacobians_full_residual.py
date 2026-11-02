#!/usr/bin/env python3
"""
Full Residual Jacobian Analysis: Compute and plot ∂(layer_N resid_post) / ∂(layer_0 resid_post).
"""

import torch
from tqdm import tqdm
import sys
import argparse
sys.path.append('./vis_plots')
sys.path.append('./common')
from utils import load_config, load_activations, generate_interpolation_results_plot, get_seed_checkpoints, get_checkpoint_dir, get_pairs_ids, aggregate_metric_data, pairs_plot_dir
from checkpoints import load_model_from_checkpoint

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_full_residual_toy(model, resid_post_layer0: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute jacobian from layer 0 to last layer (toy ResNet).
    No sequence dimension.

    Args:
        resid_post_layer0: [n_steps, hidden_dim]

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_layer0.shape[0]
    n_layers = len(model.blocks)
    jacobians = []

    def forward_through_all_layers(resid):
        activation = resid.unsqueeze(0)
        for layer_idx in range(n_layers):
            activation = model.blocks[layer_idx](activation)
        return activation.squeeze(0)

    for step_idx in tqdm(range(n_steps), desc="Steps", leave=False):
        resid = resid_post_layer0[step_idx].to(device)
        jac = torch.func.jacrev(forward_through_all_layers)(resid)
        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_full_residual_norms_for_model(checkpoint_path, pairs_ids, layer_to_interpolate, n_steps):
    """Compute full residual jacobian norms for a single model checkpoint.

    Returns:
        {pair_key: norms_tensor} — single-line format for aggregate_metric_data.
    """
    model, _, _ = load_model_from_checkpoint(checkpoint_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    jacobian_norms = {}

    for pair_ids in tqdm(pairs_ids, desc="Pairs", leave=False):
        activations = load_activations(checkpoint_path, layer_to_interpolate, pair_ids, n_steps)
        resid_post_layer0 = activations['layer0_resid_post']
        jacobians = compute_jacobian_full_residual_toy(model, resid_post_layer0, device)

        norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)

        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        jacobian_norms[pair_key] = norms

        del jacobians, resid_post_layer0, activations
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return jacobian_norms


def main():
    parser = argparse.ArgumentParser(description='Plot full residual Jacobian norms along the interpolation path')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    pairs_ids = get_pairs_ids(config)
    layer_to_interpolate = config['layer_to_interpolate']

    all_seeds_data = []
    for idx, ckpt in enumerate(checkpoints):
        print(f"  Seed {idx + 1}/{len(checkpoints)}: {ckpt}")
        all_seeds_data.append(compute_full_residual_norms_for_model(ckpt, pairs_ids, layer_to_interpolate, N_STEPS))
    mean_data, std_data = aggregate_metric_data(all_seeds_data)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_data,
        suptitle="Jacobian norm, block 0 → last block",
        ylabel="Frobenius norm",
        output_path=f"{output_dir}/jacobians_full_residual_norms.png",
        n_steps=N_STEPS,
        pairs_ids=pairs_ids,
        std_dict=std_data,
        y_floor=0,
        n_seeds=len(checkpoints),
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
