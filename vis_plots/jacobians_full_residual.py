#!/usr/bin/env python3
"""
Full Residual Jacobian Analysis: Compute and plot ∂(last block resid_post) / ∂(embedding) through all blocks.
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


def compute_jacobian_full_residual_toy(model, embed: torch.Tensor, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian from the embedding to the last block (toy ResNet): ∂(layer_{N-1} resid_post) / ∂(embed),
    i.e. of the function blocks 0, ..., N-1.
    No sequence dimension.

    Args:
        embed: [n_steps, hidden_dim] embedding (layer -1 resid_post)
        activations: Dict with recorded activations for validation

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = embed.shape[0]
    n_layers = len(model.blocks)
    jacobians = []

    def forward_through_all_layers(resid):
        activation = resid.unsqueeze(0)
        for layer_idx in range(n_layers):
            activation = model.blocks[layer_idx](activation)
        return activation.squeeze(0)

    for step_idx in tqdm(range(n_steps), desc="Steps", leave=False):
        resid = embed[step_idx].to(device)
        jac = torch.func.jacrev(forward_through_all_layers)(resid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_through_all_layers(resid)
            recorded_out = activations[f'layer{n_layers - 1}_resid_post'][step_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Full residual validation failed: step {step_idx}, diff {diff}"

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
        embed = activations['layer-1_resid_post']
        jacobians = compute_jacobian_full_residual_toy(model, embed, device, activations)

        norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)

        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        jacobian_norms[pair_key] = norms

        del jacobians, embed, activations
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
    # Activations are recorded from the interpolated layer on, so the embedding exists only for -2 (input) or -1
    if layer_to_interpolate > -1:
        raise ValueError(f"The embedding is not recorded when interpolating at layer {layer_to_interpolate}; "
                         "set layer_to_interpolate to -2 (input) or -1 (embed) in vis_plots/config.yaml")

    all_seeds_data = []
    for idx, ckpt in enumerate(checkpoints):
        print(f"  Seed {idx + 1}/{len(checkpoints)}: {ckpt}")
        all_seeds_data.append(compute_full_residual_norms_for_model(ckpt, pairs_ids, layer_to_interpolate, N_STEPS))
    mean_data, std_data = aggregate_metric_data(all_seeds_data)

    output_dir = pairs_plot_dir(get_checkpoint_dir(config, args.checkpoint_dir), pairs_ids)
    generate_interpolation_results_plot(
        data_dict=mean_data,
        suptitle="Jacobian norm, embed → last block",
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
