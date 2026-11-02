#!/usr/bin/env python3
"""
Visualize the KL divergence of the final class probabilities as a heatmap in input space.

Each grid point x is colored by KL(p(y|x*) || p(y|x)), where x* is the reference point.
"""

import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

import sys
sys.path.append('./train')
from utils import load_config, find_seed_checkpoints, load_model_from_checkpoint, plot_dir, seeds_label
from visualize_plateaus import generate_uniform_data
from visualize_contours import _draw_reference_annotations

config = load_config("vis_plateaus/kl_config.yaml")


def class_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Log class probabilities log p(y|x) from logits, shape (n, C).
    A single logit z (binary task trained with BCEWithLogitsLoss) means p(y=1|x) = sigmoid(z),
    which equals softmax([0, z]).
    """
    logits = logits.float()
    if logits.shape[1] == 1:
        logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
    return torch.log_softmax(logits, dim=1)


def compute_kl_for_checkpoint(
    checkpoint_path: str,
    data: torch.Tensor,
    reference_point: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Load checkpoint and compute KL(p(y|x*) || p(y|x)) for every x in data.

    Returns:
        kl: (n_points,)
    """
    model, _, _ = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device)

    with torch.no_grad():
        log_p = class_log_probs(model(data.to(device)))                           # (n_points, C)
        log_p_ref = class_log_probs(model(reference_point.to(device).unsqueeze(0)))  # (1, C)

    kl = (log_p_ref.exp() * (log_p_ref - log_p)).sum(dim=1)
    # Rounding can give tiny negative values; KL is non-negative
    return kl.clamp_min(0).cpu()


def visualize_kl_heatmap(
    kl: torch.Tensor,
    grid_size: int,
    radius: float,
    title: str,
    output_path: str,
    log_scale: bool = True,
    kl_floor: float = 1e-8,
    reference_point: torch.Tensor = None,
    sub_reference_points: torch.Tensor = None,
):
    """
    Plot the KL values on the 2D input grid as a heatmap (one cell per grid point).
    """
    x_coords = np.linspace(-radius, radius, grid_size)
    y_coords = np.linspace(-radius, radius, grid_size)
    # generate_uniform_data: outer loop x1, inner loop x2
    # index i*grid_size + j → (x_coords[i], y_coords[j])
    # pcolormesh expects Z[y_idx, x_idx] → transpose
    Z = kl.numpy().reshape(grid_size, grid_size).T

    if log_scale:
        Z = np.clip(Z, kl_floor, None)
        norm = mcolors.LogNorm(vmin=Z.min(), vmax=Z.max())
    else:
        norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    mesh = ax.pcolormesh(x_coords, y_coords, Z, cmap='viridis', norm=norm,
                         shading='nearest', zorder=2)
    # Values below kl_floor are clipped to it on the log scale
    plt.colorbar(mesh, ax=ax, label='KL(p(y|x*) ‖ p(y|x))', shrink=1.0, pad=0.02)

    # In input space, source coordinates are the input coordinates themselves
    _draw_reference_annotations(
        ax, reference_point, sub_reference_points,
        reference_point, sub_reference_points,
    )

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize KL divergence of class probabilities from a reference point (averaged over seed checkpoints)'
    )
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Override checkpoint_dir from config (run directory, seed directory or .pt file)')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    model_paths = find_seed_checkpoints(checkpoint_dir)

    N_POINTS        = config['n_points']
    RADIUS          = config['radius']
    REFERENCE_POINT = torch.tensor(config['reference_point'], dtype=torch.float32)
    LOG_SCALE       = config.get('log_scale', True)
    KL_FLOOR        = float(config.get('kl_floor', 1e-8))

    # Sub-reference points (None if not set or null)
    raw_sub_refs = config.get('sub_reference_points', None)
    SUB_REFERENCE_POINTS = None
    if raw_sub_refs is not None:
        SUB_REFERENCE_POINTS = torch.tensor(raw_sub_refs, dtype=torch.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = generate_uniform_data(N_POINTS, RADIUS)
    grid_size = int(math.sqrt(N_POINTS))

    print(f"{len(model_paths)} seed checkpoint(s)")
    all_kl = []
    for idx, ckpt in enumerate(model_paths):
        print(f"  Seed {idx + 1}/{len(model_paths)}: {ckpt}")
        all_kl.append(compute_kl_for_checkpoint(ckpt, data, REFERENCE_POINT, device))
    kl = torch.stack(all_kl).mean(dim=0)   # (n_points,)

    print(f"KL range: [{kl.min():.4g}, {kl.max():.4g}]")

    title = f"KL divergence of class probabilities ({seeds_label(len(model_paths))})"
    output_path = (f"{plot_dir(checkpoint_dir)}/kl/"
                   f"kl_ref_{REFERENCE_POINT.tolist()}_{'log' if LOG_SCALE else 'linear'}.png")

    visualize_kl_heatmap(
        kl=kl,
        grid_size=grid_size,
        radius=RADIUS,
        title=title,
        output_path=output_path,
        log_scale=LOG_SCALE,
        kl_floor=KL_FLOOR,
        reference_point=REFERENCE_POINT,
        sub_reference_points=SUB_REFERENCE_POINTS,
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
