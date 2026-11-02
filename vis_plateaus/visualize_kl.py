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
from utils import load_config, load_model_from_checkpoint, get_model_names, get_model_name
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
    cbar_label = 'KL(p(y|x*) || p(y|x))'
    if log_scale:
        cbar_label += f' (clipped at {kl_floor:.0e})'
    plt.colorbar(mesh, ax=ax, label=cbar_label, shrink=1.0, pad=0.02)

    # In input space, source coordinates are the input coordinates themselves
    _draw_reference_annotations(
        ax, reference_point, sub_reference_points,
        reference_point, sub_reference_points,
    )

    ax.set_xlabel('Input Dim 1')
    ax.set_ylabel('Input Dim 2')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize KL divergence of class probabilities from a reference point'
    )
    parser.add_argument('--model_type', type=str, choices=['toy_resnet'], required=True)
    parser.add_argument('--data_type', type=str, choices=['class_spiral'], required=True)
    parser.add_argument('--model_path', type=str,
                        help='Path to checkpoint file or directory')
    parser.add_argument('--multi_seed', action='store_true',
                        help='Average KL across all seed checkpoints')
    args = parser.parse_args()

    model_type = args.model_type

    if args.model_path:
        config['model_names'][model_type] = args.model_path
    model_paths = get_model_names(config, model_type)
    MODEL_NAME = get_model_name(config, model_type)

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

    output_dir = f"plots/{MODEL_NAME}/kl"

    if args.multi_seed:
        print(f"Multi-seed mode: {len(model_paths)} seeds")
        all_kl = []
        for idx, ckpt in enumerate(model_paths):
            print(f"  Seed {idx + 1}/{len(model_paths)}: {ckpt}")
            all_kl.append(compute_kl_for_checkpoint(ckpt, data, REFERENCE_POINT, device))
        kl = torch.stack(all_kl).mean(dim=0)   # (n_points,)
        title_prefix = f"KL Divergence of Class Probabilities (mean over {len(model_paths)} seeds)"
    else:
        ckpt = model_paths[0]
        print(f"Checkpoint: {ckpt}")
        kl = compute_kl_for_checkpoint(ckpt, data, REFERENCE_POINT, device)
        title_prefix = "KL Divergence of Class Probabilities"

    print(f"KL range: [{kl.min():.4g}, {kl.max():.4g}]")

    title = (
        f"{title_prefix}\n"
        f"Ref x*: {REFERENCE_POINT.tolist()} | "
        f"Scale: {'log' if LOG_SCALE else 'linear'}"
    )
    output_path = (f"{output_dir}/"
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
