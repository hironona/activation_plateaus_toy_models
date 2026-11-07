#!/usr/bin/env python3
"""
Visualize activation plateaus by coloring points by a metric (L2 norm, Jacobian norm or determinant, full or as a
layerwise product), averaged over seeds.

With the input space as the source (source_layer_idx: -2), the points form a regular grid and the metric is drawn as a
heat map with the argmax-class boundaries, plus a categorical argmax-class map, as in vis_plateaus_lm/plot_plane.py.
Other source layers are scatter plots in the (PCA-reduced) activation space with the training data in the background.
"""

from typing import Dict
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import math

import sys
sys.path.append('./train')
from utils import load_config, resolve_layer_idx, find_seed_checkpoints, load_model_from_checkpoint, plot_dir, METRIC_LABELS, layer_label, seeds_label
from compute_metrics import l2_norm_metric, jacobian_determinant_metric, jacobian_determinant_layerwise_prod_metric, jacobian_norm_metric, jacobian_norm_layerwise_prod_metric
from model import ResNetMLP
from data import ToyDataset, RingMixtureTask
from plot_style import ANNOTATION_FONTSIZE, SEQUENTIAL_BLUE, CATEGORICAL, OTHER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, SURFACE
from plot_maps import pixel_extent, draw_decision_boundaries, draw_region_labels

config = load_config("vis_plateaus/config.yaml")

METRIC_OPTIONS = ['l2_norm', 'jacobian_determinant_full', 'jacobian_determinant_layerwise_prod', 'jacobian_norm_full', 'jacobian_norm_layerwise_prod']

# Muted/pastel colors for background class overlay (up to 10 classes)
CLASS_COLORS = [
    '#F5E6A0',  # pastel yellow
    '#A8D5BA',  # pastel green
    '#F5C6A0',  # pastel orange
    '#B5C8E8',  # pastel blue
    '#E8B5D3',  # pastel pink
    '#D4D4AA',  # pastel olive
    '#C5A8D5',  # pastel purple
    '#A8D5D5',  # pastel teal
    '#D5A8A8',  # pastel red
    '#A8C5A8',  # pastel sage
]


def collect_activations_resid_post(model: ResNetMLP, data, source_layer_idx: int, target_layer_idx: int, device) -> Dict[str, torch.Tensor]:

    activations = {}

    def create_hook(layer_idx):
        def hook(module, input, output):
            activations[f'layer{layer_idx}_resid_post'] = output.cpu().clone()
        return hook

    layers = list(model.blocks) + [model.hook_input, model.hook_embed]
    hooks = []

    if source_layer_idx >= -1:  # collect activations at the input space of the source layer
        hooks.append(layers[source_layer_idx - 1].register_forward_hook(create_hook(source_layer_idx - 1)))

    for layer_idx in range(source_layer_idx, target_layer_idx + 1):
        hooks.append(layers[layer_idx].register_forward_hook(create_hook(layer_idx)))

    with torch.no_grad():
        logits = model(data)

    activations['logits'] = logits.cpu().clone()

    for hook in hooks:
        hook.remove()

    return activations

def generate_data_around_point(point: torch.Tensor, n_points: int, radius: float) -> torch.Tensor:
    """
    Generate n_points uniformly distributed around a point in the same dimension.
    Args:
        point: [dim]
        n_points: number of points to generate
        radius: maximum distance from the original point
    Returns:
        Tensor of shape [n_points, dim]
    """
    dim = point.shape[0]
    random_directions = torch.randn(n_points, dim)
    random_directions /= torch.max(torch.norm(random_directions, dim=1))   # Scale it to max 1
    random_directions /= torch.norm(random_directions, dim=1, keepdim=True)  # Normalize to unit vectors
    random_distances = torch.rand(n_points) * radius  # Random distances from the original point
    perturbations = random_directions * random_distances.unsqueeze(1)  # Scale by distances
    new_points = point.unsqueeze(0) + perturbations  # Shift by the original point
    new_points = torch.clamp(new_points, -1.0, 1.0)
    new_points = point.unsqueeze(0) + random_directions * radius  # Scale by radius and shift by the original point
    
    return new_points

def generate_uniform_data(n_points: int, radius: float) -> torch.Tensor:
    """
    Generate n_points uniformly distributed around a point in the same dimension.
    Args:
        n_points: number of points to generate
        radius: maximum distance from the original point
    Returns:
        Tensor of shape [n_points, dim]
    """
    new_points = [np.linspace([x1, -radius], [x1, radius], int(math.sqrt(n_points))) for x1 in np.linspace(-radius, radius, int(math.sqrt(n_points)))]
    new_points = torch.tensor(np.concatenate(new_points, axis=0), dtype=torch.float32)
    
    return new_points

def visualize_plateau(
    source_activations: torch.Tensor,
    metric_values: torch.Tensor,
    metric_label: str,
    title: str,
    output_path: str,
    n_pca_components: int = 3,
    reference_point_source_act: torch.Tensor = None,
    model: ResNetMLP = None,
    source_layer_idx: int = None,
    device: str = 'cpu',
    task_config: dict = None,
    mark_reference: bool = False,
    axis_limit: float = 1.0,
):
    """
    Create a scatter plot of data points in source-layer space, colored by metric value,
    with a faint background overlay of the training dataset transformed to the same space.
    When the source dimension exceeds n_pca_components, PCA is used to reduce dimensionality.

    Args:
        source_activations: (n_points, d_source) — activations at source layer for sampled points.
        metric_values: (n_points,) — computed metric value per point.
        metric_label: Label for the colorbar.
        title: Plot title.
        output_path: Where to save the plot.
        n_pca_components: Number of PCA components to reduce to (2 or 3).
        reference_point_source_act: (d_source,) — reference point in source space (for marking).
        model: Model used to transform background data to source space.
        source_layer_idx: Source layer index for transforming background data.
        device: Device string.
        task_config: Task config dict from checkpoint (keys: name, num_classes, noise_std, distribution, etc.).
        mark_reference: Whether to mark the reference point.
        axis_limit: Half-width of the plotted square when the source space is 2D (the input sampling radius).
    """
    d_source = source_activations.shape[1]
    use_pca = d_source > n_pca_components
    n_plot_dims = n_pca_components if use_pca else d_source
    use_3d = n_plot_dims >= 3


    # --- Create figure ---
    fig = plt.figure(figsize=(9, 8), dpi=300)
    if use_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # --- PCA fitting (on source activations) using sklearn ---
    pca = None
    if use_pca:
        print(f"Source dimension {d_source} exceeds {n_pca_components}, applying PCA for visualization...")
        pca = PCA(n_components=n_plot_dims)
        src_np = pca.fit_transform(source_activations.numpy())
    else:
        src_np = source_activations.numpy()


    # --- Resolve task config for background overlay ---
    num_classes = 2
    noise_std = 0.0
    distribution = 'uniform'
    task_name = 'class_spiral'
    task_params = None
    if task_config is not None:
        task_params = task_config.get('ring')
        num_classes = task_config.get('num_classes', 2)
        noise_std = task_config.get('noise_std', 0.0)
        # distribution = task_config.get('distribution', 'uniform')
        task_name = task_config.get('name', 'class_spiral')

    # --- Background: dataset overlay in source layer space ---
    bg_legend_handles = []
    if model is not None and source_layer_idx is not None:
        bg_dataset = ToyDataset(
            task_name=task_name, num_samples=10,
            num_classes=num_classes, noise_std=noise_std,
            distribution=distribution, seed=42, task_params=task_params
        )
        bg_X = bg_dataset.data   # (2500, 2)
        bg_y = bg_dataset.targets

        if source_layer_idx <= -2:
            bg_source_np = bg_X.numpy()
        else:
            bg_acts = collect_activations_resid_post(
                model, bg_X.to(device),
                source_layer_idx=-2, target_layer_idx=source_layer_idx, device=device
            )
            bg_source_np = bg_acts[f'layer{source_layer_idx-1}_resid_post'].numpy()

        if use_pca:
            bg_source_np = pca.transform(bg_source_np)

        # Get class labels
        if bg_y.dim() > 1 and bg_y.shape[-1] > 1:
            # One-hot encoded (num_classes > 2)
            bg_labels = bg_y.argmax(dim=1).numpy()
        else:
            # Scalar targets: shape (n,) or (n, 1) for binary classification
            bg_labels = bg_y.squeeze().numpy()
            # Binarize: threshold at 0.5 for sigmoid-style outputs
            bg_labels = (bg_labels >= 0.5).astype(int)

        unique_classes = sorted(set(bg_labels.tolist()))

        # Plot each class with a distinct pastel color
        for cls_idx in unique_classes:
            mask = bg_labels == cls_idx
            color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
            if use_3d:
                ax.scatter(bg_source_np[mask, 0], bg_source_np[mask, 1], bg_source_np[mask, 2],
                           c=color, s=8, alpha=0.5, zorder=1, marker='o')
            else:
                ax.scatter(bg_source_np[mask, 0], bg_source_np[mask, 1],
                           c=color, s=8, alpha=0.5, zorder=1, marker='o')

            bg_legend_handles.append(
                mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                              markersize=5, label=f'Class {cls_idx}')
            )

    # --- Main scatter: source activations colored by metric ---
    vals = metric_values.numpy()

    if use_3d:
        scatter = ax.scatter(src_np[:, 0], src_np[:, 1], src_np[:, 2], c=vals, cmap='coolwarm',
                             s=10, alpha=0.7, zorder=2, edgecolors='none')
    else:
        scatter = ax.scatter(src_np[:, 0], src_np[:, 1], c=vals, cmap='coolwarm',
                             s=10, alpha=0.7, zorder=2, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label=metric_label, shrink=0.7 if use_3d else 1.0, pad=0.1)

    # --- Mark reference point ---
    if reference_point_source_act is not None and mark_reference:
        ref = reference_point_source_act.numpy()
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if use_pca:
            ref = pca.transform(ref)
        ref = ref.squeeze()

        if use_3d:
            ax.scatter([ref[0]], [ref[1]], [ref[2]], c='black', marker='x', s=100, zorder=3, linewidths=2)
        else:
            ax.scatter(ref[0], ref[1], c='black', marker='x', s=100, zorder=3, linewidths=2)

    # --- Labels and Limits ---
    ax.set_title(title)
    if use_pca:
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        if use_3d:
            ax.set_zlabel('PC 3')
    elif source_layer_idx == -2:
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
    else:
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2')
        if use_3d:
            ax.set_zlabel('dim 3')
    
    if d_source == 2: # If the source space is 2D, assume it is the input space.
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xlim(-axis_limit, axis_limit)
        if use_3d:
            ax.set_zlim(-axis_limit, axis_limit)

    # --- Class legend (discrete, inside plot) ---
    if bg_legend_handles:
        ax.legend(handles=bg_legend_handles, title='Training data',
                  loc='lower right', framealpha=0.8)

    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()

def class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class probabilities (n, C); a single logit z (binary task) means p(y=1|x) = sigmoid(z) = softmax([0, z])."""
    logits = logits.float()
    if logits.shape[1] == 1:
        logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
    return torch.softmax(logits, dim=1)


def input_grid_axes(data: torch.Tensor):
    """
    Axes of the grid from generate_uniform_data, whose point i * G + j is (x1_i, x2_j).
    Returns (G, axis_u = x1 values, axis_v = x2 values).
    """
    G = int(round(math.sqrt(data.shape[0])))
    assert G * G == data.shape[0], f"Expected a square grid, got {data.shape[0]} points"
    return G, data[::G, 0].numpy(), data[:G, 1].numpy()


def to_image(values, G: int) -> np.ndarray:
    """(G * G,) values in generate_uniform_data order -> [G, G] image with rows = x2 and columns = x1."""
    return np.asarray(values).reshape(G, G).T


def ring_task(task_config):
    """RingMixtureTask of a class_ring_mixture checkpoint (None for other tasks)."""
    if task_config is None or task_config.get('name') != 'class_ring_mixture':
        return None
    return RingMixtureTask(task_config['num_classes'], task_config.get('ring'))


def draw_ring_overlay(ax, ring, axis_u, axis_v, density_levels=(0.1, 0.01)):
    """Dotted input-density contours (fractions of the peak) and the labeled anchors (Gaussian means)."""
    uu, vv = np.meshgrid(axis_u, axis_v)
    density = ring.density(np.stack([uu.ravel(), vv.ravel()], axis=1)).reshape(uu.shape)
    ax.contour(axis_u, axis_v, density, levels=sorted(density.max() * np.asarray(density_levels)),
               colors=TEXT_SECONDARY, linewidths=0.8, linestyles=':', alpha=0.8)
    ax.scatter(ring.means[:, 0], ring.means[:, 1], s=60, marker='o', facecolor=TEXT_PRIMARY, edgecolor=SURFACE,
               linewidth=2, zorder=5)
    for name, (x, y) in zip(ring.names, ring.means):
        ax.annotate(name, (x, y), xytext=(7, 7), textcoords='offset points', fontsize=ANNOTATION_FONTSIZE + 2,
                    fontweight='bold', color=TEXT_PRIMARY, path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)],
                    zorder=6)


def style_input_axes(ax, axis_u, axis_v):
    ax.set_aspect('equal')
    ax.set_xlim(axis_u[0], axis_u[-1])
    ax.set_ylim(axis_v[0], axis_v[-1])
    ax.set_xlabel('x₁', color=TEXT_SECONDARY)
    ax.set_ylabel('x₂', color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(OTHER_COLOR)


def class_color(c: int) -> str:
    return CATEGORICAL[c] if c < len(CATEGORICAL) else OTHER_COLOR


def plot_input_heatmap(data, metric_values, mean_class_probs, output_path, colorbar_label, title, subtitle,
                       log_scale=True, task_config=None, reference_point=None):
    """
    Heat map of a metric on the input grid (seed mean), with the boundaries of the argmax class of the seed-mean
    class probabilities, as in vis_plateaus_lm/plot_plane.py. class_ring_mixture runs also get the anchors and
    input-density contours.
    """
    G, axis_u, axis_v = input_grid_axes(data)
    values = to_image(metric_values.numpy(), G)
    if log_scale:
        values = np.log10(values + 1e-12)  # Small constant to avoid log(0)
    label_grid = to_image(mean_class_probs.argmax(dim=1).numpy(), G)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    im = ax.imshow(values, origin='lower', extent=pixel_extent(axis_u, axis_v), cmap=SEQUENTIAL_BLUE,
                   interpolation='nearest')
    ring = ring_task(task_config)
    if ring is not None:
        draw_ring_overlay(ax, ring, axis_u, axis_v)
    draw_decision_boundaries(ax, axis_u, axis_v, label_grid)
    if ring is None:  # Ring runs are identified by their labeled anchors instead
        labels = np.unique(label_grid)
        draw_region_labels(ax, axis_u, axis_v, label_grid, {c: f"class {c}" for c in labels}, labels)
    if reference_point is not None:
        ax.scatter([reference_point[0]], [reference_point[1]], s=320, marker='o', facecolor='none',
                   edgecolor=CATEGORICAL[1], linewidth=2.5, zorder=6)
    style_input_axes(ax, axis_u, axis_v)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label((r"$\log_{10}$ " if log_scale else "") + colorbar_label, color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13)
    fig.suptitle(subtitle, color=TEXT_SECONDARY, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_input_label_map(data, mean_class_probs, output_path, subtitle, task_config=None):
    """Argmax class of the seed-mean class probabilities on the input grid, as in vis_plateaus_lm/plot_plane.py."""
    G, axis_u, axis_v = input_grid_axes(data)
    n_classes = mean_class_probs.shape[1]
    label_grid = to_image(mean_class_probs.argmax(dim=1).numpy(), G)
    colors = [class_color(c) for c in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    ax.imshow(label_grid, origin='lower', extent=pixel_extent(axis_u, axis_v), cmap=ListedColormap(colors),
              vmin=-0.5, vmax=n_classes - 0.5, interpolation='nearest')
    ring = ring_task(task_config)
    if ring is not None:
        draw_ring_overlay(ax, ring, axis_u, axis_v)
    draw_decision_boundaries(ax, axis_u, axis_v, label_grid, color=SURFACE, outline=SURFACE)
    style_input_axes(ax, axis_u, axis_v)

    handles = [Patch(facecolor=class_color(c), label=f"class {c}  ({(label_grid == c).mean():.0%})")
               for c in range(min(n_classes, len(CATEGORICAL)))]
    if n_classes > len(CATEGORICAL):
        handles.append(Patch(facecolor=OTHER_COLOR, label=f"classes {len(CATEGORICAL)}–{n_classes - 1}  "
                                                         f"({(label_grid >= len(CATEGORICAL)).mean():.0%})"))
    if ring is not None:
        handles.append(mlines.Line2D([], [], color=TEXT_SECONDARY, linestyle=':', label="input density (0.1, 0.01 × peak)"))
    ax.legend(handles=handles, title="argmax class", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    ax.set_title("Argmax class on the input plane", color=TEXT_PRIMARY)
    fig.suptitle(subtitle, color=TEXT_SECONDARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def compute_metric_for_checkpoint(
    checkpoint_path: str,
    data: torch.Tensor,
    reference_point: torch.Tensor,
    metric: str,
    source_layer_idx,
    target_layer_idx,
    device: str,
) -> dict:
    """
    Load a single checkpoint and compute the metric for the given data.

    Returns:
        Dict with 'metric_values' (n_points,), 'source_activations' (n_points, d_source),
        'task_config', 'ref_source_act', and optionally 'predicted_class'.
    """
    model, model_config, full_config = load_model_from_checkpoint(checkpoint_path)
    task_config = full_config['task']
    model = model.to(device)
    n_blocks = len(model.blocks)
    target_is_logits = (target_layer_idx == 'logits')
    effective_target = n_blocks - 1 if target_is_logits else target_layer_idx

    activations = collect_activations_resid_post(
        model, data.to(device),
        source_layer_idx=source_layer_idx,
        target_layer_idx=effective_target,
        device=device,
    )

    if metric == 'l2_norm':
        point_activations = collect_activations_resid_post(
            model,
            reference_point.to(device).unsqueeze(0),
            source_layer_idx=source_layer_idx,
            target_layer_idx=effective_target,
            device=device
        )
        results = l2_norm_metric(model, point_activations, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_norm_full':
        results = jacobian_norm_metric(model, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_norm_layerwise_prod':
        results = jacobian_norm_layerwise_prod_metric(model, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_determinant_full':
        results = jacobian_determinant_metric(model, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_determinant_layerwise_prod':
        results = jacobian_determinant_layerwise_prod_metric(model, activations, source_layer_idx, target_layer_idx, device)
    else:
        raise ValueError(f"Metric {metric} not supported. Choose from {METRIC_OPTIONS}")

    # source_activations = activations[f'layer{source_layer_idx-1}_resid_post']
    source_activations = activations[f'layer{source_layer_idx}_resid_post']

    # Reference point in source space
    if source_layer_idx <= -2:
        ref_source_act = reference_point
    else:
        ref_acts = collect_activations_resid_post(
            model, reference_point.to(device).unsqueeze(0),
            source_layer_idx=-2, target_layer_idx=source_layer_idx,
            device=device,
        )
        # ref_source_act = ref_acts[f'layer{source_layer_idx-1}_resid_post'].squeeze(0)
        ref_source_act = ref_acts[f'layer{source_layer_idx}_resid_post'].squeeze(0)


    out = {
        'metric_values': results['metric_values'],
        'source_activations': source_activations,
        'ref_source_act': ref_source_act,
        'task_config': task_config,
        'model': model,
    }
    if 'predicted_class' in results:
        out['predicted_class'] = results['predicted_class']

    return out


def main():
    parser = argparse.ArgumentParser(description='Visualize activation plateaus (metric averaged over seed checkpoints)')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config (run directory, seed directory or .pt file)')
    parser.add_argument('--radius', type=float, help='Override radius from config (input grid spans [-radius, radius]^2)')
    parser.add_argument('--n_points', type=int, help='Override n_points from config (sqrt(n_points) grid points per input dim)')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    model_paths = find_seed_checkpoints(checkpoint_dir)

    N_POINTS = args.n_points if args.n_points is not None else config['n_points']
    RADIUS = args.radius if args.radius is not None else config['radius']
    REFERENCE_POINT = torch.tensor(config['reference_point'])
    METRIC = config['metric']
    SOURCE_LAYER_IDX_RAW = config['source_layer_idx']
    TARGET_LAYER_IDX_RAW = config['target_layer_idx']
    N_PCA_COMPONENTS = config.get('n_pca_components', 3)
    LOG_SCALE = config.get('log_scale', False)

    source_layer_idx = resolve_layer_idx(SOURCE_LAYER_IDX_RAW)
    target_layer_idx = resolve_layer_idx(TARGET_LAYER_IDX_RAW)
    target_is_logits = (target_layer_idx == 'logits')

    if METRIC not in METRIC_OPTIONS:
        raise ValueError(f"Metric {METRIC} not supported. Choose from {METRIC_OPTIONS}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate data once (shared across all seeds)
    data = generate_uniform_data(N_POINTS, RADIUS)

    print(f"{len(model_paths)} seed checkpoint(s) | METRIC: {METRIC}")
    all_metric_values = []
    all_class_probs = []
    first_seed_result = None
    for idx, checkpoint_path in enumerate(model_paths):
        print(f"  Seed {idx + 1}/{len(model_paths)}: {checkpoint_path}")
        result = compute_metric_for_checkpoint(
            checkpoint_path, data, REFERENCE_POINT, METRIC,
            source_layer_idx, target_layer_idx, device,
        )
        all_metric_values.append(result['metric_values'])
        with torch.no_grad():
            all_class_probs.append(class_probs(result['model'](data.to(device))).cpu())
        if first_seed_result is None:
            first_seed_result = result

    # Average metric values across seeds; use the first seed's spatial layout for plotting
    metric_values = torch.stack(all_metric_values).mean(dim=0)  # (n_points,)
    mean_class_probs = torch.stack(all_class_probs).mean(dim=0)  # (n_points, C)
    n_seeds = len(model_paths)
    source_activations = first_seed_result['source_activations']
    ref_source_act = first_seed_result['ref_source_act']
    task_config = first_seed_result['task_config']
    model = first_seed_result['model']

    title = f"{layer_label(source_layer_idx)} → {layer_label(target_layer_idx)} ({seeds_label(n_seeds)})"
    if 'predicted_class' in first_seed_result:
        print(f"Reference point {REFERENCE_POINT.tolist()}: predicted class {first_seed_result['predicted_class']}")

    output_path = f"{plot_dir(checkpoint_dir)}/plateaus/{METRIC}_src-{SOURCE_LAYER_IDX_RAW}_tgt-{TARGET_LAYER_IDX_RAW}.png"

    if source_layer_idx == -2:
        # Input space: the points form a regular grid, so draw heat maps as in vis_plateaus_lm/plot_plane.py
        print(f"Loaded model on {device} (task config: {task_config})")
        heatmap_title = f"{METRIC_LABELS[METRIC]}, {layer_label(source_layer_idx)} → {layer_label(target_layer_idx)}"
        subtitle = f"{os.path.relpath(os.path.normpath(checkpoint_dir), 'checkpoints')} · {seeds_label(n_seeds)}"
        plot_input_heatmap(
            data, metric_values, mean_class_probs, output_path,
            colorbar_label=METRIC_LABELS[METRIC],
            title=f"{heatmap_title}, with argmax-class boundaries", subtitle=subtitle, log_scale=LOG_SCALE,
            task_config=task_config, reference_point=REFERENCE_POINT if METRIC == 'l2_norm' else None,
        )
        label_map_path = os.path.join(os.path.dirname(output_path), "argmax_labels.png")
        plot_input_label_map(data, mean_class_probs, label_map_path, subtitle=subtitle, task_config=task_config)
        print("\n=== Complete ===")
        return

    metric_label = METRIC_LABELS[METRIC]
    if LOG_SCALE:
        metric_values = torch.log(metric_values + 1e-8)  # Add small constant to avoid log(0)
        metric_label = f"log {metric_label}"

    print(f"Loaded model on {device} (task config: {task_config})")

    visualize_plateau(
        source_activations=source_activations,
        metric_values=metric_values,
        metric_label=metric_label,
        title=title,
        output_path=output_path,
        n_pca_components=N_PCA_COMPONENTS,
        reference_point_source_act=ref_source_act,
        model=model,
        source_layer_idx=source_layer_idx,
        device=device,
        task_config=task_config,
        mark_reference=(METRIC == 'l2_norm'),
        axis_limit=RADIUS,
    )

    print("\n=== Complete ===")

if __name__ == "__main__":
    main()
