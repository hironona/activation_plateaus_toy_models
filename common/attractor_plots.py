"""
Plots of the attractor fields from common/attractors.py, shared by the toy (vis_plateaus/visualize_attractors.py) and
GPT-2 (vis_plateaus_lm/plot_attractors.py) pipelines.

Gradient figures take a dict with
    axis_u, axis_v [R]               grid axes; grid point (r, c) is (axis_u[c], axis_v[r])
    xlabel, ylabel, subtitle         axis labels and the figure subtitle
    labels [R, R]                    argmax class at each grid point
    label_text {id: str}             display text of each class id
    shown_labels [ids]               classes with their own color (fixed order); the rest are "other"
    anchor_coords [K, 2], anchor_names [K], anchor_labels [K]
    anchor_vectors [K, R, R, 2], anchor_norms [K, R, R]
    maxnorm_vectors [R, R, 2], maxnorm_norms [R, R], maxnorm_tokens [R, R]
    maxnorm_shown_labels (optional)  fixed color order of the max-norm classes (default: most frequent first)
    singular_vectors [R, R, 2], singular_sigma [R, R]
    overlay (optional)               callable(ax) drawing extra context (e.g. input-density contours)
Trajectory figures take a dict with
    layer_names [n_layers], frame_names [3], grid_shape (R, R), subtitle
    coords [n_layers, N, 2]          anchor-relative coordinates of every grid point at every layer
    off_plane, nearest_dist [n_layers, N]
    final_labels [N], label_text, shown_labels
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from plot_style import (ANNOTATION_FONTSIZE, SEQUENTIAL_BLUE, CATEGORICAL, OTHER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
                        SURFACE)
from plot_maps import pixel_extent, draw_decision_boundaries

REFERENCE_COLOR = CATEGORICAL[1]


# ----------------------------------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------------------------------

def label_colors(labels: np.ndarray, shown_labels) -> np.ndarray:
    """Color index per label: its slot in shown_labels, or len(shown_labels) for "other"."""
    index = np.full(labels.shape, len(shown_labels))
    for slot, label in enumerate(shown_labels):
        index[labels == label] = slot
    return index


def palette(shown_labels):
    return [CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER_COLOR for i in range(len(shown_labels))] + [OTHER_COLOR]


def legend_handles(labels: np.ndarray, shown_labels, label_text, colors):
    n = labels.size
    handles = [Patch(facecolor=colors[slot], label=f"{label_text[label]}  ({(labels == label).sum() / n:.0%})")
               for slot, label in enumerate(shown_labels) if (labels == label).any()]
    other = ~np.isin(labels, list(shown_labels))
    if other.any():
        n_other = len(np.unique(labels[other]))
        handles.append(Patch(facecolor=OTHER_COLOR, label=f"other ({n_other} class{'es' if n_other != 1 else ''}, {other.mean():.0%})"))
    return handles


def subsample_index(resolution: int, stride: int) -> np.ndarray:
    """Every stride-th grid index, centered in the grid."""
    offset = ((resolution - 1) % stride) // 2
    return np.arange(offset, resolution, stride)


def draw_field_arrows(ax, axis_u, axis_v, vectors, stride, headless=False):
    """Unit-length arrows (direction only) on every stride-th grid point; magnitudes are shown by the heat map."""
    idx = subsample_index(len(axis_u), stride)
    uu, vv = np.meshgrid(axis_u[idx], axis_v[idx])
    vec = vectors[np.ix_(idx, idx)]  # [r, c, 2]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    unit = np.where(norm > 0, vec / np.where(norm > 0, norm, 1), 0)
    length = 0.8 * stride * (axis_u[1] - axis_u[0])
    style = dict(angles='xy', scale_units='xy', scale=1, color=SURFACE, edgecolor=TEXT_PRIMARY, linewidth=0.4,
                 width=0.0035, zorder=4)
    if headless:
        style.update(headwidth=0, headlength=0, headaxislength=0, pivot='middle', width=0.003)
    else:
        style.update(headwidth=4, headlength=4, headaxislength=3.5, pivot='middle')
    ax.quiver(uu, vv, unit[..., 0] * length, unit[..., 1] * length, **style)


def draw_anchor_points(ax, coords, names, reference=None, marker='o'):
    ax.scatter(coords[:, 0], coords[:, 1], s=60 if marker == 'o' else 180, marker=marker, facecolor=TEXT_PRIMARY, edgecolor=SURFACE, linewidth=2,
               zorder=6)
    # Each label sits on the side facing away from the anchors' centroid, so nearby anchors' labels do not collide
    centroid = coords.mean(axis=0)
    for name, (u, v) in zip(names, coords):
        away = np.array([u, v]) - centroid
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 0 else np.array([1.0, 1.0]) / np.sqrt(2)
        ax.annotate(name, (u, v), xytext=tuple(9 * away), textcoords='offset points',
                    ha='left' if away[0] >= -0.3 else 'right', va='bottom' if away[1] >= -0.3 else 'top',
                    fontsize=ANNOTATION_FONTSIZE + 1, fontweight='bold', color=TEXT_PRIMARY,
                    path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)], zorder=7)
    if reference is not None:
        u, v = coords[reference]
        ax.scatter([u], [v], s=320, marker='o', facecolor='none', edgecolor=REFERENCE_COLOR, linewidth=2.5, zorder=6)


def style_axes(ax, xlabel=None, ylabel=None):
    ax.set_aspect('equal')
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(OTHER_COLOR)


def add_colorbar(fig, im, axes, label):
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(label, color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    cbar.outline.set_visible(False)


def log10(values, floor=1e-12):
    return np.log10(np.maximum(values, floor))


def heatmap_panel(ax, data, values, vmin, vmax, vectors, stride, headless=False, reference=None):
    axis_u, axis_v = data['axis_u'], data['axis_v']
    im = ax.imshow(values, origin='lower', extent=pixel_extent(axis_u, axis_v), cmap=SEQUENTIAL_BLUE, vmin=vmin,
                   vmax=vmax, interpolation='nearest')
    if data.get('overlay') is not None:
        data['overlay'](ax)
    draw_decision_boundaries(ax, axis_u, axis_v, data['labels'])
    draw_field_arrows(ax, axis_u, axis_v, vectors, stride, headless=headless)
    draw_anchor_points(ax, data['anchor_coords'], data['anchor_names'], reference=reference)
    ax.set_xlim(axis_u[0], axis_u[-1])
    ax.set_ylim(axis_v[0], axis_v[-1])
    return im


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------------------------------------------------------------------------------------------------
# Gradient-based fields
# ----------------------------------------------------------------------------------------------------------------------

def plot_anchor_gradients(data, save_path, stride):
    """One panel per anchor a: |grad ln P(y_a)| heat map with unit arrows of its direction."""
    K = len(data['anchor_names'])
    values = log10(data['anchor_norms'])
    vmin, vmax = np.percentile(values, 0.5), values.max()
    fig, axes = plt.subplots(1, K, figsize=(5.6 * K, 5.4), facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for a, ax in enumerate(axes):
        im = heatmap_panel(ax, data, values[a], vmin, vmax, data['anchor_vectors'][a], stride, reference=a)
        style_axes(ax, data['xlabel'], data['ylabel'] if a == 0 else None)
        ax.set_title(f"reference {data['anchor_names'][a]}  (ŷ = {data['label_text'][data['anchor_labels'][a]]})",
                     color=TEXT_PRIMARY, fontsize=12)
    add_colorbar(fig, im, axes.tolist(), r"$\log_{10}\, \|\nabla \ln P_x(\hat{y}_a)\|$")
    fig.suptitle("Gradient of the reference anchor's argmax log-probability (arrows: direction of steepest ascent)\n"
                 f"{data['subtitle']}", color=TEXT_SECONDARY, fontsize=13)
    save(fig, save_path)


def plot_maxnorm_gradient(data, save_path, stride, max_label_classes=7):
    """Left: the max-norm row's gradient; right: which class that row belongs to."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    values = log10(data['maxnorm_norms'])
    im = heatmap_panel(axes[0], data, values, np.percentile(values, 0.5), values.max(), data['maxnorm_vectors'], stride)
    style_axes(axes[0], data['xlabel'], data['ylabel'])
    axes[0].set_title("Max-norm row of $d \\ln P / dx$", color=TEXT_PRIMARY, fontsize=12)
    add_colorbar(fig, im, axes[0], r"$\log_{10} \max_k \|\nabla \ln P_x(k)\|$")

    tokens = data['maxnorm_tokens']
    shown = data.get('maxnorm_shown_labels')  # Fixed class order (toy); otherwise the most frequent tokens
    if shown is None:
        ids, counts = np.unique(tokens, return_counts=True)
        shown = ids[np.argsort(-counts)][:max_label_classes].tolist()
    colors = palette(shown)
    axis_u, axis_v = data['axis_u'], data['axis_v']
    axes[1].imshow(label_colors(tokens, shown), origin='lower', extent=pixel_extent(axis_u, axis_v),
                   cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5, interpolation='nearest')
    draw_decision_boundaries(axes[1], axis_u, axis_v, data['labels'])
    draw_anchor_points(axes[1], data['anchor_coords'], data['anchor_names'])
    axes[1].set_xlim(axis_u[0], axis_u[-1])
    axes[1].set_ylim(axis_v[0], axis_v[-1])
    style_axes(axes[1], data['xlabel'])
    axes[1].set_title("Class k of the max-norm row (lines: argmax boundaries)", color=TEXT_PRIMARY, fontsize=12)
    axes[1].legend(handles=legend_handles(tokens, shown, data['label_text'], colors), title="k",
                   loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, labelcolor=TEXT_PRIMARY)
    fig.suptitle(f"Fastest-changing log-probability along the surface\n{data['subtitle']}", color=TEXT_SECONDARY,
                 fontsize=13, y=1.04)
    save(fig, save_path)


def plot_singular_direction(data, save_path, stride):
    """Top singular value of d ln P / dx on the surface, with its (sign-free) direction as segments."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    values = log10(data['singular_sigma'])
    im = heatmap_panel(ax, data, values, np.percentile(values, 0.5), values.max(), data['singular_vectors'], stride,
                       headless=True)
    style_axes(ax, data['xlabel'], data['ylabel'])
    ax.set_title("Top singular direction of $d \\ln P / dx$ (segments: sign-free)", color=TEXT_PRIMARY, fontsize=12)
    add_colorbar(fig, im, ax, r"$\log_{10} \sigma_{\max}$")
    fig.suptitle(data['subtitle'], color=TEXT_SECONDARY, fontsize=12)
    save(fig, save_path)


# ----------------------------------------------------------------------------------------------------------------------
# Block-based trajectories
# ----------------------------------------------------------------------------------------------------------------------

FRAME_ANCHOR_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def trajectory_subsample(traj, stride):
    R_v, R_u = traj['grid_shape']
    rows, cols = subsample_index(R_v, stride), subsample_index(R_u, stride)
    return (rows[:, None] * R_u + cols[None, :]).ravel()


def frame_limits(coords, pad=0.15):
    """Axis limits covering the 1st-99th percentile of the plotted coordinates and the frame anchors."""
    pts = np.concatenate([coords.reshape(-1, 2), FRAME_ANCHOR_COORDS])
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    lo, hi = np.minimum(lo, FRAME_ANCHOR_COORDS.min(0)), np.maximum(hi, FRAME_ANCHOR_COORDS.max(0))
    span = (hi - lo).max() * (1 + 2 * pad)
    center = (lo + hi) / 2
    return (center[0] - span / 2, center[0] + span / 2), (center[1] - span / 2, center[1] + span / 2)


def style_frame_axes(ax, traj, xlim, ylim, labels=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    draw_anchor_points(ax, FRAME_ANCHOR_COORDS, traj['frame_names'], marker='*')
    names = traj['frame_names']
    style_axes(ax, f"toward {names[1]}  (affine)" if labels else None, f"toward {names[2]}  (affine)" if labels else None)


def plot_trajectories(traj, save_path, stride):
    """
    Trajectories of a subgrid through the blocks in anchor-relative coordinates, over the basins of attraction (the
    starting positions colored by the final argmax class).
    """
    coords = traj['coords']  # [n_layers, N, 2]
    final = traj['final_labels']
    shown = traj['shown_labels']
    colors = palette(shown)
    color_index = label_colors(final, shown)
    R_v, R_u = traj['grid_shape']

    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor=SURFACE)
    start = coords[0].reshape(R_v, R_u, 2)
    # Opaque pre-lightened colors: alpha blending leaves seams between the mesh cells
    light = [tuple(0.35 * np.array(to_rgb(c)) + 0.65 * np.array(to_rgb(SURFACE))) for c in colors]
    ax.pcolormesh(start[..., 0], start[..., 1], color_index.reshape(R_v, R_u), cmap=ListedColormap(light),
                  vmin=-0.5, vmax=len(colors) - 0.5, shading='nearest', zorder=1, edgecolors='face', linewidth=0.3,
                  rasterized=True)

    idx = trajectory_subsample(traj, stride)
    paths = coords[:, idx, :].transpose(1, 0, 2)  # [M, n_layers, 2]
    line_colors = [colors[c] for c in color_index[idx]]
    ax.add_collection(LineCollection(paths, colors=line_colors, linewidths=1.0, alpha=0.85, zorder=3,
                                     path_effects=[pe.Stroke(linewidth=1.8, foreground=SURFACE, alpha=0.6), pe.Normal()]))
    ax.scatter(paths[:, 0, 0], paths[:, 0, 1], s=6, c=TEXT_SECONDARY, zorder=4, linewidths=0)
    ax.scatter(paths[:, -1, 0], paths[:, -1, 1], s=22, c=line_colors, edgecolors=TEXT_PRIMARY, linewidths=0.5, zorder=5)

    xlim, ylim = frame_limits(coords[:, idx, :])
    style_frame_axes(ax, traj, xlim, ylim)
    handles = legend_handles(final, shown, traj['label_text'], colors)
    handles += [Line2D([], [], marker='o', linestyle='', color=TEXT_SECONDARY, markersize=3, label=f"start ({traj['layer_names'][0]})"),
                Line2D([], [], marker='o', linestyle='', markerfacecolor=OTHER_COLOR, markeredgecolor=TEXT_PRIMARY,
                       markersize=5, label=f"end ({traj['layer_names'][-1]})")]
    ax.legend(handles=handles, title="final argmax", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    ax.set_title(f"Residual-stream trajectories, {traj['layer_names'][0]} → {traj['layer_names'][-1]}\n"
                 "(shading: final argmax of each starting point)", color=TEXT_PRIMARY, fontsize=12)
    fig.suptitle(traj['subtitle'], color=TEXT_SECONDARY, fontsize=12)
    save(fig, save_path)


def plot_block_updates(traj, save_path, stride, ncols=4):
    """One panel per block: the block's update of the subgrid points, from layer l to l + 1, in anchor-relative coordinates."""
    coords = traj['coords']
    n_steps = coords.shape[0] - 1
    idx = trajectory_subsample(traj, stride)
    shown = traj['shown_labels']
    colors = palette(shown)
    arrow_colors = [colors[c] for c in label_colors(traj['final_labels'], shown)[idx]]
    xlim, ylim = frame_limits(coords[:, idx, :])
    nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), facecolor=SURFACE, squeeze=False)
    for step, ax in enumerate(axes.ravel()):
        if step >= n_steps:
            ax.axis('off')
            continue
        start, end = coords[step, idx], coords[step + 1, idx]
        ax.quiver(start[:, 0], start[:, 1], end[:, 0] - start[:, 0], end[:, 1] - start[:, 1], color=arrow_colors,
                  angles='xy', scale_units='xy', scale=1, width=0.004, headwidth=4, headlength=4, headaxislength=3.5,
                  zorder=3)
        style_frame_axes(ax, traj, xlim, ylim, labels=False)
        off = np.median(traj['off_plane'][step + 1])
        ax.set_title(f"{traj['layer_names'][step + 1]}  (off-plane {off:.2f})", color=TEXT_PRIMARY, fontsize=11)
    fig.suptitle("Update of each block in anchor-relative coordinates (arrow color: final argmax; "
                 "off-plane: median distance from the anchors' plane / mean anchor distance)\n"
                 f"{traj['subtitle']}", color=TEXT_SECONDARY, fontsize=12)
    fig.tight_layout()
    save(fig, save_path)


def plot_convergence(layer_names, series, save_path, subtitle):
    """
    series: {metric title: [(label, center [n_layers], lo [n_layers], hi [n_layers]), ...]}; the band is lo..hi.
    """
    fig, axes = plt.subplots(1, len(series), figsize=(6.5 * len(series), 4.2), facecolor=SURFACE, squeeze=False)
    x = np.arange(len(layer_names))
    for ax, (title, lines) in zip(axes[0], series.items()):
        for slot, (label, center, lo, hi) in enumerate(lines):
            color = CATEGORICAL[slot % len(CATEGORICAL)]
            ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
            ax.plot(x, center, color=color, linewidth=2, marker='o', markersize=3, label=label)
        ax.set_xticks(x, layer_names, rotation=45, ha='right')
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=OTHER_COLOR, linewidth=0.5, alpha=0.6)
        style_axes(ax)
        ax.set_aspect('auto')
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.suptitle(f"Convergence toward the anchors' trajectories\n{subtitle}", color=TEXT_SECONDARY, fontsize=12)
    fig.tight_layout()
    save(fig, save_path)
