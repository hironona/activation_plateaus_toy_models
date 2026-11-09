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
    true_labels [R, R], true_vectors [R, R, 2], true_norms [R, R] (optional; plot_true_class_gradient): the task's true
                                     argmax class at each point and the model's G^-1 grad ln P(true class)
    overlay (optional)               callable(ax) drawing extra context (e.g. input-density contours)
Trajectory figures take a dict with
    layer_names [n_layers], frame_names [3], grid_shape (R, R), subtitle
    coords [n_layers, N, 2]          anchor-relative coordinates of every grid point at every layer
    off_plane, nearest_dist [n_layers, N]
    final_labels [N], label_text, shown_labels
Boundary-zoom figures (plot_boundary_zoom) take the gradient dict above plus a list of windows, each with
    anchors (i, j), pair (a, b) = their argmax classes, shared_boundary, bounds (u0, u1, v0, v1), axis_u, axis_v [r],
    labels [r, r]
    pair_vectors [2, r, r, 2], pair_norms [2, r, r]   G^-1 grad ln P(a) and G^-1 grad ln P(b), and their magnitudes
    maxnorm_vectors, maxnorm_norms, maxnorm_tokens, singular_vectors, singular_sigma on the window's grid
Gradient arrows carry no heat map: their direction is the field's and their length is affine in log10 of its magnitude
(key beside the panels).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from plot_style import (ANNOTATION_FONTSIZE, CATEGORICAL, OTHER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
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


MIN_ARROW_FRACTION = 0.25  # The weakest arrows keep this fraction of the full length, so their direction stays visible


def log_length_range(norms):
    """(lo, hi) log10 magnitudes mapped to the shortest and longest arrows: the 5th percentile and the maximum."""
    values = log10(norms)
    return float(np.percentile(values, 5)), float(values.max())


def arrow_length(log_values, log_range, max_length):
    """Length affine in log10 magnitude: lo -> MIN_ARROW_FRACTION * max_length, hi -> max_length (clipped)."""
    lo, hi = log_range
    frac = np.clip((np.asarray(log_values) - lo) / max(hi - lo, 1e-12), 0, 1)
    return max_length * (MIN_ARROW_FRACTION + (1 - MIN_ARROW_FRACTION) * frac)


def max_arrow_length(axis_u, stride):
    return 1.5 * stride * (axis_u[1] - axis_u[0])


def draw_field_arrows(ax, axis_u, axis_v, vectors, norms, stride, log_range, colors=None, headless=False):
    """
    Arrows (no edges) on every stride-th grid point along the field, with length affine in log10 of its magnitude.
    colors: [R, R] array of arrow colors (default: dark ink). Returns the quiver (for the length key).
    """
    idx = subsample_index(len(axis_u), stride)
    uu, vv = np.meshgrid(axis_u[idx], axis_v[idx])
    vec = vectors[np.ix_(idx, idx)]  # [r, c, 2]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    unit = np.where(norm > 0, vec / np.where(norm > 0, norm, 1), 0)
    length = arrow_length(log10(norms[np.ix_(idx, idx)]), log_range, max_arrow_length(axis_u, stride))[..., None]
    style = dict(angles='xy', scale_units='xy', scale=1, linewidth=0, pivot='middle', width=0.0032, zorder=4)
    if headless:
        style.update(headwidth=0, headlength=0, headaxislength=0)
    else:
        style.update(headwidth=3.2, headlength=3.6, headaxislength=3.2)
    color = TEXT_PRIMARY if colors is None else colors[np.ix_(idx, idx)].ravel()
    return ax.quiver(uu, vv, (unit * length)[..., 0], (unit * length)[..., 1], color=color, **style)


def add_length_key(ax, quiver, log_range, max_length, title, x=1.05):
    """Reference arrows for the low, middle and high ends of the log10 magnitude range, right of the axes."""
    ax.text(x - 0.03, 1.0, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color=TEXT_SECONDARY)
    # Labels as axes text (quiverkey labels are left out of the tight bounding box and get clipped at the figure edge)
    for i, value in enumerate(np.linspace(*log_range, 3)[::-1]):
        y = 0.93 - 0.08 * i
        ax.quiverkey(quiver, x, y, float(arrow_length(value, log_range, max_length)), '', labelpos='E',
                     coordinates='axes', color=TEXT_PRIMARY)
        ax.text(x + 0.05, y, f"$10^{{{round(float(value), 1) + 0.0:.1f}}}$", transform=ax.transAxes, ha='left', va='center', fontsize=9,
                color=TEXT_SECONDARY)


def draw_anchor_points(ax, coords, names, reference=None, marker='o'):
    ax.scatter(coords[:, 0], coords[:, 1], s=60 if marker == 'o' else 180, marker=marker, facecolor=TEXT_PRIMARY,
               edgecolor=SURFACE, linewidth=2, zorder=6)
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


def log10(values, floor=1e-12):
    return np.log10(np.maximum(values, floor))


def class_color_grid(labels, shown_labels):
    """[R, R] array of category colors for a grid of class ids (other classes in gray)."""
    return np.array(palette(shown_labels), dtype=object)[label_colors(labels, shown_labels)]


def field_panel(ax, data, vectors, norms, stride, log_range, colors=None, headless=False, reference=None):
    """Arrow field on a blank background with the argmax boundaries, optional overlay and the anchors."""
    axis_u, axis_v = data['axis_u'], data['axis_v']
    ax.set_xlim(axis_u[0], axis_u[-1])
    ax.set_ylim(axis_v[0], axis_v[-1])
    if data.get('overlay') is not None:
        data['overlay'](ax)
    draw_decision_boundaries(ax, axis_u, axis_v, data['labels'], color=TEXT_SECONDARY, outline=SURFACE)
    quiver = draw_field_arrows(ax, axis_u, axis_v, vectors, norms, stride, log_range, colors=colors, headless=headless)
    draw_anchor_points(ax, data['anchor_coords'], data['anchor_names'], reference=reference)
    return quiver


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------------------------------------------------------------------------------------------------
# Gradient-based fields
# ----------------------------------------------------------------------------------------------------------------------

def plot_anchor_gradients(data, save_path, stride):
    """
    One panel per anchor a: arrows along grad ln P(y_a), length affine in log10 of its magnitude (shared across
    panels), colored by the argmax class at the arrow.
    """
    K = len(data['anchor_names'])
    log_range = log_length_range(data['anchor_norms'])
    shown = data['shown_labels']
    colors = class_color_grid(data['labels'], shown)
    fig, axes = plt.subplots(1, K, figsize=(5.6 * K, 5.6), facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for a, ax in enumerate(axes):
        quiver = field_panel(ax, data, data['anchor_vectors'][a], data['anchor_norms'][a], stride, log_range,
                             colors=colors, reference=a)
        style_axes(ax, data['xlabel'], data['ylabel'] if a == 0 else None)
        ax.set_title(f"reference {data['anchor_names'][a]}  (ŷ = {data['label_text'][data['anchor_labels'][a]]})",
                     color=TEXT_PRIMARY, fontsize=12)
    add_length_key(axes[-1], quiver, log_range, max_arrow_length(data['axis_u'], stride),
                   r"$\|\nabla \ln P_x(\hat{y}_a)\|$")
    fig.legend(handles=legend_handles(data['labels'], shown, data['label_text'], palette(shown)),
               title="arrow color: argmax at the arrow", loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(shown) + 1, 8), frameon=False, labelcolor=TEXT_PRIMARY)
    fig.suptitle("Gradient of the reference anchor's argmax log-probability (arrow length: log10 magnitude)\n"
                 f"{data['subtitle']}", color=TEXT_SECONDARY, fontsize=13)
    save(fig, save_path)


def plot_maxnorm_gradient(data, save_path, stride, max_label_classes=7):
    """Left: the max-norm row's gradient, colored by its class; right: the map of that class."""
    tokens = data['maxnorm_tokens']
    shown = data.get('maxnorm_shown_labels')  # Fixed class order (toy); otherwise the most frequent tokens
    if shown is None:
        ids, counts = np.unique(tokens, return_counts=True)
        shown = ids[np.argsort(-counts)][:max_label_classes].tolist()
    colors = palette(shown)
    axis_u, axis_v = data['axis_u'], data['axis_v']

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), facecolor=SURFACE, gridspec_kw={'wspace': 0.55})
    log_range = log_length_range(data['maxnorm_norms'])
    quiver = field_panel(axes[0], data, data['maxnorm_vectors'], data['maxnorm_norms'], stride, log_range,
                         colors=class_color_grid(tokens, shown))
    style_axes(axes[0], data['xlabel'], data['ylabel'])
    axes[0].set_title("Max-norm row of $d \\ln P / dx$ (color: its class k)", color=TEXT_PRIMARY, fontsize=12)
    add_length_key(axes[0], quiver, log_range, max_arrow_length(axis_u, stride), r"$\max_k \|\nabla \ln P_x(k)\|$")

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
    """Top singular direction of d ln P / dx on the surface as segments (sign-free), length by log10 sigma_max."""
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=SURFACE)
    log_range = log_length_range(data['singular_sigma'])
    quiver = field_panel(ax, data, data['singular_vectors'], data['singular_sigma'], stride, log_range, headless=True)
    style_axes(ax, data['xlabel'], data['ylabel'])
    ax.set_title("Top singular direction of $d \\ln P / dx$ (segments: sign-free)", color=TEXT_PRIMARY, fontsize=12)
    add_length_key(ax, quiver, log_range, max_arrow_length(data['axis_u'], stride), r"$\sigma_{\max}$")
    fig.suptitle(data['subtitle'], color=TEXT_SECONDARY, fontsize=12)
    save(fig, save_path)


def plot_true_class_gradient(data, save_path, stride):
    """
    Gradient of the model's log-probability of the task's true argmax class y*(x) at each point, colored by y*(x), with
    length affine in log10 of its magnitude; solid lines: true argmax boundaries, dashed: the model's.
    """
    true_labels, shown = data['true_labels'], data['shown_labels']
    axis_u, axis_v = data['axis_u'], data['axis_v']
    fig, ax = plt.subplots(figsize=(7.8, 6.6), facecolor=SURFACE)
    log_range = log_length_range(data['true_norms'])
    quiver = field_panel(ax, {**data, 'labels': true_labels}, data['true_vectors'], data['true_norms'], stride,
                         log_range, colors=class_color_grid(true_labels, shown))
    draw_decision_boundaries(ax, axis_u, axis_v, data['labels'], color=TEXT_PRIMARY, outline=None, linestyle='--',
                             linewidth=1.0)
    style_axes(ax, data['xlabel'], data['ylabel'])
    mismatch = (data['labels'] != true_labels).mean()
    ax.set_title(f"Model argmax ≠ true argmax on {mismatch:.1%} of the grid", color=TEXT_PRIMARY, fontsize=11)
    add_length_key(ax, quiver, log_range, max_arrow_length(axis_u, stride), r"$\|\nabla \ln P_x(y^*(x))\|$")
    handles = legend_handles(true_labels, shown, data['label_text'], palette(shown))
    handles += [Line2D([], [], color=TEXT_SECONDARY, linewidth=1.2, label="true argmax boundary"),
                Line2D([], [], color=TEXT_PRIMARY, linewidth=1.0, linestyle='--', label="model argmax boundary")]
    fig.legend(handles=handles, title="arrow color: true argmax class y*(x)", loc='upper center',
               bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False, labelcolor=TEXT_PRIMARY)
    fig.suptitle("Gradient of the model's log-probability of the true argmax class (arrow length: log10 magnitude)\n"
                 f"{data['subtitle']}", color=TEXT_SECONDARY, fontsize=12, y=1.0)
    save(fig, save_path)


# ----------------------------------------------------------------------------------------------------------------------
# Boundary zooms
# ----------------------------------------------------------------------------------------------------------------------

def light_colors(colors, weight=0.35):
    """Opaque pre-lightened colors (alpha blending leaves seams between mesh cells)."""
    return [tuple(weight * np.array(to_rgb(c)) + (1 - weight) * np.array(to_rgb(SURFACE))) for c in colors]


def maxnorm_order(data, windows, max_label_classes):
    """
    Color order of the max-norm classes: the fixed maxnorm_shown_labels when given (so colors match the full-grid
    figure), otherwise the most frequent max-norm classes inside the windows (the full grid's most frequent ones are
    often absent there).
    """
    shown = data.get('maxnorm_shown_labels')
    if shown is None:
        tokens = np.concatenate([w['maxnorm_tokens'].ravel() for w in windows])
        ids, counts = np.unique(tokens, return_counts=True)
        shown = ids[np.argsort(-counts)][:max_label_classes].tolist()
    return shown


def plot_boundary_zoom(data, windows, save_path, stride, max_label_classes=7):
    """
    One row per zoom window on the decision boundary between two anchors' argmax classes a = y_i and b = y_j,
    recomputed on a finer local grid:
        1. the full-grid argmax map with the window outlined
        2. grad ln P(a)    3. grad ln P(b)   (colored by the argmax at the arrow; one shared length scale)
        (windows with true_labels: grad ln P(y*(x)) of the task's true argmax class, colored by y*(x); solid lines:
         true argmax boundaries, dashed: the model's, as in plot_true_class_gradient)
        4. the max-norm row of d ln P / dx, colored by its class
        5. the top singular direction (sign-free segments)
    Arrow lengths share one log10 scale per field across the windows; the lines are the window's own argmax boundaries.
    """
    n = len(windows)
    shown, label_text, names = data['shown_labels'], data['label_text'], data['anchor_names']
    colors = palette(shown)
    max_shown = maxnorm_order(data, windows, max_label_classes)
    log_ranges = {
        'pair': log_length_range(np.concatenate([w['pair_norms'].ravel() for w in windows])),
        'maxnorm': log_length_range(np.concatenate([w['maxnorm_norms'].ravel() for w in windows])),
        'singular': log_length_range(np.concatenate([w['singular_sigma'].ravel() for w in windows])),
    }
    has_true = 'true_labels' in windows[0]
    if has_true:
        log_ranges['true'] = log_length_range(np.concatenate([w['true_norms'].ravel() for w in windows]))
    axis_u, axis_v = data['axis_u'], data['axis_v']
    coarse_step = axis_u[1] - axis_u[0]
    n_cols = 5 + has_true
    col_maxnorm, col_singular = 3 + has_true, 4 + has_true

    fig, axes = plt.subplots(n, n_cols, figsize=(5.7 * n_cols, 5.3 * n), facecolor=SURFACE, squeeze=False,
                             gridspec_kw={'wspace': 0.55, 'hspace': 0.36})
    for i, (w, row) in enumerate(zip(windows, axes)):
        a, b = w['pair']
        anchor_i, anchor_j = w['anchors']
        last = i == n - 1

        ax = row[0]
        ax.imshow(label_colors(data['labels'], shown), origin='lower', extent=pixel_extent(axis_u, axis_v),
                  cmap=ListedColormap(light_colors(colors)), vmin=-0.5, vmax=len(colors) - 0.5, interpolation='nearest')
        draw_decision_boundaries(ax, axis_u, axis_v, data['labels'], color=TEXT_SECONDARY, outline=SURFACE)
        for k, other in enumerate(windows):
            ou0, ou1, ov0, ov1 = other['bounds']
            current = k == i
            ax.add_patch(Rectangle((ou0, ov0), ou1 - ou0, ov1 - ov0, fill=False, zorder=5,
                                   edgecolor=TEXT_PRIMARY if current else TEXT_SECONDARY,
                                   linewidth=2.2 if current else 0.8, linestyle='-' if current else '--'))
        draw_anchor_points(ax, data['anchor_coords'], names)
        ax.set_xlim(axis_u[0], axis_u[-1])
        ax.set_ylim(axis_v[0], axis_v[-1])
        style_axes(ax, data['xlabel'] if last else None, data['ylabel'])
        where = "on their boundary" if w['shared_boundary'] else "no shared boundary: closest approach"
        ax.set_title(f"{i + 1}.  {names[anchor_i]} ({label_text[a]})  |  {names[anchor_j]} ({label_text[b]})\n{where}",
                     color=TEXT_PRIMARY, fontsize=11, loc='left')

        window_data = {'axis_u': w['axis_u'], 'axis_v': w['axis_v'], 'labels': w['labels'],
                       'overlay': data.get('overlay'), 'anchor_coords': data['anchor_coords'], 'anchor_names': names}
        argmax_colors = class_color_grid(w['labels'], shown)
        for slot, (anchor, label) in enumerate([(anchor_i, a), (anchor_j, b)]):
            ax = row[1 + slot]
            quiver = field_panel(ax, window_data, w['pair_vectors'][slot], w['pair_norms'][slot], stride,
                                 log_ranges['pair'], colors=argmax_colors, reference=anchor)
            style_axes(ax, data['xlabel'] if last else None)
            ax.set_title(rf"$\nabla \ln P$({label_text[label]})" + f"\ntoward {names[anchor]}", color=TEXT_PRIMARY,
                         fontsize=11)
            if i == 0 and slot == 1:
                add_length_key(ax, quiver, log_ranges['pair'], max_arrow_length(w['axis_u'], stride),
                               r"$\|\nabla \ln P_x(y)\|$ (both)")

        if has_true:
            ax = row[3]
            quiver = field_panel(ax, {**window_data, 'labels': w['true_labels']}, w['true_vectors'], w['true_norms'],
                                 stride, log_ranges['true'], colors=class_color_grid(w['true_labels'], shown))
            draw_decision_boundaries(ax, w['axis_u'], w['axis_v'], w['labels'], color=TEXT_PRIMARY, outline=None,
                                     linestyle='--', linewidth=1.0)
            style_axes(ax, data['xlabel'] if last else None)
            mismatch = (w['labels'] != w['true_labels']).mean()
            ax.set_title(r"$\nabla \ln P(y^*(x))$, true argmax" + f"\nmodel ≠ true on {mismatch:.1%}",
                         color=TEXT_PRIMARY, fontsize=11)
            if i == 0:
                add_length_key(ax, quiver, log_ranges['true'], max_arrow_length(w['axis_u'], stride),
                               r"$\|\nabla \ln P_x(y^*(x))\|$")

        ax = row[col_maxnorm]
        quiver = field_panel(ax, window_data, w['maxnorm_vectors'], w['maxnorm_norms'], stride, log_ranges['maxnorm'],
                             colors=class_color_grid(w['maxnorm_tokens'], max_shown))
        style_axes(ax, data['xlabel'] if last else None)
        ax.set_title("Max-norm row (color: its class k)", color=TEXT_PRIMARY, fontsize=11)
        if i == 0:
            add_length_key(ax, quiver, log_ranges['maxnorm'], max_arrow_length(w['axis_u'], stride),
                           r"$\max_k \|\nabla \ln P_x(k)\|$")

        ax = row[col_singular]
        quiver = field_panel(ax, window_data, w['singular_vectors'], w['singular_sigma'], stride,
                             log_ranges['singular'], headless=True)
        style_axes(ax, data['xlabel'] if last else None)
        ax.set_title("Top singular direction (sign-free)", color=TEXT_PRIMARY, fontsize=11)
        if i == 0:
            add_length_key(ax, quiver, log_ranges['singular'], max_arrow_length(w['axis_u'], stride), r"$\sigma_{\max}$")

    window_labels = np.concatenate([w['labels'].ravel() for w in windows])
    window_tokens = np.concatenate([w['maxnorm_tokens'].ravel() for w in windows])
    handles = legend_handles(window_labels, shown, label_text, colors)
    title = "map and class-gradient arrows: argmax class (share in the windows)"
    if has_true:
        handles += [Line2D([], [], color=TEXT_SECONDARY, linewidth=1.2, label="true argmax boundary"),
                    Line2D([], [], color=TEXT_PRIMARY, linewidth=1.0, linestyle='--', label="model argmax boundary")]
        title = "map and class-gradient arrows: argmax class (share in the windows); true-class arrows: y*(x)"
    legend = fig.legend(handles=handles, title=title, loc='upper left',
                        bbox_to_anchor=(0.12, 0.06), ncol=min(len(shown) + 1, 4), frameon=False, labelcolor=TEXT_PRIMARY)
    fig.add_artist(legend)
    fig.legend(handles=legend_handles(window_tokens, max_shown, label_text, palette(max_shown)),
               title="max-norm arrows: class k", loc='upper left', bbox_to_anchor=(0.58, 0.06),
               ncol=min(len(max_shown) + 1, 4), frameon=False, labelcolor=TEXT_PRIMARY)
    w = windows[0]  # All windows have the same size and resolution
    zoom = (axis_u[-1] - axis_u[0]) / (w['bounds'][1] - w['bounds'][0])
    density = coarse_step / (w['axis_u'][1] - w['axis_u'][0])
    fig.suptitle("Gradients of the two anchors' argmax log-probabilities around their decision boundary, recomputed on "
                 f"finer local grids ({zoom:.1f}× zoom, {density:.1f}× grid density; arrow length: log10 magnitude)\n"
                 f"{data['subtitle']}", color=TEXT_SECONDARY,
                 fontsize=14, y=0.93 + 0.02 * max(0, 4 - n))
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


def draw_transported_boundaries(ax, grid_points, labels):
    """
    Boundaries between the final-argmax regions of the grid, drawn on the grid as it sits at some layer: the 0.5 level
    set of each class's indicator over the (deformed) mesh grid_points [R_v, R_u, 2].
    """
    for label in np.unique(labels):
        mask = (labels == label).astype(float)
        if mask.all():
            continue
        cs = ax.contour(grid_points[..., 0], grid_points[..., 1], mask, levels=[0.5], colors=TEXT_SECONDARY,
                        linewidths=1.0, zorder=2)
        cs.set(path_effects=[pe.Stroke(linewidth=2.4, foreground=SURFACE), pe.Normal()])


def plot_block_updates(traj, save_path, stride, ncols=4):
    """
    One panel per block: the block's update of the subgrid points, from layer l to l + 1, in anchor-relative
    coordinates, over the final-argmax boundaries of the whole grid carried to the block's input (layer l).
    """
    coords = traj['coords']
    n_steps = coords.shape[0] - 1
    idx = trajectory_subsample(traj, stride)
    shown = traj['shown_labels']
    colors = palette(shown)
    arrow_colors = [colors[c] for c in label_colors(traj['final_labels'], shown)[idx]]
    xlim, ylim = frame_limits(coords[:, idx, :])
    R_v, R_u = traj['grid_shape']
    final_grid = traj['final_labels'].reshape(R_v, R_u)
    nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), facecolor=SURFACE, squeeze=False)
    for step, ax in enumerate(axes.ravel()):
        if step >= n_steps:
            ax.axis('off')
            continue
        draw_transported_boundaries(ax, coords[step].reshape(R_v, R_u, 2), final_grid)
        start, end = coords[step, idx], coords[step + 1, idx]
        ax.quiver(start[:, 0], start[:, 1], end[:, 0] - start[:, 0], end[:, 1] - start[:, 1], color=arrow_colors,
                  angles='xy', scale_units='xy', scale=1, width=0.004, headwidth=4, headlength=4, headaxislength=3.5,
                  zorder=3)
        style_frame_axes(ax, traj, xlim, ylim, labels=False)
        off = np.median(traj['off_plane'][step + 1])
        ax.set_title(f"{traj['layer_names'][step + 1]}  (off-plane {off:.2f})", color=TEXT_PRIMARY, fontsize=11)
    fig.suptitle("Update of each block in anchor-relative coordinates (arrow color: final argmax; lines: final-argmax "
                 "boundaries of the grid at the block's input;\noff-plane: median distance from the anchors' plane / "
                 "mean anchor distance)\n"
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
