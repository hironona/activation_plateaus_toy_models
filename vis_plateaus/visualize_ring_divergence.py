#!/usr/bin/env python3
"""
Toy counterpart of vis_plateaus_lm/compute_divergence.py + plot_divergence.py for models trained on
class_ring_mixture. The anchors are the K Gaussian means a; P_a is the class distribution at anchor a and P_x the
one at input x. For every reference anchor a (one panel each), on the 2D input plane (in nats):
    1. KL divergence KL(P_a || P_x)
    2. Surprisal of x's argmax class under the anchor, -ln P_a(y_x) with y_x = argmax_y P_x(y)
    3. Surprisal of the anchor's argmax class at x, -ln P_x(y_a) with y_a = argmax_y P_a(y)
Each figure has two rows: the trained models (metric averaged over seeds, argmax boundaries of the seed-mean
distribution) and the ground truth (the task's true P(y|x) and anchor distributions), on a shared color scale.
Also plots the argmax class maps and the three metrics along a circle through the anchors (seed mean with 95% CI).

Run from the project root:
    python vis_plateaus/visualize_ring_divergence.py [--checkpoint_dir checkpoints/class_ring_mixture/noise-0.00]
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append('./vis_plateaus')
sys.path.append('./train')
from utils import load_config, find_seed_checkpoints, load_model_from_checkpoint, plot_dir, seeds_label
from plot_style import (ANNOTATION_FONTSIZE, SEQUENTIAL_BLUE, CATEGORICAL, OTHER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
                        SURFACE)
from plot_maps import pixel_extent, draw_decision_boundaries
from data import RingMixtureTask

METRICS = {
    'kl': (r"KL$(P_a \,\|\, P_x)$", "KL divergence from the reference anchor's class distribution"),
    'surprisal': (r"$-\ln P_a(\hat{y}_x)$", r"Surprisal of each point's argmax class $\hat{y}_x$ under the reference anchor"),
    'anchor_token_surprisal': (r"$-\ln P_x(\hat{y}_a)$", r"Surprisal of the reference anchor's argmax class $\hat{y}_a$ at each point"),
}
REFERENCE_COLOR = CATEGORICAL[1]  # Orange ring around the reference anchor, distinct from the blue ramp
LEARNED_COLOR = CATEGORICAL[0]
CI_Z = 1.96  # 95% confidence interval of the seed mean


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------

def divergence_from_anchors(anchor_log_probs: torch.Tensor, log_probs: torch.Tensor) -> dict:
    """
    Args:
        anchor_log_probs: [K, C] log class probabilities at the anchors
        log_probs: [N, C] log class probabilities at the points
    Returns:
        'kl' [K, N] KL(P_a || P_x), 'surprisal' [K, N] -ln P_a(argmax P_x),
        'anchor_token_surprisal' [K, N] -ln P_x(argmax P_a), 'labels' [N]
    """
    anchor_probs = anchor_log_probs.exp()
    neg_entropy = (anchor_probs * anchor_log_probs).sum(dim=-1, keepdim=True)  # [K, 1]
    kl = (neg_entropy - anchor_probs @ log_probs.T).clamp_min(0)  # [K, N]; rounding can give tiny negatives
    labels = log_probs.argmax(dim=-1)
    anchor_labels = anchor_log_probs.argmax(dim=-1)
    return {
        'kl': kl,
        'surprisal': -anchor_log_probs[:, labels],
        'anchor_token_surprisal': -log_probs[:, anchor_labels].T,
        'labels': labels,
    }


def learned_metrics(checkpoints, ring: RingMixtureTask, points: torch.Tensor) -> dict:
    """Per-seed metrics [S, K, N] for each checkpoint, plus the seed-mean probabilities [N, C] and anchor ones [K, C]."""
    anchors = torch.from_numpy(ring.means).float()
    per_seed = {key: [] for key in METRICS}
    probs, anchor_probs = [], []
    for ckpt in checkpoints:
        model, _, _ = load_model_from_checkpoint(ckpt)
        with torch.no_grad():
            log_p = torch.log_softmax(model(points).double(), dim=-1)
            log_pa = torch.log_softmax(model(anchors).double(), dim=-1)
        out = divergence_from_anchors(log_pa, log_p)
        for key in METRICS:
            per_seed[key].append(out[key])
        probs.append(log_p.exp())
        anchor_probs.append(log_pa.exp())
    result = {key: torch.stack(vals) for key, vals in per_seed.items()}
    result['probs'] = torch.stack(probs).mean(dim=0)
    result['anchor_probs'] = torch.stack(anchor_probs).mean(dim=0)
    result['labels'] = result['probs'].argmax(dim=-1)
    return result


def true_metrics(ring: RingMixtureTask, points: torch.Tensor) -> dict:
    log_p = torch.from_numpy(np.log(ring.true_probs(points.numpy())))
    log_pa = torch.from_numpy(np.log(ring.label_dists))
    return divergence_from_anchors(log_pa, log_p)


def summarize(values: torch.Tensor):
    """[S, ...] per-seed values -> (mean, 95% CI half-width)."""
    mean = values.mean(dim=0)
    if values.shape[0] == 1:
        return mean, torch.zeros_like(mean)
    return mean, values.std(dim=0) * CI_Z / np.sqrt(values.shape[0])


def arc_angles_of_flips(labels: np.ndarray, angles_deg: np.ndarray):
    """Angles (deg) between consecutive arc points whose argmax class differs."""
    idx = np.nonzero(labels[1:] != labels[:-1])[0]
    return [0.5 * (angles_deg[i] + angles_deg[i + 1]) for i in idx]


# ----------------------------------------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------------------------------------

def class_color(c: int) -> str:
    return CATEGORICAL[c] if c < len(CATEGORICAL) else OTHER_COLOR


def draw_density(ax, axis, density_grid, levels):
    if not levels:
        return
    peak = density_grid.max()
    ax.contour(axis, axis, density_grid, levels=sorted(peak * np.asarray(levels)), colors=TEXT_SECONDARY,
               linewidths=0.8, linestyles=':', alpha=0.8)


def draw_anchors(ax, ring: RingMixtureTask, reference=None):
    ax.scatter(ring.means[:, 0], ring.means[:, 1], s=60, marker='o', facecolor=TEXT_PRIMARY, edgecolor=SURFACE,
               linewidth=2, zorder=5)
    for name, (x, y) in zip(ring.names, ring.means):
        ax.annotate(name, (x, y), xytext=(7, 7), textcoords='offset points', fontsize=ANNOTATION_FONTSIZE + 2,
                    fontweight='bold', color=TEXT_PRIMARY, path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)],
                    zorder=6)
    if reference is not None:
        x, y = ring.means[reference]
        ax.scatter([x], [y], s=320, marker='o', facecolor='none', edgecolor=REFERENCE_COLOR, linewidth=2.5, zorder=6)


def style_axes(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(OTHER_COLOR)


def task_subtitle(ring: RingMixtureTask, n_seeds: int) -> str:
    widths = ", ".join(f"{ring.names[i]}|{ring.names[(i + 1) % ring.K]} {w:g}" for i, w in enumerate(ring.widths))
    sigmas = np.unique(ring.sigmas)
    sigma = f"σ={sigmas[0]:g}" if len(sigmas) == 1 else "σ=" + "/".join(f"{s:g}" for s in ring.sigmas)
    return (f"class_ring_mixture · {sigma} · transition widths: {widths} · targets: {ring.target_mode} · "
            f"{seeds_label(n_seeds)}")


def plot_metric_panels(rows, row_labels, label_grids, ring, axis, density_grid, config, save_path, cbar_label, title,
                       subtitle, extend='neither'):
    """
    Args:
        rows: list of [K, R, R] grids (already transformed, e.g. log10), one per row, sharing one color scale
        label_grids: list of [R, R] argmax class grids, one per row (for the decision boundaries)
    """
    vmin = min(np.nanmin(r) for r in rows)
    vmax = max(np.nanmax(r) for r in rows)
    K = ring.K
    fig, axes = plt.subplots(len(rows), K, figsize=(5.2 * K, 5.0 * len(rows)), facecolor=SURFACE, squeeze=False)
    for r, (grids, row_label, label_grid) in enumerate(zip(rows, row_labels, label_grids)):
        for a in range(K):
            ax = axes[r, a]
            im = ax.imshow(grids[a], origin='lower', extent=pixel_extent(axis, axis), cmap=SEQUENTIAL_BLUE, vmin=vmin,
                           vmax=vmax, interpolation='nearest')
            draw_density(ax, axis, density_grid, config['density_levels'])
            draw_decision_boundaries(ax, axis, axis, label_grid)
            draw_anchors(ax, ring, reference=a)
            style_axes(ax, config['lim'])
            if r == 0:
                ax.set_title(f"reference {ring.names[a]}  (argmax class {ring.label_dists[a].argmax()})", color=TEXT_PRIMARY)
            if a == 0:
                ax.set_ylabel(f"{row_label}\n$x_2$", color=TEXT_SECONDARY)
            if r == len(rows) - 1:
                ax.set_xlabel("$x_1$", color=TEXT_SECONDARY)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, extend=extend)
    cbar.set_label(cbar_label, color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    fig.suptitle(f"{title}\n{subtitle}", color=TEXT_SECONDARY)
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def plot_argmax_maps(label_grids, row_labels, ring, axis, density_grid, config, save_path, subtitle):
    C = ring.num_classes
    colors = [class_color(c) for c in range(C)]
    fig, axes = plt.subplots(1, len(label_grids), figsize=(6.5 * len(label_grids), 6), facecolor=SURFACE, squeeze=False)
    for ax, label_grid, row_label in zip(axes[0], label_grids, row_labels):
        ax.imshow(label_grid, origin='lower', extent=pixel_extent(axis, axis), cmap=ListedColormap(colors), vmin=-0.5,
                  vmax=C - 0.5, interpolation='nearest')
        draw_density(ax, axis, density_grid, config['density_levels'])
        draw_decision_boundaries(ax, axis, axis, label_grid, color=SURFACE, outline=SURFACE)
        draw_anchors(ax, ring)
        style_axes(ax, config['lim'])
        ax.set_title(row_label, color=TEXT_PRIMARY)
        ax.set_xlabel("$x_1$", color=TEXT_SECONDARY)
        ax.set_ylabel("$x_2$", color=TEXT_SECONDARY)

    owners = {int(ring.label_dists[a].argmax()): ring.names[a] for a in range(ring.K)}
    handles = [Patch(facecolor=class_color(c), label=f"class {c}" + (f" (argmax of {owners[c]})" if c in owners else " (filler)"))
               for c in range(min(C, len(CATEGORICAL)))]
    if C > len(CATEGORICAL):
        handles.append(Patch(facecolor=OTHER_COLOR, label=f"classes {len(CATEGORICAL)}–{C - 1}"))
    if config['density_levels']:
        levels = ", ".join(f"{level:g}" for level in config['density_levels'])
        handles.append(Line2D([], [], color=TEXT_SECONDARY, linestyle=':', label=f"input density ({levels} × peak)"))
    axes[0, -1].legend(handles=handles, title="argmax class", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False,
                       labelcolor=TEXT_PRIMARY)
    fig.suptitle(f"Argmax class on the input plane\n{subtitle}", color=TEXT_SECONDARY, y=1.06)
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def relative_deg(deg, center_deg):
    """Angles relative to center_deg, wrapped into [-180, 180)."""
    return (np.asarray(deg) - center_deg + 180) % 360 - 180


def plot_arc_profiles(arc_deg, learned, truth, learned_flips, true_flips, ring, config, save_path, subtitle):
    """
    Rows: metrics; columns: reference anchors. Learned seed mean with 95% CI vs ground truth along the arc, each
    column centered on its reference anchor (angle relative to it, so both neighbors sit symmetrically).
    """
    K = ring.K
    floor = config['kl_floor']
    fig, axes = plt.subplots(len(METRICS), K, figsize=(4.6 * K, 3.4 * len(METRICS)), facecolor=SURFACE,
                             sharey='row', squeeze=False)
    anchor_deg = np.rad2deg(ring.angles)
    for r, (key, (label, _)) in enumerate(METRICS.items()):
        mean, ci = summarize(learned[key])
        for a in range(K):
            ax = axes[r, a]
            rel = relative_deg(arc_deg, anchor_deg[a])
            order = np.argsort(rel)
            rel = rel[order]
            lo, hi = (mean[a] - ci[a]).numpy()[order], (mean[a] + ci[a]).numpy()[order]
            m, t = mean[a].numpy()[order], truth[key][a].numpy()[order]
            if key == 'kl' and config['log_scale']:
                lo, hi, m, t = (np.maximum(v, floor) for v in (lo, hi, m, t))
                ax.set_yscale('log')
            for deg in relative_deg(true_flips, anchor_deg[a]):
                ax.axvline(deg, color=OTHER_COLOR, linewidth=1.5, zorder=0)
            for deg in relative_deg(learned_flips, anchor_deg[a]):
                ax.axvline(deg, color=LEARNED_COLOR, linewidth=1, linestyle=':', zorder=0)
            ax.axvline(0, color=REFERENCE_COLOR, linewidth=1.5, alpha=0.8, zorder=0)
            ax.fill_between(rel, lo, hi, color=LEARNED_COLOR, alpha=0.25, linewidth=0)
            ax.plot(rel, m, color=LEARNED_COLOR, linewidth=2)
            ax.plot(rel, t, color=TEXT_PRIMARY, linewidth=1.2, linestyle='--')
            ax.set_xlim(-180, 180)
            ticks = relative_deg(anchor_deg, anchor_deg[a])
            ax.set_xticks(ticks, [f"{d:g}\n{n}" for d, n in zip(ticks, ring.names)] if r == len(METRICS) - 1 else [])
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(OTHER_COLOR)
            if r == 0:
                ax.set_title(f"reference {ring.names[a]}", color=TEXT_PRIMARY)
            if a == 0:
                ax.set_ylabel(f"{label}, nats", color=TEXT_SECONDARY)
            if r == len(METRICS) - 1:
                ax.set_xlabel(f"angle from {ring.names[a]}, deg", color=TEXT_SECONDARY)
    handles = [Line2D([], [], color=LEARNED_COLOR, linewidth=2, label="trained (seed mean, 95% CI)"),
               Line2D([], [], color=TEXT_PRIMARY, linewidth=1.2, linestyle='--', label="ground truth"),
               Line2D([], [], color=OTHER_COLOR, linewidth=1.5, label="true argmax boundary"),
               Line2D([], [], color=LEARNED_COLOR, linewidth=1, linestyle=':', label="learned argmax boundary"),
               Line2D([], [], color=REFERENCE_COLOR, linewidth=1.5, label="reference anchor")]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.03),
               labelcolor=TEXT_PRIMARY)
    radius = config['arc_radius'] if config['arc_radius'] is not None else ring.radius
    fig.suptitle(f"Divergence from each anchor along the circle of radius {radius:g}\n{subtitle}", color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(save_path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def load_ring(checkpoints) -> RingMixtureTask:
    """The task of the checkpoints (all seeds must share it)."""
    task_configs = [load_model_from_checkpoint(ckpt)[2]['task'] for ckpt in checkpoints]
    task = task_configs[0]
    if task['name'] != 'class_ring_mixture':
        raise ValueError(f"Checkpoints were trained on {task['name']!r}, not 'class_ring_mixture'")
    if any(t != task for t in task_configs[1:]):
        raise ValueError("Seeds were trained on different task configs")
    return RingMixtureTask(task['num_classes'], task.get('ring'))


def print_anchor_table(ring, learned_anchor, true_anchor):
    """Anchor-vs-anchor metrics (learned seed mean ± 95% CI | ground truth)."""
    print("Anchor-to-anchor metrics, trained (seed mean ± 95% CI) | ground truth, nats:")
    for a in range(ring.K):
        for b in range(ring.K):
            if a == b:
                continue
            cells = []
            for key, short in [('kl', 'KL(a||b)'), ('surprisal', '-lnP_a(argmax b)'), ('anchor_token_surprisal', '-lnP_b(argmax a)')]:
                mean, ci = summarize(learned_anchor[key][:, a, b])
                cells.append(f"{short} {mean:.3f}±{ci:.3f} | {true_anchor[key][a, b]:.3f}")
            print(f"  a={ring.names[a]} b={ring.names[b]}: " + "; ".join(cells))


def main():
    parser = argparse.ArgumentParser(description='KL divergence and argmax surprisal w.r.t. each anchor (class_ring_mixture)')
    parser.add_argument('--config', type=str, default='./vis_plateaus/ring_divergence_config.yaml')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir (run directory, seed directory or .pt file)')
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_dir = args.checkpoint_dir or config['checkpoint_dir']
    checkpoints = find_seed_checkpoints(checkpoint_dir)
    ring = load_ring(checkpoints)
    ring.print_stats()
    print(f"{len(checkpoints)} seed checkpoint(s) in {checkpoint_dir}")

    # Input-plane grid: grid point (r, c) = (axis[c], axis[r]), flattened row-major
    R = config['grid_resolution']
    axis = np.linspace(-config['lim'], config['lim'], R)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    # Full circle through the anchors (plotted relative to each reference anchor)
    radius = config['arc_radius'] if config['arc_radius'] is not None else ring.radius
    arc_deg = np.linspace(0, 360, config['arc_points'], endpoint=False)
    arc = radius * np.stack([np.cos(np.deg2rad(arc_deg)), np.sin(np.deg2rad(arc_deg))], axis=1)

    points = torch.from_numpy(np.concatenate([grid, arc])).float()
    n_grid = grid.shape[0]
    learned = learned_metrics(checkpoints, ring, points)
    truth = true_metrics(ring, points)

    # Anchor-to-anchor metrics
    anchors = torch.from_numpy(ring.means).float()
    learned_anchor = learned_metrics(checkpoints, ring, anchors)
    true_anchor = divergence_from_anchors(torch.from_numpy(np.log(ring.label_dists)), torch.from_numpy(np.log(ring.label_dists)))
    print_anchor_table(ring, learned_anchor, true_anchor)
    learned_anchor_labels = learned_anchor['labels'].tolist()
    if learned_anchor_labels != ring.label_dists.argmax(axis=1).tolist():
        print(f"WARNING: argmax class at the anchors {learned_anchor_labels} differs from the task's "
              f"{ring.label_dists.argmax(axis=1).tolist()}")

    # Argmax boundaries along the arc
    learned_flips = arc_angles_of_flips(learned['labels'][n_grid:].numpy(), arc_deg)
    true_flips = arc_angles_of_flips(truth['labels'][n_grid:].numpy(), arc_deg)
    print(f"Argmax boundaries on the arc (radius {radius:g}), deg: trained {[round(d, 1) for d in learned_flips]} | "
          f"ground truth {[round(d, 1) for d in true_flips]}")

    subtitle = task_subtitle(ring, len(checkpoints))
    density_grid = ring.density(grid).reshape(R, R)
    label_grids = [learned['labels'][:n_grid].reshape(R, R).numpy(), truth['labels'][:n_grid].reshape(R, R).numpy()]
    row_labels = [f"trained ({seeds_label(len(checkpoints))}, mean)", "ground truth"]

    out_dir = os.path.join(plot_dir(checkpoint_dir), "divergence")
    os.makedirs(out_dir, exist_ok=True)

    for key, (label, title) in METRICS.items():
        mean, _ = summarize(learned[key][:, :, :n_grid])
        rows = [mean.reshape(ring.K, R, R).numpy(), truth[key][:, :n_grid].reshape(ring.K, R, R).numpy()]
        cbar_label, extend = f"{label}, nats", 'neither'
        if key == 'kl' and config['log_scale']:
            floor = config['kl_floor']
            clipped = any((r < floor).any() for r in rows)
            rows = [np.log10(np.maximum(r, floor)) for r in rows]
            cbar_label = r"$\log_{10}$ " + cbar_label + (f"  (clipped at {floor:g})" if clipped else "")
            extend = 'min' if clipped else 'neither'
        print(f"{key}: trained range [{rows[0].min():.3g}, {rows[0].max():.3g}] | ground truth [{rows[1].min():.3g}, {rows[1].max():.3g}]"
              + (" (log10)" if key == 'kl' and config['log_scale'] else ""))
        save_path = os.path.join(out_dir, f"{key}_heatmap.png")
        plot_metric_panels(rows, row_labels, label_grids, ring, axis, density_grid, config, save_path, cbar_label,
                           f"{title} on the input plane", subtitle, extend=extend)
        print(f"Saved: {save_path}")

    save_path = os.path.join(out_dir, "argmax_labels.png")
    plot_argmax_maps(label_grids, row_labels, ring, axis, density_grid, config, save_path, subtitle)
    print(f"Saved: {save_path}")

    arc_learned = {key: learned[key][:, :, n_grid:] for key in METRICS}
    arc_truth = {key: truth[key][:, n_grid:] for key in METRICS}
    save_path = os.path.join(out_dir, "arc_profiles.png")
    plot_arc_profiles(arc_deg, arc_learned, arc_truth, learned_flips, true_flips, ring, config, save_path, subtitle)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
