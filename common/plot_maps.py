"""
Helpers for heat maps on a regular 2D grid, shared by the GPT-2 plane plots (vis_plateaus_lm/) and the toy input-plane
plots (vis_plateaus/): pixel extents, argmax decision boundaries and region labels.
"""

import numpy as np
import matplotlib.patheffects as pe
from scipy.ndimage import distance_transform_edt

from plot_style import ANNOTATION_FONTSIZE, TEXT_PRIMARY, SURFACE


def pixel_extent(axis_u: np.ndarray, axis_v: np.ndarray):
    """imshow extent whose pixel centers sit on the grid points (so contours and fills line up)."""
    du, dv = axis_u[1] - axis_u[0], axis_v[1] - axis_v[0]
    return (axis_u[0] - du / 2, axis_u[-1] + du / 2, axis_v[0] - dv / 2, axis_v[-1] + dv / 2)


def draw_decision_boundaries(ax, axis_u, axis_v, label_grid, color='white', outline=TEXT_PRIMARY):
    """Boundaries between argmax regions: the 0.5 level set of each label's indicator on the grid [len(v), len(u)]."""
    for label in np.unique(label_grid):
        mask = (label_grid == label).astype(float)
        if mask.all():
            continue
        cs = ax.contour(axis_u, axis_v, mask, levels=[0.5], colors=color, linewidths=1.2)
        cs.set(path_effects=[pe.Stroke(linewidth=2.6, foreground=outline), pe.Normal()])


def draw_region_labels(ax, axis_u, axis_v, label_grid, label_text: dict, labels_to_annotate, min_fraction=0.01):
    """Annotate each large region with label_text[label], at the pixel farthest from the region's boundary."""
    for label in labels_to_annotate:
        mask = label_grid == label
        if mask.mean() < min_fraction:
            continue
        dist = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        r, c = np.unravel_index(np.argmax(dist), dist.shape)
        ax.text(axis_u[c], axis_v[r], label_text[label], ha='center', va='center', fontsize=ANNOTATION_FONTSIZE, color=TEXT_PRIMARY,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=SURFACE, edgecolor='none', alpha=0.85))
