"""
Shared matplotlib style and short labels for all plots. Importing this module applies the style.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'figure.titlesize': 18,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Shared palette: a sequential ramp (one hue, light -> dark), categorical slots in fixed order, and neutrals
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list('seq_blue', [
    '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER_COLOR = '#c9c8c3'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
SURFACE = '#fcfcfb'

# Font size for small in-plot annotations (contour labels, point coordinates)
ANNOTATION_FONTSIZE = 11

METRIC_LABELS = {
    'l2_norm': 'L2 distance to x*',
    'jacobian_norm_full': 'Jacobian norm',
    'jacobian_norm_layerwise_prod': 'Jacobian norm (layerwise product)',
    'jacobian_determinant_full': 'Jacobian det.',
    'jacobian_determinant_layerwise_prod': 'Jacobian det. (layerwise product)',
}


def layer_label(layer_idx) -> str:
    """-2 -> 'input', -1 -> 'embed', k -> 'block k', 'logits' -> 'logits'."""
    if layer_idx in ('input', 'embed', 'embedding', 'logits'):
        return {'embedding': 'embed'}.get(layer_idx, layer_idx)
    layer_idx = int(layer_idx)
    return {-2: 'input', -1: 'embed'}.get(layer_idx, f'block {layer_idx}')


def point_label(point) -> str:
    """[0.25, 0.0] -> '(0.25, 0)'."""
    return '(' + ', '.join(f'{float(v):g}' for v in point) + ')'


def seeds_label(n_seeds: int) -> str:
    return f'{n_seeds} seed' + ('s' if n_seeds != 1 else '')
