#!/bin/bash

# Interpolation analyses of the toy ResNet (checkpoint_dir and input pairs in vis_plots/config.yaml).
# Every script averages over all seed checkpoints of the run directory.

echo "Running activation extraction..."
uv run ./vis_plots/interpolate_and_record_activations.py

echo "Generating plots..."
uv run ./vis_plots/step_sizes_plots.py
uv run ./vis_plots/relative_distances_layerwise_plots.py
uv run ./vis_plots/relative_distances_layerwise_plots.py --single_layer 9
uv run ./vis_plots/relative_distances_logits_plots.py

uv run ./vis_plots/spline_plots.py

uv run ./vis_plots/jacobians_layerwise.py

echo "All plots completed! Check: ./plots/"
