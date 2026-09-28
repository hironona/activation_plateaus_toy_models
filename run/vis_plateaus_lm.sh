#!/bin/bash

# Plane visualization in GPT-2's residual stream: compute (GPU) then plot, for each source layer.
# Run from the project root. Plotting only needs the saved .pt files, so it can be rerun locally.

LAYERS=("-1" "0" "3" "6" "9" "11")

for layer in "${LAYERS[@]}"; do
    echo "Source layer ${layer}: computing..."
    python vis_plateaus_lm/compute_plane.py --source_layer_idx "${layer}" || exit 1
    echo "Source layer ${layer}: plotting..."
    python vis_plateaus_lm/plot_plane.py --source_layer_idx "${layer}" || exit 1
done

echo "All plots completed! Check: ./plots/gpt2-small/lm_plane/"
