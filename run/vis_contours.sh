#!/bin/bash

# Run directories are checkpoints/class_spiral/noise-{noise}{RUN_TAG} (see train/config.yaml: run_tag)
RUN_TAG="-epoch500"

echo "Visualizing activation contours..."

for noise in "0.00" "0.025" "0.05" "0.075"; do
    uv run vis_plateaus/visualize_contours.py --checkpoint_dir "checkpoints/class_spiral/noise-${noise}${RUN_TAG}"
done
