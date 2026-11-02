#!/bin/bash

echo "Visualizing training metrics..."

uv run train/plot_metrics.py --checkpoint_dir "checkpoints/class_spiral/noise-0.00"