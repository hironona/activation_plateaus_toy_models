# Activation Plateau Geometry

> [!IMPORTANT]
> **Work in Progress**: This project is actively under development on the `dev` branch.

Research project analyzing the geometry of activation plateaus and residual stream dynamics in ResNet-like MLPs trained on toy 2D tasks. Investigates how activation representations evolve through residual blocks via interpolation, Jacobian analysis, and geometric visualization of plateau structures.

Forked from [MShinkle/activation_plateau_mechanisms](https://github.com/MShinkle/activation_plateau_mechanisms).

## Pipeline Overview

The project follows a four-phase workflow, focused on toy ResNet models trained on 2D classification tasks:

1. **Train** (`train/`) — Train `ResNetMLP` on synthetic tasks (e.g., spiral classification), one model per seed.
2. **Record activations** (`vis_plots/interpolate_and_record_activations.py`) — Interpolate between input pairs (simple linear interpolation), run through the model, and record residual stream activations at every layer. Saves `.pt` files to `activations/`.
3. **Plot interpolation metrics** (`vis_plots/*_plots.py`, `vis_plots/jacobians_*.py`) — Load recorded activations and plot step sizes, relative distances, spline (Hamming) distances, and Jacobian norms across layers, as the mean over seeds with confidence intervals.
4. **Visualize plateau geometry** (`vis_plateaus/`) — Sample points in input/activation space, color by a metric (L2 norm, Jacobian norm/determinant) averaged over seeds, and plot directly. Includes contour visualization of open-ball pre-images and a KL heatmap.
5. **Visualize plateau geometry in GPT-2** (`vis_plateaus_lm/`) — Take the last-token activations of three sentences (e.g. "The house is" + " large"/" big"/" in") in the output space of a source layer, sample a grid on the plane (or the slerp-like spherical surface) through them, label each point with its argmax next token, and color it by the Frobenius norm of the layerwise Jacobian product from the source layer to the logits.

## Quick Start

```bash
# 1. Install
uv sync

# 2. Train (all seeds in train/config.yaml)
python train/train.py --config train/config.yaml

# 3. Plot training curves
python train/plot_metrics.py --checkpoint_dir checkpoints/class_spiral/noise-0.00

# 4. Record activations & generate metric plots
bash run/toy_resnet_vis.sh

# 5. Visualize plateau geometry (loops over noise levels)
bash run/vis_plateaus.sh

# 6. Visualize contour pre-images
bash run/vis_contours.sh

# 7. GPT-2 plane visualization (compute on GPU, plot anywhere from the saved .pt)
bash run/vis_plateaus_lm.sh
```

All scripts must be run from the project root.

## Checkpoints, Activations and Plots

```
checkpoints/{task}/noise-{std}[-{run_tag}]/seed-{k}/   best_model.pt, checkpoint_epoch_{N}.pt, metrics.json, config.yaml
activations/{task}/noise-{std}/seed-{k}/checkpoint_epoch_{N}/...
plots/{task}/noise-{std}/...
```

A run directory (`checkpoints/{task}/noise-{std}`) holds one `seed-{k}` directory per seed. Analysis configs point `checkpoint_dir` at a run directory; every script uses the final-epoch checkpoint of each seed and averages over them. `--checkpoint_dir` overrides the config and also accepts a single `seed-{k}` directory or `.pt` file. Training refuses to overwrite existing seed directories unless `--overwrite` is given.

Checkpoints from the old layout (`checkpoints/{task}/ResNetMLP[_tag]/{name}/{timestamp}/`) can be moved with:

```bash
python train/migrate_checkpoints.py          # dry run: prints the planned moves
python train/migrate_checkpoints.py --apply  # moves the directories
```

## Project Structure

```
.
├── train/                  # Training pipeline
│   ├── train.py           # Training entry point (--dry_run, --overwrite)
│   ├── model.py           # ResNetMLP architecture
│   ├── data.py            # ToyDataset: class_spiral, reg_sine_wave
│   ├── plot_metrics.py    # Plot training curves (mean over seeds)
│   ├── migrate_checkpoints.py  # One-off move from the old checkpoint layout
│   ├── upload_hf.py       # Upload checkpoints to HuggingFace Hub
│   └── config.yaml        # Training config
├── common/                 # Shared by all pipelines
│   ├── checkpoints.py     # Checkpoint layout, seed discovery, model loading
│   └── plot_style.py      # Matplotlib style (font sizes) and short labels
├── vis_plots/              # Interpolation-based analysis (toy ResNet)
│   ├── interpolate_and_record_activations.py  # Record activations along interpolation paths
│   ├── step_sizes_plots.py                    # L2 step sizes per layer
│   ├── relative_distances_layerwise_plots.py  # Relative distances (3 variants + --single_layer)
│   ├── relative_distances_logits_plots.py     # Relative distances in logit space
│   ├── spline_plots.py                        # Hamming distance of spline codes / step size
│   ├── jacobians_layerwise.py                 # ∂(layer i+1) / ∂(layer i) norms
│   ├── jacobians_full_residual.py             # ∂(last layer) / ∂(layer 0) norms
│   ├── jacobians_mlp.py                       # ∂(mlp_out) / ∂(resid_pre) norms per block
│   ├── utils.py           # Shared utilities (paths, metrics, seed aggregation, plotting)
│   └── config.yaml        # Analysis config (checkpoint_dir, interpolation pairs, n_steps)
├── vis_plateaus/           # Direct plateau geometry visualization (toy ResNet only)
│   ├── visualize_plateaus.py  # Scatter plot colored by metric (L2, Jacobian norm/det)
│   ├── visualize_contours.py  # Contour plot of open-ball pre-images in source space
│   ├── visualize_kl.py        # Heatmap of KL(p(y|x*) || p(y|x)) over the input grid
│   ├── compute_metrics.py     # Jacobian computation (full, layerwise, to-logits, input-to-embed)
│   ├── utils.py               # Shared utilities for this pipeline
│   ├── config.yaml            # Plateau vis settings
│   ├── contour_config.yaml   # Contour vis settings (levels, spacing, sub-reference points)
│   └── kl_config.yaml        # KL heatmap settings (reference point, log scale)
├── vis_plateaus_lm/        # Plateau geometry on a 2D plane in GPT-2's residual stream
│   ├── compute_plane.py   # Sample the plane, label by argmax next token, Jacobian norms → .pt (needs the model)
│   ├── plot_plane.py      # Heat map + decision boundaries, argmax-label map (reads the .pt only)
│   ├── compute_metrics.py # GPT-2 loading, last-token forward passes, layerwise Jacobian product to logits
│   ├── utils.py           # Plane geometry, grid, paths
│   └── config.yaml        # Model, prefix, token pairs, source layer, grid settings
├── run/                    # Shell scripts for running experiments
├── activations/            # Recorded activation .pt files (intermediate)
├── checkpoints/            # Trained model weights
└── plots/                  # Generated figures
```

## Configuration

- **`train/config.yaml`**: Model architecture (`hidden_dim`, `resblock_width`, `num_blocks`), training settings (`lr`, `batch_size`, `epochs`, `lr_scheduler`), seeds (`seed` = first seed, `n_seeds`), `checkpoint_root`, optional `run_tag`, and task (`class_spiral`/`reg_sine_wave`, `noise_std`, `num_classes`, `distribution`).
- **`vis_plots/config.yaml`**: `n_steps` (interpolation resolution), `checkpoint_dir` (run directory), `layer_to_interpolate` (hook layer index: `-2` = input, `-1` = embed, `0+` = residual blocks), input pairs.
- **`vis_plateaus/*.yaml`**: All three configs take `checkpoint_dir` (run directory).
- **`vis_plateaus/config.yaml`**: `metric` (one of `l2_norm`, `jacobian_norm_full`, `jacobian_norm_layerwise_prod`, `jacobian_determinant_full`, `jacobian_determinant_layerwise_prod`), `source_layer_idx`/`target_layer_idx`, `n_points`, `radius`, `reference_point`, `n_pca_components`, `log_scale`.
- **`vis_plateaus/contour_config.yaml`**: Same spatial settings plus `n_contour_levels`, `contour_level_spacing` (`even`/`log`), and `sub_reference_points` for marking additional points on the plot.
- **`vis_plateaus/kl_config.yaml`**: `n_points`, `radius`, `reference_point` (x* in KL(p(y|x*) ‖ p(y|x))), `sub_reference_points`, `log_scale`, `kl_floor` (lower clip for the log color scale).
- **`vis_plateaus_lm/config.yaml`**: `model_name`, `prefix` and `token_pairs` (the three unique words span the plane; each `" " + word` must be one token), `source_layer_idx` (`-1` = embedding, `k` = output of block `k`), `surface` (`flat` = affine plane; `spherical` = slerp-like curved surface whose edges from the first anchor are exactly the `slerp_rescale` paths), `grid_resolution`, `margin`, `batch_size`, `jacobian_chunk_size`, `log_scale`, `max_label_classes`.

## Key Flags

| Script | Flag | Description |
|--------|------|-------------|
| `train.py` | `--dry_run` | Single epoch for testing |
| `train.py` | `--overwrite` | Replace existing seed directories |
| `vis_plots/*.py`, `vis_plateaus/visualize_*.py` | `--checkpoint_dir PATH` | Override `checkpoint_dir` from config |
| `interpolate_and_record_activations.py` / `relative_distances_layerwise_plots.py` | `--interpolate_only_first_layer` | Only interpolate at the input layer (faster) |
| `relative_distances_layerwise_plots.py` | `--single_layer N` | Plot only for target layer N |
| `compute_plane.py` / `plot_plane.py` | `--source_layer_idx N` | Override the source layer from config |
| `compute_plane.py` / `plot_plane.py` | `--surface flat\|spherical` | Override the sampling surface from config |
| `plot_plane.py` | `--input PATH` | Plot a specific `.pt` file |

## Model Architecture

- **`ResNetMLP`**: Projects 2D input to `hidden_dim` via a linear layer, then passes through `num_blocks` residual blocks (each: LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> Linear + skip connection), followed by a final LayerNorm -> ReLU -> output linear.
- Each `ResidualBlock` has `nn.Identity()` hook points (`hook_resid_pre`, `hook_mlp_out`, `hook_resid_post`) for activation recording. The model adds `hook_input` and `hook_embed`.

## Acknowledgements

This project builds upon concepts from [MShinkle/activation_plateau_mechanisms](https://github.com/MShinkle/activation_plateau_mechanisms).
