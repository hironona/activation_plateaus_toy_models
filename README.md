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
4. **Visualize plateau geometry** (`vis_plateaus/`) — Sample points in input/activation space, color by a metric (L2 norm, Jacobian norm/determinant) averaged over seeds, and plot directly. With the input space as the source, the grid is drawn as a heat map with argmax-class boundaries (plus an argmax-class map), in the style of the GPT-2 plane plots. Includes contour visualization of open-ball pre-images and a KL heatmap.
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
│   ├── data.py            # ToyDataset: class_spiral, reg_sine_wave, class_ring_mixture
│   ├── plot_metrics.py    # Plot training curves (mean over seeds)
│   ├── migrate_checkpoints.py  # One-off move from the old checkpoint layout
│   ├── upload_hf.py       # Upload checkpoints to HuggingFace Hub
│   ├── config.yaml        # Training config
│   └── config_ring.yaml   # Training config for class_ring_mixture
├── common/                 # Shared by all pipelines
│   ├── checkpoints.py     # Checkpoint layout, seed discovery, model loading
│   ├── plot_style.py      # Matplotlib style (font sizes, shared palette) and short labels
│   ├── plot_maps.py       # Grid heat-map helpers (pixel extent, argmax boundaries, region labels)
│   ├── surfaces.py        # Plane / slerp-like sphere through three anchors, and grids on them
│   ├── attractors.py      # Attractor math: in-surface log-prob Jacobian (forward mode), pooled fields, anchor frames
│   └── attractor_plots.py # Attractor figures shared by the toy and GPT-2 pipelines
├── vis_plots/              # Interpolation-based analysis (toy ResNet)
│   ├── interpolate_and_record_activations.py  # Record activations along interpolation paths
│   ├── step_sizes_plots.py                    # L2 step sizes per layer
│   ├── relative_distances_layerwise_plots.py  # Relative distances (3 variants + --single_layer)
│   ├── relative_distances_logits_plots.py     # Relative distances in logit space
│   ├── spline_plots.py                        # Hamming distance of spline codes / step size
│   ├── jacobians_layerwise.py                 # ∂(layer i+1) / ∂(layer i) norms
│   ├── jacobians_full_residual.py             # ∂(last block) / ∂(embedding) norms
│   ├── jacobians_mlp.py                       # ∂(mlp_out) / ∂(resid_pre) norms per block
│   ├── utils.py           # Shared utilities (paths, metrics, seed aggregation, plotting)
│   └── config.yaml        # Analysis config (checkpoint_dir, interpolation pairs, n_steps)
├── vis_plateaus/           # Direct plateau geometry visualization (toy ResNet only)
│   ├── visualize_plateaus.py  # Metric heat map on the input grid (L2, Jacobian norm/det); scatter for hidden-layer sources
│   ├── visualize_contours.py  # Contour plot of open-ball pre-images in source space
│   ├── visualize_kl.py        # Heatmap of KL(p(y|x*) || p(y|x)) over the input grid
│   ├── visualize_ring_divergence.py  # class_ring_mixture: KL / surprisal w.r.t. each anchor, trained vs ground truth
│   ├── visualize_attractors.py  # Attractor fields: log-prob gradients on the source surface, block-by-block trajectories
│   ├── compute_metrics.py     # Jacobian computation (full, layerwise, to-logits, input-to-embed)
│   ├── utils.py               # Shared utilities for this pipeline
│   ├── config.yaml            # Plateau vis settings
│   ├── contour_config.yaml   # Contour vis settings (levels, spacing, sub-reference points)
│   ├── kl_config.yaml        # KL heatmap settings (reference point, log scale)
│   ├── ring_divergence_config.yaml  # Ring divergence settings (grid, arc, log scale, density contours)
│   └── attractor_config.yaml  # Attractor settings (source layer, anchors, frame anchors, grid, strides, seeds)
├── vis_plateaus_lm/        # Plateau geometry on a 2D plane in GPT-2's residual stream
│   ├── compute_plane.py   # Sample the plane, label by argmax next token, Jacobian norms → .pt (needs the model)
│   ├── plot_plane.py      # Heat map + decision boundaries, argmax-label map (reads the .pt only)
│   ├── compute_divergence.py  # KL / argmax surprisal w.r.t. each anchor on the same grid → .pt
│   ├── plot_divergence.py     # One panel per reference anchor for each divergence metric
│   ├── compute_attractors.py  # Log-prob gradient fields (two forward-mode JVPs per point) and block trajectories → .pt
│   ├── plot_attractors.py     # Gradient fields, trajectories, per-block updates, convergence curves (reads the .pt only)
│   ├── compute_metrics.py # GPT-2 loading, last-token forward passes, layerwise Jacobian product to logits
│   ├── utils.py           # Config, anchor words, paths (surface geometry re-exported from common/surfaces.py)
│   └── config.yaml        # Model, prefix, token pairs, source layer, grid settings
├── run/                    # Shell scripts for running experiments
├── activations/            # Recorded activation .pt files (intermediate)
├── checkpoints/            # Trained model weights
└── plots/                  # Generated figures
```

## Configuration

- **`train/config.yaml`**: Model architecture (`hidden_dim`, `resblock_width`, `num_blocks`), training settings (`lr`, `batch_size`, `epochs`, `lr_scheduler`), seeds (`seed` = first seed, `n_seeds`), `checkpoint_root`, optional `run_tag`, and task (`class_spiral`/`reg_sine_wave`/`class_ring_mixture`, `noise_std`, `num_classes`, `distribution`).
- **`train/config_ring.yaml`**: `class_ring_mixture`, a Gaussian mixture whose means (anchors A, B, C, ...) lie on a circle, used to relate KL divergence, surprisal and entropy between anchors to the plateau boundaries between them. `task.ring` sets the input density (`angles_deg`, `radius`, `sigmas`, `weights`) and, separately, the label function: each anchor's label distribution (`label_dists`), and per boundary between adjacent anchors a `transition_widths` entry (0 = hard step at the bisector, 1 = linear ramp from anchor to anchor along the angle), `interp_space` (`prob`/`logit`) and `targets` (`sample` = one-hot y ~ P(y|x), `soft` = P(y|x)). `noise_std` must be 0. Training prints each boundary's KL in both directions, cross-anchor argmax surprisals, entropies and where the true argmax changes. `python train/data.py --task class_ring_mixture --config train/config_ring.yaml` plots the samples and the true argmax/entropy maps.
- **`vis_plots/config.yaml`**: `n_steps` (interpolation resolution), `checkpoint_dir` (run directory), `layer_to_interpolate` (hook layer index: `-2` = input, `-1` = embed, `0+` = residual blocks), input pairs.
- **`vis_plateaus/*.yaml`**: All three configs take `checkpoint_dir` (run directory).
- **`vis_plateaus/config.yaml`**: `metric` (one of `l2_norm`, `jacobian_norm_full`, `jacobian_norm_layerwise_prod`, `jacobian_determinant_full`, `jacobian_determinant_layerwise_prod`), `source_layer_idx`/`target_layer_idx`, `n_points`, `radius`, `reference_point`, `n_pca_components`, `log_scale`. `--radius` and `--n_points` override the config, e.g. `--radius 2.0` for `class_ring_mixture`, whose anchors sit on the unit circle.
- **`vis_plateaus/contour_config.yaml`**: Same spatial settings plus `n_contour_levels`, `contour_level_spacing` (`even`/`log`), and `sub_reference_points` for marking additional points on the plot.
- **`vis_plateaus/kl_config.yaml`**: `n_points`, `radius`, `reference_point` (x* in KL(p(y|x*) ‖ p(y|x))), `sub_reference_points`, `log_scale`, `kl_floor` (lower clip for the log color scale).
- **`vis_plateaus/ring_divergence_config.yaml`**: For `class_ring_mixture` runs (the toy counterpart of `vis_plateaus_lm/compute_divergence.py` + `plot_divergence.py`, with the Gaussian means as anchors): `checkpoint_dir`, `grid_resolution` and `lim` (input-plane grid), `arc_radius`/`arc_points` (profiles along the circle through the anchors), `log_scale`, `kl_floor`, `density_levels` (dotted input-density contours). `visualize_ring_divergence.py` plots KL(P_a ‖ P_x), -ln P_a(argmax P_x) and -ln P_x(argmax P_a) for every anchor a, with the trained models (seed mean) above the task's ground truth, the argmax maps, and the arc profiles (seed mean ± 95% CI), into `plots/{run}/divergence/`.
- **Attractor fields** (`vis_plateaus/attractor_config.yaml`; `vis_plateaus_lm/compute_attractors.py` + `plot_attractors.py`): two views of error correction on the source surface, defined in `common/attractors.py`.
  - *Gradient-based*: the Jacobian of the log-probabilities along the surface (two forward-mode JVPs per point; arrows are unit steepest-ascent directions under the surface metric, the heat maps their log10 magnitude): the gradient of each anchor's argmax log-probability (one panel per anchor), the max-norm row (the class whose log-probability moves fastest, with a map of which class it is), and the top singular direction (sign-free segments). Log-probabilities rather than logits, so the shared logit shift that softmax ignores drops out.
  - *Block-based*: every grid point's residual stream at each layer from the source on (for GPT-2 the whole last-token block update, attention + MLP), in affine coordinates of the plane through the three frame anchors' activations at the same layer (anchors fixed at (0, 0), (1, 0), (0, 1)): trajectories over the basins of the final argmax, one panel per block update, and convergence curves (distance to the nearest anchor and from the anchors' plane, in mean anchor distances).
  - Toy: `python vis_plateaus/visualize_attractors.py [--source_layer_idx -1]`. `source_layer_idx` -2 uses the input plane (trajectories start at the embedding); -1 or k the plane through the frame anchors' activations. Anchors default to the ring means and `frame_anchors` to A/B/D; maps for `map_seeds`, convergence curves over all seeds, into `plots/{run}/attractors/src-{k}/`.
  - GPT-2: `attractor_batch_size`, `arrow_stride`, `trajectory_stride`; `--grid_resolution` overrides the grid for quick runs. Float32 log-probability gradients carry ~1e-2 relative noise from the softmax centering; the forward-mode Jacobian itself is validated against autograd on raw logit rows.
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
