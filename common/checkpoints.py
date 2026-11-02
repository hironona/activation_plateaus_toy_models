"""
Checkpoint layout shared by training and all analysis scripts.

    checkpoints/{task}/noise-{std}[-{run_tag}]/seed-{k}/
        best_model.pt, checkpoint_epoch_{N}.pt, metrics.json, config.yaml

A "run directory" (checkpoints/{task}/noise-{std}) holds one seed-* directory per seed. Analysis scripts take a run
directory and use the final-epoch checkpoint of every seed. Plots and recorded activations mirror the same layout
under plots/ and activations/.
"""

import os
import re
import sys
from typing import List, Optional, Tuple

import torch

sys.path.append('./train')
from model import ResNetMLP

CHECKPOINT_ROOT = "checkpoints"
_SEED_DIR = re.compile(r"^seed-(\d+)$")
_EPOCH_FILE = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")


def format_noise(noise_std: float) -> str:
    """0.0 -> '0.00', 0.05 -> '0.05', 0.025 -> '0.025'."""
    return f"{noise_std:.2f}" if round(noise_std, 2) == noise_std else f"{noise_std:g}"


def run_dir(task_name: str, noise_std: float, run_tag: Optional[str] = None, root: str = CHECKPOINT_ROOT) -> str:
    name = f"noise-{format_noise(noise_std)}"
    if run_tag:
        name += f"-{run_tag}"
    return os.path.join(root, task_name, name)


def seed_dir(run_directory: str, seed: int) -> str:
    return os.path.join(run_directory, f"seed-{seed}")


def latest_checkpoint(directory: str) -> str:
    """Path of the checkpoint_epoch_{N}.pt with the largest N in a seed directory."""
    epochs = {int(m.group(1)): f for f in os.listdir(directory) if (m := _EPOCH_FILE.match(f))}
    if not epochs:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pt files in {directory}")
    return os.path.join(directory, epochs[max(epochs)])


def list_seed_dirs(run_directory: str) -> List[str]:
    """seed-* directories of a run directory, ordered by seed."""
    seeds = sorted((int(m.group(1)), d) for d in os.listdir(run_directory) if (m := _SEED_DIR.match(d)))
    if not seeds:
        raise FileNotFoundError(f"No seed-* directories in {run_directory}")
    return [os.path.join(run_directory, d) for _, d in seeds]


def find_seed_checkpoints(path: str) -> List[str]:
    """
    Final-epoch checkpoint of every seed under a run directory, ordered by seed.
    A seed directory or a single .pt file is also accepted (one checkpoint).
    """
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    if _SEED_DIR.match(os.path.basename(os.path.normpath(path))):
        return [latest_checkpoint(path)]

    return [latest_checkpoint(d) for d in list_seed_dirs(path)]


def load_model_from_checkpoint(checkpoint_path: str) -> Tuple[ResNetMLP, dict, dict]:
    """Returns (model in eval mode, model config, full training config)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_config = checkpoint['config']
    model_config = full_config['model']
    model = ResNetMLP(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, model_config, full_config


def artifact_subdir(path: str) -> str:
    """
    Path of a run directory / checkpoint relative to checkpoints/, without the .pt extension. Used to mirror the
    checkpoint layout under plots/ and activations/.
    """
    rel = os.path.relpath(os.path.splitext(os.path.normpath(path))[0], CHECKPOINT_ROOT)
    if rel.startswith('..'):
        raise ValueError(f"Checkpoints must live under {CHECKPOINT_ROOT}/ (got {path})")
    return rel


def plot_dir(path: str) -> str:
    return os.path.join("plots", artifact_subdir(path))
