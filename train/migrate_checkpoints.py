#!/usr/bin/env python3
"""
One-off migration from the old checkpoint layout to the current one.

    old: checkpoints/{task}/{ResNetMLP[_tag]}/{checkpoint_name}/{timestamp}/
    new: checkpoints/{task}/noise-{std}[-{tag}]/seed-{k}/

Task, noise and seed are read from each run's config.yaml. A model directory with a suffix (e.g. ResNetMLP_epoch500)
becomes the run tag (noise-0.00-epoch500). ResNetMLPSkeleton runs are skipped since that model was removed.

Dry run by default; pass --apply to move the directories.
    python train/migrate_checkpoints.py [--root checkpoints] [--apply]
"""

import argparse
import os
import shutil
import sys

import yaml

sys.path.append('./common')
from checkpoints import run_dir, seed_dir


def find_old_runs(root: str):
    """Yield (model_dir_name, run_path) for every {task}/{model_dir}/{name}/{timestamp} containing config.yaml."""
    for task in sorted(os.listdir(root)):
        task_path = os.path.join(root, task)
        if not os.path.isdir(task_path):
            continue
        for model_dir in sorted(os.listdir(task_path)):
            if not model_dir.startswith("ResNetMLP"):
                continue
            for name in sorted(os.listdir(os.path.join(task_path, model_dir))):
                name_path = os.path.join(task_path, model_dir, name)
                if not os.path.isdir(name_path):
                    continue
                for timestamp in sorted(os.listdir(name_path)):
                    run_path = os.path.join(name_path, timestamp)
                    if os.path.isfile(os.path.join(run_path, "config.yaml")):
                        yield model_dir, run_path


def main():
    parser = argparse.ArgumentParser(description="Move checkpoints to the checkpoints/{task}/noise-{std}/seed-{k} layout")
    parser.add_argument("--root", type=str, default="checkpoints")
    parser.add_argument("--apply", action="store_true", help="Actually move directories (default: dry run)")
    args = parser.parse_args()

    moves = []
    for model_dir, run_path in find_old_runs(args.root):
        with open(os.path.join(run_path, "config.yaml")) as f:
            config = yaml.safe_load(f)
        if config.get('model_type', 'ResNetMLP') != 'ResNetMLP':
            print(f"skip  {run_path} (model_type {config['model_type']})")
            continue
        tag = model_dir[len("ResNetMLP"):].lstrip("_") or None
        dest = seed_dir(run_dir(config['task']['name'], config['task']['noise_std'], tag, root=args.root),
                        config['training']['seed'])
        moves.append((run_path, dest))

    destinations = [dest for _, dest in moves]
    for src, dest in moves:
        if os.path.exists(dest) or destinations.count(dest) > 1:
            print(f"CONFLICT  {src} -> {dest} (destination exists or is shared by several runs; not moved)")
            continue
        print(f"{'move' if args.apply else 'would move'}  {src} -> {dest}")
        if args.apply:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)

    if args.apply:
        # Remove the now-empty old directories
        for dirpath, _, _ in sorted(os.walk(args.root), key=lambda x: -len(x[0])):
            if os.path.isdir(dirpath) and not os.listdir(dirpath) and dirpath != args.root:
                os.rmdir(dirpath)
    else:
        print("\nDry run. Re-run with --apply to move.")


if __name__ == "__main__":
    main()
