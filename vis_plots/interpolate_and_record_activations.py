#!/usr/bin/env python3
"""
Interpolate activations between input pairs in the toy ResNet.

For each input pair and each layer:
1. Collect resid_post activations for both inputs at all layers
2. Linearly interpolate between them at the specified layer
3. Inject interpolated activations and propagate through subsequent layers
4. Record activations (resid_post, mlp_out) and logits

Runs for every seed checkpoint of the configured run directory.
Outputs: one file per (checkpoint, input pair, interpolation layer) under activations/
"""

import torch
import os
import argparse
from typing import Dict
from tqdm import tqdm

import sys
sys.path.append('./vis_plots')
sys.path.append('./train')
sys.path.append('./common')
from utils import load_config, linear_interpolation, construct_filepath, get_pairs, get_pairs_ids, get_seed_checkpoints
from checkpoints import load_model_from_checkpoint
from model import ResNetMLP

config = load_config()


def collect_original_activations_toy_resnet(model: ResNetMLP, variable_data: torch.Tensor, device: str) -> Dict[str, torch.Tensor]:
    """
    Collect activations from a toy model.
    """
    inputs = variable_data
    
    activations = {}

    def create_collection_hook(layer_idx):
        """Create hook to collect the resid_post activations at each layer."""
        def hook_fn(module, input, output):
            activations[f"layer{layer_idx}_resid_post"] = output.cpu().clone()
        return hook_fn

    hooks = []

    # input layer index: -2
    hooks.append(model.hook_input.register_forward_hook(create_collection_hook(-2)))

    # embed layer index: -1
    hooks.append(model.hook_embed.register_forward_hook(create_collection_hook(-1)))

    for layer_idx, residual_block in enumerate(model.blocks):
        hooks.append(residual_block.hook_resid_post.register_forward_hook(create_collection_hook(layer_idx)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    
    # Clean up
    del inputs, logits
    torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    return activations

def interpolate_toy_resnet_layer(
    model: ResNetMLP,
    resid_post_a: Dict[str, torch.Tensor],
    resid_post_b: Dict[str, torch.Tensor],
    interpolation_layer: int,
    n_steps: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    
    n_layers = len(model.blocks)
    # Get activations at interpolation layer for both inputs
    resid_a = resid_post_a[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]    
    resid_b = resid_post_b[f'layer{interpolation_layer}_resid_post']  # [1, hidden_dim]
    
    # Compute linear interpolations
    resid_a_device = resid_a.to(device)
    resid_b_device = resid_b.to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)
    
    # Linear interpolation
    interpolated_activations = torch.stack([
        linear_interpolation(resid_a_device, resid_b_device, alpha.item()).squeeze(0)
        for alpha in alphas
    ])  # [n_steps, hidden_dim]
    
    # Storage for collected activations
    activations = {f'layer{interpolation_layer}_resid_post': interpolated_activations.cpu().clone()}
    
    # Hook to inject interpolated activations
    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            output = list(output)
            output[0] = interpolated_activations
            return tuple(output)
        else:
            output = interpolated_activations
            return output
    
    # Hook to collect resid_post (after MLP, final layer output)
    def create_resid_post_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_resid_post'] = hidden_states.cpu().clone()
            return output
        return hook_fn
    
    def create_mlp_out_hook(target_layer):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store activation [n_steps, hidden_dim]
            activations[f'layer{target_layer}_mlp_out'] = hidden_states.cpu().clone()
            return output
        return hook_fn
    
    # get layers
    layers = list(model.blocks) + [model.hook_input, model.hook_embed]  # hook_input is layer -2, hook_embed is layer -1

    # Register hooks
    hooks = []
    
    injection_hook = layers[interpolation_layer].register_forward_hook(inject_hook)
    hooks.append(injection_hook)
    
    # Collection hooks for subsequent layers
    for i in range(interpolation_layer, n_layers):
        layer = layers[i]
        
        # Hook layer output (resid_post) - final output of the layer
        if i < 0:   # hook_input or hook_embed
            layer_hook = layer.register_forward_hook(create_resid_post_hook(i))
            hooks.append(layer_hook)
        else:   # residual blocks
            layer_hook = layer.hook_resid_post.register_forward_hook(create_resid_post_hook(i))
            hooks.append(layer_hook)
        
        # Hook mlp_out
        if i < 0:   # hook_input or hook_embed. Hamming distance will be 0
            pass
        else:   # residual blocks
            mlp_out_hook = layer.hook_mlp_out.register_forward_hook(create_mlp_out_hook(i))
            hooks.append(mlp_out_hook)
    
    # Create dummy inputs
    batched_inputs = torch.zeros((n_steps, model.input_dim), device=device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(batched_inputs)
    
    activations['logits'] = logits.cpu().clone()
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # Clean up
    del interpolated_activations, logits, batched_inputs
    torch.cuda.empty_cache()
    
    return activations

def run_for_single_model(args, config, checkpoint_path):
    """Run activation recording for a single model checkpoint."""
    n_steps = config['n_steps']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _, _ = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device)
    n_layers = len(model.blocks)
    print(f"Loaded {n_layers}-layer model on {device}")

    pairs = [torch.tensor(pair) for pair in get_pairs(config)]
    pairs_ids = get_pairs_ids(config)

    for pair, pair_ids in zip(pairs, pairs_ids):
        print(f"\nProcessing {pair_ids}")

        # Collect reference activations for both inputs
        resid_post_a, resid_post_b = [
            collect_original_activations_toy_resnet(model, point, device) for point in pair
        ]

        # Interpolate at the input layer only (-2), or at every layer from the input (-2) and embedding (-1) on
        layers = [-2] if args.interpolate_only_first_layer else range(-2, n_layers)
        for interpolation_layer in tqdm(layers, desc=f"Interpolating {pair_ids}"):
            interpolated_activations = interpolate_toy_resnet_layer(
                model=model,
                resid_post_a=resid_post_a,
                resid_post_b=resid_post_b,
                interpolation_layer=interpolation_layer,
                n_steps=n_steps,
                device=device,
            )

            filepath = construct_filepath(checkpoint_path, interpolation_layer, pair_ids, n_steps)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(interpolated_activations, filepath)


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between input pairs')
    parser.add_argument('--interpolate_only_first_layer', action='store_true', help='Only interpolate at the input layer for less data')
    parser.add_argument('--checkpoint_dir', type=str, help='Override checkpoint_dir from config')
    args = parser.parse_args()

    checkpoints = get_seed_checkpoints(config, args.checkpoint_dir)
    for idx, checkpoint_path in enumerate(checkpoints):
        print(f"\n=== Seed {idx + 1}/{len(checkpoints)}: {checkpoint_path} ===")
        run_for_single_model(args, config, checkpoint_path)

    print("\n=== Complete ===")

if __name__ == "__main__":
    main()
