#!/usr/bin/env python3
"""
Extract MLP neuron activations from the model WITH LoRA adapters.
We use the same model as LoRA experiments but look at MLP neurons instead of LoRA projections.
Extracts the first 6 neurons from each layer's up_proj for baseline comparison.
"""

import torch
import torch.nn.functional as F
import glob
import os
import h5py
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple


def process_rollout(model, tokenizer, rollout_data, device: str, n_neurons: int = 6):
    """
    Process a single rollout and extract MLP neuron activations.
    
    Args:
        model: The model WITH LoRA adapters
        tokenizer: Tokenizer
        rollout_data: Data for this rollout
        device: Device to run on
        n_neurons: Number of neurons to extract per layer (default: 6)
    
    Returns:
        activations_array: Shape (num_tokens, n_layers, n_neurons)
        num_tokens: Number of tokens
        tokens: List of decoded tokens
    """
    
    # Extract question and thinking trajectory
    question = rollout_data['question']
    thinking_trajectory = rollout_data.get('deepseek_thinking_trajectory', '')
    attempt = rollout_data.get('deepseek_attempt', '')
    
    if not thinking_trajectory or not attempt:
        return None
    
    # Format the input (same format as LoRA experiments)
    system_prompt = "You are a helpful mathematics assistant."
    full_text = (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{question}\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{thinking_trajectory}\n"
        f"<|im_start|>answer\n{attempt}<|im_end|>"
    )
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]
    num_tokens = len(input_ids)
    
    # Decode tokens
    tokens = []
    for i in range(len(input_ids)):
        decoded = tokenizer.decode(input_ids[i:i+1])
        tokens.append(decoded)
    
    # Get number of layers - access through model.model for PEFT
    if hasattr(model, 'model'):  # PEFT wrapped
        n_layers = len(model.model.model.layers)
    else:
        n_layers = len(model.model.layers)
    
    # Storage for activations
    mlp_activations = np.zeros((num_tokens, n_layers, n_neurons), dtype=np.float16)
    
    # Hook function to capture MLP up_proj output (before activation function)
    def make_mlp_hook(layer_idx):
        def hook(module, input, output):
            # Get the output of up_proj (before SiLU activation)
            # Shape: (batch, seq_len, hidden_dim)
            up_proj_output = output.detach()[0]  # Remove batch dimension
            
            # Extract first n_neurons and convert from bfloat16 to float32 first
            # BFloat16 can't be directly converted to numpy
            activations_slice = up_proj_output[:, :n_neurons].float().cpu().numpy().astype(np.float16)
            mlp_activations[:, layer_idx, :] = activations_slice
        return hook
    
    # Register hooks on up_proj layers
    hooks = []
    for layer_idx in range(n_layers):
        if hasattr(model, 'model'):  # PEFT wrapped
            up_proj = model.model.model.layers[layer_idx].mlp.up_proj
        else:
            up_proj = model.model.layers[layer_idx].mlp.up_proj
        hook = up_proj.register_forward_hook(make_mlp_hook(layer_idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return mlp_activations, num_tokens, tokens


def main():
    parser = argparse.ArgumentParser(description="Extract MLP neuron activations from LoRA-adapted model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct",
                       help="Base model name")
    parser.add_argument("--lora-dir", default="/workspace/models/ckpts_1.1/s1-lora-32B-r1-20250627_013544",
                       help="Path to LoRA adapter")
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1",
                       help="Dataset name")
    parser.add_argument("--output-dir", default="data/mlp_activations",
                       help="Output directory for H5 files")
    parser.add_argument("--num-rollouts", type=int, default=None,
                       help="Number of rollouts to process (None = all)")
    parser.add_argument("--n-neurons", type=int, default=6,
                       help="Number of neurons to extract per layer (default: 6)")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split='train')
    
    # Limit number of rollouts if specified
    if args.num_rollouts:
        dataset = dataset.select(range(min(args.num_rollouts, len(dataset))))
    
    print(f"Processing {len(dataset)} rollouts...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with LoRA
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_dir, torch_dtype=torch.bfloat16)
    model.eval()
    
    print(f"Model loaded. Number of layers: {model.config.num_hidden_layers}")
    
    # Store tokens separately (once per dataset)
    all_tokens = {}
    
    # Process each rollout
    for rollout_idx, rollout_data in enumerate(tqdm(dataset, desc="Processing rollouts")):
        try:
            result = process_rollout(model, tokenizer, rollout_data, args.device, args.n_neurons)
            
            if result is None:
                print(f"Skipping rollout {rollout_idx} (no trajectory)")
                continue
            
            activations, num_tokens, tokens = result
            
            # Save activations to H5 file
            output_file = os.path.join(args.output_dir, f"rollout_{rollout_idx}.h5")
            with h5py.File(output_file, 'w') as f:
                # Store activations
                # Shape: (num_tokens, n_layers, n_neurons)
                f.create_dataset('activations', data=activations, compression='gzip', compression_opts=1)
                
                # Store metadata
                f.attrs['num_tokens'] = num_tokens
                f.attrs['n_layers'] = activations.shape[1]
                f.attrs['n_neurons'] = args.n_neurons
                f.attrs['rollout_idx'] = rollout_idx
            
            # Store tokens
            all_tokens[rollout_idx] = tokens
            
        except Exception as e:
            print(f"Error processing rollout {rollout_idx}: {e}")
            continue
    
    # Save tokens to separate JSON file
    tokens_file = os.path.join(args.output_dir, "tokens.json")
    with open(tokens_file, 'w') as f:
        json.dump(all_tokens, f)
    
    print(f"Saved activations to {args.output_dir}")
    print(f"Saved tokens to {tokens_file}")
    
    # Print summary
    n_files = len(glob.glob(os.path.join(args.output_dir, "rollout_*.h5")))
    print(f"\nSummary:")
    print(f"  Processed rollouts: {n_files}")
    print(f"  Neurons per layer: {args.n_neurons}")
    print(f"  Total features: {model.config.num_hidden_layers * args.n_neurons} "
          f"({model.config.num_hidden_layers} layers × {args.n_neurons} neurons)")


if __name__ == "__main__":
    main()