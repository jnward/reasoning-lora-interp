#!/usr/bin/env python3
"""
Multi-GPU activation generation script for SAE training.
This script only generates H5 files with activations, no top-k tracking or JSON output.
"""

import torch
import torch.multiprocessing as mp
import glob
import os
import h5py
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import json


def extract_probe_directions(model, n_layers: int) -> Tuple[Dict[str, Dict[int, torch.Tensor]], List[int]]:
    """Extract LoRA A matrices (probe directions) from the model"""
    probe_directions = {
        'gate_proj': {},
        'up_proj': {},
        'down_proj': {},
        'q_proj': {},
        'k_proj': {},
        'v_proj': {},
        'o_proj': {}
    }
    lora_layers = set()
    
    for layer_idx in range(n_layers):
        # MLP projections
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A'):
                lora_A_weight = module.lora_A['default'].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
                lora_layers.add(layer_idx)
        
        # Attention projections
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            module = model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A'):
                lora_A_weight = module.lora_A['default'].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
                lora_layers.add(layer_idx)
    
    return probe_directions, sorted(list(lora_layers))


def process_rollout_simple(model, tokenizer, rollout_data, probe_directions: Dict, 
                           lora_layers: List[int], adapter_types: List[str], device: str):
    """Process a single rollout and return activations and tokens"""
    
    # Extract question and thinking trajectory
    question = rollout_data['question']
    thinking_trajectory = rollout_data.get('deepseek_thinking_trajectory', '')
    attempt = rollout_data.get('deepseek_attempt', '')
    
    if not thinking_trajectory or not attempt:
        return None
    
    # Format the input
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
    
    # Storage for activations
    projected_activations = {
        proj_type: {} for proj_type in adapter_types
    }
    
    # Hook functions
    def make_pre_mlp_hook(layer_idx):
        def hook(module, input, output):
            pre_mlp = output.detach()[0]
            for proj_type in ['gate_proj', 'up_proj']:
                if proj_type in adapter_types and layer_idx in probe_directions[proj_type]:
                    probe_dir = probe_directions[proj_type][layer_idx].to(device)
                    activations = torch.matmul(pre_mlp.float(), probe_dir)
                    projected_activations[proj_type][layer_idx] = activations.cpu().numpy()
        return hook
    
    def make_down_proj_hook(layer_idx):
        def hook(module, input, output):
            post_swiglu = input[0].detach()[0]
            if 'down_proj' in adapter_types and layer_idx in probe_directions['down_proj']:
                probe_dir = probe_directions['down_proj'][layer_idx].to(device)
                activations = torch.matmul(post_swiglu.float(), probe_dir)
                projected_activations['down_proj'][layer_idx] = activations.cpu().numpy()
        return hook
    
    def make_pre_attn_hook(layer_idx):
        def hook(module, input, output):
            pre_attn = output.detach()[0]
            for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                if proj_type in adapter_types and layer_idx in probe_directions[proj_type]:
                    probe_dir = probe_directions[proj_type][layer_idx].to(device)
                    activations = torch.matmul(pre_attn.float(), probe_dir)
                    projected_activations[proj_type][layer_idx] = activations.cpu().numpy()
        return hook
    
    def make_o_proj_hook(layer_idx):
        def hook(module, input, output):
            attn_output = input[0].detach()[0]
            if 'o_proj' in adapter_types and layer_idx in probe_directions['o_proj']:
                probe_dir = probe_directions['o_proj'][layer_idx].to(device)
                activations = torch.matmul(attn_output.float(), probe_dir)
                projected_activations['o_proj'][layer_idx] = activations.cpu().numpy()
        return hook
    
    # Register hooks only for layers with LoRA adapters
    hooks = []
    for layer_idx in lora_layers:
        # MLP hooks
        if any(pt in adapter_types for pt in ['gate_proj', 'up_proj']):
            layernorm = model.model.model.layers[layer_idx].post_attention_layernorm
            hook = layernorm.register_forward_hook(make_pre_mlp_hook(layer_idx))
            hooks.append(hook)
        
        if 'down_proj' in adapter_types:
            down_proj = model.model.model.layers[layer_idx].mlp.down_proj
            hook = down_proj.register_forward_hook(make_down_proj_hook(layer_idx))
            hooks.append(hook)
        
        # Attention hooks
        if any(pt in adapter_types for pt in ['q_proj', 'k_proj', 'v_proj']):
            ln_1 = model.model.model.layers[layer_idx].input_layernorm
            hook = ln_1.register_forward_hook(make_pre_attn_hook(layer_idx))
            hooks.append(hook)
        
        if 'o_proj' in adapter_types:
            o_proj = model.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_hook(make_o_proj_hook(layer_idx))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Combine activations into array
    activations_array = np.zeros((num_tokens, len(lora_layers), len(adapter_types)), dtype=np.float16)
    
    for layer_idx_pos, layer_idx in enumerate(lora_layers):
        for proj_idx, proj_type in enumerate(adapter_types):
            if layer_idx in projected_activations[proj_type]:
                activations_array[:, layer_idx_pos, proj_idx] = projected_activations[proj_type][layer_idx].astype(np.float16)
    
    return activations_array, num_tokens, tokens


def worker_process(gpu_id: int, args, start_idx: int, end_idx: int, lora_dir: str):
    """Worker process for a single GPU"""
    # Set GPU
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    print(f"GPU {gpu_id}: Processing rollouts {start_idx} to {end_idx}")
    
    # Dictionary to store tokens for this GPU
    gpu_tokens = {}
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model on specific GPU
    print(f"GPU {gpu_id}: Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={'': gpu_id},
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    
    # Extract probe directions
    probe_directions, lora_layers = extract_probe_directions(model, n_layers)
    
    # Move probe directions to GPU
    for proj_type in probe_directions:
        for layer_idx in probe_directions[proj_type]:
            probe_directions[proj_type][layer_idx] = probe_directions[proj_type][layer_idx].to(device)
    
    # Load dataset
    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    
    # Process assigned rollouts
    with tqdm(range(start_idx, end_idx), desc=f"GPU {gpu_id}", position=gpu_id) as pbar:
        for rollout_idx in pbar:
            if rollout_idx >= len(dataset):
                break
                
            rollout = dataset[rollout_idx]
            
            result = process_rollout_simple(
                model, tokenizer, rollout, probe_directions, 
                lora_layers, args.adapter_types, device
            )
            
            if result is not None:
                activations_array, num_tokens, tokens = result
                
                # Store tokens
                gpu_tokens[rollout_idx] = tokens
                
                # Save H5 file
                h5_path = os.path.join(args.output_dir, f'rollout_{rollout_idx}.h5')
                with h5py.File(h5_path, 'w') as f:
                    f.create_dataset('activations', data=activations_array, 
                                   compression='gzip', compression_opts=1)
                    f.attrs['num_tokens'] = num_tokens
                    f.attrs['num_layers'] = len(lora_layers)
                    f.attrs['projections'] = len(args.adapter_types)
                    f.attrs['rollout_idx'] = rollout_idx
                    f.attrs['adapter_types'] = args.adapter_types
            
            # Periodic memory cleanup
            if rollout_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Save tokens for this GPU
    import json
    tokens_path = os.path.join(args.output_dir, f'rollout_tokens_gpu{gpu_id}.json')
    with open(tokens_path, 'w') as f:
        json.dump(gpu_tokens, f)
    
    print(f"GPU {gpu_id}: Finished processing")


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU LoRA activation generation')
    parser.add_argument('--base-model', default="Qwen/Qwen2.5-32B-Instruct", help='Base model ID')
    parser.add_argument('--lora-path', default="/workspace/models/ckpts_1.1", help='Path to LoRA checkpoints')
    parser.add_argument('--rank', type=int, default=1, help='LoRA rank')
    parser.add_argument('--num-examples', type=int, default=1000, help='Number of examples to process')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use (default: all available)')
    parser.add_argument('--adapter-types', nargs='+', default=None,
                       choices=['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
                       help='Adapter types to process (default: all 7 types)')
    parser.add_argument('--output-dir', default=None, help='Output directory for H5 files')
    
    args = parser.parse_args()
    
    # Set adapter types
    if args.adapter_types is None:
        args.adapter_types = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    # Find LoRA checkpoint
    lora_dir = "/workspace/reasoning_interp/lora_checkpoints/s1-lora-32B-r1-20250627_013544"
    print(f"Using LoRA from: {lora_dir}")
    
    # Determine number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    else:
        args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    if args.num_gpus == 0:
        raise ValueError("No GPUs available")
    
    print(f"Using {args.num_gpus} GPUs")
    
    # Set output directory
    if args.output_dir is None:
        if set(args.adapter_types) == set(['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            args.output_dir = os.path.join(os.path.dirname(__file__), "activations_all_adapters")
        else:
            adapter_str = '-'.join([a[:1] for a in sorted(args.adapter_types)])
            args.output_dir = os.path.join(os.path.dirname(__file__), f"activations_{adapter_str}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Split work across GPUs
    examples_per_gpu = args.num_examples // args.num_gpus
    remainder = args.num_examples % args.num_gpus
    
    # Launch processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * examples_per_gpu + min(gpu_id, remainder)
        end_idx = start_idx + examples_per_gpu + (1 if gpu_id < remainder else 0)
        
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, args, start_idx, end_idx, lora_dir)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"\nAll processes completed. Generated {args.num_examples} H5 files in {args.output_dir}")
    
    # Merge token files
    print("Merging token files...")
    import json
    all_tokens = {}
    for gpu_id in range(args.num_gpus):
        tokens_path = os.path.join(args.output_dir, f'rollout_tokens_gpu{gpu_id}.json')
        if os.path.exists(tokens_path):
            with open(tokens_path, 'r') as f:
                gpu_tokens = json.load(f)
                all_tokens.update(gpu_tokens)
            # Remove temporary file
            os.remove(tokens_path)
    
    # Save merged tokens
    final_tokens_path = os.path.join(args.output_dir, 'rollout_tokens.json')
    with open(final_tokens_path, 'w') as f:
        json.dump(all_tokens, f, indent=2)
    
    print(f"Saved tokens to {final_tokens_path}")


if __name__ == "__main__":
    main()