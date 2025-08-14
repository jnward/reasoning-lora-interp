#!/usr/bin/env python3
"""Test script to debug steering hooks"""

import torch
import sys
import os
import numpy as np
from typing import Dict, List, Optional
import pandas as pd

# Setup paths
sys.path.append('/workspace/reasoning_interp')
from sae_interp.batch_topk_sae import BatchTopKSAE

# Import the IncrementalSteeringManager class
exec(open('sae_steering_experiment.py').read().split('class IncrementalSteeringManager')[1].split('def run_steering_experiment')[0])

# Load SAE
print("Loading SAE...")
sae_path = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)
sae_config = checkpoint["config"]

# Create SAE model
sae_model = BatchTopKSAE(
    d_model=sae_config["d_model"],
    dict_size=sae_config["dict_size"],
    k=sae_config["k"]
)

# Load weights
state_dict = checkpoint["model_state_dict"]
sae_model.W_enc.data = state_dict["W_enc"]
sae_model.b_enc.data = state_dict["b_enc"]
sae_model.W_dec.data = state_dict["W_dec"]
sae_model.b_dec.data = state_dict["b_dec"]
sae_model.eval()

print(f"SAE loaded: {sae_config['dict_size']} features, d={sae_config['d_model']}")

# Load model and tokenizer
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob

base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r1-*544")
lora_dir = sorted(lora_dirs)[-1]

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)
model.eval()
device = next(model.parameters()).device
sae_model = sae_model.to(device)

print("Model loaded!")

# Extract probe directions
def extract_lora_probe_directions(model, n_layers, adapter_types):
    probe_directions = {adapter_type: {} for adapter_type in adapter_types}
    
    for layer_idx in range(n_layers):
        # MLP projections
        for proj_type in ["gate_proj", "up_proj", "down_proj"]:
            if proj_type not in adapter_types:
                continue
            module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
            
            if hasattr(module, "lora_A"):
                lora_A_weight = module.lora_A["default"].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
        
        # Attention projections
        for proj_type in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if proj_type not in adapter_types:
                continue
            module = model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
            
            if hasattr(module, "lora_A"):
                lora_A_weight = module.lora_A["default"].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
    
    return probe_directions

adapter_types = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
probe_directions = extract_lora_probe_directions(model, 64, adapter_types)

print("Probe directions extracted!")

# Create a simple steering vector
test_steering = torch.randn(448, device=device) * 10.0
print(f'\nCreated test steering vector with norm: {torch.norm(test_steering).item():.2f}')
print(f'Sample values: {test_steering[:5].cpu().numpy()}')

# Create steering manager
print("\nCreating steering manager...")
manager = IncrementalSteeringManager(
    model=model,
    probe_directions=probe_directions,
    steering_vector=test_steering,
    adapter_types=adapter_types,
    n_layers=64,
    decay_factor=1.0,
    start_position=None,
    debug=True
)

print('\nRegistering hooks...')
manager.register_hooks()

# Test forward pass
print('\nRunning test forward pass...')
test_input = tokenizer('Hello world', return_tensors='pt').to(device)
print(f'Input shape: {test_input.input_ids.shape}')

with torch.no_grad():
    output = model(test_input.input_ids)
    print(f'Output shape: {output.logits.shape}')

manager.remove_hooks()

print(f'\n{"="*60}')
print('RESULTS')
print(f'{"="*60}')
print(f'Hook calls: {manager.hook_call_count}')
print(f'Activation cache entries: {len(manager.activation_cache)}')

if manager.hook_call_count == 0:
    print('ERROR: No hooks were called!')
elif len(manager.activation_cache) == 0:
    print('ERROR: Hooks called but no cache entries!')
else:
    print('SUCCESS: Hooks are working!')
    # Show first few cache entries
    for i, (key, values) in enumerate(list(manager.activation_cache.items())[:3]):
        print(f'\nCache entry {i+1}: {key}')
        print(f'  Steering scalar: {values["steering_scalar"]:.6f}')
        print(f'  Contribution norm: {values["contribution_norm"]:.6f}')
        print(f'  Output norm before: {values["output_norm_before"]:.6f}')
        print(f'  Output norm after: {values["output_norm_after"]:.6f}')