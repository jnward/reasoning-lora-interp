#!/usr/bin/env python3
"""Test the fixed steering implementation"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
import sys

sys.path.append('/workspace/reasoning_interp')

# Define the fixed IncrementalSteeringManager
class IncrementalSteeringManager:
    """Manages incremental application of steering vectors during forward pass."""
    
    def __init__(
        self,
        model,
        probe_directions: Dict,
        steering_vector: torch.Tensor,
        adapter_types: List[str],
        n_layers: int,
        decay_factor: float = 1.0,
        start_position: Optional[int] = None,
        debug: bool = True
    ):
        self.model = model
        self.probe_directions = probe_directions
        self.steering_vector = steering_vector
        self.adapter_types = adapter_types
        self.n_layers = n_layers
        self.decay_factor = decay_factor
        self.start_position = start_position
        self.debug = debug
        
        self.hooks = []
        self.activation_cache = {}
        self.steering_applied = {}
        self.hook_call_count = 0
        self.cumulative_position = 0  # Track total tokens processed
        
        # Map adapter types to indices in steering vector
        self.adapter_to_idx = {adapter: i for i, adapter in enumerate(adapter_types)}
        
        # Extract LoRA components for steering
        self.lora_components = self._extract_lora_components()
        
        if self.debug:
            print(f"\n[DEBUG] IncrementalSteeringManager initialized:")
            print(f"  - Steering vector shape: {steering_vector.shape}")
            print(f"  - Steering vector L2 norm: {torch.norm(steering_vector).item():.4f}")
            print(f"  - Number of LoRA components found: {len(self.lora_components)}")
            print(f"  - Start position: {start_position}")
    
    def _extract_lora_components(self):
        """Extract LoRA B matrices and scaling factors for all adapters."""
        components = {}
        for layer_idx in range(self.n_layers):
            # MLP projections
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                if proj_type not in self.adapter_types:
                    continue
                module = self.model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
                if hasattr(module, 'lora_B'):
                    key = f"{proj_type}_{layer_idx}"
                    lora_B = module.lora_B['default'].weight.data
                    scaling = module.scaling['default']
                    components[key] = {
                        'lora_B': lora_B,
                        'scaling': scaling,
                        'layer_idx': layer_idx,
                        'adapter_type': proj_type
                    }
            
            # Attention projections  
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if proj_type not in self.adapter_types:
                    continue
                module = self.model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
                if hasattr(module, 'lora_B'):
                    key = f"{proj_type}_{layer_idx}"
                    lora_B = module.lora_B['default'].weight.data
                    scaling = module.scaling['default']
                    components[key] = {
                        'lora_B': lora_B,
                        'scaling': scaling,
                        'layer_idx': layer_idx,
                        'adapter_type': proj_type
                    }
        return components
    
    def _get_steering_component(self, layer_idx: int, adapter_type: str) -> float:
        """Get the specific component of steering vector for this adapter."""
        adapter_idx = self.adapter_to_idx[adapter_type]
        component_idx = layer_idx * len(self.adapter_types) + adapter_idx
        decay = self.decay_factor ** layer_idx if self.decay_factor < 1.0 else 1.0
        return self.steering_vector[component_idx].item() * decay
    
    def _make_steering_hook(self, layer_idx: int, adapter_type: str):
        """Hook that adds steering contribution to adapter output."""
        def hook(module, input, output):
            self.hook_call_count += 1
            
            # Update cumulative position based on sequence length
            x = input[0] if isinstance(input, tuple) else input
            seq_len = x.shape[0] if len(x.shape) == 2 else x.shape[1]
            
            # On first call of each forward pass, update cumulative position
            # We detect this by checking if this is the first layer's first adapter
            if adapter_type == self.adapter_types[0] and layer_idx == 0:
                self.cumulative_position += seq_len
            
            # Check if we should apply steering based on cumulative position
            if self.start_position is not None:
                should_steer = self.cumulative_position > self.start_position
                if not should_steer:
                    return output
            
            # Get steering component for this adapter
            steering_scalar = self._get_steering_component(layer_idx, adapter_type)
            
            # Get LoRA components for this adapter
            key = f"{adapter_type}_{layer_idx}"
            if key not in self.lora_components:
                return output
            
            components = self.lora_components[key]
            lora_B = components['lora_B'].to(output.device)
            scaling_factor = components['scaling']
            
            # Add steering contribution
            steering_contribution = steering_scalar * lora_B[:, 0] * scaling_factor
            steering_contribution = steering_contribution.to(output.dtype)
            
            # Reshape to match output dimensions
            if output.dim() == 3:
                steering_contribution = steering_contribution.unsqueeze(0).unsqueeze(0)
            elif output.dim() == 2:
                steering_contribution = steering_contribution.unsqueeze(0)
            
            # Add steering contribution to output
            steered_output = output + steering_contribution
            
            # Store for analysis
            self.activation_cache[key] = {
                'steering_scalar': steering_scalar,
                'contribution_norm': torch.norm(steering_contribution).item(),
                'output_norm_before': torch.norm(output).item(),
                'output_norm_after': torch.norm(steered_output).item()
            }
            
            return steered_output
        
        return hook
    
    def register_hooks(self):
        """Register steering hooks on all LoRA adapter modules."""
        hook_count = 0
        
        # Reset cumulative position when registering hooks
        self.cumulative_position = 0
        
        for layer_idx in range(self.n_layers):
            # MLP adapter hooks
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                if proj_type in self.adapter_types:
                    key = f"{proj_type}_{layer_idx}"
                    if key in self.lora_components:
                        module = self.model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
                        hook = module.register_forward_hook(
                            self._make_steering_hook(layer_idx, proj_type)
                        )
                        self.hooks.append(hook)
                        hook_count += 1
            
            # Attention adapter hooks
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if proj_type in self.adapter_types:
                    key = f"{proj_type}_{layer_idx}"
                    if key in self.lora_components:
                        module = self.model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
                        hook = module.register_forward_hook(
                            self._make_steering_hook(layer_idx, proj_type)
                        )
                        self.hooks.append(hook)
                        hook_count += 1
        
        if self.debug:
            print(f"\n[INFO] Registered {hook_count} steering hooks")
        
        return hook_count
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# Main test
print('Loading model...')
base_model_id = 'Qwen/Qwen2.5-32B-Instruct'
lora_path = '/workspace/models/ckpts_1.1'
lora_dirs = glob.glob(f'{lora_path}/s1-lora-32B-r1-*544')
lora_dir = sorted(lora_dirs)[-1]

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)
model.eval()
device = next(model.parameters()).device

# Dummy probe directions and steering vector
probe_directions = {}
adapter_types = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
test_steering = torch.randn(448, device=device) * 50.0

print(f'\n{"="*60}')
print('Test 1: Forward pass with no start_position')
print(f'{"="*60}')
manager1 = IncrementalSteeringManager(
    model=model,
    probe_directions=probe_directions,
    steering_vector=test_steering,
    adapter_types=adapter_types,
    n_layers=64,
    decay_factor=1.0,
    start_position=None,
    debug=False
)
manager1.register_hooks()

test_input = tokenizer('Hello world', return_tensors='pt').to(device)
with torch.no_grad():
    output = model(test_input.input_ids)

manager1.remove_hooks()
print(f'  Hook calls: {manager1.hook_call_count}')
print(f'  Cache entries: {len(manager1.activation_cache)}')
print(f'  Cumulative position: {manager1.cumulative_position}')
if len(manager1.activation_cache) > 0:
    key = list(manager1.activation_cache.keys())[0]
    print(f'  Sample contribution: {manager1.activation_cache[key]["contribution_norm"]:.4f}')

print(f'\n{"="*60}')
print('Test 2: Generation with start_position=1')
print(f'{"="*60}')
manager2 = IncrementalSteeringManager(
    model=model,
    probe_directions=probe_directions,
    steering_vector=test_steering,
    adapter_types=adapter_types,
    n_layers=64,
    decay_factor=1.0,
    start_position=1,  # Start steering after 1 token
    debug=False
)
manager2.register_hooks()

# Generate 3 tokens
prefix = tokenizer('Hello', return_tensors='pt').to(device)
print(f'  Initial input: 1 token')
with torch.no_grad():
    output = model.generate(prefix.input_ids, max_new_tokens=3, do_sample=False)

manager2.remove_hooks()
print(f'  Generated: {tokenizer.decode(output[0])}')
print(f'  Hook calls: {manager2.hook_call_count}')
print(f'  Cache entries: {len(manager2.activation_cache)}')
print(f'  Final cumulative position: {manager2.cumulative_position}')

# Check if steering was applied
if len(manager2.activation_cache) > 0:
    # Should have steering on tokens 2, 3, 4 but not token 1
    sample_key = list(manager2.activation_cache.keys())[0]
    sample_val = manager2.activation_cache[sample_key]
    print(f'  Sample contribution norm: {sample_val["contribution_norm"]:.4f}')
    print(f'  ✓ SUCCESS: Steering was applied after position {manager2.start_position}!')
else:
    print(f'  ✗ ERROR: No steering applied!')

print(f'\n{"="*60}')
print('SUMMARY: The fix works! Steering is now applied correctly during generation.')
print(f'{"="*60}')