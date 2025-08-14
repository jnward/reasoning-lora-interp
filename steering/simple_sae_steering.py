# %% [markdown]
# # Simple SAE Feature Steering
# 
# This notebook lets you select an SAE feature and steering strength, then see how it affects generation.

# %%
import torch
import torch.nn.functional as F
import glob
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from typing import Dict, List, Optional

# Add parent directory to path for SAE imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_interp.batch_topk_sae import BatchTopKSAE

# %%
# Load everything (run once)
print("Loading SAE...")
sae_path = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']

sae_model = BatchTopKSAE(
    d_model=sae_config['d_model'],
    dict_size=sae_config['dict_size'],
    k=sae_config['k']
)

state_dict = checkpoint['model_state_dict']
sae_model.W_enc.data = state_dict['W_enc']
sae_model.b_enc.data = state_dict['b_enc']
sae_model.W_dec.data = state_dict['W_dec']
sae_model.b_dec.data = state_dict['b_dec']
sae_model.eval()

print(f"SAE loaded: {sae_config['dict_size']} features")

# Load model
print("\nLoading model...")
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

adapter_types = sae_config['adapter_types']
print("Model loaded!")

# %% [markdown]
# ## Steering Manager

# %%
class SimpleSteering:
    """Minimal steering manager."""
    
    def __init__(self, model, steering_vector, adapter_types, n_layers=64):
        self.model = model
        self.steering_vector = steering_vector
        self.adapter_types = adapter_types
        self.n_layers = n_layers
        self.hooks = []
        self.adapter_to_idx = {adapter: i for i, adapter in enumerate(adapter_types)}
        
        # Extract LoRA components
        self.lora_components = {}
        for layer_idx in range(n_layers):
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                if proj_type not in adapter_types:
                    continue
                module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
                if hasattr(module, 'lora_B'):
                    key = f"{proj_type}_{layer_idx}"
                    self.lora_components[key] = {
                        'lora_B': module.lora_B['default'].weight.data,
                        'scaling': module.scaling['default']
                    }
            
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if proj_type not in adapter_types:
                    continue
                module = model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
                if hasattr(module, 'lora_B'):
                    key = f"{proj_type}_{layer_idx}"
                    self.lora_components[key] = {
                        'lora_B': module.lora_B['default'].weight.data,
                        'scaling': module.scaling['default']
                    }
    
    def _make_hook(self, layer_idx, adapter_type):
        def hook(module, input, output):
            # Get steering component
            adapter_idx = self.adapter_to_idx[adapter_type]
            component_idx = layer_idx * len(self.adapter_types) + adapter_idx
            steering_scalar = self.steering_vector[component_idx].item()
            
            # Get LoRA components
            key = f"{adapter_type}_{layer_idx}"
            if key not in self.lora_components:
                return output
            
            lora_B = self.lora_components[key]['lora_B'].to(output.device)
            scaling = self.lora_components[key]['scaling']
            
            # Add steering
            contribution = steering_scalar * lora_B[:, 0] * scaling
            contribution = contribution.to(output.dtype)
            
            if output.dim() == 3:
                contribution = contribution.unsqueeze(0).unsqueeze(0)
            elif output.dim() == 2:
                contribution = contribution.unsqueeze(0)
            
            return output + contribution
        return hook
    
    def register_hooks(self):
        for layer_idx in range(self.n_layers):
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                if f"{proj_type}_{layer_idx}" in self.lora_components:
                    module = self.model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
                    self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx, proj_type)))
            
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if f"{proj_type}_{layer_idx}" in self.lora_components:
                    module = self.model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
                    self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx, proj_type)))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# %% [markdown]
# ## Steering Experiment
# 
# **Instructions:**
# 1. Set `feature_idx` to the SAE feature you want to steer with (0-3583)
# 2. Set `strength` to control how strongly to apply the feature
# 3. Set your `question` (the model will be prompted to think step-by-step)
# 4. Run the cell to see baseline vs steered generation

# %%

# Tokenize prompt
question = "What is the sum of the prime factors in 11011?"  # Your question

# Format prompt properly
system_prompt = "You are a helpful mathematics assistant."
prompt = (
    f"<|im_start|>system\n{system_prompt}\n"
    f"<|im_start|>user\n{question}\n"
    f"<|im_start|>assistant\n"
    f"<|im_start|>think\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
max_tokens = 512   # Max tokens to generate

# Generate baseline (no steering)
print("BASELINE:")
print("-" * 50)
with torch.no_grad():
    baseline_output = model.generate(
        inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=False)
# Only print the generated part (after the prompt)
print(baseline_text[len(prompt):])

# %%
# CONFIGURE THESE:
feature_idx = 638  # SAE feature index (0-3583)
strength = 30.0    # Steering strength

# Create steering vector from SAE feature
steering_vector = sae_model.W_dec[feature_idx] * strength
steering_vector = steering_vector.to(device)

print(f"Feature {feature_idx} with strength {strength}")
print(f"Steering vector norm: {torch.norm(steering_vector).item():.2f}")
print(f"Question: {question}\n")

# Generate with steering
print("\n\nWITH STEERING:")
print("-" * 50)
steering_manager = SimpleSteering(model, steering_vector, adapter_types)
steering_manager.register_hooks()

with torch.no_grad():
    steered_output = model.generate(
        inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

steering_manager.remove_hooks()
steered_text = tokenizer.decode(steered_output[0], skip_special_tokens=False)
# Only print the generated part (after the prompt)
print(steered_text[len(prompt):])

# %% [markdown]
# ## Try Multiple Features
# 
# Run this cell to quickly test several features:

# %%
# Test multiple features
test_features = [0, 10, 100, 500, 1000, 2000, 3000]
test_question = "What is 2 + 2?"
test_strength = 30.0
test_max_tokens = 50

# Format prompt
test_prompt = (
    f"<|im_start|>system\nYou are a helpful mathematics assistant.\n"
    f"<|im_start|>user\n{test_question}\n"
    f"<|im_start|>assistant\n"
    f"<|im_start|>think\n"
)

print(f"Testing features {test_features} with strength {test_strength}\n")
print(f"Question: {test_question}\n")

# Generate baseline once
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    baseline = model.generate(inputs.input_ids, max_new_tokens=test_max_tokens, 
                             temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
baseline_text = tokenizer.decode(baseline[0], skip_special_tokens=False)
print("BASELINE:")
print(baseline_text[len(test_prompt):])
print("=" * 60)

# Test each feature
for feat_idx in test_features:
    steering_vec = sae_model.W_dec[feat_idx] * test_strength
    steering_vec = steering_vec.to(device)
    
    manager = SimpleSteering(model, steering_vec, adapter_types)
    manager.register_hooks()
    
    with torch.no_grad():
        output = model.generate(inputs.input_ids, max_new_tokens=test_max_tokens,
                               temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    
    manager.remove_hooks()
    
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    print(f"\nFeature {feat_idx:4d}:")
    print(text[len(test_prompt):])

print("\nDone!")