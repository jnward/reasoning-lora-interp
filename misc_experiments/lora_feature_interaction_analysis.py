# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc
from dataclasses import dataclass
from tabulate import tabulate
from tqdm import tqdm
import types
import math
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-2*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# %%
class AttentionLinearizer:
    """Linearizes attention by treating attention patterns as constants"""
    
    def __init__(self, model):
        self.model = model
        self.original_attention_forwards = {}
        
    def linearize_all_attention_modules(self):
        """Monkey-patch all attention modules to linearize attention patterns"""
        count = 0
        
        # Check attention implementation type
        sample_attn = self.model.model.model.layers[0].self_attn
        attn_class = sample_attn.__class__.__name__
        print(f"Detected attention implementation: {attn_class}")
        
        # Monkey-patch each attention layer
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            attn_module = layer.self_attn
            
            # Store original forward
            self.original_attention_forwards[layer_idx] = attn_module.forward
            
            # Create linearized forward
            linearized_forward = self._create_linearized_attention_forward(
                attn_module.forward, attn_module, layer_idx
            )
            
            # Replace forward method
            attn_module.forward = linearized_forward
            count += 1
            
        print(f"Linearized {count} attention modules")
        
    def _create_linearized_attention_forward(self, original_forward, attn_module, layer_idx):
        """Create a linearized forward function for attention"""
        
        def linearized_forward(self, hidden_states, *args, **kwargs):
            # For simplicity, we'll call the original forward but with a patched F.scaled_dot_product_attention
            
            # Store the original SDPA function
            original_sdpa = F.scaled_dot_product_attention
            
            # Create a patched version that detaches attention weights
            def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                # Compute attention weights normally
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                if is_causal:
                    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)
                    
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias += attn_mask
                        
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                
                # CRITICAL: Detach attention weights here
                attn_weight = attn_weight.detach()
                
                # Apply dropout if needed
                if dropout_p > 0.0:
                    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                    
                # Apply attention to values
                return attn_weight @ value
            
            # Temporarily replace F.scaled_dot_product_attention
            F.scaled_dot_product_attention = patched_sdpa
            
            try:
                # Call original forward with patched SDPA
                outputs = original_forward(hidden_states, *args, **kwargs)
            finally:
                # Always restore original SDPA
                F.scaled_dot_product_attention = original_sdpa
                
            return outputs
            
        # Bind to the attention module
        return linearized_forward.__get__(attn_module, attn_module.__class__)


class LinearizedLayerNorm:
    """Manages linearization of LayerNorm modules via monkey-patching"""
    
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        
    def _create_linearized_forward(self, original_forward):
        """Create a linearized forward function for LayerNorm/RMSNorm"""
        def linearized_forward(self, input):
            # Check if this is RMSNorm (no mean subtraction, no bias)
            is_rmsnorm = not hasattr(self, 'bias')
            
            if is_rmsnorm:
                # RMSNorm: only uses RMS, no mean subtraction
                variance = input.pow(2).mean(-1, keepdim=True)
                rms = torch.sqrt(variance + self.variance_epsilon).detach()
                normalized = input / rms
                return self.weight * normalized
            else:
                # Standard LayerNorm
                mean = input.mean(-1, keepdim=True).detach()
                var = input.var(-1, keepdim=True, unbiased=False).detach()
                normalized = (input - mean) / torch.sqrt(var + self.variance_epsilon)
                return self.weight * normalized + self.bias
            
        return linearized_forward
    
    def linearize_all_layernorms(self):
        """Monkey-patch all LayerNorm modules to use linearized forward"""
        count = 0
        
        # Linearize LayerNorms in transformer layers
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            # Linearize input LayerNorm (pre-attention)
            if hasattr(layer, 'input_layernorm'):
                ln = layer.input_layernorm
                self.original_forwards[f'layer{layer_idx}_input'] = ln.forward
                ln.forward = self._create_linearized_forward(ln.forward).__get__(ln, ln.__class__)
                count += 1
            
            # Linearize post-attention LayerNorm (pre-MLP)
            if hasattr(layer, 'post_attention_layernorm'):
                ln = layer.post_attention_layernorm
                self.original_forwards[f'layer{layer_idx}_post'] = ln.forward
                ln.forward = self._create_linearized_forward(ln.forward).__get__(ln, ln.__class__)
                count += 1
        
        # Linearize final LayerNorm
        if hasattr(self.model.model.model, 'norm'):
            ln = self.model.model.model.norm
            self.original_forwards['final_norm'] = ln.forward
            ln.forward = self._create_linearized_forward(ln.forward).__get__(ln, ln.__class__)
            count += 1
        
        print(f"Linearized {count} LayerNorm modules")
    
    def restore_original_forwards(self):
        """Restore original LayerNorm forward methods"""
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            # Restore input LayerNorm
            key = f'layer{layer_idx}_input'
            if key in self.original_forwards and hasattr(layer, 'input_layernorm'):
                layer.input_layernorm.forward = self.original_forwards[key]
                
            # Restore post-attention LayerNorm
            key = f'layer{layer_idx}_post'
            if key in self.original_forwards and hasattr(layer, 'post_attention_layernorm'):
                layer.post_attention_layernorm.forward = self.original_forwards[key]
        
        # Restore final LayerNorm
        if 'final_norm' in self.original_forwards and hasattr(self.model.model.model, 'norm'):
            self.model.model.model.norm.forward = self.original_forwards['final_norm']
            
        print(f"Restored {len(self.original_forwards)} original LayerNorm modules")
        self.original_forwards = {}


class LoRAFeatureTracker:
    """Tracks LoRA feature activations and computes mean activations over tokens"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}  # {layer_name: tensor}
        self.mean_activations = {}  # {layer_name: mean_tensor}
        self.hooks = []
        self.feature_names = []  # List of all feature names in order
        
    def _create_hook(self, layer_name: str, adapter_name: str = 'default'):
        """Create a forward hook that captures activations and maintains gradient flow"""
        
        def hook_fn(module, input, output):
            # output shape: [batch_size, seq_len, 1] for rank-1
            # CRITICAL: We need to ensure the tensor requires grad
            if not output.requires_grad:
                output.requires_grad_(True)
            
            # Now we can retain gradients
            output.retain_grad()
            
            # Store reference to the original tensor (not a clone!)
            key = f"{layer_name}.{adapter_name}"
            self.activations[key] = output
            
            # Return unchanged to preserve computation graph
            return output
            
        return hook_fn
    
    def register_hooks(self):
        """Register hooks on MLP LoRA A matrices only"""
        
        # Navigate through model structure
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            # Only check MLP projections (skip attention)
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                module = getattr(layer.mlp, proj_name, None)
                if module and hasattr(module, 'lora_A'):
                    for adapter_name, lora_A_module in module.lora_A.items():
                        feature_name = f"layer{layer_idx}.mlp.{proj_name}.{adapter_name}"
                        self.feature_names.append(feature_name)
                        hook = self._create_hook(f"layer{layer_idx}.mlp.{proj_name}", adapter_name)
                        handle = lora_A_module.register_forward_hook(hook)
                        self.hooks.append(handle)
        
        print(f"Registered {len(self.hooks)} hooks on MLP LoRA A matrices")
        print(f"Total features tracked: {len(self.feature_names)}")
    
    def compute_mean_activations(self):
        """Compute mean activation over all token positions for each feature"""
        self.mean_activations = {}
        
        for name, activation in self.activations.items():
            # activation shape: [batch_size, seq_len, 1]
            # Compute mean over sequence length dimension
            mean_act = activation.mean(dim=1, keepdim=True)  # [batch_size, 1, 1]
            mean_act.retain_grad()  # Ensure we can get gradients
            self.mean_activations[name] = mean_act
            
        return self.mean_activations
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# %%
# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)

# %%
import json
import os

# Configuration for MATH-500
example_idx = 10  # 10th example as requested
max_new_tokens = 512  # Shorter for efficiency in this analysis
generation_cache_file = f"math500_generation_example_{example_idx}_short.json"

# %%
# Load MATH-500 dataset
print("Loading MATH-500 dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# Get the 10th example
example = dataset[example_idx]
problem = example['problem']

print(f"\nUsing example {example_idx}:")
print(f"Problem: {problem[:200]}..." if len(problem) > 200 else f"Problem: {problem}")

# %%
# Check if generation already exists
if os.path.exists(generation_cache_file):
    print(f"\nLoading cached generation from {generation_cache_file}")
    with open(generation_cache_file, 'r') as f:
        cache_data = json.load(f)
    prompt = cache_data['full_text']
    generated_text = cache_data['generated_text']
    input_prompt = cache_data['input_prompt']
else:
    # Format prompt for generation
    system_prompt = "You are a helpful mathematics assistant. Please think step by step to solve the problem."
    input_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    print("\nGenerating response...")
    
    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
    
    # Generate with the model
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generation
    prompt = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    
    # Save to cache
    cache_data = {
        'example_idx': example_idx,
        'problem': problem,
        'input_prompt': input_prompt,
        'generated_text': generated_text,
        'full_text': prompt
    }
    
    with open(generation_cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Generation saved to {generation_cache_file}")

print(f"\nGenerated response preview: {generated_text[:200]}...")

# %%
# Tokenize for analysis (limit length for efficiency)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs['input_ids'][:, :256]  # Limit to 256 tokens for efficiency
seq_len = input_ids.shape[1]

print(f"\nUsing {seq_len} tokens for analysis")

# %%
# Compute feature interaction attribution matrix
print("\nComputing LoRA feature interaction matrix...")

# Setup linearized LayerNorm
print("Linearizing LayerNorm modules...")
layernorm_linearizer = LinearizedLayerNorm(model)
layernorm_linearizer.linearize_all_layernorms()

# Setup linearized attention
print("Linearizing attention patterns...")
attention_linearizer = AttentionLinearizer(model)
attention_linearizer.linearize_all_attention_modules()

# Setup tracker
tracker = LoRAFeatureTracker(model)
tracker.register_hooks()

# Enable gradient computation
model.eval()
torch.set_grad_enabled(True)

# Clear any existing gradients
model.zero_grad()

# Forward pass - hooks will capture activations
print("\nRunning forward pass...")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(input_ids=input_ids)

# Compute mean activations over all token positions
print("Computing mean activations...")
mean_activations = tracker.compute_mean_activations()

# Get the number of features
n_features = len(tracker.feature_names)
print(f"\nTotal features: {n_features}")

# Initialize attribution matrix
attribution_matrix = torch.zeros(n_features, n_features, device=model.device)

# %%
# Debug: Analyze feature ordering
print("\nDEBUG: Analyzing feature ordering...")
print(f"Total features: {n_features}")
print("\nFirst 10 features:")
for i in range(min(10, n_features)):
    print(f"  {i}: {tracker.feature_names[i]}")
print("\nLast 10 features:")
for i in range(max(0, n_features-10), n_features):
    print(f"  {i}: {tracker.feature_names[i]}")

# Extract layer indices from feature names
import re
feature_layers = []
for name in tracker.feature_names:
    match = re.search(r'layer(\d+)', name)
    if match:
        feature_layers.append(int(match.group(1)))
    else:
        feature_layers.append(-1)

print(f"\nLayer distribution:")
print(f"  Min layer: {min(feature_layers)}")
print(f"  Max layer: {max(feature_layers)}")
print(f"  Unique layers: {len(set(feature_layers))}")

# %%
# Debug: First compute attribution for a single token position
debug_position = 100  # Choose a position in the middle of the sequence
print(f"\nDEBUG: Computing attributions for single position {debug_position}")

# Initialize debug attribution matrix for single position
single_pos_attribution_matrix = torch.zeros(n_features, n_features, device=model.device)

for i, source_feature in enumerate(tqdm(tracker.feature_names, desc="Debug single position")):
    # Get activation at specific position for this feature
    source_activation_full = tracker.activations[source_feature]  # [batch, seq_len, 1]
    source_activation_at_pos = source_activation_full[0, debug_position, 0]
    
    # Clear gradients - BOTH parameter gradients AND activation gradients
    model.zero_grad()
    for name, act in tracker.activations.items():
        if act.grad is not None:
            act.grad = None
    
    # Backward pass from this specific position's activation
    source_activation_at_pos.backward(retain_graph=True)
    
    # Collect gradients on all features at all positions
    for j, target_feature in enumerate(tracker.feature_names):
        if target_feature in tracker.activations:
            target_activation = tracker.activations[target_feature]
            
            if target_activation.grad is not None:
                # Get gradient and activation at all positions
                grad = target_activation.grad[0, :, 0]  # [seq_len]
                act = target_activation[0, :, 0]  # [seq_len]
                
                # Attribution = gradient * activation, averaged over target positions
                attribution = (grad * act).mean().item()
                single_pos_attribution_matrix[i, j] = attribution

# Visualize single position attribution matrix
plt.figure(figsize=(12, 10))
single_pos_matrix_np = single_pos_attribution_matrix.cpu().numpy()
vmax = np.percentile(np.abs(single_pos_matrix_np), 99)
plt.imshow(single_pos_matrix_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(label='Attribution')
plt.title(f'Single Position Attribution Matrix (Position {debug_position})\n(How each feature at pos {debug_position} influences others)')
plt.xlabel('Target Feature Index')
plt.ylabel('Source Feature Index') 
plt.tight_layout()
plt.savefig('/workspace/reasoning_interp/single_position_attribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Debug: Check gradient flow pattern - DIRECT TEST
print("\nDEBUG: Direct gradient flow test...")

# First, let's verify our understanding by testing gradient flow WITHOUT using mean activations
test_layer = 30
test_features = [i for i, name in enumerate(tracker.feature_names) if f"layer{test_layer}" in name]

if test_features:
    test_feature_idx = test_features[0]
    test_feature_name = tracker.feature_names[test_feature_idx]
    print(f"\nTest 1: Direct backward from feature at specific position")
    print(f"Source: {test_feature_name} at position {debug_position}")
    
    # Clear ALL gradients
    model.zero_grad()
    for name, act in tracker.activations.items():
        if act.grad is not None:
            act.grad = None
    
    # Get the ORIGINAL activation (not mean)
    source_activation = tracker.activations[test_feature_name]
    source_at_pos = source_activation[0, debug_position, 0]
    
    # Create a fresh scalar to backward from
    scalar_target = source_at_pos.sum()  # Ensure it's a scalar
    scalar_target.backward(retain_graph=True)
    
    # Check gradients
    print("\nGradients received by layer:")
    layer_gradient_info = {}
    for j, target_name in enumerate(tracker.feature_names):
        target_layer = feature_layers[j]
        if target_name in tracker.activations:
            target_act = tracker.activations[target_name]
            if target_act.grad is not None:
                grad_norm = target_act.grad.abs().mean().item()
                if target_layer not in layer_gradient_info:
                    layer_gradient_info[target_layer] = []
                layer_gradient_info[target_layer].append(grad_norm)
    
    for layer in sorted(layer_gradient_info.keys()):
        if layer_gradient_info[layer]:
            avg_grad = np.mean(layer_gradient_info[layer])
            if layer > test_layer:
                print(f"  Layer {layer}: avg gradient = {avg_grad:.6f} ⚠️ SHOULD BE ZERO!")
            else:
                print(f"  Layer {layer}: avg gradient = {avg_grad:.6f}")

print("\n" + "="*80)
print("Test 2: Backward from mean activation")

# Now test with mean activation
if test_features:
    # Clear gradients again
    model.zero_grad()
    for name, act in tracker.activations.items():
        if act.grad is not None:
            act.grad = None
    
    # Use mean activation
    mean_act = mean_activations[test_feature_name]
    mean_act.backward(torch.ones_like(mean_act), retain_graph=True)
    
    # Check gradients
    layer_gradient_info2 = {}
    for j, target_name in enumerate(tracker.feature_names):
        target_layer = feature_layers[j]
        if target_name in tracker.activations:
            target_act = tracker.activations[target_name]
            if target_act.grad is not None:
                grad_norm = target_act.grad.abs().mean().item()
                if target_layer not in layer_gradient_info2:
                    layer_gradient_info2[target_layer] = []
                layer_gradient_info2[target_layer].append(grad_norm)
    
    print(f"\nGradients from mean activation of layer {test_layer}:")
    for layer in sorted(layer_gradient_info2.keys()):
        if layer_gradient_info2[layer]:
            avg_grad = np.mean(layer_gradient_info2[layer])
            if layer > test_layer:
                print(f"  Layer {layer}: avg gradient = {avg_grad:.6f} ⚠️ SHOULD BE ZERO!")
            else:
                print(f"  Layer {layer}: avg gradient = {avg_grad:.6f}")

# %%
# Debug: Let's also check the raw gradients without multiplication
print(f"\nDEBUG: Checking raw gradients from a single source feature")

# Pick a specific source feature to examine
debug_source_idx = n_features // 2  # Middle feature
debug_source_name = tracker.feature_names[debug_source_idx]
source_activation_full = tracker.activations[debug_source_name]
source_activation_at_pos = source_activation_full[0, debug_position, 0]

# Clear and compute gradients
model.zero_grad()
source_activation_at_pos.backward(retain_graph=True)

# Collect just the gradients (not multiplied by activations)
gradient_matrix = []
for target_feature in tracker.feature_names:
    if target_feature in tracker.activations:
        target_activation = tracker.activations[target_feature]
        if target_activation.grad is not None:
            # Average gradient across positions
            avg_grad = target_activation.grad[0, :, 0].mean().item()
            gradient_matrix.append(avg_grad)
        else:
            gradient_matrix.append(0.0)
    else:
        gradient_matrix.append(0.0)

# Plot the gradient values
plt.figure(figsize=(12, 6))
plt.bar(range(len(gradient_matrix)), gradient_matrix)
plt.xlabel('Target Feature Index')
plt.ylabel('Average Gradient')
plt.title(f'Raw Gradients from Source Feature {debug_source_idx} at Position {debug_position}')
plt.tight_layout()
plt.savefig('/workspace/reasoning_interp/raw_gradients_debug.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Gradient statistics:")
print(f"  Mean: {np.mean(gradient_matrix):.6f}")
print(f"  Std: {np.std(gradient_matrix):.6f}")
print(f"  Min: {np.min(gradient_matrix):.6f}")
print(f"  Max: {np.max(gradient_matrix):.6f}")

# %%
# Debug: Check LoRA contribution magnitude
print("\nDEBUG: Checking LoRA contribution magnitudes...")

# Check a few LoRA modules to understand their scale
for layer_idx in [0, 15, 30, 45, 60]:
    if layer_idx < model.config.num_hidden_layers:
        layer = model.model.model.layers[layer_idx]
        
        # Check gate_proj as example
        if hasattr(layer.mlp, 'gate_proj'):
            module = layer.mlp.gate_proj
            
            # Get base weight norm
            base_weight_norm = module.base_layer.weight.norm().item()
            
            # Get LoRA contribution norm
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                for adapter_name in module.lora_A.keys():
                    lora_A = module.lora_A[adapter_name]
                    lora_B = module.lora_B[adapter_name]
                    
                    # LoRA contribution is B @ A * scaling
                    lora_weight = lora_B.weight @ lora_A.weight
                    scaling = module.scaling[adapter_name] if hasattr(module, 'scaling') else 1.0
                    lora_contrib_norm = (lora_weight * scaling).norm().item()
                    
                    print(f"\nLayer {layer_idx} gate_proj:")
                    print(f"  Base weight norm: {base_weight_norm:.4f}")
                    print(f"  LoRA contribution norm: {lora_contrib_norm:.4f}")
                    print(f"  LoRA/Base ratio: {lora_contrib_norm/base_weight_norm:.6f}")
                    break

# %%
# Debug: Let's check the actual activation magnitudes
print("\nDEBUG: Checking activation magnitudes...")

# Sample some features and check their activation scales
sample_features = [0, n_features//4, n_features//2, 3*n_features//4, n_features-1]

for idx in sample_features:
    if idx < n_features:
        feature_name = tracker.feature_names[idx]
        activation = tracker.activations[feature_name]
        
        act_mean = activation.abs().mean().item()
        act_std = activation.std().item()
        act_max = activation.abs().max().item()
        
        print(f"\nFeature {idx} ({feature_name}):")
        print(f"  Mean |activation|: {act_mean:.6f}")
        print(f"  Std activation: {act_std:.6f}")
        print(f"  Max |activation|: {act_max:.6f}")

# %%
# Debug: Let's check the computation graph structure
print("\nDEBUG: Checking computation graph structure...")

# Check if mean activations are creating unexpected connections
print("\nChecking mean activation dependencies...")
test_mean_act = mean_activations[tracker.feature_names[n_features//2]]
print(f"Mean activation shape: {test_mean_act.shape}")
print(f"Requires grad: {test_mean_act.requires_grad}")

# Check if there are any shared parameters between LoRA modules
print("\nChecking for shared parameters between LoRA modules...")
lora_params = {}
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param_id = id(param)
        if param_id not in lora_params:
            lora_params[param_id] = []
        lora_params[param_id].append(name)

shared_params = {k: v for k, v in lora_params.items() if len(v) > 1}
if shared_params:
    print(f"Found {len(shared_params)} shared parameters!")
    for param_id, names in list(shared_params.items())[:5]:
        print(f"  Parameter shared by: {names[:3]}...")
else:
    print("No shared parameters found between LoRA modules.")

# %%
# Debug: Check if gradients are uniform across source features
print("\nDEBUG: Checking gradient patterns across different source features...")

# Pick a few source features from different layers
test_sources = [0, n_features//4, n_features//2, 3*n_features//4, n_features-1]
gradient_patterns = []

for src_idx in test_sources:
    if src_idx < n_features:
        src_name = tracker.feature_names[src_idx]
        src_layer = feature_layers[src_idx]
        
        # Clear gradients
        model.zero_grad()
        
        # Backward from mean activation
        src_mean = mean_activations[src_name]
        src_mean.backward(torch.ones_like(src_mean), retain_graph=True)
        
        # Collect gradients for all targets
        grads = []
        for tgt_name in tracker.feature_names:
            if tgt_name in tracker.activations:
                tgt_act = tracker.activations[tgt_name]
                if tgt_act.grad is not None:
                    # Average gradient magnitude
                    avg_grad = tgt_act.grad[0, :, 0].abs().mean().item()
                    grads.append(avg_grad)
                else:
                    grads.append(0.0)
            else:
                grads.append(0.0)
        
        gradient_patterns.append(grads)
        print(f"Source {src_idx} (layer {src_layer}): mean grad = {np.mean(grads):.6f}, std = {np.std(grads):.6f}")

# Check correlation between gradient patterns
print("\nCorrelation between gradient patterns from different sources:")
for i in range(len(test_sources)-1):
    for j in range(i+1, len(test_sources)):
        corr = np.corrcoef(gradient_patterns[i], gradient_patterns[j])[0, 1]
        print(f"  Source {test_sources[i]} vs {test_sources[j]}: correlation = {corr:.4f}")

# %%
# Debug: Analyze which features show vertical bar pattern
print("\nDEBUG: Analyzing vertical bar patterns...")

# For each target feature, compute how uniform its attribution column is
column_uniformity = []
for j in range(n_features):
    column = attribution_matrix[:, j].cpu().numpy()
    # Measure uniformity as std/mean ratio (lower = more uniform)
    if np.abs(np.mean(column)) > 1e-10:
        uniformity = np.std(column) / np.abs(np.mean(column))
    else:
        uniformity = float('inf')
    column_uniformity.append({
        'feature': tracker.feature_names[j],
        'uniformity': uniformity,
        'mean': np.mean(column),
        'std': np.std(column)
    })

# Sort by uniformity (most uniform first)
column_uniformity.sort(key=lambda x: x['uniformity'])

print("\nMost uniform columns (vertical bars):")
for i, info in enumerate(column_uniformity[:10]):
    print(f"{i+1}. {info['feature']}: uniformity={info['uniformity']:.4f}, mean={info['mean']:.6f}, std={info['std']:.6f}")

print("\nLeast uniform columns:")
for i, info in enumerate(column_uniformity[-10:]):
    print(f"{i+1}. {info['feature']}: uniformity={info['uniformity']:.4f}, mean={info['mean']:.6f}, std={info['std']:.6f}")

# %%
# Now compute the full attribution matrix with averaging
print("\nComputing feature-to-feature attributions with averaging...")

for i, source_feature in enumerate(tqdm(tracker.feature_names, desc="Computing attributions")):
    # Get mean activation for this feature (to use as backward target)
    source_mean_activation = mean_activations[source_feature]
    
    # Clear gradients - BOTH parameter gradients AND activation gradients
    model.zero_grad()
    for name, act in tracker.activations.items():
        if act.grad is not None:
            act.grad = None
    
    # Backward pass from this feature's mean activation
    # This computes gradients for all features at all token positions
    source_mean_activation.backward(torch.ones_like(source_mean_activation), retain_graph=True)
    
    # Collect gradients on all features AT ALL TOKEN POSITIONS
    for j, target_feature in enumerate(tracker.feature_names):
        if target_feature in tracker.activations:  # Use full activations, not mean
            target_activation = tracker.activations[target_feature]  # [batch, seq_len, 1]
            
            if target_activation.grad is not None:
                # Get gradients at all positions
                grad = target_activation.grad[0, :, 0]  # [seq_len]
                act = target_activation[0, :, 0]  # [seq_len]
                
                # Attribution = gradient * activation at each position
                position_attributions = grad * act  # [seq_len]
                
                # NOW take max absolute attribution over positions
                max_attribution = position_attributions.abs().max().item()
                # Keep the sign of the max absolute value
                max_idx = position_attributions.abs().argmax()
                max_attribution_signed = position_attributions[max_idx].item()
                attribution_matrix[i, j] = max_attribution_signed

print("Attribution computation complete!")

# %%
# Convert to numpy for analysis
attribution_matrix_np = attribution_matrix.cpu().numpy()

# Basic statistics
print("\nAttribution Matrix Statistics:")
print(f"Shape: {attribution_matrix_np.shape}")
print(f"Mean attribution: {np.mean(attribution_matrix_np):.6f}")
print(f"Std deviation: {np.std(attribution_matrix_np):.6f}")
print(f"Max attribution: {np.max(attribution_matrix_np):.6f}")
print(f"Min attribution: {np.min(attribution_matrix_np):.6f}")

# Diagonal vs off-diagonal
diagonal = np.diag(attribution_matrix_np)
off_diagonal = attribution_matrix_np[~np.eye(attribution_matrix_np.shape[0], dtype=bool)]

print(f"\nDiagonal (self-attribution):")
print(f"  Mean: {np.mean(diagonal):.6f}")
print(f"  Std: {np.std(diagonal):.6f}")

print(f"\nOff-diagonal (cross-attribution):")
print(f"  Mean: {np.mean(off_diagonal):.6f}")
print(f"  Std: {np.std(off_diagonal):.6f}")

# Create a version with masked diagonal for visualization
attribution_matrix_masked = attribution_matrix_np.copy()
np.fill_diagonal(attribution_matrix_masked, 0)

# %%
# Visualize the attribution matrix
plt.figure(figsize=(12, 10))

# Use a diverging colormap centered at 0
# Use the off-diagonal values to compute vmax (excluding the diagonal)
vmax = np.percentile(np.abs(off_diagonal), 99)
plt.imshow(attribution_matrix_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(label='Attribution')
plt.title('LoRA Feature Interaction Matrix - Max Attribution (Diagonal Masked)\n(attribution[i,j] = max |attribution| of feature j to feature i across positions)')
plt.xlabel('Source Feature Index (j)')
plt.ylabel('Target Feature Index (i)')
plt.tight_layout()
plt.show()

# %%
# Analyze most interacting feature pairs
print("\nTop 20 positive interactions (excluding self-interactions):")
print("(Note: attribution[i,j] = how feature j influences feature i)")

# Mask out diagonal
masked_attributions = attribution_matrix_np.copy()
np.fill_diagonal(masked_attributions, -np.inf)

# Find top positive interactions
flat_indices = np.argsort(masked_attributions.ravel())[::-1][:20]
top_interactions = []

for idx in flat_indices:
    i, j = np.unravel_index(idx, attribution_matrix_np.shape)
    attr = attribution_matrix_np[i, j]
    if attr != -np.inf:
        # CORRECTED: j influences i, not i influences j
        top_interactions.append({
            'source': tracker.feature_names[j],  # j is the source
            'target': tracker.feature_names[i],  # i is the target
            'attribution': attr
        })

for rank, interaction in enumerate(top_interactions):
    print(f"{rank+1:2d}. {interaction['source']} → {interaction['target']}: {interaction['attribution']:.6f}")

# %%
# Analyze most negative interactions
print("\nTop 20 negative interactions:")

# Find most negative interactions
flat_indices_neg = np.argsort(masked_attributions.ravel())[:20]
negative_interactions = []

for idx in flat_indices_neg:
    i, j = np.unravel_index(idx, attribution_matrix_np.shape)
    attr = attribution_matrix_np[i, j]
    if attr != -np.inf:
        negative_interactions.append({
            'source': tracker.feature_names[i],
            'target': tracker.feature_names[j],
            'attribution': attr
        })

for rank, interaction in enumerate(negative_interactions):
    print(f"{rank+1:2d}. {interaction['source']} → {interaction['target']}: {interaction['attribution']:.6f}")

# %%
# Analyze feature influence patterns
# Compute total influence (sum of absolute attributions) for each feature

# How much each feature influences others (row sums)
influence_on_others = np.sum(np.abs(attribution_matrix_np), axis=1)

# How much each feature is influenced by others (column sums)  
influenced_by_others = np.sum(np.abs(attribution_matrix_np), axis=0)

# Create a summary dataframe
feature_influence_data = []
for i, feature_name in enumerate(tracker.feature_names):
    feature_influence_data.append({
        'feature': feature_name,
        'influences_others': influence_on_others[i],
        'influenced_by_others': influenced_by_others[i],
        'self_attribution': diagonal[i]
    })

# Sort by total influence
feature_influence_data.sort(key=lambda x: x['influences_others'], reverse=True)

print("\nTop 15 most influential features (affect many others):")
headers = ["Rank", "Feature", "Influences Others", "Influenced by Others", "Self Attribution"]
table_data = []
for i, data in enumerate(feature_influence_data[:15]):
    table_data.append([
        i+1,
        data['feature'],
        f"{data['influences_others']:.4f}",
        f"{data['influenced_by_others']:.4f}",
        f"{data['self_attribution']:.4f}"
    ])
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# %%
# Visualize feature connectivity as a heatmap of top interactions
plt.figure(figsize=(14, 10))

# Create a filtered version showing only strong interactions
# Use only off-diagonal values for threshold computation
threshold = np.percentile(np.abs(off_diagonal), 95)
strong_interactions = attribution_matrix_np.copy()
strong_interactions[np.abs(strong_interactions) < threshold] = 0

# Plot - mask diagonal in strong interactions too
strong_interactions_masked = strong_interactions.copy()
np.fill_diagonal(strong_interactions_masked, 0)

plt.subplot(1, 2, 1)
plt.imshow(strong_interactions_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(label='Attribution')
plt.title('Strong Feature Interactions Only\n(Top 5% by magnitude, diagonal masked)')
plt.xlabel('Target Feature Index')
plt.ylabel('Source Feature Index')

# Plot histogram of attribution values
plt.subplot(1, 2, 2)
plt.hist(off_diagonal, bins=100, alpha=0.7, color='blue', label='Off-diagonal')
plt.hist(diagonal, bins=50, alpha=0.7, color='red', label='Diagonal')
plt.xlabel('Attribution Value')
plt.ylabel('Count')
plt.title('Distribution of Attribution Values')
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.show()

# %%
# Compute covariance matrix of feature activations
print("\nComputing feature activation covariance matrix...")

# Collect all activations into a matrix [n_features, n_positions]
activation_matrix = []
for feature_name in tracker.feature_names:
    if feature_name in tracker.activations:
        act = tracker.activations[feature_name]
        # Flatten across batch and sequence dimensions
        flat_act = act[0, :, 0].float().detach().cpu().numpy()  # [seq_len]
        activation_matrix.append(flat_act)

activation_matrix = np.array(activation_matrix)  # [n_features, seq_len]

# Compute covariance
covariance_matrix = np.cov(activation_matrix)  # [n_features, n_features]

# Also compute correlation for normalized view
correlation_matrix = np.corrcoef(activation_matrix)

# Visualize covariance
plt.figure(figsize=(12, 10))
vmax_cov = np.percentile(np.abs(covariance_matrix), 99)
plt.imshow(covariance_matrix, cmap='RdBu_r', vmin=-vmax_cov, vmax=vmax_cov, aspect='auto')
plt.colorbar(label='Covariance')
plt.title('LoRA Feature Activation Covariance Matrix')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.savefig('/workspace/reasoning_interp/covariance_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualize correlation
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(label='Correlation')
plt.title('LoRA Feature Activation Correlation Matrix')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.savefig('/workspace/reasoning_interp/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze covariance patterns
print("\nCovariance Matrix Statistics:")
print(f"Shape: {covariance_matrix.shape}")

# Off-diagonal statistics
off_diag_mask = ~np.eye(n_features, dtype=bool)
off_diag_cov = covariance_matrix[off_diag_mask]
off_diag_corr = correlation_matrix[off_diag_mask]

print(f"\nOff-diagonal covariance:")
print(f"  Mean: {np.mean(off_diag_cov):.6f}")
print(f"  Std: {np.std(off_diag_cov):.6f}")
print(f"  Max: {np.max(off_diag_cov):.6f}")
print(f"  Min: {np.min(off_diag_cov):.6f}")

print(f"\nOff-diagonal correlation:")
print(f"  Mean: {np.mean(off_diag_corr):.6f}")
print(f"  Std: {np.std(off_diag_corr):.6f}")
print(f"  Max: {np.max(off_diag_corr):.6f}")
print(f"  Min: {np.min(off_diag_corr):.6f}")

# Find most correlated feature pairs
correlations = []
for i in range(n_features):
    for j in range(i+1, n_features):
        correlations.append({
            'feature1': tracker.feature_names[i],
            'feature2': tracker.feature_names[j],
            'correlation': correlation_matrix[i, j],
            'covariance': covariance_matrix[i, j]
        })

correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

print("\nTop 10 most correlated feature pairs:")
for i, pair in enumerate(correlations[:10]):
    print(f"{i+1}. {pair['feature1']} <-> {pair['feature2']}: "
          f"corr={pair['correlation']:.4f}, cov={pair['covariance']:.4f}")

print("\nTop 10 most anti-correlated feature pairs:")
correlations.sort(key=lambda x: x['correlation'])
for i, pair in enumerate(correlations[:10]):
    print(f"{i+1}. {pair['feature1']} <-> {pair['feature2']}: "
          f"corr={pair['correlation']:.4f}, cov={pair['covariance']:.4f}")

# %%
print("\nFeature interaction analysis complete!")
# %%
