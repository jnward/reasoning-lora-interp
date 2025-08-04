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
    """Tracks LoRA feature activations"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}  # {layer_name: tensor}
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
# NEW APPROACH: Compute feature-to-output attribution for each feature,
# then analyze the correlation/interaction between these attribution patterns

print("\nComputing LoRA feature interactions via output attributions...")

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

# Get logits at the last position
logits = outputs.logits[0, -1]  # [vocab_size]

# Get the number of features
n_features = len(tracker.feature_names)
print(f"\nTotal features: {n_features}")

# %%
# Compute attribution patterns for each feature to the output
print("\nComputing feature attribution patterns...")

# We'll compute how each feature at each position contributes to the final logits
# Store attribution patterns: [n_features, seq_len]
attribution_patterns = torch.zeros(n_features, seq_len, device=model.device)

# Use top-k logits as targets to get diverse gradients
top_k = 10
top_logits, top_indices = torch.topk(logits, top_k)

for k in range(top_k):
    print(f"\nComputing attributions for logit {k+1}/{top_k}")
    target_logit = logits[top_indices[k]]
    
    # Clear gradients
    model.zero_grad()
    
    # Backward from this logit
    target_logit.backward(retain_graph=True)
    
    # Collect gradients for all features
    for i, feature_name in enumerate(tracker.feature_names):
        if feature_name in tracker.activations:
            activation = tracker.activations[feature_name]  # [batch, seq_len, 1]
            
            if activation.grad is not None:
                # Get gradient and activation at each position
                grad = activation.grad[0, :, 0]  # [seq_len]
                act = activation[0, :, 0]  # [seq_len]
                
                # Attribution at each position
                attr = grad * act  # [seq_len]
                
                # Add to pattern (accumulating over different logits)
                attribution_patterns[i, :] += attr

# Average over the k logits
attribution_patterns /= top_k

print("\nAttribution pattern computation complete!")

# %%
# Now compute feature interaction as correlation between attribution patterns
print("\nComputing feature interaction matrix from attribution patterns...")

# Convert to numpy for easier manipulation
attr_patterns_np = attribution_patterns.cpu().numpy()

# Compute correlation matrix between feature attribution patterns
# This tells us which features tend to contribute similarly to outputs
from scipy.stats import pearsonr

correlation_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        # Correlate the attribution patterns of features i and j
        corr, _ = pearsonr(attr_patterns_np[i, :], attr_patterns_np[j, :])
        correlation_matrix[i, j] = corr

print("Feature correlation computation complete!")

# %%
# Visualize the correlation matrix
plt.figure(figsize=(12, 10))

# Use a diverging colormap
plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(label='Correlation')
plt.title('LoRA Feature Interaction Matrix\n(Correlation of attribution patterns to output)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.savefig('/workspace/reasoning_interp/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Alternative: Compute interaction as covariance of activations
print("\nComputing feature activation covariance...")

# Get activation patterns [n_features, seq_len]
activation_matrix = torch.zeros(n_features, seq_len, device=model.device)

for i, feature_name in enumerate(tracker.feature_names):
    if feature_name in tracker.activations:
        activation = tracker.activations[feature_name]  # [batch, seq_len, 1]
        activation_matrix[i, :] = activation[0, :, 0]

# Convert to numpy
activation_matrix_np = activation_matrix.cpu().numpy()

# Compute covariance matrix
# Center the activations first
centered_activations = activation_matrix_np - activation_matrix_np.mean(axis=1, keepdims=True)
covariance_matrix = np.cov(centered_activations)

# %%
# Visualize the covariance matrix
plt.figure(figsize=(12, 10))

# Use a diverging colormap
vmax = np.percentile(np.abs(covariance_matrix), 99)
plt.imshow(covariance_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(label='Covariance')
plt.title('LoRA Feature Covariance Matrix\n(How features co-activate across positions)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.savefig('/workspace/reasoning_interp/covariance_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Analyze the results
print("\nCorrelation Matrix Statistics:")
print(f"Mean correlation: {np.mean(correlation_matrix[~np.eye(n_features, dtype=bool)]):.4f}")
print(f"Std correlation: {np.std(correlation_matrix[~np.eye(n_features, dtype=bool)]):.4f}")

print("\nMost correlated feature pairs:")
# Get upper triangle indices (to avoid duplicates)
triu_indices = np.triu_indices(n_features, k=1)
correlations = correlation_matrix[triu_indices]
sorted_indices = np.argsort(correlations)[::-1]

for i in range(10):
    idx = sorted_indices[i]
    feat_i = triu_indices[0][idx]
    feat_j = triu_indices[1][idx]
    corr = correlations[idx]
    print(f"{i+1}. {tracker.feature_names[feat_i]} <-> {tracker.feature_names[feat_j]}: {corr:.4f}")

print("\nMost anti-correlated feature pairs:")
for i in range(10):
    idx = sorted_indices[-(i+1)]
    feat_i = triu_indices[0][idx]
    feat_j = triu_indices[1][idx]
    corr = correlations[idx]
    print(f"{i+1}. {tracker.feature_names[feat_i]} <-> {tracker.feature_names[feat_j]}: {corr:.4f}")

# %%
# Save results
results = {
    'correlation_matrix': correlation_matrix,
    'covariance_matrix': covariance_matrix,
    'attribution_patterns': attr_patterns_np,
    'feature_names': tracker.feature_names
}

np.savez('/workspace/reasoning_interp/feature_interaction_results_v2.npz', **results)
print("\nResults saved!")