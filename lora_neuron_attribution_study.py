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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*544")
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


class LoRANeuronTracker:
    """Tracks LoRA neuron activations during forward pass while preserving gradient flow"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}  # {layer_name: tensor}
        self.hooks = []
        
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
                        hook = self._create_hook(f"layer{layer_idx}.mlp.{proj_name}", adapter_name)
                        handle = lora_A_module.register_forward_hook(hook)
                        self.hooks.append(handle)
        
        print(f"Registered {len(self.hooks)} hooks on MLP LoRA A matrices")
    
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
example_idx = 16  # 10th example as requested
max_new_tokens = 256
generation_cache_file = f"math500_generation_example_{example_idx}_{max_new_tokens}.json"

# %%
# Load MATH-500 dataset
print("Loading MATH-500 dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# Get the 10th example
example = dataset[example_idx]
problem = example['problem']

# print(f"\nUsing example {example_idx}:")
# print(f"Problem: {problem[:200]}..." if len(problem) > 200 else f"Problem: {problem}")

prompt = f"""<|im_start|>system
You are a helpful mathematics assistant.
<|im_end|>
<|im_start|>user
{problem}
<|im_end|>
<|im_start|>assistant
<|im_start|>think
Okay, so I need to compute this sum: 1 - 2 + 3 - 4 + 5 - ... + 99 - 100. Hmm, let me think about how to approach this.

First, I notice that the signs alternate between positive and negative. The first term is positive, then negative, then positive again, and so on. So it's an alternating series where each term alternates between adding and subtracting consecutive integers starting from 1 up to 100. But wait, the last term is -100, right? Because the pattern is n - (n+1). Let me check: the first pair is 1 - 2, then 3 - 4, ..., up to 99 - 100. So there are 100 terms in total, but arranged in 50 pairs, each consisting of an odd number minus the next even number.

So maybe I can group them into these pairs and compute the sum of each pair first. Let's try that. Each pair is (2k - 1) - 2k for k = 1 to 50. Wait, when k=1: 1 - """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# %%
with torch.no_grad():
    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        # temperature=0.7,
        do_sample=False,
        # top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
generated_text = tokenizer.decode(*generated_ids)
print(f"\nGenerated response: {generated_text}...")

# %%
# Tokenize and display tokens with positions
inputs = tokenizer(generated_text, return_tensors="pt").to(model.device)
input_ids = inputs['input_ids'][:, :512]

# %%
print("\n" + "="*80)
print("TOKENIZED PROMPT - TOKENS WITH POSITION INDICES")
print("="*80)

# Decode tokens properly to handle special characters
tokens = []
for i in range(len(input_ids[0])):
    # Decode each token individually to get the exact string representation
    token_str = tokenizer.decode(input_ids[0][i:i+1])
    tokens.append(token_str)

for idx, (token_id, token) in enumerate(zip(input_ids[0], tokens)):
    # Escape special characters for display
    display_token = repr(token)[1:-1]  # Remove quotes from repr
    print(f"Position {idx:4d}: token={token:<20} (display: {display_token:<20}) id={token_id.item():<6}")

print("="*80)
print(f"Total tokens: {len(tokens)}")
print("="*80 + "\n")

# %%
# USER: HARDCODE YOUR TARGET POSITION HERE BASED ON THE PRINTED TOKENS ABOVE
target_position = 288  # Use -1 for last token, or specify a position

# %%
# Compute attribution
print("Computing LoRA neuron attributions...")

# Setup linearized LayerNorm
# This linearizes LayerNorm by treating mean/variance as constants during backward pass
# Each token's normalization is independent, so no cross-token gradients exist
# This preserves gradient flow while making LayerNorm act as a simple linear scaling
print("Linearizing LayerNorm modules...")
layernorm_linearizer = LinearizedLayerNorm(model)
layernorm_linearizer.linearize_all_layernorms()

# Setup linearized attention
# This treats attention patterns as constants during backward pass
# Gradients flow through V but not through the softmax(QK^T) computation
# This preserves OV circuit gradients while linearizing attention patterns
print("Linearizing attention patterns...")
attention_linearizer = AttentionLinearizer(model)
attention_linearizer.linearize_all_attention_modules()

# Setup tracker
tracker = LoRANeuronTracker(model)
tracker.register_hooks()

# Enable gradient computation
model.eval()
torch.set_grad_enabled(True)

# Clear any existing gradients
model.zero_grad()

# Forward pass - hooks will capture activations with retain_grad()
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(input_ids=input_ids)

# Handle target position
if target_position == -1:
    target_position = input_ids.shape[1] - 1

print(f"\nTarget position: {target_position}")
print(f"Target token: '{tokens[target_position]}' (display: {repr(tokens[target_position])[1:-1]})")

# Get logits at target position
logits = outputs.logits[0, target_position]  # [vocab_size]

# Find top 10 tokens by logit value
top_k = 10
top_logits, top_indices = torch.topk(logits, top_k)

print(f"\nTop {top_k} predictions at position {target_position}:")
print("="*70)
print(f"{'Rank':<6} {'Token ID':<10} {'Token':<30} {'Logit':<10}")
print("="*70)

for i in range(top_k):
    token_id = top_indices[i].item()
    token = tokenizer.decode([token_id])
    logit_value = top_logits[i].item()
    display_token = repr(token)[1:-1][:30]
    print(f"{i:<6} {token_id:<10} {display_token:<30} {logit_value:<10.3f}")

print("="*70)

# %%
# USER: SPECIFY POSITIVE AND NEGATIVE LOGIT TOKEN IDS HERE
# Based on the table above, choose:
# - positive_token_id: The token you want the model to predict (target)
# - negative_token_id: The token you want to contrast against (counterfactual)

positive_token_id = 13824  # Set this to the token ID for positive logit
negative_token_id = 2055  # Set this to the token ID for negative logit

# If not specified, use top 2 tokens as default
if positive_token_id is None:
    positive_token_id = top_indices[0].item()
    print(f"\nUsing default positive token: {positive_token_id} (top prediction)")
if negative_token_id is None:
    negative_token_id = top_indices[1].item()
    print(f"Using default negative token: {negative_token_id} (second prediction)")

positive_token = tokenizer.decode([positive_token_id])
negative_token = tokenizer.decode([negative_token_id])
positive_logit = logits[positive_token_id]
negative_logit = logits[negative_token_id]

print(f"\nComputing attribution for logit difference:")
print(f"Positive token: '{positive_token}' (id: {positive_token_id}, logit: {positive_logit.item():.3f})")
print(f"Negative token: '{negative_token}' (id: {negative_token_id}, logit: {negative_logit.item():.3f})")
print(f"Logit difference: {positive_logit.item() - negative_logit.item():.3f}")

# Compute logit difference as target metric
# target_metric = positive_logit - negative_logit
target_metric = positive_logit

# %%
# Compute gradients and attributions for ALL positions
print("\nComputing gradients for all token positions...")
all_attributions = []

# One backward pass computes ALL gradients
# model.zero_grad()
target_metric.backward(retain_graph=True)

# outputs = model(input_ids=input_ids)
# logits = outputs.logits[0, target_position]
# target_metric = positive_logit - negative_logit
# target_metric.backward(retain_graph=True)

# Then just access the stored gradients
for name, activation in tracker.activations.items():
    grad = activation.grad  # Already computed!
    
    # Process as before...
    for pos in range(min(target_position + 1, activation.shape[1])):
        activation_value = activation[0, pos, 0].item()
        gradient_value = grad[0, pos, 0].item()
        attribution = gradient_value * activation_value
        
        # Store with position and token information
        all_attributions.append({
            'layer': name,
            'position': pos,
            'token': tokens[pos],
            'token_id': input_ids[0][pos].item(),
            'attribution': attribution,
            'activation': activation_value,
            'gradient': gradient_value
        })

print(f"Computed {len(all_attributions)} total attributions")

# %%
# Sort by attribution (not absolute)
sorted_attributions = sorted(
    all_attributions,
    key=lambda x: x['attribution'],
    reverse=True
)


# %%
# Create attribution heatmap visualization
print("\nCreating interactive attribution heatmap...")

# Organize attribution data into a matrix
# First, get unique layers and their order
unique_layers = []
seen_layers = set()
for entry in all_attributions:
    if entry['layer'] not in seen_layers:
        unique_layers.append(entry['layer'])
        seen_layers.add(entry['layer'])

# Create mapping from layer to index
layer_to_idx = {layer: idx for idx, layer in enumerate(unique_layers)}

# Get number of positions to visualize (up to target position + 1)
num_positions = min(target_position + 1, input_ids.shape[1])

# Initialize attribution matrix: [num_features, num_positions]
attribution_matrix = np.zeros((len(unique_layers), num_positions))

# Fill the matrix
for entry in all_attributions:
    layer_idx = layer_to_idx[entry['layer']]
    pos = entry['position']
    if pos < num_positions:
        attribution_matrix[layer_idx, pos] = entry['attribution']

# Prepare data for Plotly
# Token labels for x-axis
token_labels = [f"{i}: {repr(tokens[i])[1:-1][:30]}" for i in range(num_positions)]
# Feature labels for y-axis
feature_labels = [f"{layer.split('layer')[1].split('.')[0]}.{'.'.join(layer.split('.')[2:])}" for layer in unique_layers]

# Create hover text with detailed information
hover_text = []
for i, layer in enumerate(unique_layers):
    row_hover = []
    for j in range(num_positions):
        row_hover.append(f"Feature: {layer}<br>Position: {j}<br>Token: {repr(tokens[j])[1:-1]}<br>Attribution: {attribution_matrix[i, j]:.6f}")
    hover_text.append(row_hover)

# Create interactive heatmap
# Use actual max value for color scale
vmax = np.abs(attribution_matrix).max()

fig = go.Figure(data=go.Heatmap(
    z=attribution_matrix,
    x=token_labels,
    y=feature_labels,
    colorscale='RdBu_r',
    zmid=0,
    zmin=-vmax,
    zmax=vmax,
    hovertext=hover_text,
    hoverinfo='text',
    colorbar=dict(title="Attribution")
))

# Add vertical line at target position
fig.add_vline(x=target_position, line_width=2, line_dash="dash", line_color="green", opacity=0.7)

# Update layout
fig.update_layout(
    title=dict(
        text=f'LoRA Feature Attribution Heatmap<br>Target: position {target_position} ("{repr(tokens[target_position])[1:-1]}")<br>'
             f'Metric: logit("{positive_token}") - logit("{negative_token}") = {positive_logit.item() - negative_logit.item():.3f}',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Token Position",
    yaxis_title="LoRA Feature (Layer.Module)",
    height=max(600, len(unique_layers) * 15),
    width=max(800, num_positions * 15),
    xaxis=dict(tickangle=-90)
)

# Save as HTML
heatmap_file = f'lora_attribution_heatmap_example_{example_idx}_pos_{target_position}.html'
fig.write_html(heatmap_file)
print(f"Interactive heatmap saved to {heatmap_file}")
fig.show()

# %%
# Create a focused heatmap showing only the most important features
print("\nCreating focused interactive attribution heatmap (top features only)...")

# Calculate total absolute attribution per feature
feature_importance = np.abs(attribution_matrix).sum(axis=1)
top_k_features = 50  # Show top 50 most important features

# Get indices of top features
top_feature_indices = np.argsort(feature_importance)[-top_k_features:][::-1]

# Create subset matrix
focused_matrix = attribution_matrix[top_feature_indices, :]
focused_layers = [unique_layers[i] for i in top_feature_indices]

# Prepare data for focused heatmap
focused_feature_labels = [f"{layer.split('layer')[1].split('.')[0]}.{'.'.join(layer.split('.')[2:])}" for layer in focused_layers]

# Create hover text for focused heatmap
focused_hover_text = []
for i, layer in enumerate(focused_layers):
    row_hover = []
    for j in range(num_positions):
        row_hover.append(f"Feature: {layer}<br>Position: {j}<br>Token: {repr(tokens[j])[1:-1]}<br>Attribution: {focused_matrix[i, j]:.6f}")
    focused_hover_text.append(row_hover)

# Create focused interactive heatmap
# Use actual max value for the focused matrix
vmax_focused = np.abs(focused_matrix).max()

fig_focused = go.Figure(data=go.Heatmap(
    z=focused_matrix,
    x=token_labels,
    y=focused_feature_labels,
    colorscale='RdBu_r',
    zmid=0,
    zmin=-vmax_focused,
    zmax=vmax_focused,
    hovertext=focused_hover_text,
    hoverinfo='text',
    colorbar=dict(title="Attribution")
))

# Add vertical line at target position
fig_focused.add_vline(x=target_position, line_width=2, line_dash="dash", line_color="green", opacity=0.7)

# Update layout
fig_focused.update_layout(
    title=dict(
        text=f'Top {top_k_features} LoRA Features by Attribution<br>Target: position {target_position} ("{repr(tokens[target_position])[1:-1]}")',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Token Position",
    yaxis_title="LoRA Feature (Layer.Module.Adapter)",
    height=800,
    width=max(800, num_positions * 15),
    xaxis=dict(tickangle=-90)
)

# Save focused heatmap
focused_heatmap_file = f'lora_attribution_heatmap_focused_example_{example_idx}_pos_{target_position}.html'
fig_focused.write_html(focused_heatmap_file)
print(f"Focused interactive heatmap saved to {focused_heatmap_file}")
fig_focused.show()

# %%
# Create position-wise attribution summary
print("\nCreating interactive position-wise attribution summary...")

# Sum attributions by position
position_attributions = attribution_matrix.sum(axis=0)

# Create bar colors
bar_colors = ['red' if x < 0 else 'blue' for x in position_attributions]

# Create hover text for bars
bar_hover_text = [f"Position: {i}<br>Token: {repr(tokens[i])[1:-1]}<br>Total Attribution: {position_attributions[i]:.6f}" 
                  for i in range(num_positions)]

# Create interactive bar chart
fig_bar = go.Figure()

# Add bars
fig_bar.add_trace(go.Bar(
    x=token_labels,
    y=position_attributions,
    marker_color=bar_colors,
    hovertext=bar_hover_text,
    hoverinfo='text',
    name='Attribution'
))

# Add vertical line at target position
fig_bar.add_vline(x=target_position, line_width=2, line_dash="dash", line_color="green", opacity=0.7)

# Update layout
fig_bar.update_layout(
    title=dict(
        text=f'Total Attribution by Token Position<br>Target: position {target_position}',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Token Position",
    yaxis_title="Total Attribution",
    height=600,
    width=max(800, num_positions * 15),
    xaxis=dict(tickangle=-90),
    showlegend=False
)

# Save bar chart
position_summary_file = f'lora_attribution_by_position_example_{example_idx}_pos_{target_position}.html'
fig_bar.write_html(position_summary_file)
print(f"Interactive position summary saved to {position_summary_file}")
fig_bar.show()

# %%
# Create a combined visualization with subplots
print("\nCreating combined interactive visualization...")

# Create subplots
fig_combined = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    shared_xaxes=True,
    subplot_titles=(f'Top {top_k_features} LoRA Features Attribution', 'Total Attribution by Position'),
    vertical_spacing=0.1
)

# Add focused heatmap to top subplot
fig_combined.add_trace(
    go.Heatmap(
        z=focused_matrix,
        x=token_labels,
        y=focused_feature_labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-vmax_focused,
        zmax=vmax_focused,
        hovertext=focused_hover_text,
        hoverinfo='text',
        colorbar=dict(title="Attribution", x=1.02)
    ),
    row=1, col=1
)

# Add bar chart to bottom subplot
fig_combined.add_trace(
    go.Bar(
        x=token_labels,
        y=position_attributions,
        marker_color=bar_colors,
        hovertext=bar_hover_text,
        hoverinfo='text',
        showlegend=False
    ),
    row=2, col=1
)

# Add vertical lines at target position
fig_combined.add_vline(x=target_position, line_width=2, line_dash="dash", line_color="green", opacity=0.7)

# Update layout
fig_combined.update_layout(
    title=dict(
        text=f'LoRA Attribution Analysis<br>Target: position {target_position} ("{repr(tokens[target_position])[1:-1]}")<br>'
             f'Metric: logit("{positive_token}") - logit("{negative_token}") = {positive_logit.item() - negative_logit.item():.3f}',
        x=0.5,
        xanchor='center'
    ),
    height=1000,
    width=max(1000, num_positions * 20),
    showlegend=False
)

# Update x-axes
fig_combined.update_xaxes(title_text="Token Position", tickangle=-90, row=2, col=1)
fig_combined.update_yaxes(title_text="LoRA Feature", row=1, col=1)
fig_combined.update_yaxes(title_text="Total Attribution", row=2, col=1)

# Save combined visualization
combined_file = f'lora_attribution_combined_example_{example_idx}_pos_{target_position}.html'
fig_combined.write_html(combined_file)
print(f"Combined interactive visualization saved to {combined_file}")
fig_combined.show()

# %%
print("\nAttribution study complete!")

# %%
