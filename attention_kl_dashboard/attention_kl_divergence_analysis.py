# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import gc
import os
from typing import Dict, List, Tuple, Optional

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
# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# %%
# Hook storage for attention patterns
attention_patterns_storage = {}

def create_attention_hook(model_name: str, layer_idx: int):
    """Create a hook to capture attention patterns from a specific layer"""
    def hook(module, input, output):
        # For Qwen2 models, the attention output is a tuple
        # (hidden_states, attention_weights, past_key_values)
        if len(output) >= 2 and output[1] is not None:
            # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
            attention_weights = output[1].detach().cpu()
            key = f"{model_name}_layer_{layer_idx}"
            attention_patterns_storage[key] = attention_weights
    return hook

# %%
# Load base model
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Use eager attention to get attention weights
)

# Get number of layers and heads
n_layers = base_model.config.num_hidden_layers
n_heads = base_model.config.num_attention_heads
print(f"Model has {n_layers} layers and {n_heads} attention heads")

# %%
# Load LoRA model
print("\nLoading LoRA adapter...")
lora_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Use eager attention to get attention weights
)
lora_model = PeftModel.from_pretrained(lora_model, lora_dir, torch_dtype=torch.bfloat16)

# %%
# Generate or load a reasoning trace
print("\nPreparing reasoning trace...")

# Option 1: Use a cached generation if available
generation_cache_file = "math500_generation_example_10.json"
if os.path.exists(generation_cache_file):
    print(f"Loading cached generation from {generation_cache_file}")
    with open(generation_cache_file, 'r') as f:
        cache_data = json.load(f)
    full_text = cache_data['full_text']
    print(f"Loaded reasoning trace with {len(tokenizer.encode(full_text))} tokens")
else:
    # Option 2: Generate a new reasoning trace
    print("Generating new reasoning trace...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problem = dataset[10]['problem']
    
    system_prompt = "You are a helpful mathematics assistant. Please think step by step to solve the problem."
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
    
    with torch.no_grad():
        generated_ids = lora_model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

# Tokenize for analysis
inputs = tokenizer(full_text, return_tensors="pt", max_length=512, truncation=True)
input_ids = inputs.input_ids.to(base_model.device)
seq_len = input_ids.shape[1]

print(f"\nAnalyzing sequence of length: {seq_len}")

# %%
# Register hooks on both models
print("\nRegistering attention hooks...")

base_hooks = []
lora_hooks = []

for layer_idx in range(n_layers):
    # Hook for base model
    base_hook = base_model.model.layers[layer_idx].self_attn.register_forward_hook(
        create_attention_hook("base", layer_idx)
    )
    base_hooks.append(base_hook)
    
    # Hook for LoRA model
    lora_hook = lora_model.model.model.layers[layer_idx].self_attn.register_forward_hook(
        create_attention_hook("lora", layer_idx)
    )
    lora_hooks.append(lora_hook)

# %%
# Run forward pass on base model
print("\nRunning forward pass on base model...")
attention_patterns_storage.clear()

with torch.no_grad():
    base_outputs = base_model(input_ids, output_attentions=True)

# Store base attention patterns
base_attention_patterns = {}
for layer_idx in range(n_layers):
    key = f"base_layer_{layer_idx}"
    if key in attention_patterns_storage:
        base_attention_patterns[layer_idx] = attention_patterns_storage[key]

# Clear storage
attention_patterns_storage.clear()

# %%
# Run forward pass on LoRA model
print("\nRunning forward pass on LoRA model...")

with torch.no_grad():
    lora_outputs = lora_model(input_ids, output_attentions=True)

# Store LoRA attention patterns
lora_attention_patterns = {}
for layer_idx in range(n_layers):
    key = f"lora_layer_{layer_idx}"
    if key in attention_patterns_storage:
        lora_attention_patterns[layer_idx] = attention_patterns_storage[key]

# Remove hooks
for hook in base_hooks + lora_hooks:
    hook.remove()

# %%
# Compute KL divergence for each position, layer, and head
print("\nComputing KL divergences...")

kl_divergences = np.zeros((n_layers, n_heads, seq_len))

for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
    if layer_idx not in base_attention_patterns or layer_idx not in lora_attention_patterns:
        continue
    
    base_attn = base_attention_patterns[layer_idx][0]  # [n_heads, seq_len, seq_len]
    lora_attn = lora_attention_patterns[layer_idx][0]  # [n_heads, seq_len, seq_len]
    
    for head_idx in range(n_heads):
        for pos_idx in range(seq_len):
            # Get attention distributions for this position
            base_dist = base_attn[head_idx, pos_idx, :pos_idx+1]  # Only look at previous positions
            lora_dist = lora_attn[head_idx, pos_idx, :pos_idx+1]
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            base_dist = base_dist + epsilon
            lora_dist = lora_dist + epsilon
            
            # Renormalize
            base_dist = base_dist / base_dist.sum()
            lora_dist = lora_dist / lora_dist.sum()
            
            # Compute KL divergence: KL(base || lora)
            kl_div = (base_dist * (base_dist.log() - lora_dist.log())).sum().item()
            kl_divergences[layer_idx, head_idx, pos_idx] = kl_div

# %%
# Analyze results
print("\nAnalyzing KL divergence results...")

# Average KL divergence per layer
avg_kl_per_layer = kl_divergences.mean(axis=(1, 2))
print("\nAverage KL divergence per layer:")
for layer_idx, avg_kl in enumerate(avg_kl_per_layer):
    print(f"Layer {layer_idx}: {avg_kl:.4f}")

# Average KL divergence per head (across all layers)
avg_kl_per_head = kl_divergences.mean(axis=(0, 2))
print(f"\nAverage KL divergence per head (across layers):")
print(f"Min: {avg_kl_per_head.min():.4f}, Max: {avg_kl_per_head.max():.4f}, Mean: {avg_kl_per_head.mean():.4f}")

# Average KL divergence per position
avg_kl_per_position = kl_divergences.mean(axis=(0, 1))

# %%
# Create visualizations
print("\nCreating visualizations...")

# 1. Heatmap of KL divergence by layer and position
fig = go.Figure()

# Average across heads for each layer
kl_by_layer_position = kl_divergences.mean(axis=1)  # [n_layers, seq_len]

fig.add_trace(go.Heatmap(
    z=kl_by_layer_position,
    x=list(range(seq_len)),
    y=list(range(n_layers)),
    colorscale='Viridis',
    colorbar=dict(title="KL Divergence"),
    hovertemplate="Layer: %{y}<br>Position: %{x}<br>KL: %{z:.4f}<extra></extra>"
))

fig.update_layout(
    title="KL Divergence of Attention Patterns: Base vs LoRA Model",
    xaxis_title="Token Position",
    yaxis_title="Layer",
    width=1200,
    height=800
)

# Display the figure
fig.show()
print("Displayed KL divergence heatmap")

# %%
# 2. Line plot of average KL divergence per position
fig2 = go.Figure()

# Decode tokens for hover information
tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

fig2.add_trace(go.Scatter(
    x=list(range(seq_len)),
    y=avg_kl_per_position,
    mode='lines+markers',
    name='Average KL Divergence',
    text=[f"Token: {repr(token)}" for token in tokens],
    hovertemplate="Position: %{x}<br>KL: %{y:.4f}<br>%{text}<extra></extra>"
))

fig2.update_layout(
    title="Average KL Divergence by Token Position",
    xaxis_title="Token Position",
    yaxis_title="Average KL Divergence",
    width=1200,
    height=600
)

# Display the figure
fig2.show()
print("Displayed KL divergence by position")

# %%
# 3. Find top positions with highest KL divergence
top_k = 20
print(f"\nTop {top_k} positions with highest average KL divergence:")

top_positions = np.argsort(avg_kl_per_position)[-top_k:][::-1]
for rank, pos in enumerate(top_positions):
    token = tokens[pos]
    kl_value = avg_kl_per_position[pos]
    print(f"{rank+1:2d}. Position {pos:3d}: KL={kl_value:.4f}, Token: {repr(token)}")

# %%
# 4. Find layers and heads with highest divergence
print(f"\nTop layer-head combinations with highest average KL divergence:")

# Reshape to get all layer-head combinations
kl_per_layer_head = kl_divergences.mean(axis=2)  # [n_layers, n_heads]
layer_head_pairs = []
for layer in range(n_layers):
    for head in range(n_heads):
        layer_head_pairs.append((layer, head, kl_per_layer_head[layer, head]))

# Sort by KL divergence
layer_head_pairs.sort(key=lambda x: x[2], reverse=True)

for rank, (layer, head, kl_value) in enumerate(layer_head_pairs[:20]):
    print(f"{rank+1:2d}. Layer {layer:2d}, Head {head:2d}: KL={kl_value:.4f}")

# %%
# 5. Create detailed heatmap for top divergent layers
top_layers = np.argsort(avg_kl_per_layer)[-5:][::-1]

fig3 = make_subplots(
    rows=len(top_layers), cols=1,
    subplot_titles=[f"Layer {layer} (Avg KL: {avg_kl_per_layer[layer]:.4f})" 
                   for layer in top_layers],
    vertical_spacing=0.05
)

for idx, layer in enumerate(top_layers):
    fig3.add_trace(
        go.Heatmap(
            z=kl_divergences[layer],  # [n_heads, seq_len]
            x=list(range(seq_len)),
            y=list(range(n_heads)),
            colorscale='Viridis',
            showscale=(idx == 0),
            colorbar=dict(title="KL Divergence") if idx == 0 else None,
            hovertemplate=f"Layer {layer}<br>Head: %{{y}}<br>Position: %{{x}}<br>KL: %{{z:.4f}}<extra></extra>"
        ),
        row=idx+1, col=1
    )
    
    fig3.update_xaxes(title_text="Token Position" if idx == len(top_layers)-1 else "", row=idx+1, col=1)
    fig3.update_yaxes(title_text="Head", row=idx+1, col=1)

fig3.update_layout(
    title="KL Divergence by Head and Position for Top Divergent Layers",
    height=300 * len(top_layers),
    width=1200
)

# Display the figure
fig3.show()
print("Displayed detailed layer analysis")

# %%
# Save results for further analysis
results = {
    "model_info": {
        "base_model": base_model_id,
        "lora_checkpoint": lora_dir,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len
    },
    "kl_divergences": {
        "full_tensor_shape": list(kl_divergences.shape),
        "avg_per_layer": avg_kl_per_layer.tolist(),
        "avg_per_position": avg_kl_per_position.tolist(),
        "avg_per_head": avg_kl_per_head.tolist(),
        "top_positions": [
            {
                "position": int(pos),
                "token": tokens[pos],
                "avg_kl": float(avg_kl_per_position[pos])
            }
            for pos in top_positions
        ],
        "top_layer_head_pairs": [
            {
                "layer": int(layer),
                "head": int(head),
                "avg_kl": float(kl)
            }
            for layer, head, kl in layer_head_pairs[:50]
        ]
    },
    "text_info": {
        "full_text_preview": full_text[:500] + "...",
        "tokens_preview": tokens[:50]
    }
}

with open("attention_kl_divergence_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to attention_kl_divergence_results.json")

# %%
# Memory cleanup
del base_attention_patterns
del lora_attention_patterns
del base_model
del lora_model
gc.collect()
torch.cuda.empty_cache()

print("\nAnalysis complete!")

# %%