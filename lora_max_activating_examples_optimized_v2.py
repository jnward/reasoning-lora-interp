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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import heapq
import gc

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1
context_window = 10  # Number of tokens before and after
top_k = 16  # Number of top activating examples

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# %%
# Load tokenizer
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
# Extract A matrices for all projections
print("Extracting LoRA A matrices...")

probe_directions = {
    'gate_proj': {},
    'up_proj': {},
    'down_proj': {}
}

# Get the number of layers
n_layers = model.config.num_hidden_layers

for layer_idx in range(n_layers):
    # Extract A matrices for all projections
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        # Access the module directly
        module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
        
        # Extract the LoRA A matrix (which is a vector for rank-1)
        if hasattr(module, 'lora_A'):
            # Get the A matrix from the LoRA adapter
            lora_A_weight = module.lora_A['default'].weight.data
            # For rank-1, this could be shape [input_dim, 1] or [1, input_dim]
            # We want a 1D vector of shape [input_dim]
            probe_direction = lora_A_weight.squeeze()
            probe_directions[proj_type][layer_idx] = probe_direction

print(f"Extracted directions for {len(probe_directions['gate_proj'])} layers")

# %%
# Load s1K-1.1 dataset
print("Loading s1K-1.1 dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"Dataset has {len(dataset)} examples")

# %%
# Optimized storage - just store minimal info
@dataclass
class TopKEntry:
    """Minimal storage for top-k tracking"""
    activation: float
    rollout_idx: int
    token_idx: int
    
class OptimizedTopKTracker:
    """Memory-efficient top-k tracker that stores only essential info"""
    def __init__(self, k: int):
        self.k = k
        self.top_positive = []  # min heap of (activation, counter, rollout_idx, token_idx)
        self.top_negative = []  # max heap (negated) 
        self.counter = 0  # Tie-breaker
        
    def add(self, activation: float, rollout_idx: int, token_idx: int):
        self.counter += 1
        
        if activation >= 0:
            if len(self.top_positive) < self.k:
                heapq.heappush(self.top_positive, (activation, self.counter, rollout_idx, token_idx))
            elif activation > self.top_positive[0][0]:
                heapq.heapreplace(self.top_positive, (activation, self.counter, rollout_idx, token_idx))
        else:
            # Use negative to create max heap for negative values
            if len(self.top_negative) < self.k:
                heapq.heappush(self.top_negative, (-activation, self.counter, rollout_idx, token_idx))
            elif -activation > self.top_negative[0][0]:
                heapq.heapreplace(self.top_negative, (-activation, self.counter, rollout_idx, token_idx))
    
    def get_top_positive(self) -> List[Tuple[float, int, int]]:
        # Return sorted from highest to lowest: (activation, rollout_idx, token_idx)
        return [(act, rid, tid) for act, _, rid, tid in sorted(self.top_positive, key=lambda x: x[0], reverse=True)]
    
    def get_top_negative(self) -> List[Tuple[float, int, int]]:
        # Return sorted from lowest to highest
        return [(-act, rid, tid) for act, _, rid, tid in sorted(self.top_negative, key=lambda x: x[0])]

# Initialize trackers for each projection type and layer
top_k_trackers = {
    proj_type: {layer: OptimizedTopKTracker(top_k) for layer in range(n_layers)}
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']
}

# Store tokens and metadata for each rollout (for later context extraction)
rollout_data = {}

# Also keep lightweight statistics
activation_stats = {
    proj_type: {layer: {'min': float('inf'), 'max': float('-inf')} for layer in range(n_layers)}
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']
}

# %%
# Process rollouts
num_examples = min(100, len(dataset))
print(f"Processing {num_examples} rollouts...")

# Process rollouts - using DeepSeek traces and attempts
for rollout_idx in tqdm(range(num_examples), desc="Processing rollouts"):
    # Get the rollout
    rollout = dataset[rollout_idx]
    
    # Extract question and DeepSeek thinking trajectory + attempt
    question = rollout['question']
    thinking_trajectory = rollout.get('deepseek_thinking_trajectory', '')
    attempt = rollout.get('deepseek_attempt', '')
    
    if not thinking_trajectory or not attempt:
        print(f"Skipping rollout {rollout_idx}: missing DeepSeek thinking trajectory or attempt")
        continue
    
    # Use the exact format for thinking traces
    system_prompt = "You are a helpful mathematics assistant."
    
    full_text = (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{question}\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{thinking_trajectory}\n"
        f"<|im_start|>answer\n{attempt}<|im_end|>"
    )
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    
    # Decode tokens properly to handle Ġ (space) tokens
    # Use batch_decode to get proper text representation
    tokens = []
    for i in range(len(input_ids)):
        # Decode each token individually to get the exact string representation
        decoded = tokenizer.decode(input_ids[i:i+1])
        tokens.append(decoded)
    
    # Store rollout data for later context extraction
    rollout_data[rollout_idx] = {
        'tokens': tokens,
        'input_ids': input_ids.cpu(),  # Store input_ids for later use
        'full_text': full_text
    }
    
    # Storage for projected activations - we need to keep these for context visualization
    projected_activations = {
        'gate_proj': {},
        'up_proj': {},
        'down_proj': {}
    }
    
    # Hook function to compute gate/up projections from pre-MLP residual
    def make_pre_mlp_hook(layer_idx):
        def hook(module, input, output):
            pre_mlp = output.detach()[0]  # [seq_len, hidden_size]
            # Compute projections for gate and up immediately
            for proj_type in ['gate_proj', 'up_proj']:
                probe_dir = probe_directions[proj_type][layer_idx]
                activations = torch.matmul(pre_mlp.float(), probe_dir)  # [seq_len]
                projected_activations[proj_type][layer_idx] = activations.cpu().numpy()
        return hook
    
    # Hook function to compute down_proj projections from post-SwiGLU
    def make_down_proj_hook(layer_idx):
        def hook(module, input, output):
            # Get the post-SwiGLU activations (input to down_proj)
            post_swiglu = input[0].detach()[0]  # [seq_len, intermediate_size]
            # Project onto the A matrix
            probe_dir = probe_directions['down_proj'][layer_idx]
            activations = torch.matmul(post_swiglu.float(), probe_dir)  # [seq_len]
            projected_activations['down_proj'][layer_idx] = activations.cpu().numpy()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in range(n_layers):
        # Pre-MLP hook (computes gate/up projections)
        layernorm = model.model.model.layers[layer_idx].post_attention_layernorm
        hook = layernorm.register_forward_hook(make_pre_mlp_hook(layer_idx))
        hooks.append(hook)
        
        # Down-proj hook (computes down projections)
        down_proj = model.model.model.layers[layer_idx].mlp.down_proj
        hook = down_proj.register_forward_hook(make_down_proj_hook(layer_idx))
        hooks.append(hook)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Store activations for top-k examples only
    rollout_data[rollout_idx]['activations'] = {}
    
    # Process activations efficiently
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        rollout_data[rollout_idx]['activations'][proj_type] = {}
        
        for layer_idx in range(n_layers):
            # Get pre-computed activations from hooks
            activations = projected_activations[proj_type][layer_idx]
            
            # Update statistics
            activation_stats[proj_type][layer_idx]['min'] = min(
                activation_stats[proj_type][layer_idx]['min'], 
                float(np.min(activations))
            )
            activation_stats[proj_type][layer_idx]['max'] = max(
                activation_stats[proj_type][layer_idx]['max'], 
                float(np.max(activations))
            )
            
            # Update top-k tracker
            for token_idx in range(len(tokens)):
                activation_value = float(activations[token_idx])
                
                # Check if this might be a top-k activation before adding
                tracker = top_k_trackers[proj_type][layer_idx]
                
                # Add to tracker
                tracker.add(activation_value, rollout_idx, token_idx)
            
            # Store the full activation array for this rollout/layer/proj
            # This allows us to get context activations later
            rollout_data[rollout_idx]['activations'][proj_type][layer_idx] = activations
    
    # Clear GPU memory after each rollout to prevent OOM
    torch.cuda.empty_cache()
    
    # Periodic garbage collection to free up memory
    if rollout_idx % 10 == 0:
        gc.collect()

print("Finished processing all rollouts")

# %%
# Extract final top-k results and create context on-demand
print("\nExtracting top activating examples and creating contexts...")

from IPython.display import HTML, display
import html as html_lib

def get_context_with_activations(rollout_idx: int, token_idx: int, 
                               proj_type: str, layer_idx: int) -> Tuple[List[str], List[str], Dict[int, float]]:
    """Extract context with activations for visualization"""
    tokens = rollout_data[rollout_idx]['tokens']
    activations = rollout_data[rollout_idx]['activations'][proj_type][layer_idx]
    
    context_start = max(0, token_idx - context_window)
    context_end = min(len(tokens), token_idx + context_window + 1)
    
    context_before = tokens[context_start:token_idx]
    context_after = tokens[token_idx+1:context_end]
    
    # Get activations for context tokens
    context_activations = {}
    for ctx_idx in range(context_start, context_end):
        if ctx_idx != token_idx:  # Don't duplicate the main token
            context_activations[ctx_idx] = float(activations[ctx_idx])
    
    return context_before, context_after, context_activations

def create_html_examples_optimized(examples: List[Tuple[float, int, int]], 
                                 proj_type: str, layer: int, title: str) -> str:
    """Create HTML visualization with context activations"""
    
    html_parts = [f"<h3>{title}</h3>"]
    
    # Get activation range for normalization
    min_act = activation_stats[proj_type][layer]['min']
    max_act = activation_stats[proj_type][layer]['max']
    
    for activation, rollout_idx, token_idx in examples:
        # Get rollout data
        if rollout_idx not in rollout_data:
            continue
            
        tokens = rollout_data[rollout_idx]['tokens']
        token = tokens[token_idx]
        
        # Extract context with activations
        context_before, context_after, context_activations = get_context_with_activations(
            rollout_idx, token_idx, proj_type, layer
        )
        
        # Build context HTML
        context_html = []
        
        # Add context before with activation coloring
        for j, ctx_token in enumerate(context_before[-10:]):  # Show last 10 tokens
            # Get activation for this context token
            ctx_position = token_idx - len(context_before[-10:]) + j
            ctx_activation = context_activations.get(ctx_position, 0)
            
            # Normalize activation for coloring
            intensity = min(abs(ctx_activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
            if ctx_activation > 0:
                bg_color = f"rgba(255, 0, 0, {intensity})"
            else:
                bg_color = f"rgba(0, 0, 255, {intensity})"
            
            token_display = html_lib.escape(ctx_token).replace('\n', '\\n')
            context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
        
        # Add the target token with red outline
        intensity = min(abs(activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
        if activation > 0:
            bg_color = f"rgba(255, 0, 0, {intensity})"
        else:
            bg_color = f"rgba(0, 0, 255, {intensity})"
        
        token_display = html_lib.escape(token).replace('\n', '\\n')
        context_html.append(f'<span style="border: 2px solid red; background-color: {bg_color}; padding: 2px; font-weight: bold;">{token_display}</span>')
        
        # Add context after with activation coloring
        for j, ctx_token in enumerate(context_after[:10]):  # Show first 10 tokens
            ctx_position = token_idx + j + 1
            ctx_activation = context_activations.get(ctx_position, 0)
            
            intensity = min(abs(ctx_activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
            if ctx_activation > 0:
                bg_color = f"rgba(255, 0, 0, {intensity})"
            else:
                bg_color = f"rgba(0, 0, 255, {intensity})"
            
            token_display = html_lib.escape(ctx_token).replace('\n', '\\n')
            context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
        
        # Combine into a single line
        html_parts.append(f'<div class="example">')
        html_parts.append(f'<small style="color: #666;">Rollout {rollout_idx}, Act: {activation:.3f}</small><br>')
        html_parts.append(''.join(context_html))
        html_parts.append('</div>')
    
    return '\n'.join(html_parts)

# %%
# Create HTML output
html_output = """
<!DOCTYPE html>
<html>
<head>
    <title>LoRA Probe Activations - Side by Side (Optimized)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; text-align: center; }
        h2 { color: #666; margin-top: 40px; margin-bottom: 20px; }
        h3 { color: #888; margin-top: 20px; }
        .layer-section { margin-bottom: 50px; }
        .projections-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr 1fr; 
            gap: 20px;
            margin-top: 20px;
        }
        .projection-column { 
            border: 1px solid #ddd; 
            padding: 15px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .projection-column h3 { 
            margin-top: 0; 
            text-align: center;
            background-color: #e8e8e8;
            padding: 10px;
            margin: -15px -15px 15px -15px;
            border-radius: 5px 5px 0 0;
        }
        .example { margin: 10px 0; font-family: monospace; line-height: 1.8; }
        hr { margin: 30px 0; border: 1px solid #ddd; }
        .activation-type { 
            font-weight: bold; 
            color: #555; 
            margin-top: 15px; 
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
<h1>LoRA Probe Activations - Gate, Up, and Down Projections</h1>
<p style="text-align: center; color: #666; margin-bottom: 30px;">
    All projections show: input activations ⋅ A matrix (rank-1 LoRA neuron activations)
</p>
"""

print("Generating HTML...")

# Process each layer
for layer_idx in tqdm(range(n_layers), desc="Generating HTML"):
    html_output += f'<div class="layer-section">'
    html_output += f'<h2>Layer {layer_idx}</h2>'
    html_output += '<div class="projections-grid">'
    
    # Process all three projection types for this layer
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        # Get top examples from optimized tracker
        top_positive = top_k_trackers[proj_type][layer_idx].get_top_positive()
        top_negative = top_k_trackers[proj_type][layer_idx].get_top_negative()
        
        # Create column for this projection type
        html_output += f'<div class="projection-column">'
        html_output += f'<h3>{proj_type.upper()}</h3>'
        
        # Add positive examples
        html_output += f'<div class="activation-type">Top {top_k} Positive Activations</div>'
        html_output += create_html_examples_optimized(top_positive, proj_type, layer_idx, "").replace("<h3></h3>", "")
        
        # Add negative examples
        html_output += f'<div class="activation-type">Top {top_k} Negative Activations</div>'
        html_output += create_html_examples_optimized(top_negative, proj_type, layer_idx, "").replace("<h3></h3>", "")
        
        html_output += '</div>'  # Close projection-column
    
    html_output += '</div>'  # Close projections-grid
    html_output += '</div>'  # Close layer-section
    
    if layer_idx < n_layers - 1:
        html_output += '<hr>'

html_output += """
</body>
</html>
"""

# Save to file
filename = "lora_activations_optimized_v2.html"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(html_output)
print(f"Saved optimized activations to {filename}")

# Also display in notebook
display(HTML(html_output))

# %%
# Clean up memory - only keep essential data
print("\nCleaning up memory...")
# Remove activation arrays from rollout_data to save memory
for rollout_idx in rollout_data:
    if 'activations' in rollout_data[rollout_idx]:
        # Only keep activations for rollouts that have top-k examples
        rollouts_with_topk = set()
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            for layer_idx in range(n_layers):
                for _, rid, _ in top_k_trackers[proj_type][layer_idx].get_top_positive():
                    rollouts_with_topk.add(rid)
                for _, rid, _ in top_k_trackers[proj_type][layer_idx].get_top_negative():
                    rollouts_with_topk.add(rid)
        
        # Remove activations for rollouts not in top-k
        if rollout_idx not in rollouts_with_topk:
            del rollout_data[rollout_idx]['activations']

gc.collect()

print("\nDone! Optimization complete.")
print(f"Memory usage: {len(rollout_data)} rollouts stored")
print(f"Rollouts with full activations: {len([r for r in rollout_data.values() if 'activations' in r])}")
print(f"Total top-k entries: {n_layers * 3 * top_k * 2} (vs {num_examples * 10000 * n_layers * 3} total activations processed)")

# %%
