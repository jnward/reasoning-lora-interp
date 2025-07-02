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
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json
import heapq

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
print(f"Gate proj directions: {len(probe_directions['gate_proj'])}")
print(f"Up proj directions: {len(probe_directions['up_proj'])}")
print(f"Down proj directions: {len(probe_directions['down_proj'])}")

# %%
# Load s1K-1.1 dataset
print("Loading s1K-1.1 dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"Dataset has {len(dataset)} examples")

# Check the structure of the dataset
print(f"Dataset features: {dataset.features}")
print(f"First example keys: {list(dataset[0].keys())}")
print(f"First example preview: {str(dataset[0])[:500]}...")

# %%
@dataclass
class ActivationExample:
    """Store activation with context"""
    rollout_idx: int
    token_idx: int
    token: str
    activation: float
    context_before: List[str]
    context_after: List[str]
    layer: int
    proj_type: str
    context_activations: Dict[int, float] = None  # Maps token_idx to activation for context tokens

# Storage for top activations only - using min heaps for efficiency
class TopKTracker:
    def __init__(self, k):
        self.k = k
        self.top_positive = []  # min heap of positive activations
        self.top_negative = []  # max heap (negated) of negative activations
        self.counter = 0  # Tie-breaker to avoid comparing ActivationExample objects
        
    def add(self, example: ActivationExample):
        act = example.activation
        self.counter += 1
        
        if act >= 0:
            if len(self.top_positive) < self.k:
                heapq.heappush(self.top_positive, (act, self.counter, example))
            elif act > self.top_positive[0][0]:
                heapq.heapreplace(self.top_positive, (act, self.counter, example))
        else:
            # Use negative to create max heap for negative values
            if len(self.top_negative) < self.k:
                heapq.heappush(self.top_negative, (-act, self.counter, example))
            elif -act > self.top_negative[0][0]:
                heapq.heapreplace(self.top_negative, (-act, self.counter, example))
    
    def get_top_positive(self):
        # Return sorted from highest to lowest
        return [ex for _, _, ex in sorted(self.top_positive, key=lambda x: x[0], reverse=True)]
    
    def get_top_negative(self):
        # Return sorted from lowest to highest
        return [ex for _, _, ex in sorted(self.top_negative, key=lambda x: -x[0])]

# Initialize trackers for each projection type and layer
top_k_trackers = {
    proj_type: {layer: TopKTracker(top_k) for layer in range(n_layers)}
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']
}

# Also keep track of all activations for statistics and visualization
all_activations_stats = {
    proj_type: {layer: [] for layer in range(n_layers)}
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']
}

# %%
# Process rollouts
num_examples = min(16, len(dataset))
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
    
    # Get tokens for context
    tokens = []
    for token_id in input_ids:
        decoded = tokenizer.decode([token_id])
        tokens.append(decoded)
    
    # Storage for projected activations only (to save memory)
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
    
    # Process activations and update top-k trackers
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        for layer_idx in range(n_layers):
            # Get pre-computed activations from hooks
            activations = projected_activations[proj_type][layer_idx]
            
            # Store statistics
            all_activations_stats[proj_type][layer_idx].extend(activations.tolist())
            
            # Update top-k tracker for each token
            for token_idx in range(len(tokens)):
                # Get context window
                context_start = max(0, token_idx - context_window)
                context_end = min(len(tokens), token_idx + context_window + 1)
                
                context_before = tokens[context_start:token_idx]
                context_after = tokens[token_idx+1:context_end]
                
                # Store context activations
                context_activations = {}
                for ctx_idx in range(context_start, context_end):
                    if ctx_idx != token_idx:  # Don't duplicate the main token
                        context_activations[ctx_idx] = activations[ctx_idx]
                
                example = ActivationExample(
                    rollout_idx=rollout_idx,
                    token_idx=token_idx,
                    token=tokens[token_idx],
                    activation=activations[token_idx],
                    context_before=context_before,
                    context_after=context_after,
                    layer=layer_idx,
                    proj_type=proj_type,
                    context_activations=context_activations
                )
                
                # Add to top-k tracker
                top_k_trackers[proj_type][layer_idx].add(example)
    
    # Clear GPU memory after each rollout to prevent OOM
    torch.cuda.empty_cache()

print("Finished processing all rollouts")

# %%
# Extract final top-k results
print("\nExtracting top activating examples...")

top_activations = {
    proj_type: {
        layer: {
            'positive': tracker.get_top_positive(),
            'negative': tracker.get_top_negative()
        }
        for layer, tracker in top_k_trackers[proj_type].items()
    }
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']
}

# %%
# Create HTML visualization
print("\nCreating side-by-side visualization...")

from IPython.display import HTML, display
import html as html_lib

def create_html_examples(examples: List[ActivationExample], title: str, 
                        min_act: float, max_act: float) -> str:
    """Create HTML visualization of activation examples with context"""
    
    html_parts = [f"<h3>{title}</h3>"]
    
    for i, ex in enumerate(examples):
        # Build context with colored tokens
        context_html = []
        
        # Add context before
        for j, token in enumerate(ex.context_before[-10:]):  # Show last 10 tokens
            # Get activation for this context token if available
            ctx_position = ex.token_idx - len(ex.context_before[-10:]) + j
            ctx_activation = ex.context_activations.get(ctx_position, 0) if ex.context_activations else 0
            
            # Normalize activation for coloring
            intensity = min(abs(ctx_activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
            if ctx_activation > 0:
                bg_color = f"rgba(255, 0, 0, {intensity})"
            else:
                bg_color = f"rgba(0, 0, 255, {intensity})"
            
            token_display = html_lib.escape(token).replace('\n', '\\n')
            context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
        
        # Add the target token with red outline
        intensity = min(abs(ex.activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
        if ex.activation > 0:
            bg_color = f"rgba(255, 0, 0, {intensity})"
        else:
            bg_color = f"rgba(0, 0, 255, {intensity})"
        
        token_display = html_lib.escape(ex.token).replace('\n', '\\n')
        context_html.append(f'<span style="border: 2px solid red; background-color: {bg_color}; padding: 2px; font-weight: bold;">{token_display}</span>')
        
        # Add context after
        for j, token in enumerate(ex.context_after[:10]):  # Show first 10 tokens
            ctx_position = ex.token_idx + j + 1
            ctx_activation = ex.context_activations.get(ctx_position, 0) if ex.context_activations else 0
            
            intensity = min(abs(ctx_activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
            if ctx_activation > 0:
                bg_color = f"rgba(255, 0, 0, {intensity})"
            else:
                bg_color = f"rgba(0, 0, 255, {intensity})"
            
            token_display = html_lib.escape(token).replace('\n', '\\n')
            context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
        
        # Combine into a single line
        html_parts.append(f'<div class="example">')
        html_parts.append(f'<small style="color: #666;">Rollout {ex.rollout_idx}</small><br>')
        html_parts.append(''.join(context_html))
        html_parts.append('</div>')
    
    return '\n'.join(html_parts)

# Pre-compute activation ranges for normalization
print("Computing activation ranges...")
activation_ranges = {}
for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
    for layer_idx in range(n_layers):
        all_acts = all_activations_stats[proj_type][layer_idx]
        if all_acts:
            activation_ranges[(proj_type, layer_idx)] = (min(all_acts), max(all_acts))
        else:
            activation_ranges[(proj_type, layer_idx)] = (0, 0)

# No need for separate activation lookups - we have context_activations in each example

# Create HTML output
html_output = """
<!DOCTYPE html>
<html>
<head>
    <title>LoRA Probe Activations - Side by Side</title>
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
        # Get pre-computed top examples
        top_positive = top_activations[proj_type][layer_idx]['positive']
        top_negative = top_activations[proj_type][layer_idx]['negative']
        
        # Get ranges
        min_act, max_act = activation_ranges[(proj_type, layer_idx)]
        
        # Create column for this projection type
        html_output += f'<div class="projection-column">'
        html_output += f'<h3>{proj_type.upper()}</h3>'
        
        # Add positive examples
        html_output += f'<div class="activation-type">Top {top_k} Positive Activations</div>'
        html_output += create_html_examples(top_positive, "", min_act, max_act).replace("<h3></h3>", "")
        
        # Add negative examples
        html_output += f'<div class="activation-type">Top {top_k} Negative Activations</div>'
        html_output += create_html_examples(top_negative, "", min_act, max_act).replace("<h3></h3>", "")
        
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
filename = "lora_activations_side_by_side.html"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(html_output)
print(f"Saved side-by-side activations to {filename}")

# Also display in notebook
display(HTML(html_output))

# %%
print("\nDone!")
