# Generate and visualize LoRA activations across every token
# in a generated context. This script produces heatmaps of
# activations across all layers and token positions.

# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import textwrap
from IPython.display import HTML, display

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/s1_peft/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dir = "/workspace/s1_peft/ckpts_lora/s1-lora-32B-r1-20250627_013544"
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
# Extract A matrices for gate_proj and up_proj from each layer
print("Extracting A matrices for gate_proj and up_proj...")

probe_directions = {
    'gate_proj': {},
    'up_proj': {}
}

# Get the number of layers
n_layers = model.config.num_hidden_layers

for layer_idx in range(n_layers):
    for proj_type in ['gate_proj', 'up_proj']:
        # Access the module directly
        module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
        
        # Extract the LoRA A matrix (which is a vector for rank-1)
        if hasattr(module, 'lora_A'):
            # Get the A matrix from the LoRA adapter
            lora_A_weight = module.lora_A['default'].weight.data
            # For rank-1, this could be shape [hidden_size, 1] or [1, hidden_size]
            # We want a 1D vector of shape [hidden_size]
            probe_direction = lora_A_weight.squeeze()
            probe_directions[proj_type][layer_idx] = probe_direction
            print(f"  Layer {layer_idx} {proj_type}: shape {probe_direction.shape}")

print(f"Extracted directions for {len(probe_directions['gate_proj'])} layers")

# %%
# Load a sample problem from MATH-500 dataset
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
problem_idx = 0
problem = dataset[problem_idx]['problem']
solution = dataset[problem_idx]['solution']

print("Sample problem:")
print("-" * 80)
print(textwrap.fill(problem, width=80))
print("-" * 80)

# %%
# Format the prompt
messages = [
    {"role": "system", "content": "You are a helpful mathematics assistant."},
    {"role": "user", "content": problem}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs.input_ids

print(f"Prompt length: {input_ids.shape[1]} tokens")

# %%
# Generate rollout and capture residual stream activations
print("Generating rollout with adapted model...")

# First, generate the full response
with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
    )

# Get the full sequence
full_sequence = generated.sequences[0]
response_start = len(input_ids[0])
response_text = tokenizer.decode(full_sequence[response_start:], skip_special_tokens=True)

print("Generated response:")
print("-" * 80)
# Split by newlines, wrap each line, then rejoin
lines = response_text[:2000].split('\n')
wrapped_lines = [textwrap.fill(line, width=80) if line else '' for line in lines]
print('\n'.join(wrapped_lines))
if len(response_text) > 2000:
    print("...")
print("-" * 80)

# %%
# Now run a forward pass to capture residual stream activations
print("Capturing residual stream activations...")

# Use the full generated sequence for analysis
analysis_input_ids = full_sequence.unsqueeze(0)

# Storage for residual stream activations
residual_streams = {}

# Hook function to capture post-layernorm (pre-MLP) residual stream
def make_hook(layer_idx):
    def hook(module, input, output):
        # The output of post_attention_layernorm is the normalized residual
        # that gets fed into the MLP
        residual_streams[layer_idx] = output.detach()  # [seq_len, hidden_size]
    return hook

# Register hooks on each layer's post_attention_layernorm
hooks = []
for layer_idx in range(n_layers):
    layernorm = model.model.model.layers[layer_idx].post_attention_layernorm
    hook = layernorm.register_forward_hook(make_hook(layer_idx))
    hooks.append(hook)

# Run forward pass
with torch.no_grad():
    outputs = model(analysis_input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

print(f"Captured post-layernorm (pre-MLP) residual streams for {len(residual_streams)} layers")
print(f"Post-layernorm residual stream shape: {residual_streams[0].shape}")

# %%
# Compute probe activations
print("Computing probe activations...")

probe_activations = {
    'gate_proj': {},
    'up_proj': {}
}

# For each probe type and layer, compute activations
for proj_type in ['gate_proj', 'up_proj']:
    for layer_idx in range(n_layers):
        # Get the probe direction and residual stream
        probe_dir = probe_directions[proj_type][layer_idx]
        residual = residual_streams[layer_idx][0]  # [seq_len, hidden_size]
        
        # Compute dot product between residual stream and probe direction
        # This gives us the activation for each token position
        activations = torch.matmul(residual.float(), probe_dir)  # [seq_len]
        
        probe_activations[proj_type][layer_idx] = activations.cpu().numpy()

print("Computed probe activations for all layers and projection types")

# %%
# Create dataframe with probe activations
print("Creating activation dataframe...")

# Get tokens for display
# Use convert_ids_to_tokens to get raw tokens (for understanding boundaries)
raw_tokens = tokenizer.convert_ids_to_tokens(analysis_input_ids[0])
# Also decode each token individually for cleaner display
tokens = []
for token_id in analysis_input_ids[0]:
    decoded = tokenizer.decode([token_id])
    tokens.append(decoded)

# Create a list to store all data
data_rows = []

for token_idx, token in enumerate(tokens):
    for layer_idx in range(n_layers):
        for proj_type in ['gate_proj', 'up_proj']:
            activation = probe_activations[proj_type][layer_idx][token_idx]
            data_rows.append({
                'token_idx': token_idx,
                'token': token,
                'layer': layer_idx,
                'proj_type': proj_type,
                'activation': activation
            })

df = pd.DataFrame(data_rows)
print(f"Created dataframe with {len(df)} rows")
print(df.head())

# %%
# Analyze activation statistics
print("\nActivation statistics:")
for proj_type in ['gate_proj', 'up_proj']:
    proj_df = df[df['proj_type'] == proj_type]
    print(f"\n{proj_type}:")
    print(f"  Mean activation: {proj_df['activation'].mean():.4f}")
    print(f"  Std activation: {proj_df['activation'].std():.4f}")
    print(f"  Min activation: {proj_df['activation'].min():.4f}")
    print(f"  Max activation: {proj_df['activation'].max():.4f}")

# %%
# Create HTML visualization
print("Creating HTML visualization...")

def create_html_visualization(tokens, probe_activations, proj_type, layer_subset=None, center_activations=False):
    """Create an HTML visualization of probe activations.
    
    Args:
        tokens: List of tokens
        probe_activations: Dict of probe activations by layer
        proj_type: Type of projection ('gate_proj' or 'up_proj')
        layer_subset: Optional subset of layers to visualize
        center_activations: Whether to subtract mean activation per layer (default: True)
    """
    
    # If layer_subset is provided, only show those layers
    if layer_subset is None:
        layer_subset = list(range(n_layers))
    
    html_parts = [f"<h3>Probe Activations: {proj_type}</h3>"]
    
    for layer_idx in layer_subset:
        activations = probe_activations[proj_type][layer_idx].copy()
        
        # Optionally center activations by subtracting mean
        if center_activations:
            mean_act = activations.mean()
            activations_for_color = activations - mean_act
        else:
            activations_for_color = activations
        
        # Normalize activations for coloring (preserving zero)
        # We want to scale by absolute magnitude, keeping 0 at 0
        max_abs = np.abs(activations_for_color).max()
        if max_abs > 0:
            norm_activations = np.abs(activations_for_color) / max_abs
        else:
            norm_activations = np.zeros_like(activations_for_color)
        
        html_parts.append(f"<div style='margin: 10px 0;'>")
        html_parts.append(f"<strong>Layer {layer_idx}</strong> ")
        html_parts.append(f"<span style='color: #666;'>(min: {activations.min():.3f}, max: {activations.max():.3f}, mean: {activations.mean():.3f})</span><br>")
        
        # Create colored tokens
        for token_idx, (token, activation, norm_act) in enumerate(zip(tokens, activations, norm_activations)):
            # Use centered activation for color determination
            centered_val = activations_for_color[token_idx]
            
            # Color intensity based on activation
            if centered_val > 0:
                # Positive activations: shades of red
                color = f"rgba(255, 0, 0, {norm_act:.2f})"
            else:
                # Negative activations: shades of blue
                color = f"rgba(0, 0, 255, {norm_act:.2f})"
            
            # Clean token for display
            display_token = token.replace('\n', '\\n')
            
            html_parts.append(
                f"<span style='background-color: {color}; padding: 2px 4px; margin: 1px; "
                f"border-radius: 3px; display: inline-block; font-family: monospace;' "
                f"title='Token {token_idx}: {activation:.3f}'>{display_token}</span>"
            )
            
            # Add line breaks for newlines
            newline_count = token.count('\n')
            for _ in range(newline_count):
                html_parts.append("<br>")
        
        html_parts.append("</div>")
    
    html_parts.append("<div style='margin-top: 20px; color: #666;'>")
    html_parts.append("<strong>Color legend:</strong> ")
    html_parts.append("<span style='color: red;'>Red = positive activation</span>, ")
    html_parts.append("<span style='color: blue;'>Blue = negative activation</span>. ")
    html_parts.append("Hover over tokens to see exact values.")
    html_parts.append("</div>")
    
    return "".join(html_parts)

# %%
# Display visualizations for a subset of layers
# Show first, middle, and last few layers
interesting_layers = [30, 31, 32]
interesting_layers = [l for l in interesting_layers if l < n_layers]

# %%
# Up proj visualization
html_up = create_html_visualization(tokens, probe_activations, 'up_proj', interesting_layers)
display(HTML(html_up))

# %%
# Gate proj visualization
html_gate = create_html_visualization(tokens, probe_activations, 'gate_proj', interesting_layers)
display(HTML(html_gate))

# %%
# Plot activation heatmap across all layers and tokens
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Find double newline positions (step boundaries)
double_newline_positions = []
for idx, token in enumerate(tokens):
    if token.count('\n') >= 2:
        double_newline_positions.append(idx)

# Gate proj heatmap
gate_matrix = np.array([probe_activations['gate_proj'][i] for i in range(n_layers)])
im1 = ax1.imshow(gate_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax1.set_title('Gate Projection Probe Activations')
ax1.set_xlabel('Token Position')
ax1.set_ylabel('Layer')

# Add vertical lines at double newline positions
for pos in double_newline_positions:
    ax1.axvline(x=pos, color='green', alpha=0.6, linestyle='-', linewidth=1.5)

# Add double newline markers on x-axis
if double_newline_positions:
    ax1.set_xticks(double_newline_positions, minor=True)
    ax1.tick_params(axis='x', which='minor', length=8, color='green', width=2)

plt.colorbar(im1, ax=ax1)

# Up proj heatmap
up_matrix = np.array([probe_activations['up_proj'][i] for i in range(n_layers)])
im2 = ax2.imshow(up_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax2.set_title('Up Projection Probe Activations')
ax2.set_xlabel('Token Position')
ax2.set_ylabel('Layer')

# Add vertical lines at double newline positions
for pos in double_newline_positions:
    ax2.axvline(x=pos, color='green', alpha=0.6, linestyle='-', linewidth=1.5)

# Add double newline markers on x-axis
if double_newline_positions:
    ax2.set_xticks(double_newline_positions, minor=True)
    ax2.tick_params(axis='x', which='minor', length=8, color='green', width=2)

plt.colorbar(im2, ax=ax2)

plt.tight_layout()

# Add legend for double newlines
fig.text(0.5, 0.01, 'Green lines indicate double newline tokens (step boundaries)', 
         ha='center', color='green', fontsize=10)

plt.show()

# %%