# %%
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# %%
# Load the LoRA adapter
lora_path = Path("/workspace/models/ckpts_1.1/s1-lora-32B-r1-20250627_013544/adapter_model.safetensors")

# Dictionary to store all weights
lora_weights = {}

# Load safetensors file
with safe_open(lora_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        lora_weights[key] = f.get_tensor(key)

print(f"Loaded {len(lora_weights)} weight tensors")
print("Sample keys:", list(lora_weights.keys())[:10])

# %%
# First, let's examine the key structure
print("\nExamining MLP key structure:")
mlp_keys = [k for k in lora_weights.keys() if 'mlp' in k and ('lora_A' in k or 'lora_B' in k)]
for i, key in enumerate(mlp_keys[:5]):
    print(f"  {key}")
    print(f"  Split: {key.split('.')}")

# %%
# Extract MLP adapter weights and compute norms
mlp_norms = {
    'up_proj': {'lora_A': [], 'lora_B': []},
    'down_proj': {'lora_A': [], 'lora_B': []},
    'gate_proj': {'lora_A': [], 'lora_B': []}
}

layer_indices = []

# Parse weights and compute norms
for key, tensor in lora_weights.items():
    if 'mlp' in key and ('lora_A' in key or 'lora_B' in key):
        # Extract layer index - format is like "base_model.model.layers.0.mlp.up_proj.lora_A.weight"
        parts = key.split('.')
        try:
            # Find the numeric part after 'layers'
            layers_idx = parts.index('layers')
            layer_idx = int(parts[layers_idx + 1])
        except (ValueError, IndexError):
            print(f"Could not parse layer index from key: {key}")
            continue
        
        # Extract projection type and lora type
        if 'up_proj' in key:
            proj_type = 'up_proj'
        elif 'down_proj' in key:
            proj_type = 'down_proj'
        elif 'gate_proj' in key:
            proj_type = 'gate_proj'
        else:
            continue
            
        lora_type = 'lora_A' if 'lora_A' in key else 'lora_B'
        
        # Compute Frobenius norm
        norm = torch.norm(tensor).item()
        
        # Store the norm with its layer index
        if layer_idx not in layer_indices:
            layer_indices.append(layer_idx)
            
        mlp_norms[proj_type][lora_type].append((layer_idx, norm))

# Sort by layer index
layer_indices.sort()
for proj_type in mlp_norms:
    for lora_type in ['lora_A', 'lora_B']:
        mlp_norms[proj_type][lora_type].sort(key=lambda x: x[0])

# %%
# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('LoRA MLP Adapter Norms by Layer', fontsize=16)

projections = ['up_proj', 'gate_proj', 'down_proj']
colors = {'lora_A': 'blue', 'lora_B': 'red'}

for idx, (ax, proj_type) in enumerate(zip(axes, projections)):
    # Extract layer indices and norms
    layers_A = [x[0] for x in mlp_norms[proj_type]['lora_A']]
    norms_A = [x[1] for x in mlp_norms[proj_type]['lora_A']]
    
    layers_B = [x[0] for x in mlp_norms[proj_type]['lora_B']]
    norms_B = [x[1] for x in mlp_norms[proj_type]['lora_B']]
    
    # Plot both lora_A and lora_B
    ax.plot(layers_A, norms_A, 'o-', color=colors['lora_A'], label='lora_A', markersize=6, linewidth=2)
    ax.plot(layers_B, norms_B, 's-', color=colors['lora_B'], label='lora_B', markersize=6, linewidth=2)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Frobenius Norm', fontsize=12)
    ax.set_title(f'{proj_type} LoRA Adapter Norms', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.1)

plt.tight_layout()
plt.show()

# %%
# Print statistics
print("\nStatistics Summary:")
print("=" * 60)

for proj_type in projections:
    print(f"\n{proj_type}:")
    for lora_type in ['lora_A', 'lora_B']:
        norms = [x[1] for x in mlp_norms[proj_type][lora_type]]
        if norms:
            print(f"  {lora_type}:")
            print(f"    Mean: {np.mean(norms):.6f}")
            print(f"    Std:  {np.std(norms):.6f}")
            print(f"    Min:  {np.min(norms):.6f}")
            print(f"    Max:  {np.max(norms):.6f}")

# %%
# Create a combined plot showing all projections on one plot
plt.figure(figsize=(14, 8))

line_styles = {'lora_A': '-', 'lora_B': '--'}
markers = {'up_proj': 'o', 'gate_proj': 's', 'down_proj': '^'}
colors_proj = {'up_proj': 'blue', 'gate_proj': 'green', 'down_proj': 'red'}

for proj_type in projections:
    for lora_type in ['lora_A', 'lora_B']:
        layers = [x[0] for x in mlp_norms[proj_type][lora_type]]
        norms = [x[1] for x in mlp_norms[proj_type][lora_type]]
        
        label = f'{proj_type} ({lora_type})'
        plt.plot(layers, norms, 
                line_styles[lora_type], 
                color=colors_proj[proj_type],
                marker=markers[proj_type],
                label=label,
                markersize=6,
                linewidth=2,
                alpha=0.8)

plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Frobenius Norm', fontsize=12)
plt.title('All MLP LoRA Adapter Norms by Layer', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Check which layers have the highest norms
print("\nTop 5 layers by norm for each projection and LoRA type:")
print("=" * 60)

for proj_type in projections:
    print(f"\n{proj_type}:")
    for lora_type in ['lora_A', 'lora_B']:
        data = mlp_norms[proj_type][lora_type]
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:5]
        print(f"  {lora_type} top layers:")
        for layer_idx, norm in sorted_data:
            print(f"    Layer {layer_idx}: {norm:.6f}")

# %%