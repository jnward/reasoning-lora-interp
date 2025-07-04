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

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# Ablation strategy: if True, ablate most important layers first; if False, ablate least important first
ABLATE_MOST_IMPORTANT_FIRST = True
print(f"\nAblation strategy: {'Most' if ABLATE_MOST_IMPORTANT_FIRST else 'Least'} important layers first")

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
# Load multiple math problems
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
n_prompts = 1
problems = [dataset[i]['problem'] for i in range(n_prompts)]

print(f"Loaded {n_prompts} problems")

# %%
# Generate ground truth rollouts
print("Generating ground truth rollouts...")
all_rollouts = []

for i, problem in enumerate(tqdm(problems, desc="Generating rollouts")):
    messages = [
        {"role": "system", "content": "You are a helpful mathematics assistant."},
        {"role": "user", "content": problem}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,  # Shorter for faster iteration
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
    
    full_sequence = generated.sequences[0]
    rollout_input_ids = full_sequence[:-1].unsqueeze(0)
    rollout_labels = full_sequence[1:]
    
    with torch.no_grad():
        outputs = model(rollout_input_ids, return_dict=True)
        ground_truth_logits = outputs.logits[0]
    
    ground_truth_probs = F.softmax(ground_truth_logits, dim=-1)
    
    all_rollouts.append({
        'input_ids': rollout_input_ids,
        'labels': rollout_labels,
        'ground_truth_logits': ground_truth_logits,
        'ground_truth_probs': ground_truth_probs
    })

print(f"Generated {len(all_rollouts)} rollouts")

# %%
# Ablation functions
def ablate_matrix_type_in_layer(model, layer_idx, matrix_type):
    """Ablate specific matrix type in a specific layer."""
    ablated_modules = []
    
    for name, module in model.named_modules():
        if (f"layers.{layer_idx}." in name and 
            f".{matrix_type}" in name and 
            hasattr(module, 'lora_A')):
            
            original_A = {}
            original_B = {}
            
            for key in module.lora_A.keys():
                original_A[key] = module.lora_A[key].weight.data.clone()
                original_B[key] = module.lora_B[key].weight.data.clone()
                
                module.lora_A[key].weight.data.zero_()
                module.lora_B[key].weight.data.zero_()
            
            ablated_modules.append((module, original_A, original_B))
    
    return ablated_modules

def ablate_all_matrices_of_type(model, matrix_type):
    """Ablate all matrices of a specific type across all layers."""
    ablated_modules = []
    
    for name, module in model.named_modules():
        if f".{matrix_type}" in name and hasattr(module, 'lora_A'):
            original_A = {}
            original_B = {}
            
            for key in module.lora_A.keys():
                original_A[key] = module.lora_A[key].weight.data.clone()
                original_B[key] = module.lora_B[key].weight.data.clone()
                
                module.lora_A[key].weight.data.zero_()
                module.lora_B[key].weight.data.zero_()
            
            ablated_modules.append((module, original_A, original_B))
    
    return ablated_modules

def restore_modules(ablated_modules):
    """Restore original LoRA weights."""
    for module, original_A, original_B in ablated_modules:
        for key in module.lora_A.keys():
            module.lora_A[key].weight.data = original_A[key]
            module.lora_B[key].weight.data = original_B[key]

def compute_metrics(model, rollouts):
    """Compute KL divergence and CE loss for current model state."""
    all_kls = []
    all_ces = []
    
    for rollout in rollouts:
        with torch.no_grad():
            outputs = model(rollout['input_ids'], return_dict=True)
            logits = outputs.logits[0]
        
        probs = F.softmax(logits, dim=-1)
        
        kl_div = F.kl_div(
            probs.log(),
            rollout['ground_truth_probs'],
            reduction='none',
            log_target=False
        ).sum(dim=-1)
        
        ce_loss = F.cross_entropy(
            logits,
            rollout['labels'],
            reduction='none'
        )
        
        all_kls.append(kl_div)
        all_ces.append(ce_loss)
    
    # Return both mean and max across all prompts and positions
    all_kls_concat = torch.cat(all_kls)
    all_ces_concat = torch.cat(all_ces)
    
    return {
        'mean_kl': all_kls_concat.mean().item(),
        'max_kl': all_kls_concat.max().item(),
        'mean_ce': all_ces_concat.mean().item(),
        'max_ce': all_ces_concat.max().item()
    }

# %%
# Step 1: Ablate all attention matrices as baseline
print("Step 1: Ablating all attention matrices...")
attention_matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
attention_ablated = []

for matrix_type in attention_matrices:
    modules = ablate_all_matrices_of_type(model, matrix_type)
    attention_ablated.extend(modules)

print(f"Ablated {len(attention_ablated)} attention modules")

# Compute baseline metrics with attention ablated
baseline_metrics = compute_metrics(model, all_rollouts)
print(f"Baseline (attention ablated) - Mean KL: {baseline_metrics['mean_kl']:.4f}, Max KL: {baseline_metrics['max_kl']:.4f}")

# %%
# Save attention-ablated version
# print("\nSaving LoRA with attention matrices ablated...")

# # Create new directory name
# import os
# original_dir_name = os.path.basename(lora_dir)
# attn_ablated_dir_name = original_dir_name + "_attn_ablated"
# attn_ablated_dir_path = os.path.join(os.path.dirname(lora_dir), attn_ablated_dir_name)

# print(f"Saving to: {attn_ablated_dir_path}")

# # Save the model with attention ablated
# model.save_pretrained(attn_ablated_dir_path)

# Also copy the tokenizer files if they exist
# import shutil
# tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
# for file in tokenizer_files:
#     src_path = os.path.join(lora_dir, file)
#     if os.path.exists(src_path):
#         shutil.copy2(src_path, os.path.join(attn_ablated_dir_path, file))

# print(f"Successfully saved attention-ablated LoRA to: {attn_ablated_dir_path}")

# %%
# Step 2: Measure importance of each MLP layer
print("\nStep 2: Measuring importance of each MLP layer...")
mlp_matrices = ['gate_proj', 'up_proj', 'down_proj']
n_layers = 64

layer_importance = []

for layer_idx in tqdm(range(n_layers), desc="Testing MLP layers"):
    # Ablate this layer's MLP
    layer_ablated = []
    for matrix_type in mlp_matrices:
        modules = ablate_matrix_type_in_layer(model, layer_idx, matrix_type)
        layer_ablated.extend(modules)
    
    if layer_ablated:  # Only if we found MLP modules in this layer
        # Compute metrics with this layer ablated
        metrics = compute_metrics(model, all_rollouts)
        
        # Calculate increase in KL from baseline (using mean KL)
        kl_increase = metrics['mean_kl'] - baseline_metrics['mean_kl']
        
        layer_importance.append({
            'layer': layer_idx,
            'kl_increase': kl_increase,
            'max_kl': metrics['max_kl'],
            'mean_kl': metrics['mean_kl']
        })
        
        # Restore this layer
        restore_modules(layer_ablated)

# Create dataframe and sort by importance
df_importance = pd.DataFrame(layer_importance)
df_importance = df_importance.sort_values('kl_increase', ascending=not ABLATE_MOST_IMPORTANT_FIRST)

print(f"\n{'Most' if ABLATE_MOST_IMPORTANT_FIRST else 'Least'} important MLP layers (by mean KL increase):")
print(df_importance.head(10))

# %%
# Step 3: Iteratively ablate layers based on importance
print(f"\nStep 3: Iteratively ablating {'most' if ABLATE_MOST_IMPORTANT_FIRST else 'least'} important MLP layers...")

# kl_threshold = 0.1
kl_threshold = 1.0

# Track ablation progress
ablation_history = []
permanently_ablated = []  # Keep track of ablated modules
ablated_layers = set()  # Keep track of which layers are ablated

current_mean_kl = baseline_metrics['mean_kl']
iteration = 0

# Get all non-ablated layers
remaining_layer_indices = set(range(n_layers))

while current_mean_kl < kl_threshold and len(remaining_layer_indices) > len(ablated_layers):
    print(f"\nIteration {iteration + 1}: Recomputing importance for {len(remaining_layer_indices) - len(ablated_layers)} remaining layers...")
    
    # Recompute importance for all remaining (non-ablated) layers
    layer_importance_current = []
    
    for layer_idx in tqdm(sorted(remaining_layer_indices - ablated_layers), desc="Testing remaining MLP layers"):
        # Temporarily ablate this layer's MLP
        layer_ablated = []
        for matrix_type in mlp_matrices:
            modules = ablate_matrix_type_in_layer(model, layer_idx, matrix_type)
            layer_ablated.extend(modules)
        
        if layer_ablated:  # Only if we found MLP modules in this layer
            # Compute metrics with this layer ablated
            metrics = compute_metrics(model, all_rollouts)
            
            # Calculate increase in KL from current state
            kl_increase = metrics['mean_kl'] - current_mean_kl
            
            layer_importance_current.append({
                'layer': layer_idx,
                'kl_increase': kl_increase,
                'max_kl': metrics['max_kl'],
                'mean_kl': metrics['mean_kl']
            })
            
            # Restore this layer
            restore_modules(layer_ablated)
    
    # Sort by importance based on ablation strategy
    layer_importance_current.sort(key=lambda x: x['kl_increase'], reverse=ABLATE_MOST_IMPORTANT_FIRST)
    
    if not layer_importance_current:
        print("No more layers to test")
        break
    
    # Get the next layer to ablate based on strategy
    next_layer_info = layer_importance_current[0]
    next_layer = next_layer_info['layer']
    
    importance_desc = "Most" if ABLATE_MOST_IMPORTANT_FIRST else "Least"
    print(f"\n{importance_desc} important layer: {next_layer} (KL increase: {next_layer_info['kl_increase']:.4f})")
    
    # Ablate this layer's MLP permanently
    print(f"Testing permanent ablation of layer {next_layer}")
    
    temp_ablated = []
    for matrix_type in mlp_matrices:
        modules = ablate_matrix_type_in_layer(model, next_layer, matrix_type)
        temp_ablated.extend(modules)
    
    # Get the actual KL with this layer ablated
    test_mean_kl = next_layer_info['mean_kl']
    
    print(f"  Mean KL with layer {next_layer} ablated: {test_mean_kl:.4f} (ratio: {test_mean_kl / baseline_metrics['mean_kl']:.2f}x)")
    
    # Check if we should keep this ablation
    if test_mean_kl < kl_threshold:
        # Keep the ablation
        print(f"  ✓ Keeping layer {next_layer} ablated (KL below threshold)")
        ablated_layers.add(next_layer)
        permanently_ablated.extend(temp_ablated)
        current_mean_kl = test_mean_kl
        
        # Calculate instantaneous KL increase (from previous state)
        instantaneous_kl_increase = test_mean_kl - (ablation_history[-1]['mean_kl'] if ablation_history else baseline_metrics['mean_kl'])
        
        ablation_history.append({
            'iteration': iteration + 1,
            'layer_ablated': next_layer,
            'n_layers_ablated': len(ablated_layers),
            'max_kl': next_layer_info['max_kl'],
            'mean_kl': current_mean_kl,
            'kl_ratio': current_mean_kl / baseline_metrics['mean_kl'],
            'kl_increase': next_layer_info['kl_increase'],
            'instantaneous_kl_increase': instantaneous_kl_increase,
            'kept_ablation': True
        })
    else:
        # Restore this layer - KL exceeded threshold
        print(f"  ✗ Restoring layer {next_layer} (KL exceeded threshold: {test_mean_kl:.4f} > {kl_threshold:.4f})")
        restore_modules(temp_ablated)
        
        # Calculate instantaneous KL increase (from previous state)
        instantaneous_kl_increase = test_mean_kl - (ablation_history[-1]['mean_kl'] if ablation_history else baseline_metrics['mean_kl'])
        
        ablation_history.append({
            'iteration': iteration + 1,
            'layer_ablated': next_layer,
            'n_layers_ablated': len(ablated_layers),
            'max_kl': next_layer_info['max_kl'],
            'mean_kl': test_mean_kl,
            'kl_ratio': test_mean_kl / baseline_metrics['mean_kl'],
            'kl_increase': next_layer_info['kl_increase'],
            'instantaneous_kl_increase': instantaneous_kl_increase,
            'kept_ablation': False
        })
        
        # Stop here - we've found the limit
        print(f"\nStopping: Found the minimal set of {len(ablated_layers)} essential MLP layers")
        break
    
    print(f"  Total layers ablated: {len(ablated_layers)}")
    iteration += 1

# %%
# Create results dataframe
df_history = pd.DataFrame(ablation_history)

print(f"\nAblation stopped at iteration {len(ablation_history)}")
print(f"Final mean KL: {current_mean_kl:.4f} ({current_mean_kl / baseline_metrics['mean_kl']:.2f}x baseline)")
print(f"Total MLP layers ablated: {len(ablated_layers)} out of {n_layers}")
print(f"Percentage of MLP layers remaining: {(n_layers - len(ablated_layers)) / n_layers * 100:.1f}%")

# %%
# Plot ablation progress
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot KL divergence progression
# Separate kept and restored ablations
kept_mask = df_history['kept_ablation']
kept_data = df_history[kept_mask]
restored_data = df_history[~kept_mask]

# Plot kept ablations
if len(kept_data) > 0:
    ax1.plot(kept_data['iteration'], kept_data['mean_kl'], 'bo-', label='Kept ablations', markersize=8)

# Plot restored ablations (if any)
if len(restored_data) > 0:
    ax1.scatter(restored_data['iteration'], restored_data['mean_kl'], color='red', marker='x', 
                s=100, label='Restored (exceeded threshold)', zorder=5)

ax1.axhline(y=baseline_metrics['mean_kl'], color='g', linestyle='--', label='Baseline (attention only ablated)')
ax1.axhline(y=kl_threshold, color='r', linestyle='--', label=f'Threshold')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Mean KL Divergence')
ax1.set_title('Progressive MLP Layer Ablation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot number of layers ablated
ax2.plot(df_history['iteration'], df_history['n_layers_ablated'], 'orange', marker='s')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Number of MLP Layers Ablated')
ax2.set_title('Cumulative Layers Ablated')
ax2.grid(True, alpha=0.3)

# Plot instantaneous KL increase for each layer
ax3.bar(df_history['iteration'], df_history['instantaneous_kl_increase'], 
        color=['green' if kept else 'red' for kept in df_history['kept_ablation']],
        alpha=0.7, edgecolor='black')

# Add layer numbers on top of bars
for idx, row in df_history.iterrows():
    ax3.text(row['iteration'], row['instantaneous_kl_increase'] + 0.001, 
             str(row['layer_ablated']), ha='center', va='bottom', fontsize=8)

ax3.set_xlabel('Iteration')
ax3.set_ylabel('Instantaneous KL Increase')
ax3.set_title('Impact of Each Layer Ablation (Green=Kept, Red=Restored)')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

# %%
# Show which layers were kept
remaining_layers = sorted(set(range(n_layers)) - ablated_layers)
print(f"\nRemaining MLP layers ({len(remaining_layers)} total):")
print(remaining_layers)

# %%
# Plot layers sorted by their impact
fig, ax = plt.subplots(figsize=(12, 6))

# Get only the kept ablations and sort by instantaneous KL increase
kept_ablations = df_history[df_history['kept_ablation']].copy()
# Sort based on ablation strategy - if we ablated most important first, we want to show them in reverse order
kept_ablations_sorted = kept_ablations.sort_values('instantaneous_kl_increase', ascending=not ABLATE_MOST_IMPORTANT_FIRST)

# Create bar plot
bars = ax.bar(range(len(kept_ablations_sorted)), 
               kept_ablations_sorted['instantaneous_kl_increase'],
               color='steelblue', alpha=0.7, edgecolor='black')

# Add layer numbers on top of bars
for i, (idx, row) in enumerate(kept_ablations_sorted.iterrows()):
    ax.text(i, row['instantaneous_kl_increase'] + 0.0005, 
            f"L{row['layer_ablated']}", ha='center', va='bottom', fontsize=8, rotation=90)

ax.set_xlabel('Layers (sorted by impact)', fontsize=12)
ax.set_ylabel('Instantaneous KL Increase', fontsize=12)
sort_order = "Most to Least" if ABLATE_MOST_IMPORTANT_FIRST else "Least to Most"
ax.set_title(f'MLP Layers Sorted by Impact When Ablated ({sort_order} Important)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Add a text box with statistics
avg_impact = kept_ablations_sorted['instantaneous_kl_increase'].mean()
median_impact = kept_ablations_sorted['instantaneous_kl_increase'].median()
textstr = f'Avg impact: {avg_impact:.4f}\nMedian impact: {median_impact:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %%
# Plot impact by layer number
fig, ax = plt.subplots(figsize=(14, 6))

# Create a mapping of layer number to impact
layer_to_impact = {}
for _, row in df_history.iterrows():
    if row['kept_ablation']:
        layer_to_impact[row['layer_ablated']] = row['instantaneous_kl_increase']

# Create arrays for plotting
layer_numbers = list(range(n_layers))
impacts = []
colors_impact = []

for layer in layer_numbers:
    if layer in layer_to_impact:
        impacts.append(layer_to_impact[layer])
        colors_impact.append('darkred')
    else:
        impacts.append(0)  # Not ablated
        colors_impact.append('lightgray')

# Create bar plot
bars = ax.bar(layer_numbers, impacts, color=colors_impact, alpha=0.7, edgecolor='black', linewidth=0.5)

# Highlight non-zero bars with their values
for layer, impact in layer_to_impact.items():
    if impact > 0.005:  # Only label significant impacts
        ax.text(layer, impact + 0.0005, f'{impact:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xlabel('Layer Number', fontsize=12)
ax.set_ylabel('Instantaneous KL Increase When Ablated', fontsize=12)
ax.set_title('Impact of Ablating Each MLP Layer by Position in Model', fontsize=14)
ax.set_xlim(-0.5, n_layers - 0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add horizontal lines to divide model sections
for i in range(1, 4):
    ax.axvline(x=i*16 - 0.5, color='gray', linestyle='--', alpha=0.5)

# Add section labels
section_y = ax.get_ylim()[1] * 0.95
ax.text(8, section_y, 'Early', ha='center', fontsize=10, style='italic')
ax.text(24, section_y, 'Early-Mid', ha='center', fontsize=10, style='italic')
ax.text(40, section_y, 'Late-Mid', ha='center', fontsize=10, style='italic')
ax.text(56, section_y, 'Late', ha='center', fontsize=10, style='italic')

# Add statistics box
n_ablated = len(layer_to_impact)
total_impact = sum(layer_to_impact.values())
textstr = f'Layers ablated: {n_ablated}/{n_layers}\nTotal KL increase: {total_impact:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %%
# Visualize which layers were ablated with ablation order
fig, ax = plt.subplots(figsize=(14, 4))

# Create mapping of layer to ablation order
layer_to_order = {}
for idx, row in df_history.iterrows():
    if row['kept_ablation']:
        layer_to_order[row['layer_ablated']] = idx + 1

# Create colors based on ablation order
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Use a colormap for ablated layers - red to yellow
cmap = cm.get_cmap('YlOrRd_r')  # Reversed so red is first, yellow is last
norm = mcolors.Normalize(vmin=1, vmax=len(layer_to_order))

colors = []
for layer in range(n_layers):
    if layer in layer_to_order:
        # Color based on ablation order
        order = layer_to_order[layer]
        colors.append(cmap(norm(order)))
    else:
        # Active layer - light green
        colors.append('lightgreen')

# Plot bars
bars = ax.bar(range(n_layers), np.ones(n_layers), color=colors, edgecolor='black', linewidth=0.5)

# Add ablation order numbers for ablated layers
for layer, order in layer_to_order.items():
    ax.text(layer, 0.5, str(order), ha='center', va='center', fontsize=7, 
            color='white', fontweight='bold')

ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('')
ax.set_title(f'MLP Layer Ablation Order ({len(ablated_layers)} ablated, {len(remaining_layers)} remaining)', fontsize=14)
ax.set_ylim(0, 1.3)
ax.set_xlim(-0.5, n_layers - 0.5)
ax.set_yticks([])

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
cbar.set_label('Ablation Order (1 = first ablated)', fontsize=10)

# Add legend for active layers
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgreen', edgecolor='black', label='Active MLP layer')
]
ax.legend(handles=legend_elements, loc='upper right')

# Add text summary
ax.text(n_layers/2, 1.15, f'Total ablated: {len(ablated_layers)} ({len(ablated_layers)/n_layers*100:.1f}%)', 
        ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Show distribution of ablated layers across model depth
fig, ax = plt.subplots(figsize=(10, 4))

# Divide layers into quarters
quarters = ['Early (0-15)', 'Early-Mid (16-31)', 'Late-Mid (32-47)', 'Late (48-63)']
quarter_counts = [0, 0, 0, 0]

for layer in ablated_layers:
    quarter_idx = layer // 16
    quarter_counts[quarter_idx] += 1

# Plot bar chart
bars = ax.bar(quarters, quarter_counts, color=['#ff9999', '#ffcc99', '#99ccff', '#cc99ff'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{int(height)}', ha='center', va='bottom')

ax.set_ylabel('Number of Ablated Layers', fontsize=12)
ax.set_title('Distribution of Ablated MLP Layers Across Model Depth', fontsize=14)
ax.set_ylim(0, max(quarter_counts) + 2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %%
# Generate sample output with minimal MLP layers
print("\nGenerating sample with minimal MLP layers...")

problem = problems[0]
messages = [
    {"role": "system", "content": "You are a helpful mathematics assistant."},
    {"role": "user", "content": problem}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    minimal_outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

minimal_text = tokenizer.decode(minimal_outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print("\n" + "="*80)
print("PROBLEM:")
print(problem[:300] + "..." if len(problem) > 300 else problem)
print("\n" + "="*80)
print(f"OUTPUT WITH MINIMAL MLP LAYERS ({len(remaining_layers)} layers, no attention):")
for line in minimal_text.split('\n')[:15]:
    print(textwrap.fill(line, width=80) if line else '')

# %%
# Save minimal layers version
# if len(ablated_layers) > 0:
#     print(f"\nSaving minimal LoRA with {len(remaining_layers)} MLP layers...")
    
#     # The model already has attention ablated and minimal MLP layers
#     # Create new directory name
#     min_layers_dir_name = original_dir_name + "_min_layers"
#     min_layers_dir_path = os.path.join(os.path.dirname(lora_dir), min_layers_dir_name)
    
#     print(f"Saving to: {min_layers_dir_path}")
    
#     # Save the model with minimal layers
#     model.save_pretrained(min_layers_dir_path)
    
#     # Copy tokenizer files if they exist
#     for file in tokenizer_files:
#         src_path = os.path.join(lora_dir, file)
#         if os.path.exists(src_path):
#             shutil.copy2(src_path, os.path.join(min_layers_dir_path, file))
    
#     # Save metadata about which layers are active
#     import json
#     metadata = {
#         'original_model': original_dir_name,
#         'total_layers': n_layers,
#         'ablated_layers': sorted(list(ablated_layers)),
#         'remaining_layers': remaining_layers,
#         'n_remaining': len(remaining_layers),
#         'percent_remaining': len(remaining_layers) / n_layers * 100,
#         'final_mean_kl': current_mean_kl,
#         'kl_threshold': kl_threshold
#     }
    
#     with open(os.path.join(min_layers_dir_path, 'ablation_metadata.json'), 'w') as f:
#         json.dump(metadata, f, indent=2)
    
#     print(f"Successfully saved minimal LoRA to: {min_layers_dir_path}")
#     print(f"Active MLP layers: {len(remaining_layers)} ({len(remaining_layers)/n_layers*100:.1f}%)")

# %%