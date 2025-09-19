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
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dir = "/workspace/reasoning_interp/lora_checkpoints/s1-lora-32B-r1-20250627_013544"
print(f"Using LoRA from: {lora_dir}")

# Ablation strategy: if True, ablate most important layers first; if False, ablate least important first
ABLATE_MOST_IMPORTANT_FIRST = False
print(f"\nAblation strategy: {'Most' if ABLATE_MOST_IMPORTANT_FIRST else 'Least'} important layers first")

# KL metric to use: "mean_kl" or "max_kl"
# Note: When using multiple prompts (n_prompts > 1), max_kl automatically uses average of per-prompt max KLs
KL_METRIC = "max_kl"  # Change this to "mean_kl" or "max_kl"
print(f"Using KL metric: {KL_METRIC}")

# Whether to ablate attention adapters as baseline
ABLATE_ATTENTION = False  # Set to False to keep attention adapters active
print(f"Ablate attention adapters: {ABLATE_ATTENTION}")

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
n_prompts = 8
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
    per_prompt_max_kls = []
    
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
        
        # Track max KL for this prompt
        per_prompt_max_kls.append(kl_div.max().item())
    
    # Return both mean and max across all prompts and positions
    all_kls_concat = torch.cat(all_kls)
    all_ces_concat = torch.cat(all_ces)
    
    # When using multiple prompts, use average of per-prompt max KLs for max_kl metric
    if len(rollouts) > 1:
        max_kl_value = sum(per_prompt_max_kls) / len(per_prompt_max_kls)
    else:
        max_kl_value = all_kls_concat.max().item()
    
    return {
        'mean_kl': all_kls_concat.mean().item(),
        'max_kl': max_kl_value,
        'mean_ce': all_ces_concat.mean().item(),
        'max_ce': all_ces_concat.max().item()
    }

# %%
# Step 1: Optionally ablate all attention matrices as baseline
if ABLATE_ATTENTION:
    print("Step 1: Ablating all attention matrices...")
    attention_matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    attention_ablated = []
    
    for matrix_type in attention_matrices:
        modules = ablate_all_matrices_of_type(model, matrix_type)
        attention_ablated.extend(modules)
    
    print(f"Ablated {len(attention_ablated)} attention modules")
else:
    print("Step 1: Skipping attention ablation (keeping attention adapters active)")
    attention_ablated = []

# Compute baseline metrics
baseline_metrics = compute_metrics(model, all_rollouts)
if ABLATE_ATTENTION:
    print(f"Baseline (attention ablated) - Mean KL: {baseline_metrics['mean_kl']:.4f}, Max KL: {baseline_metrics['max_kl']:.4f}")
else:
    print(f"Baseline (no ablation) - Mean KL: {baseline_metrics['mean_kl']:.4f}, Max KL: {baseline_metrics['max_kl']:.4f}")
baseline_kl = baseline_metrics[KL_METRIC]
if KL_METRIC == "max_kl" and n_prompts > 1:
    print(f"Baseline {KL_METRIC} (avg of per-prompt max): {baseline_kl:.4f}")
else:
    print(f"Baseline {KL_METRIC}: {baseline_kl:.4f}")

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
# Step 2: Measure importance of each MLP matrix separately
print("\nStep 2: Measuring importance of each MLP matrix separately...")
mlp_matrices = ['gate_proj', 'up_proj', 'down_proj']
n_layers = 64

matrix_importance = []

for layer_idx in tqdm(range(n_layers), desc="Testing MLP layers"):
    for matrix_type in mlp_matrices:
        # Ablate only this specific matrix in this layer
        modules = ablate_matrix_type_in_layer(model, layer_idx, matrix_type)
        
        if modules:  # Only if we found this matrix type in this layer
            # Compute metrics with this matrix ablated
            metrics = compute_metrics(model, all_rollouts)
            
            # Calculate increase in KL from baseline
            kl_increase = metrics[KL_METRIC] - baseline_metrics[KL_METRIC]
            
            matrix_importance.append({
                'layer': layer_idx,
                'matrix_type': matrix_type,
                'kl_increase': kl_increase,
                'max_kl': metrics['max_kl'],
                'mean_kl': metrics['mean_kl']
            })
            
            # Restore this matrix
            restore_modules(modules)

# Create dataframe and sort by importance
df_importance = pd.DataFrame(matrix_importance)
df_importance = df_importance.sort_values('kl_increase', ascending=not ABLATE_MOST_IMPORTANT_FIRST)

print(f"\n{'Most' if ABLATE_MOST_IMPORTANT_FIRST else 'Least'} important MLP matrices (by {KL_METRIC} increase):")
print(df_importance.head(15))  # Show more since we have 3x as many entries

# %%
# Plot individual layer ablation impacts
fig, ax = plt.subplots(figsize=(14, 6))

# Create arrays for plotting
layer_numbers = list(range(n_layers))
impacts = []

# Map matrix importance data to positions
matrix_colors = {'gate_proj': 'steelblue', 'up_proj': 'forestgreen', 'down_proj': 'crimson'}
bar_width = 0.25

# Create separate lists for each matrix type
gate_impacts = [0] * n_layers
up_impacts = [0] * n_layers
down_impacts = [0] * n_layers

for _, row in df_importance.iterrows():
    layer = row['layer']
    impact = row['kl_increase']
    if row['matrix_type'] == 'gate_proj':
        gate_impacts[layer] = impact
    elif row['matrix_type'] == 'up_proj':
        up_impacts[layer] = impact
    elif row['matrix_type'] == 'down_proj':
        down_impacts[layer] = impact

# Create grouped bar plot
x = np.arange(n_layers)
bars1 = ax.bar(x - bar_width, gate_impacts, bar_width, label='gate_proj', 
                color=matrix_colors['gate_proj'], alpha=0.7, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, up_impacts, bar_width, label='up_proj',
                color=matrix_colors['up_proj'], alpha=0.7, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + bar_width, down_impacts, bar_width, label='down_proj',
                color=matrix_colors['down_proj'], alpha=0.7, edgecolor='black', linewidth=0.5)

# Highlight high-impact matrices with their values
for layer in range(n_layers):
    for offset, impacts, matrix_type in [(-bar_width, gate_impacts, 'gate_proj'), 
                                         (0, up_impacts, 'up_proj'), 
                                         (bar_width, down_impacts, 'down_proj')]:
        if impacts[layer] > 0.005:  # Only label significant impacts
            ax.text(layer + offset, impacts[layer] + 0.0005, f'{impacts[layer]:.3f}', 
                    ha='center', va='bottom', fontsize=6, rotation=90)

ax.set_xlabel('Layer Number', fontsize=12)
ax.set_ylabel(f'{KL_METRIC.replace("_", " ").title()} Increase When Ablated', fontsize=12)
attention_note = " (baseline: attention ablated)" if ABLATE_ATTENTION else " (baseline: no ablation)"
ax.set_title(f'Individual MLP Matrix Ablation Impact by Type{attention_note}', fontsize=14)
ax.legend(loc='upper right')
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
# Calculate statistics per matrix type
stats_text = []
for matrix_type in mlp_matrices:
    df_matrix = df_importance[df_importance['matrix_type'] == matrix_type]
    mean_impact = df_matrix['kl_increase'].mean()
    max_impact = df_matrix['kl_increase'].max()
    if len(df_matrix) > 0:
        max_layer = df_matrix.loc[df_matrix['kl_increase'].idxmax(), 'layer']
        stats_text.append(f'{matrix_type}: mean={mean_impact:.4f}, max={max_impact:.4f} (L{max_layer})')
textstr = '\n'.join(stats_text)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %%
# Step 3: Iteratively ablate individual matrices based on importance
print(f"\nStep 3: Iteratively ablating {'most' if ABLATE_MOST_IMPORTANT_FIRST else 'least'} important MLP matrices...")

kl_threshold = 0

if kl_threshold == 0:
    print("No KL threshold set - will test ablating all layers")
else:
    print(f"KL threshold: {kl_threshold}")

# Track ablation progress
ablation_history = []
permanently_ablated = []  # Keep track of ablated modules
ablated_matrices = set()  # Keep track of which (layer, matrix_type) pairs are ablated

current_kl = baseline_metrics[KL_METRIC]
iteration = 0

# Get all possible matrix positions
all_matrix_positions = [(layer, matrix_type) for layer in range(n_layers) for matrix_type in mlp_matrices]
remaining_positions = set(all_matrix_positions)

# Calculate total expected tests
total_matrices = len(all_matrix_positions)
if kl_threshold == 0:
    # If no threshold, worst case is we test all matrices: n + (n-1) + (n-2) + ... + 1 = n(n+1)/2
    max_expected_tests = total_matrices * (total_matrices + 1) // 2
else:
    # With threshold, estimate conservatively
    max_expected_tests = min(total_matrices * 20, total_matrices * (total_matrices + 1) // 2)

# Create overall progress bar
overall_pbar = tqdm(total=max_expected_tests, desc="Overall ablation progress")
tests_completed = 0

while (kl_threshold == 0 or current_kl < kl_threshold) and len(ablated_matrices) < len(all_matrix_positions):
    remaining_count = len(remaining_positions) - len(ablated_matrices)
    print(f"\nIteration {iteration + 1}: Testing {remaining_count} remaining matrices...")
    
    # Recompute importance for all remaining (non-ablated) matrices
    matrix_importance_current = []
    
    for layer_idx, matrix_type in sorted(remaining_positions - ablated_matrices):
        # Temporarily ablate this specific matrix
        modules = ablate_matrix_type_in_layer(model, layer_idx, matrix_type)
        
        if modules:  # Only if we found this matrix
            # Compute metrics with this matrix ablated
            metrics = compute_metrics(model, all_rollouts)
            
            # Calculate increase in KL from current state
            kl_increase = metrics[KL_METRIC] - current_kl
            
            matrix_importance_current.append({
                'layer': layer_idx,
                'matrix_type': matrix_type,
                'kl_increase': kl_increase,
                'max_kl': metrics['max_kl'],
                'mean_kl': metrics['mean_kl']
            })
            
            # Restore this matrix
            restore_modules(modules)
        
        tests_completed += 1
        overall_pbar.update(1)
    
    # Sort by importance based on ablation strategy
    matrix_importance_current.sort(key=lambda x: x['kl_increase'], reverse=ABLATE_MOST_IMPORTANT_FIRST)
    
    if not matrix_importance_current:
        print("No more matrices to test")
        break
    
    # Get the next matrix to ablate based on strategy
    next_matrix_info = matrix_importance_current[0]
    next_layer = next_matrix_info['layer']
    next_matrix_type = next_matrix_info['matrix_type']
    
    importance_desc = "Most" if ABLATE_MOST_IMPORTANT_FIRST else "Least"
    print(f"\n{importance_desc} important matrix: L{next_layer}.{next_matrix_type} (KL increase: {next_matrix_info['kl_increase']:.4f})")
    
    # Ablate this specific matrix permanently
    print(f"Testing permanent ablation of L{next_layer}.{next_matrix_type}")
    
    temp_ablated = ablate_matrix_type_in_layer(model, next_layer, next_matrix_type)
    
    # Get the actual KL with this matrix ablated
    test_kl = next_matrix_info[KL_METRIC]
    
    print(f"  {KL_METRIC} with L{next_layer}.{next_matrix_type} ablated: {test_kl:.4f} (ratio: {test_kl / baseline_metrics[KL_METRIC]:.2f}x)")
    
    # Check if we should keep this ablation
    if kl_threshold == 0 or test_kl < kl_threshold:
        # Keep the ablation
        print(f"  ✓ Keeping L{next_layer}.{next_matrix_type} ablated (KL below threshold)")
        ablated_matrices.add((next_layer, next_matrix_type))
        permanently_ablated.extend(temp_ablated)
        current_kl = test_kl
        
        # Calculate instantaneous KL increase (from previous state)
        instantaneous_kl_increase = test_kl - (ablation_history[-1][KL_METRIC] if ablation_history else baseline_metrics[KL_METRIC])
        
        ablation_history.append({
            'iteration': iteration + 1,
            'layer_ablated': next_layer,
            'matrix_ablated': next_matrix_type,
            'n_matrices_ablated': len(ablated_matrices),
            'mean_kl': next_matrix_info['mean_kl'],
            'max_kl': next_matrix_info['max_kl'],
            'kl_ratio': current_kl / baseline_metrics[KL_METRIC],
            'kl_increase': next_matrix_info['kl_increase'],
            'instantaneous_kl_increase': instantaneous_kl_increase,
            'kept_ablation': True
        })
    else:
        # Restore this matrix - KL exceeded threshold
        print(f"  ✗ Restoring L{next_layer}.{next_matrix_type} (KL exceeded threshold: {test_kl:.4f} > {kl_threshold:.4f})")
        restore_modules(temp_ablated)
        
        # Calculate instantaneous KL increase (from previous state)
        instantaneous_kl_increase = test_kl - (ablation_history[-1][KL_METRIC] if ablation_history else baseline_metrics[KL_METRIC])
        
        ablation_history.append({
            'iteration': iteration + 1,
            'layer_ablated': next_layer,
            'matrix_ablated': next_matrix_type,
            'n_matrices_ablated': len(ablated_matrices),
            'mean_kl': next_matrix_info['mean_kl'],
            'max_kl': next_matrix_info['max_kl'],
            'kl_ratio': test_kl / baseline_metrics[KL_METRIC],
            'kl_increase': next_matrix_info['kl_increase'],
            'instantaneous_kl_increase': instantaneous_kl_increase,
            'kept_ablation': False
        })
        
        # Stop here - we've found the limit
        print(f"\nStopping: Found the minimal set of {len(ablated_matrices)} essential MLP matrices")
        overall_pbar.close()
        break
    
    print(f"  Total matrices ablated: {len(ablated_matrices)}")
    iteration += 1

# Close progress bar when done
overall_pbar.close()

# %%
# Create results dataframe
df_history = pd.DataFrame(ablation_history)

print(f"\nAblation stopped at iteration {len(ablation_history)}")
print(f"Final {KL_METRIC}: {current_kl:.4f} ({current_kl / baseline_metrics[KL_METRIC]:.2f}x baseline)")
print(f"Total MLP matrices ablated: {len(ablated_matrices)} out of {len(all_matrix_positions)}")
print(f"Percentage of MLP matrices remaining: {(len(all_matrix_positions) - len(ablated_matrices)) / len(all_matrix_positions) * 100:.1f}%")

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
    ax1.plot(kept_data['iteration'], kept_data[KL_METRIC], 'bo-', label='Kept ablations', markersize=8)

# Plot restored ablations (if any)
if len(restored_data) > 0:
    ax1.scatter(restored_data['iteration'], restored_data[KL_METRIC], color='red', marker='x', 
                s=100, label='Restored (exceeded threshold)', zorder=5)

baseline_label = 'Baseline (attention ablated)' if ABLATE_ATTENTION else 'Baseline (no ablation)'
ax1.axhline(y=baseline_metrics[KL_METRIC], color='g', linestyle='--', label=baseline_label)
if kl_threshold > 0:
    ax1.axhline(y=kl_threshold, color='r', linestyle='--', label=f'Threshold')
ax1.set_xlabel('Iteration')
ax1.set_ylabel(f'{KL_METRIC.replace("_", " ").title()} Divergence')
title_suffix = " (with attention ablated)" if ABLATE_ATTENTION else " (attention active)"
ax1.set_title(f'Progressive MLP Matrix Ablation{title_suffix}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot number of layers ablated
ax2.plot(df_history['iteration'], df_history['n_matrices_ablated'], 'orange', marker='s')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Number of MLP Matrices Ablated')
ax2.set_title('Cumulative Matrices Ablated')
ax2.grid(True, alpha=0.3)

# Plot instantaneous KL increase for each layer
ax3.bar(df_history['iteration'], df_history['instantaneous_kl_increase'], 
        color=['green' if kept else 'red' for kept in df_history['kept_ablation']],
        alpha=0.7, edgecolor='black')

# Add matrix labels on top of bars
for idx, row in df_history.iterrows():
    label = f"L{row['layer_ablated']}.{row['matrix_ablated'][:1]}"
    ax3.text(row['iteration'], row['instantaneous_kl_increase'] + 0.001, 
             label, ha='center', va='bottom', fontsize=7, rotation=90)

ax3.set_xlabel('Iteration')
ax3.set_ylabel('Instantaneous KL Increase')
ax3.set_title('Impact of Each Matrix Ablation (Green=Kept, Red=Restored)')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

# %%
# Show which matrices were kept
remaining_matrices = sorted(set(all_matrix_positions) - ablated_matrices)
print(f"\nRemaining MLP matrices ({len(remaining_matrices)} total):")
# Group by layer for cleaner display
remaining_by_layer = {}
for layer, matrix in remaining_matrices:
    if layer not in remaining_by_layer:
        remaining_by_layer[layer] = []
    remaining_by_layer[layer].append(matrix)
for layer in sorted(remaining_by_layer.keys()):
    print(f"  Layer {layer}: {', '.join(remaining_by_layer[layer])}")

# %%
# Plot matrices sorted by their impact
fig, ax = plt.subplots(figsize=(12, 6))

# Get only the kept ablations and sort by instantaneous KL increase
kept_ablations = df_history[df_history['kept_ablation']].copy()
# Sort based on ablation strategy - if we ablated most important first, we want to show them in reverse order
kept_ablations_sorted = kept_ablations.sort_values('instantaneous_kl_increase', ascending=not ABLATE_MOST_IMPORTANT_FIRST)

# Color by matrix type
matrix_colors = {'gate_proj': 'steelblue', 'up_proj': 'forestgreen', 'down_proj': 'crimson'}
bar_colors = [matrix_colors[row['matrix_ablated']] for _, row in kept_ablations_sorted.iterrows()]

# Create bar plot
bars = ax.bar(range(len(kept_ablations_sorted)), 
               kept_ablations_sorted['instantaneous_kl_increase'],
               color=bar_colors, alpha=0.7, edgecolor='black')

# Add matrix labels on top of bars
for i, (idx, row) in enumerate(kept_ablations_sorted.iterrows()):
    label = f"L{row['layer_ablated']}.{row['matrix_ablated'][:1]}"
    ax.text(i, row['instantaneous_kl_increase'] + 0.0005, 
            label, ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xlabel('Matrices (sorted by impact)', fontsize=12)
ax.set_ylabel('Instantaneous KL Increase', fontsize=12)
sort_order = "Most to Least" if ABLATE_MOST_IMPORTANT_FIRST else "Least to Most"
ax.set_title(f'MLP Matrices Sorted by Impact When Ablated ({sort_order} Important)', fontsize=14)

# Add legend for matrix types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=matrix_colors[mt], label=mt) for mt in mlp_matrices]
ax.legend(handles=legend_elements, loc='upper right')
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
# Plot impact by layer number with matrix types
fig, ax = plt.subplots(figsize=(14, 6))

# Create mappings for each matrix type
matrix_colors = {'gate_proj': 'steelblue', 'up_proj': 'forestgreen', 'down_proj': 'crimson'}
bar_width = 0.25

# Initialize impact arrays for each matrix type
gate_impacts = [0] * n_layers
up_impacts = [0] * n_layers
down_impacts = [0] * n_layers

# Fill impacts from kept ablations
for _, row in df_history.iterrows():
    if row['kept_ablation']:
        layer = row['layer_ablated']
        impact = row['instantaneous_kl_increase']
        if row['matrix_ablated'] == 'gate_proj':
            gate_impacts[layer] = impact
        elif row['matrix_ablated'] == 'up_proj':
            up_impacts[layer] = impact
        elif row['matrix_ablated'] == 'down_proj':
            down_impacts[layer] = impact

# Create grouped bar plot
x = np.arange(n_layers)
bars1 = ax.bar(x - bar_width, gate_impacts, bar_width, label='gate_proj',
                color=matrix_colors['gate_proj'], alpha=0.7, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, up_impacts, bar_width, label='up_proj',
                color=matrix_colors['up_proj'], alpha=0.7, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + bar_width, down_impacts, bar_width, label='down_proj',
                color=matrix_colors['down_proj'], alpha=0.7, edgecolor='black', linewidth=0.5)

# Highlight non-zero bars with their values
for layer in range(n_layers):
    for offset, impacts, matrix_type in [(-bar_width, gate_impacts, 'gate_proj'),
                                         (0, up_impacts, 'up_proj'),
                                         (bar_width, down_impacts, 'down_proj')]:
        if impacts[layer] > 0.005:  # Only label significant impacts
            ax.text(layer + offset, impacts[layer] + 0.0005, f'{impacts[layer]:.3f}',
                    ha='center', va='bottom', fontsize=6, rotation=90)

ax.set_xlabel('Layer Number', fontsize=12)
ax.set_ylabel('Instantaneous KL Increase When Ablated', fontsize=12)
ax.set_title('Impact of Ablating Each MLP Matrix by Position in Model', fontsize=14)
ax.legend(loc='upper left')
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
n_ablated = len(ablated_matrices)
total_impact = sum(gate_impacts) + sum(up_impacts) + sum(down_impacts)
textstr = f'Matrices ablated: {n_ablated}/{len(all_matrix_positions)}\nTotal KL increase: {total_impact:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %%
# Visualize which matrices were ablated with ablation order
fig, ax = plt.subplots(figsize=(14, 6))

# Create mapping of (layer, matrix_type) to ablation order
matrix_to_order = {}
for idx, row in df_history.iterrows():
    if row['kept_ablation']:
        matrix_to_order[(row['layer_ablated'], row['matrix_ablated'])] = idx + 1

# Matrix type positions and colors
matrix_colors = {'gate_proj': 'steelblue', 'up_proj': 'forestgreen', 'down_proj': 'crimson'}
matrix_positions = {'gate_proj': 0, 'up_proj': 1, 'down_proj': 2}
bar_height = 0.8

# Create the visualization
for layer in range(n_layers):
    for matrix_type in mlp_matrices:
        y_pos = layer * 3 + matrix_positions[matrix_type]
        
        if (layer, matrix_type) in matrix_to_order:
            # Ablated matrix - use darker version of the color
            color = matrix_colors[matrix_type]
            order = matrix_to_order[(layer, matrix_type)]
            ax.barh(y_pos, 1, bar_height, color=color, alpha=0.8, edgecolor='black', linewidth=1)
            # Add order number
            ax.text(0.5, y_pos, str(order), ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold')
        else:
            # Active matrix - light gray
            ax.barh(y_pos, 1, bar_height, color='lightgray', alpha=0.3, edgecolor='black', linewidth=0.5)

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, n_layers * 3 - 0.5)
ax.set_xlabel('')
ax.set_xticks([])

# Set y-axis labels
y_labels = []
y_positions = []
for layer in range(n_layers):
    if layer % 4 == 0:  # Show every 4th layer
        y_labels.append(f'L{layer}')
        y_positions.append(layer * 3 + 1)

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels)
ax.set_ylabel('Layer', fontsize=12)

# Add matrix type labels
for i, (matrix_type, color) in enumerate(matrix_colors.items()):
    ax.text(1.02, i, matrix_type, transform=ax.transData, fontsize=10,
            color=color, fontweight='bold', va='center')

attention_note = " (attention ablated)" if ABLATE_ATTENTION else " (attention active)"
ax.set_title(f'MLP Matrix Ablation Order ({len(ablated_matrices)} ablated, {len(remaining_matrices)} remaining){attention_note}', fontsize=14)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgray', alpha=0.3, edgecolor='black', label='Active matrix'),
    Patch(facecolor='gray', edgecolor='black', label='Ablated matrix (number shows order)')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

# Add statistics for each matrix type
stats_y_start = n_layers * 3 + 2
for i, matrix_type in enumerate(mlp_matrices):
    count = sum(1 for (l, m) in ablated_matrices if m == matrix_type)
    total = n_layers
    ax.text(0.5, stats_y_start + i, f'{matrix_type}: {count}/{total} ablated',
            transform=ax.transData, fontsize=9, ha='center')

plt.tight_layout()
plt.show()

# %%
# Show distribution of ablated matrices across model depth
fig, ax = plt.subplots(figsize=(10, 4))

# Divide layers into quarters
quarters = ['Early (0-15)', 'Early-Mid (16-31)', 'Late-Mid (32-47)', 'Late (48-63)']
quarter_counts = {mt: [0, 0, 0, 0] for mt in mlp_matrices}

for layer, matrix_type in ablated_matrices:
    quarter_idx = layer // 16
    quarter_counts[matrix_type][quarter_idx] += 1

# Create grouped bar chart
bar_width = 0.25
x = np.arange(len(quarters))

for i, (matrix_type, color) in enumerate(matrix_colors.items()):
    counts = quarter_counts[matrix_type]
    bars = ax.bar(x + i * bar_width - bar_width, counts, bar_width,
                   label=matrix_type, color=color, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Model Section', fontsize=12)
ax.set_ylabel('Number of Ablated Matrices', fontsize=12)
ax.set_title('Distribution of Ablated MLP Matrices Across Model Depth', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.legend()
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
# Count remaining matrices by type
remaining_by_type = {'gate_proj': 0, 'up_proj': 0, 'down_proj': 0}
for _, matrix_type in remaining_matrices:
    remaining_by_type[matrix_type] += 1
attention_status = "no attention" if ABLATE_ATTENTION else "with attention"
print(f"OUTPUT WITH MINIMAL MLP MATRICES ({len(remaining_matrices)} matrices - G:{remaining_by_type['gate_proj']}, U:{remaining_by_type['up_proj']}, D:{remaining_by_type['down_proj']}, {attention_status}):")
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
#         'final_kl': current_kl,
#         'kl_metric': KL_METRIC,
#         'kl_threshold': kl_threshold
#     }
    
#     with open(os.path.join(min_layers_dir_path, 'ablation_metadata.json'), 'w') as f:
#         json.dump(metadata, f, indent=2)
    
#     print(f"Successfully saved minimal LoRA to: {min_layers_dir_path}")
#     print(f"Active MLP layers: {len(remaining_layers)} ({len(remaining_layers)/n_layers*100:.1f}%)")

# %%