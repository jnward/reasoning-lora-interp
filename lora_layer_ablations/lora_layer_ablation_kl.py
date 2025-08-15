# %%
"""
LoRA Layer Ablation Study with KL Divergence

This notebook performs a leave-one-out ablation study on LoRA layers,
measuring the KL divergence between the ablated model and the full LoRA model.
It also includes a baseline comparison between the base model (no LoRA) and full LoRA.
"""

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
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Experiment configuration
RANDOM_SEED = 42  # Set seed for reproducible sampling
N_EXAMPLES = 16   # Number of examples to sample from dataset
MAX_SEQ_LENGTH = 2048  # Maximum sequence length (for memory constraints)

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*544")
lora_dir = sorted(lora_dirs)[-1] if lora_dirs else None
print(f"Using LoRA from: {lora_dir}")

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

# Get model info
n_layers = model.config.num_hidden_layers
print(f"Model has {n_layers} layers")

# %%
# Load s1K-1.1 dataset (has pre-formatted text, but needs tokenization)
print("\nLoading s1K-1.1 dataset...")
# Try the tokenized version first, fall back to regular if needed
try:
    dataset = load_dataset("simplescaling/s1K-1.1_tokenized", split="train")
    print("Using s1K-1.1_tokenized dataset")
except:
    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    print("Using s1K-1.1 dataset")
    
print(f"Dataset has {len(dataset)} examples")
print(f"Dataset columns: {dataset.column_names}")

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Sample indices from the full dataset
all_indices = np.arange(len(dataset))
example_indices = np.random.choice(all_indices, size=N_EXAMPLES, replace=False)
example_indices = sorted(example_indices.tolist())

print(f"\nRandomly sampled {N_EXAMPLES} examples from {len(dataset)} total")
print(f"Example indices: {example_indices[:10]}..." if len(example_indices) > 10 else f"Example indices: {example_indices}")

# %%
# Prepare tokenized sequences from dataset
all_sequences = []
skipped_count = 0

for i, idx in enumerate(example_indices):
    example = dataset[idx]
    
    # Debug: show structure of first example
    if i == 0:
        print(f"\nFirst example keys: {example.keys()}")
    
    # Check if we have the 'text' field (pre-formatted) or need to build it
    if 'text' in example:
        # Use pre-formatted text
        full_text = example['text']
    else:
        # Build from components
        question = example.get('question', '')
        thinking = example.get('deepseek_thinking_trajectory', '')
        answer = example.get('deepseek_attempt', '')
        
        system_prompt = "You are a helpful mathematics assistant."
        full_text = (
            f"<|im_start|>system\n{system_prompt}\n"
            f"<|im_start|>user\n{question}\n"
            f"<|im_start|>assistant\n"
            f"<|im_start|>think\n{thinking}\n"
            f"<|im_start|>answer\n{answer}<|im_end|>"
        )
    
    # Tokenize the text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = inputs.input_ids[0]  # Remove batch dimension
    
    # Only use sequences with reasonable length (at least 100 tokens)
    if len(input_ids) > 100:
        all_sequences.append(input_ids.to(model.device))
        if len(all_sequences) <= 3:  # Show info for first few examples
            print(f"Example {idx}: {len(input_ids)} tokens")
    else:
        skipped_count += 1
        if skipped_count <= 3:
            print(f"Skipped example {idx}: too short ({len(input_ids)} tokens)")

print(f"\n✓ Prepared {len(all_sequences)} sequences for evaluation")
if skipped_count > 0:
    print(f"  (Skipped {skipped_count} sequences that were too short)")

# Check if we have enough sequences
if len(all_sequences) == 0:
    raise ValueError("No valid sequences found! Check the dataset format and tokenization.")
elif len(all_sequences) < 5:
    print(f"⚠ Warning: Only {len(all_sequences)} sequences available, results may be noisy")

# %%
# Compute ground truth distributions for all sequences
print("\nComputing ground truth distributions (full LoRA) for all sequences...")

all_ground_truth = []
all_input_ids = []
all_labels = []

for seq_idx, full_sequence in enumerate(all_sequences):
    # Prepare for next-token prediction
    rollout_input_ids = full_sequence[:-1].unsqueeze(0)
    rollout_labels = full_sequence[1:]
    
    # Forward pass with full LoRA
    with torch.no_grad():
        outputs_full_lora = model(rollout_input_ids, return_dict=True)
        logits_full_lora = outputs_full_lora.logits[0]  # [seq_len, vocab_size]
    
    # Compute probabilities
    probs_full_lora = F.softmax(logits_full_lora, dim=-1)
    
    all_ground_truth.append({
        'logits': logits_full_lora,
        'probs': probs_full_lora
    })
    all_input_ids.append(rollout_input_ids)
    all_labels.append(rollout_labels)
    
    print(f"Sequence {seq_idx}: {rollout_input_ids.shape[1]} tokens")

# %%
# Skip baseline computation - focus on layer-wise ablation only
print("\nSkipping baseline (no LoRA) computation - focusing on layer-wise effects only")

# %%
# Helper functions for layer ablation
def ablate_layer(model, layer_idx):
    """
    Zero out all LoRA weights for a specific layer.
    This includes all 7 adapter types: gate_, up_, down_, q_, k_, v_, o_proj
    """
    ablated_modules = []
    
    for name, module in model.named_modules():
        if f"layers.{layer_idx}." in name and hasattr(module, 'lora_A'):
            # Store original weights
            original_A = {}
            original_B = {}
            
            for key in module.lora_A.keys():
                original_A[key] = module.lora_A[key].weight.data.clone()
                original_B[key] = module.lora_B[key].weight.data.clone()
                
                # Zero out the weights
                module.lora_A[key].weight.data.zero_()
                module.lora_B[key].weight.data.zero_()
            
            ablated_modules.append((name, module, original_A, original_B))
    
    return ablated_modules

def restore_layer(ablated_modules):
    """Restore original LoRA weights."""
    for name, module, original_A, original_B in ablated_modules:
        for key in module.lora_A.keys():
            module.lora_A[key].weight.data = original_A[key]
            module.lora_B[key].weight.data = original_B[key]

# %%
# Perform layer-wise ablation study
print(f"\nRunning layer-wise ablation study across {n_layers} layers...")
print(f"Averaging over {len(all_sequences)} sequences...\n")

results = []

for layer_idx in tqdm(range(n_layers), desc="Ablating layers"):
    # Ablate the layer
    ablated_modules = ablate_layer(model, layer_idx)
    n_ablated = len(ablated_modules)
    
    # Collect metrics for all sequences
    layer_kls = []
    layer_ces = []
    
    for seq_idx in range(len(all_sequences)):
        # Forward pass with ablated layer
        with torch.no_grad():
            outputs_ablated = model(all_input_ids[seq_idx], return_dict=True)
            logits_ablated = outputs_ablated.logits[0]  # [seq_len, vocab_size]
        
        # Compute probabilities
        probs_ablated = F.softmax(logits_ablated, dim=-1)
        
        # Compute KL divergence: KL(ablated || full_lora)
        kl_div = F.kl_div(
            probs_ablated.log(),
            all_ground_truth[seq_idx]['probs'],
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # Sum over vocab dimension
        
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(
            logits_ablated,
            all_labels[seq_idx],
            reduction='none'
        )
        
        layer_kls.append(kl_div.mean().item())
        layer_ces.append(ce_loss.mean().item())
    
    # Average across sequences (with safety checks)
    if layer_kls:
        mean_kl = np.mean(layer_kls)
        std_kl = np.std(layer_kls)
        max_kl = np.max(layer_kls)
    else:
        mean_kl = std_kl = max_kl = 0.0
        print(f"Warning: No KL values computed for layer {layer_idx}")
    
    if layer_ces:
        mean_ce = np.mean(layer_ces)
        max_ce = np.max(layer_ces)
    else:
        mean_ce = max_ce = 0.0
    
    # Store results
    results.append({
        'layer': layer_idx,
        'mean_kl': mean_kl,
        'max_kl': max_kl,
        'std_kl': std_kl,
        'mean_ce': mean_ce,
        'max_ce': max_ce,
        'n_modules_ablated': n_ablated
    })
    
    # Restore the layer
    restore_layer(ablated_modules)
    
    # Verify restoration worked correctly (only check every 10 layers for speed)
    if layer_idx % 10 == 0:
        with torch.no_grad():
            outputs_test = model(all_input_ids[0], return_dict=True)
            logits_test = outputs_test.logits[0]
        assert torch.allclose(logits_test, all_ground_truth[0]['logits'], atol=1e-5), f"Layer {layer_idx} restoration failed"

print("\n✓ Ablation study completed!")

# %%
# Create dataframe with results
df = pd.DataFrame(results)

print("\nAblation Results Summary:")
print(df.describe())

# Save layer-wise ablation results to JSON
layer_ablation_file = f'layer_ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
layer_ablation_data = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'model': base_model_id,
        'lora_path': lora_dir,
        'dataset': 'simplescaling/s1K-1.1_tokenized',
        'n_sequences': len(all_sequences),
        'n_total_dataset': len(dataset),
        'example_indices': example_indices,
        'random_seed': RANDOM_SEED,
        'max_seq_length': MAX_SEQ_LENGTH,
        'n_layers': n_layers
    },
    'results': df.to_dict('records')
}
with open(layer_ablation_file, 'w') as f:
    json.dump(layer_ablation_data, f, indent=2)
print(f"✓ Layer ablation results saved to {layer_ablation_file}")

# %%
# Visualization: Create comprehensive plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Mean KL divergence by layer
ax = axes[0, 0]
ax.plot(df['layer'], df['mean_kl'], marker='o', markersize=4, linewidth=2)
ax.set_xlabel('Layer')
ax.set_ylabel('Mean KL Divergence')
ax.set_title('Mean KL Divergence by Layer (Leave-One-Out)')
ax.grid(True, alpha=0.3)

# 2. Max KL divergence by layer
ax = axes[0, 1]
ax.plot(df['layer'], df['max_kl'], marker='o', markersize=4, color='orange', linewidth=2)
ax.set_xlabel('Layer')
ax.set_ylabel('Max KL Divergence')
ax.set_title('Max KL Divergence by Layer')
ax.grid(True, alpha=0.3)

# 3. Bar chart of mean KL by layer
ax = axes[0, 2]
colors = plt.cm.viridis(df['mean_kl'] / df['mean_kl'].max())
ax.bar(df['layer'], df['mean_kl'], color=colors, alpha=0.7)
ax.set_xlabel('Layer')
ax.set_ylabel('Mean KL Divergence')
ax.set_title('Layer Importance (Mean KL)')
ax.grid(True, alpha=0.3, axis='y')

# 4. Mean CE loss by layer
ax = axes[1, 0]
ax.plot(df['layer'], df['mean_ce'], marker='o', markersize=4, color='purple')
ax.set_xlabel('Layer')
ax.set_ylabel('Mean Cross-Entropy Loss')
ax.set_title('Mean CE Loss by Layer')
ax.grid(True, alpha=0.3)

# 5. Standard deviation of KL
ax = axes[1, 1]
ax.plot(df['layer'], df['std_kl'], marker='o', markersize=4, color='brown')
ax.set_xlabel('Layer')
ax.set_ylabel('Std Dev of KL Divergence')
ax.set_title('KL Divergence Variability by Layer')
ax.grid(True, alpha=0.3)

# 6. Heatmap of layer importance
ax = axes[1, 2]
# Create a 2D representation for visualization (reshape into 8x8 for 64 layers)
importance_matrix = df['mean_kl'].values.reshape(8, 8)
im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
ax.set_title('Layer Importance Heatmap (8x8 grid)')
ax.set_xlabel('Layer index (within group)')
ax.set_ylabel('Layer group')
plt.colorbar(im, ax=ax, label='Mean KL')

plt.suptitle(f'LoRA Layer Ablation Study - Averaged over {len(all_sequences)} sequences', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Batch ablation: Ablate layers in groups of 8
print("\n" + "="*60)
print("BATCH ABLATION STUDY (Groups of 8 Layers)")
print("="*60)

batch_size = 8
n_batches = n_layers // batch_size
batch_results = []

print(f"\nAblating layers in {n_batches} batches of {batch_size} layers each...")

for batch_idx in tqdm(range(n_batches), desc="Ablating batches"):
    start_layer = batch_idx * batch_size
    end_layer = start_layer + batch_size
    
    # Ablate all layers in this batch
    all_ablated_modules = []
    for layer_idx in range(start_layer, end_layer):
        ablated_modules = ablate_layer(model, layer_idx)
        all_ablated_modules.extend(ablated_modules)
    
    # Collect metrics for all sequences
    batch_kls = []
    batch_ces = []
    
    for seq_idx in range(len(all_sequences)):
        # Forward pass with batch ablated
        with torch.no_grad():
            outputs_ablated = model(all_input_ids[seq_idx], return_dict=True)
            logits_ablated = outputs_ablated.logits[0]
        
        # Compute probabilities
        probs_ablated = F.softmax(logits_ablated, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            probs_ablated.log(),
            all_ground_truth[seq_idx]['probs'],
            reduction='none',
            log_target=False
        ).sum(dim=-1)
        
        # Compute CE loss
        ce_loss = F.cross_entropy(
            logits_ablated,
            all_labels[seq_idx],
            reduction='none'
        )
        
        batch_kls.append(kl_div.mean().item())
        batch_ces.append(ce_loss.mean().item())
    
    # Store results (averaged across sequences)
    batch_results.append({
        'batch': batch_idx,
        'layers': f"{start_layer}-{end_layer-1}",
        'mean_kl': np.mean(batch_kls),
        'max_kl': np.max(batch_kls),
        'std_kl': np.std(batch_kls),
        'mean_ce': np.mean(batch_ces),
        'n_layers_ablated': batch_size
    })
    
    # Restore all layers in the batch
    for modules in all_ablated_modules:
        if isinstance(modules, tuple):
            restore_layer([modules])
        else:
            restore_layer(modules)

# Create dataframe for batch results
df_batch = pd.DataFrame(batch_results)

print("\nBatch Ablation Results:")
print(df_batch)

# Save batch ablation results to JSON
batch_ablation_file = f'batch_ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
batch_ablation_data = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'model': base_model_id,
        'lora_path': lora_dir,
        'dataset': 'simplescaling/s1K-1.1_tokenized',
        'n_sequences': len(all_sequences),
        'n_total_dataset': len(dataset),
        'example_indices': example_indices,
        'random_seed': RANDOM_SEED,
        'max_seq_length': MAX_SEQ_LENGTH,
        'batch_size': batch_size,
        'n_batches': n_batches
    },
    'results': df_batch.to_dict('records')
}
with open(batch_ablation_file, 'w') as f:
    json.dump(batch_ablation_data, f, indent=2)
print(f"✓ Batch ablation results saved to {batch_ablation_file}")

# %%
# Visualize batch ablation results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Mean KL by batch - Bar chart
ax = axes[0]
colors = plt.cm.coolwarm(df_batch['mean_kl'] / df_batch['mean_kl'].max())
bars = ax.bar(df_batch['batch'], df_batch['mean_kl'], color=colors, alpha=0.8)
ax.set_xlabel('Batch Index')
ax.set_ylabel('Mean KL Divergence')
ax.set_title(f'Mean KL Divergence by Layer Batch (Size={batch_size})')
ax.set_xticks(df_batch['batch'])
ax.set_xticklabels(df_batch['layers'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, df_batch['mean_kl']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Max KL by batch
ax = axes[1]
colors = plt.cm.plasma(df_batch['max_kl'] / df_batch['max_kl'].max())
bars = ax.bar(df_batch['batch'], df_batch['max_kl'], color=colors, alpha=0.8)
ax.set_xlabel('Batch Index')
ax.set_ylabel('Max KL Divergence')
ax.set_title(f'Max KL Divergence by Layer Batch')
ax.set_xticks(df_batch['batch'])
ax.set_xticklabels(df_batch['layers'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, df_batch['max_kl']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# 3. Comparison: Individual vs Batch ablation
ax = axes[2]
# Average KL per batch from individual ablations
individual_batch_means = []
for batch_idx in range(n_batches):
    start_layer = batch_idx * batch_size
    end_layer = start_layer + batch_size
    batch_layers_df = df[(df['layer'] >= start_layer) & (df['layer'] < end_layer)]
    individual_batch_means.append(batch_layers_df['mean_kl'].mean())

x = np.arange(n_batches)
width = 0.35

bars1 = ax.bar(x - width/2, individual_batch_means, width, label='Avg of Individual', alpha=0.8, color='skyblue')
bars2 = ax.bar(x + width/2, df_batch['mean_kl'], width, label='Batch Ablation', alpha=0.8, color='coral')

ax.set_xlabel('Batch Index')
ax.set_ylabel('Mean KL Divergence')
ax.set_title('Individual vs Batch Ablation Effects')
ax.set_xticks(x)
ax.set_xticklabels(df_batch['layers'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Batch Ablation Study - Ablating Groups of 8 Layers', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

# %%
# Analyze batch ablation patterns
print("\n" + "="*60)
print("BATCH ABLATION ANALYSIS")
print("="*60)

# Find most and least important batches
most_important_batch = df_batch.nlargest(1, 'mean_kl').iloc[0]
least_important_batch = df_batch.nsmallest(1, 'mean_kl').iloc[0]

print(f"\nMost important batch: Layers {most_important_batch['layers']}")
print(f"  Mean KL: {most_important_batch['mean_kl']:.4f}")
print(f"  Max KL: {most_important_batch['max_kl']:.4f}")

print(f"\nLeast important batch: Layers {least_important_batch['layers']}")
print(f"  Mean KL: {least_important_batch['mean_kl']:.4f}")
print(f"  Max KL: {least_important_batch['max_kl']:.4f}")

# Compare batch effects to sum of individual effects
print("\n" + "-"*40)
print("Interaction Effects:")
print("-"*40)

for batch_idx in range(n_batches):
    start_layer = batch_idx * batch_size
    end_layer = start_layer + batch_size
    
    # Get individual effects
    batch_layers_df = df[(df['layer'] >= start_layer) & (df['layer'] < end_layer)]
    sum_individual = batch_layers_df['mean_kl'].sum()
    avg_individual = batch_layers_df['mean_kl'].mean()
    
    # Get batch effect
    batch_effect = df_batch.iloc[batch_idx]['mean_kl']
    
    # Compute interaction (super-additive if batch > sum)
    interaction = batch_effect - sum_individual
    ratio = batch_effect / avg_individual if avg_individual > 0 else 0
    
    print(f"Layers {start_layer:2d}-{end_layer-1:2d}: "
          f"Batch KL = {batch_effect:.4f}, "
          f"Avg Individual = {avg_individual:.4f}, "
          f"Ratio = {ratio:.2f}x")

# %%
# Individual adapter ablation: Ablate each adapter type per layer
print("\n" + "="*60)
print("INDIVIDUAL ADAPTER ABLATION (Per Layer, Per Type)")
print("="*60)

# Define adapter types to ablate
adapter_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# Helper function to ablate specific adapter type in specific layer
def ablate_adapter_type(model, layer_idx, adapter_type):
    """Zero out LoRA weights for a specific adapter type in a specific layer."""
    ablated_modules = []
    
    for name, module in model.named_modules():
        # The actual structure is: base_model.model.model.layers.{layer_idx}.{self_attn|mlp}.{adapter_type}
        # Check if this is the right layer and adapter type
        if f"layers.{layer_idx}." in name and name.endswith(adapter_type) and hasattr(module, 'lora_A'):
            # Store original weights
            original_A = {}
            original_B = {}
            
            for key in module.lora_A.keys():
                original_A[key] = module.lora_A[key].weight.data.clone()
                original_B[key] = module.lora_B[key].weight.data.clone()
                
                # Zero out the weights
                module.lora_A[key].weight.data.zero_()
                module.lora_B[key].weight.data.zero_()
            
            ablated_modules.append((name, module, original_A, original_B))
    
    return ablated_modules

# Create matrix to store results
adapter_ablation_matrix = np.zeros((len(adapter_types), n_layers))

print(f"\nAblating {len(adapter_types)} adapter types across {n_layers} layers...")
print("This will take a while (448 forward passes)...\n")

# First, let's check what modules exist for debugging
if False:  # Set to True to see module structure
    print("Checking module structure for layer 0...")
    for name, module in model.named_modules():
        if "layers.0." in name and hasattr(module, 'lora_A'):
            print(f"  Found LoRA module: {name}")
    print()

# Progress tracking
total_ablations = len(adapter_types) * n_layers
with tqdm(total=total_ablations, desc="Adapter ablations") as pbar:
    for adapter_idx, adapter_type in enumerate(adapter_types):
        for layer_idx in range(n_layers):
            # Ablate specific adapter in specific layer
            ablated_modules = ablate_adapter_type(model, layer_idx, adapter_type)
            
            if ablated_modules:  # Only if we found something to ablate
                # Collect KL for all sequences
                adapter_kls = []
                
                for seq_idx in range(len(all_sequences)):
                    # Forward pass
                    with torch.no_grad():
                        outputs_ablated = model(all_input_ids[seq_idx], return_dict=True)
                        logits_ablated = outputs_ablated.logits[0]
                    
                    # Compute KL divergence
                    probs_ablated = F.softmax(logits_ablated, dim=-1)
                    kl_div = F.kl_div(
                        probs_ablated.log(),
                        all_ground_truth[seq_idx]['probs'],
                        reduction='none',
                        log_target=False
                    ).sum(dim=-1)
                    
                    adapter_kls.append(kl_div.mean().item())
                
                # Store mean KL (averaged across sequences)
                adapter_ablation_matrix[adapter_idx, layer_idx] = np.mean(adapter_kls)
                
                # Restore
                restore_layer(ablated_modules)
            else:
                # No adapter of this type in this layer (shouldn't happen, but just in case)
                # This shouldn't happen with the corrected function
                adapter_ablation_matrix[adapter_idx, layer_idx] = 0.0
            
            pbar.update(1)

print("\n✓ Individual adapter ablation completed!")

# Save adapter ablation matrix to JSON
adapter_ablation_file = f'adapter_ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
adapter_ablation_data = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'model': base_model_id,
        'lora_path': lora_dir,
        'dataset': 'simplescaling/s1K-1.1_tokenized',
        'n_sequences': len(all_sequences),
        'n_total_dataset': len(dataset),
        'example_indices': example_indices,
        'random_seed': RANDOM_SEED,
        'max_seq_length': MAX_SEQ_LENGTH,
        'adapter_types': adapter_types,
        'n_layers': n_layers
    },
    'matrix': adapter_ablation_matrix.tolist(),  # Convert numpy array to list for JSON
    'mean_per_adapter': adapter_ablation_matrix.mean(axis=1).tolist(),
    'mean_per_layer': adapter_ablation_matrix.mean(axis=0).tolist()
}
with open(adapter_ablation_file, 'w') as f:
    json.dump(adapter_ablation_data, f, indent=2)
print(f"✓ Adapter ablation results saved to {adapter_ablation_file}")

# %%
# Visualize adapter ablation heatmap
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})

# Main heatmap
im = ax1.imshow(adapter_ablation_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Adapter Type', fontsize=12)
ax1.set_title('KL Divergence from Ablating Individual Adapters', fontsize=14)
ax1.set_yticks(range(len(adapter_types)))
ax1.set_yticklabels(adapter_types)
ax1.set_xticks(range(0, n_layers, 4))
ax1.set_xticklabels(range(0, n_layers, 4))

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1, fraction=0.05)
cbar.set_label('Mean KL Divergence', fontsize=11)

# Add grid for better readability
ax1.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, len(adapter_types), 1), minor=True)
ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)

# Summary plot: Average KL per adapter type
mean_per_adapter = adapter_ablation_matrix.mean(axis=1)
colors = plt.cm.YlOrRd(mean_per_adapter / mean_per_adapter.max())
bars = ax2.bar(range(len(adapter_types)), mean_per_adapter, color=colors, alpha=0.8)
ax2.set_xlabel('Adapter Type', fontsize=12)
ax2.set_ylabel('Mean KL', fontsize=12)
ax2.set_title('Average KL Divergence Across All Layers (Per Adapter Type)', fontsize=13)
ax2.set_xticks(range(len(adapter_types)))
ax2.set_xticklabels(adapter_types, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, mean_per_adapter):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Individual Adapter Ablation Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Analyze adapter importance patterns
print("\n" + "="*60)
print("ADAPTER IMPORTANCE ANALYSIS")
print("="*60)

# Find most important adapter types (averaged across layers)
adapter_importance = pd.DataFrame({
    'adapter': adapter_types,
    'mean_kl': adapter_ablation_matrix.mean(axis=1),
    'max_kl': adapter_ablation_matrix.max(axis=1),
    'std_kl': adapter_ablation_matrix.std(axis=1)
}).sort_values('mean_kl', ascending=False)

print("\nAdapter types ranked by importance (mean KL across all layers):")
for _, row in adapter_importance.iterrows():
    print(f"  {row['adapter']:10s}: Mean KL = {row['mean_kl']:.5f}, Max KL = {row['max_kl']:.5f}, Std = {row['std_kl']:.5f}")

# Find most important layer for each adapter type
print("\n" + "-"*40)
print("Most important layer for each adapter type:")
print("-"*40)
for adapter_idx, adapter_type in enumerate(adapter_types):
    layer_kls = adapter_ablation_matrix[adapter_idx, :]
    most_important_layer = np.argmax(layer_kls)
    max_kl = layer_kls[most_important_layer]
    print(f"  {adapter_type:10s}: Layer {most_important_layer:2d} (KL = {max_kl:.5f})")

# Analyze attention vs MLP adapters
attention_adapters = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
mlp_adapters = ['gate_proj', 'up_proj', 'down_proj']

attention_indices = [adapter_types.index(a) for a in attention_adapters]
mlp_indices = [adapter_types.index(a) for a in mlp_adapters]

attention_mean = adapter_ablation_matrix[attention_indices, :].mean()
mlp_mean = adapter_ablation_matrix[mlp_indices, :].mean()

print("\n" + "-"*40)
print("Attention vs MLP Adapter Importance:")
print("-"*40)
print(f"  Attention adapters (Q,K,V,O): Mean KL = {attention_mean:.5f}")
print(f"  MLP adapters (Gate,Up,Down):  Mean KL = {mlp_mean:.5f}")
print(f"  Ratio (MLP/Attention):         {mlp_mean/attention_mean:.2f}x")

# Find layers where specific adapters are particularly important
print("\n" + "-"*40)
print("Layers with exceptionally high impact for specific adapters:")
print("-"*40)
threshold = np.percentile(adapter_ablation_matrix, 90)  # Top 10% of values
for adapter_idx, adapter_type in enumerate(adapter_types):
    high_impact_layers = np.where(adapter_ablation_matrix[adapter_idx, :] > threshold)[0]
    if len(high_impact_layers) > 0:
        print(f"  {adapter_type:10s}: Layers {high_impact_layers.tolist()}")

# %%
# Create a more detailed heatmap with annotations for top values
fig, ax = plt.subplots(figsize=(24, 8))

# Create heatmap
im = ax.imshow(adapter_ablation_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

# Annotate cells with high values
threshold = np.percentile(adapter_ablation_matrix, 95)  # Top 5%
for i in range(len(adapter_types)):
    for j in range(n_layers):
        if adapter_ablation_matrix[i, j] > threshold:
            ax.text(j, i, f'{adapter_ablation_matrix[i, j]:.3f}', 
                   ha='center', va='center', color='white', fontsize=6, fontweight='bold')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Adapter Type', fontsize=12)
ax.set_title('Individual Adapter Ablation Heatmap (Top 5% values annotated)', fontsize=14)
ax.set_yticks(range(len(adapter_types)))
ax.set_yticklabels(adapter_types)
ax.set_xticks(range(0, n_layers, 2))
ax.set_xticklabels(range(0, n_layers, 2))

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mean KL Divergence', fontsize=11)

# Add grid
ax.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(adapter_types), 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
