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
lora_path = "/root/s1_peft/ckpts_1.1"
rank = 1

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
# Load multiple math problems
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
n_prompts = 1
problems = [dataset[i]['problem'] for i in range(n_prompts)]

print(f"Loaded {n_prompts} problems for averaging")

# %%
# Generate ground truth rollouts for all prompts
print("Generating ground truth rollouts for all prompts...")

all_rollouts = []
all_ground_truth = []

for i, problem in enumerate(tqdm(problems, desc="Generating rollouts")):
    # Format prompt
    messages = [
        {"role": "system", "content": "You are a helpful mathematics assistant."},
        {"role": "user", "content": problem}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get the full sequence
    full_sequence = generated.sequences[0]
    
    # Prepare for forward pass
    rollout_input_ids = full_sequence[:-1].unsqueeze(0)
    rollout_labels = full_sequence[1:]
    
    # Get ground truth logits
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
# Function to ablate all matrices of a specific type across all layers
def ablate_all_matrices_of_type(model, matrix_type):
    """Zero out LoRA weights for all matrices of a specific type across all layers."""
    ablated_modules = []
    
    for name, module in model.named_modules():
        # Check if this module is the target matrix type
        if f".{matrix_type}" in name and hasattr(module, 'lora_A'):
            # Store original weights
            original_A = {}
            original_B = {}
            
            for key in module.lora_A.keys():
                original_A[key] = module.lora_A[key].weight.data.clone()
                original_B[key] = module.lora_B[key].weight.data.clone()
                
                # Zero out the weights
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

# %%
# Define matrix types to ablate
matrix_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# Store results
results = []

print("Running ablation experiment for each matrix type (all layers at once)...")
for matrix_type in tqdm(matrix_types, desc="Matrix types"):
    # Ablate all matrices of this type
    ablated_modules = ablate_all_matrices_of_type(model, matrix_type)
    print(f"  Ablated {len(ablated_modules)} {matrix_type} modules")
    
    # Compute metrics for each prompt
    prompt_kls = []
    prompt_ces = []
    
    for rollout in all_rollouts:
        # Forward pass with ablated matrices
        with torch.no_grad():
            ablated_outputs = model(rollout['input_ids'], return_dict=True)
            ablated_logits = ablated_outputs.logits[0]
        
        # Compute probabilities
        ablated_probs = F.softmax(ablated_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            ablated_probs.log(),
            rollout['ground_truth_probs'],
            reduction='none',
            log_target=False
        ).sum(dim=-1)
        
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(
            ablated_logits,
            rollout['labels'],
            reduction='none'
        )
        
        prompt_kls.append(kl_div.mean().item())
        prompt_ces.append(ce_loss.mean().item())
    
    # Average across prompts
    mean_kl = np.mean(prompt_kls)
    mean_ce = np.mean(prompt_ces)
    
    # Store results
    results.append({
        'matrix_type': matrix_type,
        'mean_kl': mean_kl,
        'std_kl': np.std(prompt_kls),
        'mean_ce': mean_ce,
        'std_ce': np.std(prompt_ces),
        'n_modules_ablated': len(ablated_modules)
    })
    
    # Restore the modules
    restore_modules(ablated_modules)
    
    # Verify restoration on first rollout
    with torch.no_grad():
        test_outputs = model(all_rollouts[0]['input_ids'], return_dict=True)
        test_logits = test_outputs.logits[0]
    assert torch.allclose(test_logits, all_rollouts[0]['ground_truth_logits'], atol=1e-6), f"Restoration failed for {matrix_type}"

# %%
# Create dataframe
df = pd.DataFrame(results)
print("\nAblation Results:")
print(df)

# %%
# Create bar plot for mean KL divergence with error bars
plt.figure(figsize=(10, 6))
bars = plt.bar(df['matrix_type'], df['mean_kl'], yerr=df['std_kl'],
                color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'],
                capsize=5)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + df['std_kl'].iloc[i],
             f'{height:.3f}', ha='center', va='bottom')

plt.xlabel('Matrix Type', fontsize=12)
plt.ylabel('Mean KL Divergence', fontsize=12)
plt.title(f'Impact of Ablating All Matrices of Each Type (n={n_prompts})', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Create bar plot for mean CE loss with error bars
plt.figure(figsize=(10, 6))
bars = plt.bar(df['matrix_type'], df['mean_ce'], yerr=df['std_ce'],
                color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'],
                capsize=5)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + df['std_ce'].iloc[i],
             f'{height:.3f}', ha='center', va='bottom')

plt.xlabel('Matrix Type', fontsize=12)
plt.ylabel('Mean Cross-Entropy Loss', fontsize=12)
plt.title(f'Cross-Entropy Loss from Ablating All Matrices of Each Type (n={n_prompts})', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Sort by importance
df_sorted = df.sort_values('mean_kl', ascending=False)
print("\nMatrix types ranked by importance (mean KL divergence):")
for _, row in df_sorted.iterrows():
    print(f"{row['matrix_type']:10s}: KL = {row['mean_kl']:.4f}, CE = {row['mean_ce']:.4f} ({row['n_modules_ablated']} modules)")

# %%
# Ablate all attention matrices vs all MLP matrices
print("\nAblating attention vs MLP matrices...")

# Define matrix groups
attention_matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
mlp_matrices = ['gate_proj', 'up_proj', 'down_proj']

group_results = []

# Ablate all attention matrices
print("Ablating all attention matrices...")
ablated_modules = []
for matrix_type in attention_matrices:
    modules = ablate_all_matrices_of_type(model, matrix_type)
    ablated_modules.extend(modules)
print(f"  Total attention modules ablated: {len(ablated_modules)}")

# Compute metrics for each prompt
prompt_kls = []
prompt_ces = []

for rollout in all_rollouts:
    with torch.no_grad():
        ablated_outputs = model(rollout['input_ids'], return_dict=True)
        ablated_logits = ablated_outputs.logits[0]
    
    ablated_probs = F.softmax(ablated_logits, dim=-1)
    kl_div = F.kl_div(
        ablated_probs.log(),
        rollout['ground_truth_probs'],
        reduction='none',
        log_target=False
    ).sum(dim=-1)
    
    ce_loss = F.cross_entropy(
        ablated_logits,
        rollout['labels'],
        reduction='none'
    )
    
    prompt_kls.append(kl_div.mean().item())
    prompt_ces.append(ce_loss.mean().item())

group_results.append({
    'group': 'Attention (Q,K,V,O)',
    'mean_kl': np.mean(prompt_kls),
    'std_kl': np.std(prompt_kls),
    'mean_ce': np.mean(prompt_ces),
    'std_ce': np.std(prompt_ces),
    'n_modules': len(ablated_modules)
})

# Restore all modules
for modules in ablated_modules:
    if isinstance(modules, tuple):
        restore_modules([modules])
    else:
        restore_modules(modules)

# Verify restoration
with torch.no_grad():
    test_outputs = model(all_rollouts[0]['input_ids'], return_dict=True)
    test_logits = test_outputs.logits[0]
assert torch.allclose(test_logits, all_rollouts[0]['ground_truth_logits'], atol=1e-6), "Restoration failed for attention matrices"

# Ablate all MLP matrices
print("\nAblating all MLP matrices...")
ablated_modules = []
for matrix_type in mlp_matrices:
    modules = ablate_all_matrices_of_type(model, matrix_type)
    ablated_modules.extend(modules)
print(f"  Total MLP modules ablated: {len(ablated_modules)}")

# Compute metrics for each prompt
prompt_kls = []
prompt_ces = []

for rollout in all_rollouts:
    with torch.no_grad():
        ablated_outputs = model(rollout['input_ids'], return_dict=True)
        ablated_logits = ablated_outputs.logits[0]
    
    ablated_probs = F.softmax(ablated_logits, dim=-1)
    kl_div = F.kl_div(
        ablated_probs.log(),
        rollout['ground_truth_probs'],
        reduction='none',
        log_target=False
    ).sum(dim=-1)
    
    ce_loss = F.cross_entropy(
        ablated_logits,
        rollout['labels'],
        reduction='none'
    )
    
    prompt_kls.append(kl_div.mean().item())
    prompt_ces.append(ce_loss.mean().item())

group_results.append({
    'group': 'MLP (Gate,Up,Down)',
    'mean_kl': np.mean(prompt_kls),
    'std_kl': np.std(prompt_kls),
    'mean_ce': np.mean(prompt_ces),
    'std_ce': np.std(prompt_ces),
    'n_modules': len(ablated_modules)
})

# Restore all modules
for modules in ablated_modules:
    if isinstance(modules, tuple):
        restore_modules([modules])
    else:
        restore_modules(modules)

# %%
# Create comparison plot
df_groups = pd.DataFrame(group_results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# KL divergence comparison
bars1 = ax1.bar(df_groups['group'], df_groups['mean_kl'], yerr=df_groups['std_kl'],
                color=['skyblue', 'lightcoral'], capsize=5)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + df_groups['std_kl'].iloc[i],
             f'{height:.3f}', ha='center', va='bottom')

ax1.set_ylabel('Mean KL Divergence', fontsize=12)
ax1.set_title(f'KL Divergence: Attention vs MLP Ablation (n={n_prompts})', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# CE loss comparison  
bars2 = ax2.bar(df_groups['group'], df_groups['mean_ce'], yerr=df_groups['std_ce'],
                color=['skyblue', 'lightcoral'], capsize=5)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + df_groups['std_ce'].iloc[i],
             f'{height:.3f}', ha='center', va='bottom')

ax2.set_ylabel('Mean Cross-Entropy Loss', fontsize=12)
ax2.set_title(f'CE Loss: Attention vs MLP Ablation (n={n_prompts})', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nAttention vs MLP ablation results:")
print(df_groups)

# %%
# Generate a rollout using only MLP LoRA (attention ablated)
print("\nGenerating rollout with MLP-only LoRA...")

# Ablate all attention matrices
attention_ablated = []
for matrix_type in attention_matrices:
    modules = ablate_all_matrices_of_type(model, matrix_type)
    attention_ablated.extend(modules)
print(f"Ablated {len(attention_ablated)} attention modules")

# Use the first problem for demonstration
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

# Generate with MLP-only LoRA
print("Generating with MLP-only LoRA...")
with torch.no_grad():
    mlp_only_outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

mlp_only_text = tokenizer.decode(mlp_only_outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

# Restore attention modules
for modules in attention_ablated:
    if isinstance(modules, tuple):
        restore_modules([modules])
    else:
        restore_modules(modules)

# Generate with full LoRA for comparison
print("\nGenerating with full LoRA for comparison...")
with torch.no_grad():
    full_lora_outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

full_lora_text = tokenizer.decode(full_lora_outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print("\n" + "="*80)
print("PROBLEM:")
print(problem[:300] + "..." if len(problem) > 300 else problem)
print("\n" + "="*80)
print("MLP-ONLY LoRA OUTPUT:")
for line in mlp_only_text.split('\n')[:20]:  # First 20 lines
    print(textwrap.fill(line, width=80) if line else '')
if len(mlp_only_text.split('\n')) > 20:
    print("...")
print("\n" + "="*80)
print("FULL LoRA OUTPUT:")
for line in full_lora_text.split('\n')[:20]:  # First 20 lines
    print(textwrap.fill(line, width=80) if line else '')
if len(full_lora_text.split('\n')) > 20:
    print("...")

# %%
