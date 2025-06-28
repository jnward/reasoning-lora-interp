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
# Load a math problem
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
problem = dataset[0]['problem']

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

print(f"Problem: {problem[:200]}...")

# %%
# Generate ground truth rollout with full LoRA
print("Generating ground truth rollout...")

# First generate the full response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    # Generate tokens
    generated = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

# Get the full sequence (prompt + generated)
full_sequence = generated.sequences[0]
prompt_length = inputs.input_ids.shape[1]
generated_tokens = full_sequence[prompt_length:]

print(f"Generated {len(generated_tokens)} tokens")
print("Generated text:", tokenizer.decode(generated_tokens, skip_special_tokens=True)[:200] + "...")

# %%
# Get ground truth logits by running forward pass on the full sequence
print("\nGetting ground truth distributions...")

# Prepare input for forward pass (exclude last token to get predictions)
rollout_input_ids = full_sequence[:-1].unsqueeze(0)
rollout_labels = full_sequence[1:]  # Shifted for next-token prediction

# Forward pass with full LoRA
with torch.no_grad():
    outputs = model(rollout_input_ids, return_dict=True)
    ground_truth_logits = outputs.logits[0]  # [seq_len, vocab_size]

# Save the ground truth distributions
ground_truth_probs = F.softmax(ground_truth_logits, dim=-1)
print(f"Ground truth logits shape: {ground_truth_logits.shape}")

# %%
# First, let's examine the model structure to understand the naming
print("Examining model structure for LoRA modules...")
lora_modules = []
for name, module in model.named_modules():
    if hasattr(module, 'lora_A'):
        lora_modules.append(name)
        if len(lora_modules) <= 10:  # Print first 10
            print(f"  {name}")

print(f"\nTotal LoRA modules found: {len(lora_modules)}")

# %%
# Function to ablate specific matrix type in a layer
def ablate_matrix_type(model, layer_idx, matrix_type):
    """Zero out LoRA weights for a specific matrix type in a specific layer."""
    ablated_modules = []
    
    for name, module in model.named_modules():
        # Check if this module is in the target layer and is the target matrix type
        if (f"layers.{layer_idx}." in name and 
            f".{matrix_type}" in name and  # Removed the extra dot
            hasattr(module, 'lora_A')):
            
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
n_layers = 64

# Store results for each matrix type
results_by_matrix = {matrix_type: [] for matrix_type in matrix_types}

print("Running ablation experiment for each individual matrix...")
for layer_idx in tqdm(range(n_layers), desc="Layers"):
    for matrix_type in matrix_types:
        # Ablate the specific matrix type in this layer
        ablated_modules = ablate_matrix_type(model, layer_idx, matrix_type)
        
        # Skip if no modules were ablated (some layers might not have all matrix types)
        if not ablated_modules:
            continue
        
        # Forward pass with ablated matrix
        with torch.no_grad():
            ablated_outputs = model(rollout_input_ids, return_dict=True)
            ablated_logits = ablated_outputs.logits[0]
        
        # Compute probabilities
        ablated_probs = F.softmax(ablated_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            ablated_probs.log(),
            ground_truth_probs,
            reduction='none',
            log_target=False
        ).sum(dim=-1)
        
        # Store results
        results_by_matrix[matrix_type].append({
            'layer': layer_idx,
            'mean_kl': kl_div.mean().item(),
            'max_kl': kl_div.max().item()
        })
        
        # Restore the modules
        restore_modules(ablated_modules)
        
        # Verify restoration
        with torch.no_grad():
            test_outputs = model(rollout_input_ids, return_dict=True)
            test_logits = test_outputs.logits[0]
        assert torch.allclose(test_logits, ground_truth_logits, atol=1e-6), f"Restoration failed for {matrix_type} layer {layer_idx}"

# %%
# Create dataframes for each matrix type
dfs = {}
for matrix_type, results in results_by_matrix.items():
    if results:
        dfs[matrix_type] = pd.DataFrame(results)

# %%
# Plot multi-line graph with each matrix type as a separate line
plt.figure(figsize=(12, 6))

# Define colors for each matrix type
colors = {
    'q_proj': 'blue',
    'k_proj': 'orange', 
    'v_proj': 'green',
    'o_proj': 'red',
    'gate_proj': 'purple',
    'up_proj': 'brown',
    'down_proj': 'pink'
}

# Plot each matrix type
for matrix_type, df in dfs.items():
    plt.plot(df['layer'], df['mean_kl'], 
             marker='o', markersize=3, 
             label=matrix_type, 
             color=colors.get(matrix_type, 'gray'),
             linewidth=2)

plt.xlabel('Layer', fontsize=12)
plt.ylabel('Mean KL Divergence', fontsize=12)
plt.title('Layer Importance by Matrix Type (Mean KL Divergence)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Create a second plot for max KL divergence
plt.figure(figsize=(12, 6))

for matrix_type, df in dfs.items():
    plt.plot(df['layer'], df['max_kl'], 
             marker='o', markersize=3, 
             label=matrix_type, 
             color=colors.get(matrix_type, 'gray'),
             linewidth=2, alpha=0.7)

plt.xlabel('Layer', fontsize=12)
plt.ylabel('Max KL Divergence', fontsize=12)
plt.title('Layer Importance by Matrix Type (Max KL Divergence)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Summary statistics
print("\nAverage KL divergence across all layers by matrix type:")
for matrix_type, df in dfs.items():
    print(f"{matrix_type:10s}: {df['mean_kl'].mean():.4f}")

# %%
# Find most important layers for each matrix type
print("\nMost important layer for each matrix type:")
for matrix_type, df in dfs.items():
    top_layer = df.loc[df['mean_kl'].idxmax()]
    print(f"{matrix_type:10s}: Layer {int(top_layer['layer'])} (KL: {top_layer['mean_kl']:.4f})")

# %%