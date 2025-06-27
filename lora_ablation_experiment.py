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
# Function to zero out LoRA weights for a specific layer
def ablate_layer(model, layer_idx):
    """Zero out LoRA weights for a specific layer."""
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
            
            ablated_modules.append((module, original_A, original_B))
    
    return ablated_modules

def restore_layer(ablated_modules):
    """Restore original LoRA weights."""
    for module, original_A, original_B in ablated_modules:
        for key in module.lora_A.keys():
            module.lora_A[key].weight.data = original_A[key]
            module.lora_B[key].weight.data = original_B[key]

# %%
# Ablate each layer and compute metrics
results = []
n_layers = 64  # Qwen-2.5-32B has 64 layers

print("Running ablation experiment across all layers...")
for layer_idx in tqdm(range(n_layers)):
    # Ablate the layer
    ablated_modules = ablate_layer(model, layer_idx)
    
    # Forward pass with ablated layer
    with torch.no_grad():
        ablated_outputs = model(rollout_input_ids, return_dict=True)
        ablated_logits = ablated_outputs.logits[0]  # [seq_len, vocab_size]
    
    # Compute probabilities
    ablated_probs = F.softmax(ablated_logits, dim=-1)
    
    # Compute KL divergence
    kl_div = F.kl_div(
        ablated_probs.log(),
        ground_truth_probs,
        reduction='none',
        log_target=False
    ).sum(dim=-1)  # Sum over vocab dimension
    
    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(
        ablated_logits,
        rollout_labels,
        reduction='none'
    )
    
    # Store results
    results.append({
        'layer': layer_idx,
        'mean_kl': kl_div.mean().item(),
        'max_kl': kl_div.max().item(),
        'mean_ce': ce_loss.mean().item(),
        'max_ce': ce_loss.max().item(),
        'n_modules_ablated': len(ablated_modules)
    })
    
    # Restore the layer
    restore_layer(ablated_modules)
    
    # Verify restoration worked correctly
    with torch.no_grad():
        test_outputs = model(rollout_input_ids, return_dict=True)
        test_logits = test_outputs.logits[0]
    assert torch.allclose(test_logits, ground_truth_logits, atol=1e-6), f"Layer {layer_idx} restoration failed"

# %%
# Create dataframe with results
df = pd.DataFrame(results)
print("\nAblation Results Summary:")
print(df.head(10))

# %%
# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Mean KL divergence
ax = axes[0, 0]
ax.plot(df['layer'], df['mean_kl'], marker='o', markersize=4)
ax.set_xlabel('Layer')
ax.set_ylabel('Mean KL Divergence')
ax.set_title('Mean KL Divergence by Layer')
ax.grid(True, alpha=0.3)

# Max KL divergence
ax = axes[0, 1]
ax.plot(df['layer'], df['max_kl'], marker='o', markersize=4, color='orange')
ax.set_xlabel('Layer')
ax.set_ylabel('Max KL Divergence')
ax.set_title('Max KL Divergence by Layer')
ax.grid(True, alpha=0.3)

# Mean CE loss
ax = axes[1, 0]
ax.plot(df['layer'], df['mean_ce'], marker='o', markersize=4, color='green')
ax.set_xlabel('Layer')
ax.set_ylabel('Mean Cross-Entropy Loss')
ax.set_title('Mean CE Loss by Layer')
ax.grid(True, alpha=0.3)

# Max CE loss
ax = axes[1, 1]
ax.plot(df['layer'], df['max_ce'], marker='o', markersize=4, color='red')
ax.set_xlabel('Layer')
ax.set_ylabel('Max Cross-Entropy Loss')
ax.set_title('Max CE Loss by Layer')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Identify most and least important layers
print("\nMost important layers (by mean KL divergence):")
print(df.nlargest(10, 'mean_kl')[['layer', 'mean_kl', 'mean_ce']])

print("\nLeast important layers (by mean KL divergence):")
print(df.nsmallest(10, 'mean_kl')[['layer', 'mean_kl', 'mean_ce']])

# %%