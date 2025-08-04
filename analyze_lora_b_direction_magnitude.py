# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
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
model.eval()

# %%
# Extract LoRA B directions from down_proj
n_layers = len(model.model.model.layers)
lora_b_directions = {}

for layer_idx in range(n_layers):
    # Access the down_proj module
    down_proj = model.model.model.layers[layer_idx].mlp.down_proj
    
    # Extract the LoRA B matrix (which is a vector for rank-1)
    if hasattr(down_proj, 'lora_B'):
        # Get the B matrix from the LoRA adapter
        lora_B_weight = down_proj.lora_B['default'].weight.data
        # For rank-1, this should be shape [hidden_size, 1]
        # We want a 1D vector of shape [hidden_size]
        lora_b_direction = lora_B_weight.squeeze()
        lora_b_directions[layer_idx] = lora_b_direction
        
print(f"Extracted LoRA B directions for {len(lora_b_directions)} layers")

# %%
# Load s1K-1.1 dataset
print("Loading s1K-1.1 dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"Dataset has {len(dataset)} examples")

# %%
# Get a single example
example_idx = 0
example = dataset[example_idx]

# Extract question and thinking trajectory
question = example['question']
thinking_trajectory = example.get('deepseek_thinking_trajectory', '')
attempt = example.get('deepseek_attempt', '')

# Format the input
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
input_ids = inputs.input_ids
seq_len = input_ids.shape[1]

print(f"Sequence length: {seq_len} tokens")
print(f"Number of layers: {n_layers}")

# %%
# Storage for projections of residual stream onto LoRA B directions
lora_b_projections = torch.zeros(n_layers, seq_len)
residual_norms = torch.zeros(n_layers, seq_len)

# Hook function to capture residual stream before MLP
def make_pre_mlp_hook(layer_idx):
    def hook(module, input, output):
        # output is the residual stream after attention + residual connection
        # This is what the MLP will add to
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        # Compute norm of residual stream
        res_norms = torch.norm(residual, p=2, dim=-1)  # [seq_len]
        residual_norms[layer_idx, :] = res_norms.cpu()
        
        # Project onto LoRA B direction
        if layer_idx in lora_b_directions:
            lora_b_dir = lora_b_directions[layer_idx]
            # Normalize the LoRA B direction
            lora_b_dir_normalized = lora_b_dir / torch.norm(lora_b_dir)
            # Project residual onto normalized LoRA B direction
            projections = torch.matmul(residual.float(), lora_b_dir_normalized.float())  # [seq_len]
            lora_b_projections[layer_idx, :] = projections.cpu()
    return hook

# Register hooks on the LayerNorm before MLP
hooks = []
for layer_idx in range(n_layers):
    # Hook the output of the layer norm that feeds into the MLP
    hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
        make_pre_mlp_hook(layer_idx)
    )
    hooks.append(hook)

# %%
# Forward pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Convert to numpy for plotting
lora_b_projections_np = lora_b_projections.numpy()
residual_norms_np = residual_norms.numpy()

# Compute statistics
print(f"\nLoRA B direction projection statistics:")
print(f"  Mean: {lora_b_projections_np.mean():.4f}")
print(f"  Std: {lora_b_projections_np.std():.4f}")
print(f"  Min: {lora_b_projections_np.min():.4f}")
print(f"  Max: {lora_b_projections_np.max():.4f}")

print(f"\nResidual stream norm statistics:")
print(f"  Mean: {residual_norms_np.mean():.2f}")
print(f"  Std: {residual_norms_np.std():.2f}")
print(f"  Min: {residual_norms_np.min():.2f}")
print(f"  Max: {residual_norms_np.max():.2f}")

# %%
# Create heatmap of projections (with appropriate scale)
plt.figure(figsize=(20, 12))
# Use a diverging colormap centered at 0
vmax = np.abs(lora_b_projections_np).max()
sns.heatmap(lora_b_projections_np, 
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={'label': 'Projection onto LoRA B direction'},
            xticklabels=False,
            yticklabels=5)
plt.xlabel('Token Position')
plt.ylabel('Layer')
plt.title('Residual Stream Projection onto LoRA B Direction (Before MLP)')
plt.tight_layout()
plt.savefig('lora_b_projections_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot mean projection by layer
mean_projections_by_layer = lora_b_projections_np.mean(axis=1)
std_projections_by_layer = lora_b_projections_np.std(axis=1)

plt.figure(figsize=(12, 6))
plt.errorbar(range(n_layers), mean_projections_by_layer, yerr=std_projections_by_layer, 
             fmt='b-', linewidth=2, alpha=0.7, capsize=3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Layer')
plt.ylabel('Mean Projection onto LoRA B Direction')
plt.title('Mean Residual Stream Projection onto LoRA B Direction by Layer')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mean_lora_b_projections_by_layer.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Compare projection magnitude to residual norm
relative_projections = np.abs(lora_b_projections_np) / (residual_norms_np + 1e-8)
print(f"\nRelative projection magnitude (|projection| / |residual|):")
print(f"  Mean: {relative_projections.mean():.4f}")
print(f"  Std: {relative_projections.std():.4f}")
print(f"  Max: {relative_projections.max():.4f}")

# %%
# Plot distribution of projections
plt.figure(figsize=(12, 6))
all_projections = lora_b_projections_np.flatten()
plt.hist(all_projections, bins=100, alpha=0.7, density=True)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Projection Value')
plt.ylabel('Density')
plt.title('Distribution of Residual Stream Projections onto LoRA B Directions')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lora_b_projection_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Show some specific layer patterns
interesting_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
plt.figure(figsize=(16, 10))
for i, layer_idx in enumerate(interesting_layers):
    plt.subplot(len(interesting_layers), 1, i+1)
    plt.plot(lora_b_projections_np[layer_idx, :], linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.ylabel(f'Layer {layer_idx}')
    if i == 0:
        plt.title('Residual Stream Projection onto LoRA B Direction for Selected Layers')
    if i == len(interesting_layers) - 1:
        plt.xlabel('Token Position')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lora_b_projections_selected_layers.png', dpi=300, bbox_inches='tight')
plt.show()