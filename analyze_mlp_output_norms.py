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
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*544")
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
n_layers = len(model.model.model.layers)

print(f"Sequence length: {seq_len} tokens")
print(f"Number of layers: {n_layers}")

# %%
# Storage for MLP output norms
mlp_output_norms = torch.zeros(n_layers, seq_len)

# Hook function to capture MLP outputs
def make_mlp_hook(layer_idx):
    def hook(module, input, output):
        # output shape: [batch_size, seq_len, hidden_size]
        mlp_out = output[0].detach()  # [seq_len, hidden_size]
        # Compute L2 norm for each token position
        norms = torch.norm(mlp_out, p=2, dim=-1)  # [seq_len]
        mlp_output_norms[layer_idx, :] = norms.cpu()
    return hook

# Register hooks
hooks = []
for layer_idx in range(n_layers):
    hook = model.model.model.layers[layer_idx].mlp.register_forward_hook(make_mlp_hook(layer_idx))
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
mlp_output_norms_np = mlp_output_norms.numpy()

# Compute statistics
print(f"MLP output norm statistics:")
print(f"  Mean: {mlp_output_norms_np.mean():.2f}")
print(f"  Std: {mlp_output_norms_np.std():.2f}")
print(f"  Min: {mlp_output_norms_np.min():.2f}")
print(f"  Max: {mlp_output_norms_np.max():.2f}")

# %%
# Create heatmap with log scale
plt.figure(figsize=(20, 12))
# Add small epsilon to avoid log(0)
log_norms = np.log10(mlp_output_norms_np + 1e-8)
sns.heatmap(log_norms, 
            cmap='viridis',
            cbar_kws={'label': 'log10(MLP Output L2 Norm)'},
            xticklabels=False,  # Too many token positions to show
            yticklabels=5)  # Show every 5th layer
plt.xlabel('Token Position')
plt.ylabel('Layer')
plt.title('MLP Output L2 Norms by Layer and Token Position (log scale)')
plt.tight_layout()
plt.savefig('mlp_output_norms_heatmap_log.png', dpi=300, bbox_inches='tight')
plt.show()

# Also create regular scale heatmap for comparison
plt.figure(figsize=(20, 12))
sns.heatmap(mlp_output_norms_np, 
            cmap='viridis',
            cbar_kws={'label': 'MLP Output L2 Norm'},
            xticklabels=False,  # Too many token positions to show
            yticklabels=5)  # Show every 5th layer
plt.xlabel('Token Position')
plt.ylabel('Layer')
plt.title('MLP Output L2 Norms by Layer and Token Position')
plt.tight_layout()
plt.savefig('mlp_output_norms_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot mean norm by layer
mean_norms_by_layer = mlp_output_norms_np.mean(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(range(n_layers), mean_norms_by_layer, 'b-', linewidth=2)
plt.xlabel('Layer')
plt.ylabel('Mean MLP Output L2 Norm')
plt.title('Mean MLP Output L2 Norm by Layer')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mean_mlp_norms_by_layer.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot mean norm by token position
mean_norms_by_position = mlp_output_norms_np.mean(axis=0)
plt.figure(figsize=(16, 6))
plt.plot(range(seq_len), mean_norms_by_position, 'g-', linewidth=1)
plt.xlabel('Token Position')
plt.ylabel('Mean MLP Output L2 Norm')
plt.title('Mean MLP Output L2 Norm by Token Position')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mean_mlp_norms_by_position.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Show some specific layer patterns
interesting_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
plt.figure(figsize=(16, 10))
for i, layer_idx in enumerate(interesting_layers):
    plt.subplot(len(interesting_layers), 1, i+1)
    plt.plot(mlp_output_norms_np[layer_idx, :], linewidth=1)
    plt.ylabel(f'Layer {layer_idx}')
    if i == 0:
        plt.title('MLP Output L2 Norms for Selected Layers')
    if i == len(interesting_layers) - 1:
        plt.xlabel('Token Position')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_norms_selected_layers.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Decode some tokens to understand what's happening at high-norm positions
# Find top 10 highest norm positions
top_positions = np.unravel_index(np.argsort(mlp_output_norms_np.ravel())[-10:], mlp_output_norms_np.shape)
print("\nTop 10 highest MLP output norm positions:")
for idx in range(len(top_positions[0])-1, -1, -1):  # Reverse to show highest first
    layer = top_positions[0][idx]
    pos = top_positions[1][idx]
    norm = mlp_output_norms_np[layer, pos]
    token = tokenizer.decode(input_ids[0, pos:pos+1])
    print(f"  Layer {layer}, Position {pos}: norm={norm:.2f}, token='{token}'")

# %%
# Extract LoRA B directions from down_proj
print("\n=== Analyzing LoRA B Direction Magnitude in Residual Stream ===")
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
# Storage for projections of residual stream onto LoRA B directions
lora_b_projections = torch.zeros(n_layers, seq_len)
residual_norms = torch.zeros(n_layers, seq_len)

# Hook function to capture residual stream before MLP
def make_pre_mlp_hook_v2(layer_idx):
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
        make_pre_mlp_hook_v2(layer_idx)
    )
    hooks.append(hook)

# %%
# Forward pass
print("Running forward pass for LoRA B analysis...")
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
# Analyze number of zero eigenvalues in residual stream
print("\n=== Analyzing Zero Eigenvalues in Residual Stream ===")

# Storage for number of zero eigenvalues at each layer
num_zero_eigenvalues = []
sampled_layers = []

# Hook function to count zero eigenvalues
def make_eigenvalue_hook(layer_idx):
    def hook(module, input, output):
        # output is the residual stream after attention + residual connection
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        # Compute covariance matrix
        residual_centered = residual - residual.mean(dim=0, keepdim=True)
        cov_matrix = torch.matmul(residual_centered.T, residual_centered) / (residual.shape[0] - 1)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov_matrix.float())
        eigenvalues_np = eigenvalues.cpu().numpy()
        
        # Count zero eigenvalues (using threshold relative to max eigenvalue)
        max_eigenvalue = np.max(np.abs(eigenvalues_np))
        threshold = 1e-6 * max_eigenvalue
        n_zeros = np.sum(np.abs(eigenvalues_np) < threshold)
        
        num_zero_eigenvalues.append(n_zeros)
        sampled_layers.append(layer_idx)
    return hook

# Register hooks for eigenvalue analysis
hooks = []
for layer_idx in range(n_layers):
    if layer_idx % 2 == 0:  # Sample every other layer
        hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
            make_eigenvalue_hook(layer_idx)
        )
        hooks.append(hook)

# %%
# Forward pass for eigenvalue analysis
print("Running forward pass for eigenvalue analysis...")
num_zero_eigenvalues = []
sampled_layers = []

with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Plot number of zero eigenvalues
hidden_size = model.config.hidden_size
plt.figure(figsize=(12, 6))
plt.semilogy(sampled_layers, num_zero_eigenvalues, 'ro-', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('Number of Zero Eigenvalues (log scale)')
plt.title(f'Number of Zero Eigenvalues in Residual Stream Covariance (threshold = 1e-6 * max eigenvalue)')
plt.grid(True, alpha=0.3)

# Set up log scale ticks to show 1,2,3...10,20,30...100,200,300... automatically
from matplotlib.ticker import LogLocator, ScalarFormatter, FixedLocator
ax = plt.gca()

# Ensure we have major ticks at 1, 10, 100, etc.
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)

# Force display of 10 by setting specific ticks
yticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.yticks(yticks)

# Set y-axis limits to ensure all data is visible
min_val = min(num_zero_eigenvalues) if num_zero_eigenvalues else 1
max_val = max(num_zero_eigenvalues) if num_zero_eigenvalues else 100

# Extend range to include values below 1 if needed
if min_val < 1:
    plt.ylim(0.1, max_val * 1.2)
else:
    plt.ylim(min_val * 0.8, max_val * 1.2)

# Add text showing total dimension
plt.text(0.02, 0.95, f'Hidden size: {hidden_size}', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('residual_stream_zero_eigenvalues.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nNumber of zero eigenvalues statistics:")
print(f"  Start (Layer 0): {num_zero_eigenvalues[0]}")
print(f"  End (Layer {sampled_layers[-1]}): {num_zero_eigenvalues[-1]}")
print(f"  Min: {min(num_zero_eigenvalues)}")
print(f"  Max: {max(num_zero_eigenvalues)}")

# %%
# Analyze if LoRA B directions align with null space
print("\n=== Analyzing LoRA B Direction Alignment with Null Space ===")

# Storage for alignment scores
lora_null_space_alignment = []
lora_used_space_alignment = []
analyzed_layers = []

# Hook function to analyze LoRA B alignment with eigenspaces
def make_alignment_hook(layer_idx):
    def hook(module, input, output):
        # output is the residual stream after attention + residual connection
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        # Compute covariance matrix
        residual_centered = residual - residual.mean(dim=0, keepdim=True)
        cov_matrix = torch.matmul(residual_centered.T, residual_centered) / (residual.shape[0] - 1)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix.float())
        eigenvalues_np = eigenvalues.cpu().numpy()
        eigenvectors_np = eigenvectors.cpu().numpy()
        
        # Identify null space (zero eigenvalue directions)
        max_eigenvalue = np.max(np.abs(eigenvalues_np))
        threshold = 1e-6 * max_eigenvalue
        null_space_mask = np.abs(eigenvalues_np) < threshold
        
        if layer_idx in lora_b_directions and np.sum(null_space_mask) > 0:
            # Get LoRA B direction
            lora_b_dir = lora_b_directions[layer_idx].cpu().numpy()
            lora_b_dir_normalized = lora_b_dir / np.linalg.norm(lora_b_dir)
            
            # Project LoRA B onto null space
            null_space_eigenvectors = eigenvectors_np[:, null_space_mask]
            null_space_projection = np.linalg.norm(null_space_eigenvectors.T @ lora_b_dir_normalized)**2
            
            # Project LoRA B onto used space (non-null)
            used_space_eigenvectors = eigenvectors_np[:, ~null_space_mask]
            used_space_projection = np.linalg.norm(used_space_eigenvectors.T @ lora_b_dir_normalized)**2
            
            lora_null_space_alignment.append(null_space_projection)
            lora_used_space_alignment.append(used_space_projection)
            analyzed_layers.append(layer_idx)
    return hook

# Register hooks for alignment analysis
hooks = []
for layer_idx in range(n_layers):
    if layer_idx % 2 == 0:  # Sample every other layer
        hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
            make_alignment_hook(layer_idx)
        )
        hooks.append(hook)

# %%
# Forward pass for alignment analysis
print("Running forward pass for alignment analysis...")
lora_null_space_alignment = []
lora_used_space_alignment = []
analyzed_layers = []

with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Plot alignment results
plt.figure(figsize=(14, 6))

# Plot 1: Alignment with null space vs used space
plt.subplot(1, 2, 1)
plt.plot(analyzed_layers, lora_null_space_alignment, 'ro-', label='Null space alignment', linewidth=2, markersize=6)
plt.plot(analyzed_layers, lora_used_space_alignment, 'bo-', label='Used space alignment', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('Alignment (projection squared)')
plt.title('LoRA B Direction Alignment with Residual Stream Subspaces')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Percentage in null space
plt.subplot(1, 2, 2)
null_space_percentage = np.array(lora_null_space_alignment) / (np.array(lora_null_space_alignment) + np.array(lora_used_space_alignment) + 1e-10) * 100
plt.plot(analyzed_layers, null_space_percentage, 'go-', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('% of LoRA B in Null Space')
plt.title('Percentage of LoRA B Direction in Residual Stream Null Space')
plt.grid(True, alpha=0.3)
plt.ylim(0, 105)

plt.tight_layout()
plt.savefig('lora_b_null_space_alignment.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nLoRA B null space alignment statistics:")
print(f"  Mean % in null space: {np.mean(null_space_percentage):.1f}%")
print(f"  Max % in null space: {np.max(null_space_percentage):.1f}% (Layer {analyzed_layers[np.argmax(null_space_percentage)]})")
print(f"  Min % in null space: {np.min(null_space_percentage):.1f}% (Layer {analyzed_layers[np.argmin(null_space_percentage)]})")

# %%
# Directly compute variance of residual stream in LoRA B direction
print("\n=== Analyzing Variance of Residual Stream in LoRA B Direction ===")

# Storage for variance analysis
lora_b_variances = []
lora_b_means = []
residual_total_variances = []
variance_layers = []

# Hook function to compute variance in LoRA B direction
def make_variance_hook(layer_idx):
    def hook(module, input, output):
        # output is the residual stream BEFORE MLP writes to it
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        if layer_idx in lora_b_directions:
            # Get LoRA B direction
            lora_b_dir = lora_b_directions[layer_idx]
            lora_b_dir_normalized = lora_b_dir / torch.norm(lora_b_dir)
            
            # Project residual onto LoRA B direction
            projections = torch.matmul(residual.float(), lora_b_dir_normalized.float())  # [seq_len]
            projections_np = projections.cpu().numpy()
            
            # Compute mean and variance
            mean_proj = np.mean(projections_np)
            var_proj = np.var(projections_np)
            
            # Also compute total variance of residual stream for comparison
            residual_float = residual.float().cpu().numpy()
            total_var = np.mean(np.var(residual_float, axis=0))  # Average variance across all dimensions
            
            lora_b_variances.append(var_proj)
            lora_b_means.append(mean_proj)
            residual_total_variances.append(total_var)
            variance_layers.append(layer_idx)
    return hook

# Register hooks for variance analysis
hooks = []
for layer_idx in range(n_layers):
    if layer_idx in lora_b_directions:  # Only analyze layers with LoRA
        hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
            make_variance_hook(layer_idx)
        )
        hooks.append(hook)

# %%
# Forward pass for variance analysis
print("Running forward pass for variance analysis...")
lora_b_variances = []
lora_b_means = []
residual_total_variances = []
variance_layers = []

with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Plot variance results
plt.figure(figsize=(16, 10))

# Plot 1: Variance in LoRA B direction
plt.subplot(2, 2, 1)
plt.plot(variance_layers, lora_b_variances, 'ro-', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('Variance')
plt.title('Variance of Residual Stream in LoRA B Direction')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Mean squared vs Variance
plt.subplot(2, 2, 2)
mean_squared = np.array(lora_b_means)**2
plt.scatter(variance_layers, mean_squared, label='Mean²', alpha=0.7, s=50)
plt.scatter(variance_layers, lora_b_variances, label='Variance', alpha=0.7, s=50)
plt.xlabel('Layer')
plt.ylabel('Value')
plt.title('Mean² vs Variance in LoRA B Direction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 3: Relative variance (compared to average dimension)
plt.subplot(2, 2, 3)
relative_variance = np.array(lora_b_variances) / np.array(residual_total_variances)
plt.plot(variance_layers, relative_variance, 'go-', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('Relative Variance')
plt.title('LoRA B Direction Variance / Average Dimension Variance')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 4: Coefficient of variation (std/mean)
plt.subplot(2, 2, 4)
coeff_variation = np.sqrt(lora_b_variances) / (np.abs(lora_b_means) + 1e-10)
plt.plot(variance_layers, coeff_variation, 'mo-', linewidth=2, markersize=6)
plt.xlabel('Layer')
plt.ylabel('Coefficient of Variation')
plt.title('Std Dev / |Mean| in LoRA B Direction')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('lora_b_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Print statistics
print(f"\nLoRA B direction variance statistics:")
print(f"  Layers analyzed: {len(variance_layers)}")
print(f"\nVariance in LoRA B direction:")
print(f"  Min: {np.min(lora_b_variances):.2e} (Layer {variance_layers[np.argmin(lora_b_variances)]})")
print(f"  Max: {np.max(lora_b_variances):.2e} (Layer {variance_layers[np.argmax(lora_b_variances)]})")
print(f"  Median: {np.median(lora_b_variances):.2e}")

print(f"\nRelative variance (vs average dimension):")
print(f"  Min: {np.min(relative_variance):.2e} (Layer {variance_layers[np.argmin(relative_variance)]})")
print(f"  Max: {np.max(relative_variance):.2e} (Layer {variance_layers[np.argmax(relative_variance)]})")
print(f"  Median: {np.median(relative_variance):.2e}")

# Count how many layers have "low" variance
low_var_threshold = 0.01  # 1% of average dimension variance
n_low_var = np.sum(relative_variance < low_var_threshold)
print(f"\nLayers with relative variance < {low_var_threshold}: {n_low_var}/{len(variance_layers)} ({n_low_var/len(variance_layers)*100:.1f}%)")

# %%
# Baseline comparison: Top eigenvectors
print("\n=== Baseline: Top Eigenvector Projections ===")

# Storage for projections
n_top_eigenvectors = 5
top_eigenvector_projections = {layer: [] for layer in range(n_layers)}
lora_b_abs_projections = []
eigenvector_cache = {}

# Hook function to compute absolute projections onto top eigenvectors and LoRA B
def make_eigenvector_projection_hook(layer_idx):
    def hook(module, input, output):
        # output is the residual stream after attention + residual connection
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        # Compute covariance and eigenvectors (only once per layer)
        if layer_idx not in eigenvector_cache:
            residual_centered = residual - residual.mean(dim=0, keepdim=True)
            cov_matrix = torch.matmul(residual_centered.T, residual_centered) / (residual.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix.float())
            # Sort by eigenvalue (descending)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvector_cache[layer_idx] = (eigenvectors, eigenvalues)
        else:
            eigenvectors, eigenvalues = eigenvector_cache[layer_idx]
        
        # Project onto top eigenvectors and take mean of absolute values
        for i in range(min(n_top_eigenvectors, eigenvectors.shape[1])):
            eigenvec = eigenvectors[:, i]
            projections = torch.matmul(residual.float(), eigenvec)  # [seq_len]
            mean_abs_proj = torch.abs(projections).mean().item()
            top_eigenvector_projections[layer_idx].append(mean_abs_proj)
        
        # Also compute for LoRA B direction if it exists
        if layer_idx in lora_b_directions:
            lora_b_dir = lora_b_directions[layer_idx]
            lora_b_dir_normalized = lora_b_dir / torch.norm(lora_b_dir)
            projections = torch.matmul(residual.float(), lora_b_dir_normalized.float())
            mean_abs_proj = torch.abs(projections).mean().item()
            lora_b_abs_projections.append((layer_idx, mean_abs_proj))
    return hook

# Register hooks for eigenvector analysis
hooks = []
for layer_idx in range(n_layers):
    if layer_idx % 2 == 0:  # Sample every other layer to save computation
        hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
            make_eigenvector_projection_hook(layer_idx)
        )
        hooks.append(hook)

# %%
# Forward pass with eigenvector baseline
print("Running forward pass for eigenvector baseline...")
lora_b_abs_projections = []
eigenvector_cache = {}
with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Process LoRA B absolute projections
lora_b_abs_by_layer = {layer: 0.0 for layer in range(n_layers)}
for layer, abs_proj in lora_b_abs_projections:
    lora_b_abs_by_layer[layer] = abs_proj

# %%
# Compare absolute projections
plt.figure(figsize=(14, 6))

# Plot absolute projections
# LoRA B absolute projections
lora_layers = sorted([layer for layer, _ in lora_b_abs_projections])
lora_abs_values = [lora_b_abs_by_layer[layer] for layer in lora_layers]
plt.plot(lora_layers, lora_abs_values, 'ro-', label='LoRA B direction', linewidth=2, markersize=6)

# Top eigenvector projections
sampled_layers = sorted(top_eigenvector_projections.keys())
for i in range(n_top_eigenvectors):
    eigenvec_trace = []
    for layer in sampled_layers:
        if len(top_eigenvector_projections[layer]) > i:
            eigenvec_trace.append(top_eigenvector_projections[layer][i])
        else:
            eigenvec_trace.append(np.nan)
    plt.plot(sampled_layers, eigenvec_trace, '--', marker='s', markersize=4, 
             label=f'Eigenvector {i+1}', alpha=0.7, linewidth=1.5)

plt.xlabel('Layer')
plt.ylabel('Mean |Projection|')
plt.title('Mean Absolute Projections: LoRA B vs Top Eigenvectors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lora_b_vs_eigenvector_projections.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Statistical comparison
print("\nStatistical comparison (absolute projections):")

# Get LoRA values for sampled layers only
sampled_lora_values = [lora_b_abs_by_layer[layer] for layer in sampled_layers if layer in lora_b_directions]
sampled_lora_layers = [layer for layer in sampled_layers if layer in lora_b_directions]

# Compare with top eigenvector
top_eigenvec_values = [top_eigenvector_projections[layer][0] if top_eigenvector_projections[layer] else 0 
                       for layer in sampled_lora_layers]

print(f"Layers where LoRA B < Top eigenvector: {sum(a < b for a, b in zip(sampled_lora_values, top_eigenvec_values))}/{len(sampled_lora_values)}")

# Compute ratios
lora_to_top_eigenvec_ratio = np.array(sampled_lora_values) / (np.array(top_eigenvec_values) + 1e-10)
print(f"\nRatio LoRA B / Top eigenvector:")
print(f"  Min: {lora_to_top_eigenvec_ratio.min():.3f}")
print(f"  Max: {lora_to_top_eigenvec_ratio.max():.3f}")
print(f"  Median: {np.median(lora_to_top_eigenvec_ratio):.3f}")
print(f"  Mean: {lora_to_top_eigenvec_ratio.mean():.3f}")

# Also show average projections for each eigenvector
print(f"\nAverage projections across layers:")
for i in range(n_top_eigenvectors):
    avg_proj = np.mean([top_eigenvector_projections[layer][i] 
                        for layer in sampled_layers 
                        if len(top_eigenvector_projections[layer]) > i])
    print(f"  Eigenvector {i+1}: {avg_proj:.3f}")
lora_avg = np.mean([lora_b_abs_by_layer[layer] for layer in sampled_layers if layer in lora_b_directions])
print(f"  LoRA B: {lora_avg:.3f}")

# %%
# Track each LoRA B direction through all subsequent layers
print("\n=== Tracking LoRA B Directions Through Layers ===")

# Storage for projections of each LoRA B direction at all layers
lora_b_tracking = {}  # {source_layer: {target_layer: mean_abs_projection}}

# Hook function to track all LoRA B directions
def make_tracking_hook(layer_idx, all_lora_b_directions):
    def hook(module, input, output):
        # output is the residual stream after attention + residual connection
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        # Project onto each LoRA B direction from earlier layers
        for source_layer, lora_b_dir in all_lora_b_directions.items():
            if source_layer <= layer_idx:  # Only track from where it's written onwards
                lora_b_dir_normalized = lora_b_dir / torch.norm(lora_b_dir)
                projections = torch.matmul(residual.float(), lora_b_dir_normalized.float())
                mean_abs_proj = torch.abs(projections).mean().item()
                
                if source_layer not in lora_b_tracking:
                    lora_b_tracking[source_layer] = {}
                lora_b_tracking[source_layer][layer_idx] = mean_abs_proj
    return hook

# Register hooks for tracking
hooks = []
for layer_idx in range(n_layers):
    hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
        make_tracking_hook(layer_idx, lora_b_directions)
    )
    hooks.append(hook)

# %%
# Forward pass for tracking
print("Running forward pass for tracking analysis...")
lora_b_tracking = {}
with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Plot tracking results
plt.figure(figsize=(14, 8), dpi=300)

# Plot each LoRA B direction's journey through the network
colors = plt.cm.viridis(np.linspace(0, 1, len(lora_b_directions)))
for idx, (source_layer, tracking_data) in enumerate(sorted(lora_b_tracking.items())):
    layers = sorted(tracking_data.keys())
    values = [tracking_data[l] for l in layers]
    
    # Only plot from where the direction is written
    valid_layers = [l for l in layers if l >= source_layer]
    valid_values = [tracking_data[l] for l in valid_layers]
    
    plt.plot(valid_layers, valid_values, '-', color=colors[idx], 
             linewidth=2, alpha=0.7, label=f'LoRA B from layer {source_layer}')
    
    # Mark the starting point
    plt.scatter([source_layer], [tracking_data[source_layer]], 
                color=colors[idx], s=100, zorder=5, edgecolor='black', linewidth=1)

plt.xlabel('Layer')
plt.ylabel('Mean |Projection|')
plt.title('Evolution of LoRA B Direction Projections Through Network')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('lora_b_direction_tracking.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Analyze persistence of directions
print("\nDirection persistence analysis:")
for source_layer in sorted(lora_b_tracking.keys()):
    tracking_data = lora_b_tracking[source_layer]
    # Get projection at source layer and final layer
    source_proj = tracking_data[source_layer]
    final_layers = [l for l in tracking_data.keys() if l > source_layer]
    if final_layers:
        final_proj = tracking_data[max(final_layers)]
        ratio = final_proj / source_proj if source_proj > 0 else 0
        print(f"  Layer {source_layer}: {source_proj:.3f} → {final_proj:.3f} (ratio: {ratio:.2f})")

# %%
# Track projections at a single token position
print("\n=== Single Token Position Analysis ===")

# Choose a token position to track (e.g., middle of sequence)
token_position = seq_len // 2 + 5
print(f"Tracking token position {token_position} (out of {seq_len} total tokens)")

# Storage for single token projections
single_token_tracking = {}  # {source_layer: {target_layer: projection}}

# Hook function to track single token projections
def make_single_token_hook(layer_idx, all_lora_b_directions, token_pos):
    def hook(module, input, output):
        # output is the residual stream after attention + residual connection
        residual = output[0].detach()  # [seq_len, hidden_size]
        
        # Get residual at specific token position
        residual_at_token = residual[token_pos]  # [hidden_size]
        
        # Project onto each LoRA B direction from earlier layers
        for source_layer, lora_b_dir in all_lora_b_directions.items():
            if source_layer <= layer_idx:  # Only track from where it's written onwards
                lora_b_dir_normalized = lora_b_dir / torch.norm(lora_b_dir)
                projection = torch.dot(residual_at_token.float(), lora_b_dir_normalized.float()).item()
                
                if source_layer not in single_token_tracking:
                    single_token_tracking[source_layer] = {}
                single_token_tracking[source_layer][layer_idx] = projection
    return hook

# Register hooks for single token tracking
hooks = []
for layer_idx in range(n_layers):
    hook = model.model.model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
        make_single_token_hook(layer_idx, lora_b_directions, token_position)
    )
    hooks.append(hook)

# %%
# Forward pass for single token tracking
print("Running forward pass for single token analysis...")
single_token_tracking = {}
with torch.no_grad():
    outputs = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# %%
# Plot single token tracking results
plt.figure(figsize=(14, 8))

# Plot each LoRA B direction's projection at single token
colors = plt.cm.viridis(np.linspace(0, 1, len(lora_b_directions)))
for idx, (source_layer, tracking_data) in enumerate(sorted(single_token_tracking.items())):
    layers = sorted(tracking_data.keys())
    values = [tracking_data[l] for l in layers]
    
    # Only plot from where the direction is written
    valid_layers = [l for l in layers if l >= source_layer]
    valid_values = [tracking_data[l] for l in valid_layers]
    
    plt.plot(valid_layers, valid_values, '-', color=colors[idx], 
             linewidth=2, alpha=0.7, label=f'LoRA B from layer {source_layer}')
    
    # Mark the starting point
    plt.scatter([source_layer], [tracking_data[source_layer]], 
                color=colors[idx], s=100, zorder=5, edgecolor='black', linewidth=1)

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Layer')
plt.ylabel('Projection (not absolute)')
plt.title(f'Single Token (Position {token_position}) Projection onto LoRA B Directions')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('lora_b_single_token_tracking.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Decode the token we're tracking
tracked_token_id = input_ids[0, token_position].item()
tracked_token = tokenizer.decode([tracked_token_id])
print(f"\nTracked token: '{tracked_token}' (ID: {tracked_token_id})")

# Show the projection values at key points
print("\nProjection values for selected layers:")
for source_layer in [43, 52, 58]:  # Key layers from earlier analysis
    if source_layer in single_token_tracking:
        tracking_data = single_token_tracking[source_layer]
        initial_proj = tracking_data.get(source_layer, 0)
        if initial_proj != 0:
            print(f"\nLayer {source_layer} LoRA B direction:")
            print(f"  At layer {source_layer}: {initial_proj:.3f}")
            for offset in [5, 10, 15]:
                target_layer = source_layer + offset
                if target_layer in tracking_data:
                    proj = tracking_data[target_layer]
                    print(f"  At layer {target_layer} (+{offset}): {proj:.3f}")

# %%
# Compute projection matrix between all LoRA B and down_proj weights
print("\n=== Projection Matrix: LoRA B × Down Projection Weights ===")

# First extract all down_proj weight matrices
down_proj_weights = {}
for layer_idx in range(n_layers):
    down_proj = model.model.model.layers[layer_idx].mlp.down_proj
    # Extract the base weight matrix (not LoRA)
    base_weight = down_proj.base_layer.weight.data.float()  # [hidden_size, intermediate_size]
    down_proj_weights[layer_idx] = base_weight

# Create projection matrix
layers_with_lora = sorted(lora_b_directions.keys())
projection_matrix = np.zeros((len(layers_with_lora), n_layers))

for i, lora_layer in enumerate(layers_with_lora):
    lora_b_dir = lora_b_directions[lora_layer].float()  # [hidden_size]
    
    for j, down_proj_layer in enumerate(range(n_layers)):
        down_proj_weight = down_proj_weights[down_proj_layer]  # [hidden_size, intermediate_size]
        
        # Compute projection: sum(LoRA_B * down_proj)
        # Element-wise multiply and sum over all dimensions
        projection = torch.sum(lora_b_dir.unsqueeze(1) * down_proj_weight).item()
        projection_matrix[i, j] = projection

# %%
# Plot projection matrix heatmap
plt.figure(figsize=(16, 12))
vmax = np.percentile(np.abs(projection_matrix), 99)  # Use 99th percentile for better contrast
sns.heatmap(projection_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=5,  # Show every 5th layer
            yticklabels=[f'LoRA L{l}' for l in layers_with_lora],
            cbar_kws={'label': 'Projection Value'})
plt.xlabel('Down-Projection Layer')
plt.ylabel('LoRA B Layer')
plt.title('Projection Matrix: LoRA B • Down-Projection Weights')

# Add diagonal line for reference
for i, lora_layer in enumerate(layers_with_lora):
    if lora_layer < n_layers:
        plt.axhline(y=i+0.5, xmin=lora_layer/n_layers, xmax=(lora_layer+1)/n_layers, 
                   color='green', linewidth=2, alpha=0.5)
        plt.axvline(x=lora_layer+0.5, ymin=i/len(layers_with_lora), ymax=(i+1)/len(layers_with_lora), 
                   color='green', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('lora_b_downproj_projection_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Analyze diagonal vs off-diagonal
diagonal_values = []
off_diagonal_values = []

for i, lora_layer in enumerate(layers_with_lora):
    for j in range(n_layers):
        value = projection_matrix[i, j]
        if j == lora_layer:
            diagonal_values.append(value)
        else:
            off_diagonal_values.append(value)

plt.figure(figsize=(10, 6))
plt.hist(diagonal_values, bins=30, alpha=0.7, label=f'Diagonal (n={len(diagonal_values)})', density=True)
plt.hist(off_diagonal_values, bins=100, alpha=0.7, label=f'Off-diagonal (n={len(off_diagonal_values)})', density=True)
plt.xlabel('Projection Value')
plt.ylabel('Density')
plt.title('Distribution of Diagonal vs Off-Diagonal Projections')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diagonal_vs_offdiagonal_projections.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Print statistics
print("\nProjection Matrix Statistics:")
print(f"Shape: {projection_matrix.shape}")
print(f"Overall stats:")
print(f"  Mean: {np.mean(projection_matrix):.3f}")
print(f"  Std: {np.std(projection_matrix):.3f}")
print(f"  Min: {np.min(projection_matrix):.3f}")
print(f"  Max: {np.max(projection_matrix):.3f}")

print(f"\nDiagonal projections (LoRA B layer i → down_proj layer i):")
print(f"  Mean: {np.mean(diagonal_values):.3f}")
print(f"  Std: {np.std(diagonal_values):.3f}")
print(f"  Min: {np.min(diagonal_values):.3f}")
print(f"  Max: {np.max(diagonal_values):.3f}")

print(f"\nOff-diagonal projections:")
print(f"  Mean: {np.mean(off_diagonal_values):.3f}")
print(f"  Std: {np.std(off_diagonal_values):.3f}")
print(f"  % near zero (|proj| < 1): {np.mean(np.abs(off_diagonal_values) < 1)*100:.1f}%")

# %%
# Compute projection matrix between LoRA B and up_proj weights
print("\n=== Projection Matrix: LoRA B × Up-Projection Weights ===")

# Extract all up_proj weight matrices
up_proj_weights = {}
for layer_idx in range(n_layers):
    up_proj = model.model.model.layers[layer_idx].mlp.up_proj
    # Extract the base weight matrix (not LoRA)
    base_weight = up_proj.base_layer.weight.data.float()  # [intermediate_size, hidden_size]
    up_proj_weights[layer_idx] = base_weight

# Create projection matrix for up_proj
up_projection_matrix = np.zeros((len(layers_with_lora), n_layers))

for i, lora_layer in enumerate(layers_with_lora):
    lora_b_dir = lora_b_directions[lora_layer].float()  # [hidden_size]
    
    for j, up_proj_layer in enumerate(range(n_layers)):
        up_proj_weight = up_proj_weights[up_proj_layer]  # [intermediate_size, hidden_size]
        
        # Compute projection: sum(LoRA_B * up_proj)
        # up_proj: [intermediate_size, hidden_size]
        # lora_b_dir: [hidden_size]
        # We want to see how much each intermediate neuron reads from LoRA B direction
        # So we compute: up_proj @ lora_b_dir, then sum
        projections_per_neuron = torch.matmul(up_proj_weight, lora_b_dir)  # [intermediate_size]
        projection = torch.sum(projections_per_neuron).item()
        up_projection_matrix[i, j] = projection

# %%
# Plot up_proj projection matrix heatmap
plt.figure(figsize=(16, 12))
vmax = np.percentile(np.abs(up_projection_matrix), 99)
sns.heatmap(up_projection_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=5,
            yticklabels=[f'LoRA L{l}' for l in layers_with_lora],
            cbar_kws={'label': 'Projection Value'})
plt.xlabel('Up-Projection Layer (MLP that reads)')
plt.ylabel('LoRA B Layer (writes to residual)')
plt.title('Projection Matrix: LoRA B • Up-Projection Weights\n(Shows which MLPs read from LoRA-written directions)')

# Add diagonal line for reference
for i, lora_layer in enumerate(layers_with_lora):
    if lora_layer < n_layers:
        plt.axhline(y=i+0.5, xmin=lora_layer/n_layers, xmax=(lora_layer+1)/n_layers, 
                   color='green', linewidth=2, alpha=0.5)
        plt.axvline(x=lora_layer+0.5, ymin=i/len(layers_with_lora), ymax=(i+1)/len(layers_with_lora), 
                   color='green', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig('lora_b_upproj_projection_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Compare down_proj and up_proj patterns
plt.figure(figsize=(16, 6))

# Subplot 1: Down-projection
plt.subplot(1, 2, 1)
sns.heatmap(projection_matrix,
            cmap='RdBu_r',
            center=0,
            vmax=np.percentile(np.abs(projection_matrix), 99),
            vmin=-np.percentile(np.abs(projection_matrix), 99),
            xticklabels=10,
            yticklabels=[f'L{l}' for l in layers_with_lora],
            cbar_kws={'label': 'Projection'})
plt.xlabel('Layer')
plt.ylabel('LoRA B Layer')
plt.title('LoRA B → Down-Proj')

# Subplot 2: Up-projection
plt.subplot(1, 2, 2)
sns.heatmap(up_projection_matrix,
            cmap='RdBu_r',
            center=0,
            vmax=np.percentile(np.abs(up_projection_matrix), 99),
            vmin=-np.percentile(np.abs(up_projection_matrix), 99),
            xticklabels=10,
            yticklabels=[f'L{l}' for l in layers_with_lora],
            cbar_kws={'label': 'Projection'})
plt.xlabel('Layer')
plt.ylabel('LoRA B Layer')
plt.title('LoRA B → Up-Proj')

plt.tight_layout()
plt.savefig('lora_b_proj_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Analyze which layers read from LoRA directions
print("\nUp-Projection Analysis (which MLPs read LoRA directions):")
for i, lora_layer in enumerate(layers_with_lora):
    projections = up_projection_matrix[i, :]
    # Find layers with significant projections
    significant_readers = np.where(np.abs(projections) > np.std(projections) * 2)[0]
    
    print(f"\nLoRA B from layer {lora_layer}:")
    print(f"  Strongly read by layers: {significant_readers.tolist()}")
    print(f"  Max projection at layer {np.argmax(np.abs(projections))}: {projections[np.argmax(np.abs(projections))]:.2f}")
    
    # Check if subsequent layers read it
    if lora_layer < n_layers - 1:
        subsequent_projections = projections[lora_layer+1:]
        if len(subsequent_projections) > 0:
            print(f"  Mean |projection| in subsequent layers: {np.mean(np.abs(subsequent_projections)):.2f}")
# %%

