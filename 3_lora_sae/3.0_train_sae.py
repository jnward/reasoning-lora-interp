# %% [markdown]
# # SAE Training Script - One Epoch
# 
# This notebook trains a BatchTopKSAE on activation vectors from rollout data.

# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import h5py
from glob import glob
import os
from tqdm import tqdm
import time
import wandb
import random
from batch_topk_sae import BatchTopKSAE

# %% Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Dead Latent Tracker
class LatentTracker:
    """Track which latent features are active during training."""
    
    def __init__(self, n_features, device, dead_threshold=1e7):
        self.n_features = n_features
        self.device = device
        self.dead_threshold = dead_threshold
        
        # Track last seen position for each feature
        self.last_seen_at = torch.zeros(n_features, device=device)
        self.total_tokens_seen = 0
    
    def update(self, sparse_activations):
        """Update last seen positions based on current batch."""
        # sparse_activations: [batch_size, n_features]
        # Find which features are active in this batch
        feature_active = (sparse_activations != 0).any(dim=0)
        
        # Update last seen position for active features
        self.last_seen_at[feature_active] = self.total_tokens_seen
        
        # Update total tokens seen
        self.total_tokens_seen += sparse_activations.shape[0]
    
    def get_dead_latents(self):
        """Get indices of dead latents based on threshold."""
        # Calculate how long since each feature was last seen
        tokens_since_last_seen = self.total_tokens_seen - self.last_seen_at
        
        # Dead latents are those not seen in the last dead_threshold tokens
        dead_mask = tokens_since_last_seen > self.dead_threshold
        dead_indices = torch.where(dead_mask)[0]
        
        # For compatibility, also return activation rates (tokens since last seen / total tokens)
        activation_rates = 1.0 - (tokens_since_last_seen / max(1, self.total_tokens_seen))
        
        return dead_indices, activation_rates
    
    def reset(self):
        """Reset tracking statistics."""
        self.last_seen_at.zero_()
        self.total_tokens_seen = 0

# %% Auxiliary Loss Function
def auxiliary_loss(dead_latents, error, model, k, alpha=1/32):
    """
    Compute auxiliary loss to reactivate dead latents.
    
    Args:
        dead_latents: Tensor of dead latent indices
        error: Reconstruction error (x - x_hat)
        model: The SAE model
        k: Top-k value for dead latent selection
        alpha: Scaling coefficient for auxiliary loss
    """
    if len(dead_latents) == 0:
        return torch.tensor(0.0, device=error.device)
    
    # Get pre-activations for dead latents only
    # error: [batch_size, d_model]
    # W_enc[:, dead_latents]: [d_model, n_dead]
    dead_pre_activations = torch.einsum(
        "bd,dn->bn", 
        error, 
        model.W_enc[:, dead_latents]
    ) + model.b_enc[dead_latents]
    
    # Calculate decoder norms for dead latents
    dead_decoder_norms = torch.norm(model.W_dec[dead_latents], dim=1)
    
    # Calculate value scores for dead latents
    dead_latent_values = dead_pre_activations * dead_decoder_norms
    
    # Select top k dead latents per sample
    k_dead = min(k, dead_latents.shape[0])
    topk_values, topk_indices = torch.topk(
        dead_latent_values, k=k_dead, dim=1
    )
    
    # Create mask for selected dead latents
    mask = torch.zeros_like(dead_pre_activations)
    batch_indices = torch.arange(mask.shape[0], device=mask.device).unsqueeze(1)
    mask[batch_indices, topk_indices] = 1.0
    
    # Apply mask to get sparse dead activations
    masked_dead_pre_activations = dead_pre_activations * mask
    
    # Reconstruct using only dead latents
    # masked_dead_pre_activations: [batch_size, n_dead]
    # W_dec[dead_latents]: [n_dead, d_model]
    dead_reconstruction = torch.einsum(
        "bn,nd->bd",
        masked_dead_pre_activations,
        model.W_dec[dead_latents]
    )
    
    # MSE between dead reconstruction and error
    aux_loss = nn.functional.mse_loss(dead_reconstruction, error)
    
    return alpha * aux_loss

# %% Learning rate scheduler with warmup and decay
def get_schedule_with_warmup_and_decay(optimizer, num_warmup_steps, total_steps):
    """
    Create a schedule with:
    - Linear warmup for num_warmup_steps
    - Constant LR until half of total steps
    - 10x reduction after half of training
    """
    # half_steps = total_steps * 0.9
    half_steps = total_steps * 1.1 # don't decay ever
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < half_steps:
            # Constant phase
            return 1.0
        else:
            # Decay phase - reduce by 10x
            return 0.1
    
    return LambdaLR(optimizer, lr_lambda)

# %% Load all activations
def load_all_activations(activation_dir, adapter_types=None):
    """Load all activations from H5 files and flatten them.
    
    Args:
        activation_dir: Directory containing H5 files
        adapter_types: List of adapter types to include (default: all 7)
                      Options: ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    """
    print("Loading all activations into memory...")
    
    # Default to all adapter types
    if adapter_types is None:
        adapter_types = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    # Map adapter names to indices
    all_adapters = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    adapter_indices = [all_adapters.index(a) for a in adapter_types]
    
    all_activations = []
    h5_files = sorted(glob(os.path.join(activation_dir, 'rollout_*.h5')))
    
    # Check first file to determine format
    with h5py.File(h5_files[0], 'r') as f:
        projections = f.attrs.get('projections', 3)  # Default to 3 for backward compatibility
        num_layers = f.attrs['num_layers']
    
    print(f"Detected {projections} projections per layer")
    print(f"Loading adapters: {adapter_types}")
    
    for file_path in tqdm(h5_files, desc="Loading files"):
        with h5py.File(file_path, 'r') as f:
            # Load activations: shape (n_tokens, num_layers, projections)
            acts = f['activations'][:]
            
            if projections == 3 and len(adapter_indices) > 3:
                # Old format - only has MLP adapters
                print("Warning: Old activation format detected. Only MLP adapters available.")
                adapter_indices = [i for i in adapter_indices if i < 3]
                if not adapter_indices:
                    raise ValueError("No MLP adapters requested but file only contains MLP activations")
            
            # Select only requested adapters
            acts_selected = acts[:, :, adapter_indices]
            
            # Flatten to (n_tokens, num_layers * len(adapter_types))
            acts_flat = acts_selected.reshape(-1, num_layers * len(adapter_indices))
            all_activations.append(acts_flat)
    
    # Verify adapter consistency
    if len(adapter_indices) != len(adapter_types):
        print(f"\nWarning: Only {len(adapter_indices)} of {len(adapter_types)} requested adapters found in data.")
        print(f"Requested: {adapter_types}")
        print(f"Available indices: {adapter_indices} (from {all_adapters[:projections]})")
    
    # Concatenate all activations
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"Loaded {len(all_activations):,} activation vectors")
    print(f"Shape: {all_activations.shape}")
    print(f"Memory usage: {all_activations.nbytes / 1e9:.2f} GB")
    
    return all_activations, num_layers * len(adapter_indices)

# %% Simple DataLoader
class SimpleDataLoader:
    """Simple dataloader that shuffles data once and iterates in batches."""
    
    def __init__(self, activations, batch_size=1024, shuffle=True, device='cpu'):
        # Convert to float32 tensor (from float16)
        self.data = torch.from_numpy(activations).float()
        self.device = device
        
        if shuffle:
            print("Shuffling data...")
            indices = torch.randperm(len(self.data))
            self.data = self.data[indices]
        
        self.batch_size = batch_size
        self.num_batches = (len(self.data) + batch_size - 1) // batch_size
    
    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size].to(self.device)
            yield batch
    
    def __len__(self):
        return self.num_batches

k = 16
lr = 5e-4
batch_size = 512
expansion_factor = 8
alpha = 1/32
seed = 42  # Random seed for reproducibility

# %% Training configuration
# Adapter types to train on (default: all)
adapter_types = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
# adapter_types = ['gate_proj', 'up_proj', 'down_proj']  # MLP only
# adapter_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Attention only

# Determine activation directory based on adapter types
if set(adapter_types) == set(['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
    # Full 7-adapter mode
    activation_dir = '../2_lora_activation_interp/activations_all_adapters'
else:
    # Custom adapter selection
    adapter_str = '-'.join([a[:1] for a in sorted(adapter_types)])
    activation_dir = f'../2_lora_activation_interp/activations_{adapter_str}'

config = {
    'activation_dir': activation_dir,
    'adapter_types': adapter_types,
    'd_model': None,  # Will be set based on loaded data
    'dict_size': None,  # Will be set based on d_model
    'k': k,  # top-k sparsity
    'batch_size': batch_size,
    'learning_rate': lr,  # Lower learning rate as in reference
    'warmup_steps': 100,
    'aux_loss_alpha': alpha,  # Auxiliary loss coefficient
    'dead_threshold': 1e6,
    'seed': seed,  # Random seed for reproducibility
    'use_wandb': True,  # Enable wandb logging
    'wandb_project': 'lora-interp',
    'wandb_run_name': None,  # Will be set based on adapter types
}

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# Check if activation directory exists
if not os.path.exists(config['activation_dir']):
    print(f"\nWarning: Activation directory '{config['activation_dir']}' not found!")
    print("Please run generate_activations_multigpu.py or generate_activations_data.py first with the appropriate adapter types.")
    print(f"Example: python generate_activations_multigpu.py --adapter-types {' '.join(adapter_types)}")
    raise FileNotFoundError(f"Activation directory not found: {config['activation_dir']}")

# %% Set random seeds for reproducibility
print(f"Setting random seed: {config['seed']}")
torch.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
# For deterministic behavior (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% Initialize wandb
if config['use_wandb']:
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config
    )
    print("Wandb initialized")

# %% Load data
activations, d_model = load_all_activations(config['activation_dir'], config['adapter_types'])

# Update config with actual d_model
config['d_model'] = d_model
config['dict_size'] = int(d_model * expansion_factor)

# Update wandb run name
adapter_str = '-'.join([a[:1] for a in sorted(config['adapter_types'])])  # e.g., 'd-g-k-o-q-u-v'
config['wandb_run_name'] = f'sae_k{k}_dict{d_model*expansion_factor}_lr{lr}_batch{batch_size}_alpha{alpha}_adapters_{adapter_str}'

print(f"\nUpdated configuration:")
print(f"  d_model: {config['d_model']}")
print(f"  dict_size: {config['dict_size']}")
print(f"  adapters: {config['adapter_types']}")

# %% Create dataloader
dataloader = SimpleDataLoader(
    activations,
    batch_size=config['batch_size'],
    shuffle=True,
    device=device
)

print(f"Created dataloader with {len(dataloader)} batches")

# %% Initialize model
model = BatchTopKSAE(
    d_model=config['d_model'],
    dict_size=config['dict_size'],
    k=config['k']
).to(device)

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# %% Define loss function
def sae_loss(model_output, x):
    """Compute SAE loss with MSE reconstruction only (sparsity handled by top-k)."""
    reconstruction = model_output['reconstruction']
    
    # MSE reconstruction loss
    mse_loss = nn.functional.mse_loss(reconstruction, x)
    
    return mse_loss

# %% Initialize optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
total_steps = len(dataloader)
scheduler = get_schedule_with_warmup_and_decay(optimizer, config['warmup_steps'], total_steps)

# %% Initialize latent tracker
latent_tracker = LatentTracker(
    n_features=config['dict_size'],
    device=device,
    dead_threshold=config['dead_threshold']
)

# %% Training loop
print("\nStarting training for 1 epoch...")
model.train()

# Track metrics
mse_loss_sum = 0
aux_loss_sum = 0
total_loss_sum = 0
fvu_sum = 0
num_batches = 0
num_dead_latents_history = []

# Track FVU over last 100 batches
from collections import deque
fvu_window = deque(maxlen=100)

# Progress bar
pbar = tqdm(dataloader, desc="Training")
start_time = time.time()

for batch_idx, batch in enumerate(pbar):
    # Forward pass
    output = model(batch)
    reconstruction = output['reconstruction']
    sparse_activations = output['sparse_latent_activations']
    
    # Compute MSE loss
    mse_loss = sae_loss(output, batch)
    
    # Compute FVU (Fraction of Variance Unexplained)
    error = reconstruction - batch
    total_variance = (batch - batch.mean(0)).pow(2).sum()
    squared_error = error.pow(2)
    fvu = squared_error.sum() / total_variance
    
    # Update latent tracker
    latent_tracker.update(sparse_activations)
    
    # Get dead latents and compute auxiliary loss
    dead_latents, activation_rates = latent_tracker.get_dead_latents()
    error = batch - reconstruction
    aux_loss = auxiliary_loss(
        dead_latents, error, model, 
        k=config['k'], 
        alpha=config['aux_loss_alpha']
    )
    
    # Total loss
    total_loss = mse_loss + aux_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Track metrics
    mse_loss_sum += mse_loss.item()
    aux_loss_sum += aux_loss.item()
    total_loss_sum += total_loss.item()
    fvu_sum += fvu.item()
    num_batches += 1
    num_dead_latents_history.append(len(dead_latents))
    
    # Add FVU to rolling window
    fvu_window.append(fvu.item())
    
    # Log to wandb every 10th batch
    if config['use_wandb'] and batch_idx % 10 == 0:
        wandb.log({
            'train/total_loss': total_loss.item(),
            'train/mse_loss': mse_loss.item(),
            'train/aux_loss': aux_loss.item(),
            'train/fvu': fvu.item(),
            'train/num_dead_latents': len(dead_latents),
            'train/dead_latent_rate': len(dead_latents) / config['dict_size'],
            'train/learning_rate': scheduler.get_last_lr()[0],
        }, step=batch_idx)
    
    # Update progress bar
    if batch_idx % 10 == 0:
        avg_mse = mse_loss_sum / num_batches
        avg_aux = aux_loss_sum / num_batches
        avg_total = total_loss_sum / num_batches
        # Use rolling window average for FVU
        avg_fvu_window = sum(fvu_window) / len(fvu_window) if fvu_window else fvu.item()
        current_lr = scheduler.get_last_lr()[0]
        
        pbar.set_postfix({
            'loss': f'{avg_total:.4f}',
            'fvu': f'{avg_fvu_window:.4f}',
            'dead': len(dead_latents),
            'lr': f'{current_lr:.2e}'
        })

# %% Training complete
end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining completed in {training_time:.1f} seconds")
print(f"Final metrics:")
print(f"  Total loss: {total_loss_sum / num_batches:.4f}")
print(f"  MSE loss: {mse_loss_sum / num_batches:.4f}")
print(f"  Auxiliary loss: {aux_loss_sum / num_batches:.6f}")
print(f"  FVU: {fvu_sum / num_batches:.4f}")
print(f"  Average dead latents: {np.mean(num_dead_latents_history):.1f}")
print(f"  Final dead latents: {num_dead_latents_history[-1]}")

# %% Analyze feature activation rates
dead_latents_final, activation_rates_final = latent_tracker.get_dead_latents()
print(f"\nFeature activation analysis:")
print(f"  Total features: {config['dict_size']}")
print(f"  Dead features: {len(dead_latents_final)}")
print(f"  Active features: {config['dict_size'] - len(dead_latents_final)}")
print(f"  Dead feature rate: {len(dead_latents_final) / config['dict_size']:.2%}")

# Log final metrics to wandb
if config['use_wandb']:
    wandb.log({
        'final/total_loss': total_loss_sum / num_batches,
        'final/mse_loss': mse_loss_sum / num_batches,
        'final/aux_loss': aux_loss_sum / num_batches,
        'final/fvu': fvu_sum / num_batches,
        'final/num_dead_latents': len(dead_latents_final),
        'final/dead_latent_rate': len(dead_latents_final) / config['dict_size'],
        'final/training_time': training_time,
    })
    
    # Log histogram of activation rates
    wandb.log({
        'final/activation_rates_histogram': wandb.Histogram(activation_rates_final.cpu().numpy()),
        'final/dead_latents_over_time': wandb.plot.line_series(
            xs=list(range(len(num_dead_latents_history))),
            ys=[num_dead_latents_history],
            keys=['Dead Latents'],
            title='Dead Latents Over Training',
            xname='Batch'
        )
    })

# %% Save model
adapter_str = '-'.join([a[:1] for a in config['adapter_types']])
save_path = f'trained_sae_adapters_{adapter_str}.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'final_metrics': {
        'total_loss': total_loss_sum / num_batches,
        'mse_loss': mse_loss_sum / num_batches,
        'aux_loss': aux_loss_sum / num_batches,
        'fvu': fvu_sum / num_batches,
        'num_dead_latents': len(dead_latents_final),
        'dead_latent_rate': len(dead_latents_final) / config['dict_size'],
        'training_time': training_time
    },
    'activation_rates': activation_rates_final.cpu().numpy(),
    'dead_latents': dead_latents_final.cpu().numpy()
}, save_path)

print(f"\nModel saved to {save_path}")

# Log model artifact to wandb
if config['use_wandb']:
    # artifact = wandb.Artifact('trained_sae', type='model')
    # artifact.add_file(save_path)
    # wandb.log_artifact(artifact)
    wandb.finish()