# %% [markdown]
# # SAE Training Script - Multi-Epoch with Train/Validation Split
# 
# This notebook trains a BatchTopKSAE on activation vectors from rollout data,
# with proper train/validation split and multi-epoch training.

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
from batch_topk_sae import BatchTopKSAE
import random

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

# %% Load train/validation split activations
def load_train_val_activations(activation_dir, adapter_types=None, validation_split=0.1, random_seed=42):
    """Load activations with train/validation split.
    
    Args:
        activation_dir: Directory containing H5 files
        adapter_types: List of adapter types to include (default: all 7)
        validation_split: Fraction of rollouts for validation (default: 0.1)
        random_seed: Random seed for reproducible splits
    """
    print(f"Loading activations with {validation_split:.0%} validation split...")
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Default to all adapter types
    if adapter_types is None:
        adapter_types = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    # Map adapter names to indices
    all_adapters = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    adapter_indices = [all_adapters.index(a) for a in adapter_types]
    
    # Get all H5 files
    h5_files = sorted(glob(os.path.join(activation_dir, 'rollout_*.h5')))
    print(f"Found {len(h5_files)} rollout files")
    
    # Check first file to determine format
    with h5py.File(h5_files[0], 'r') as f:
        projections = f.attrs.get('projections', 3)  # Default to 3 for backward compatibility
        num_layers = f.attrs['num_layers']
    
    print(f"Detected {projections} projections per layer")
    print(f"Loading adapters: {adapter_types}")
    
    # Split files into train/validation
    n_val = max(1, int(len(h5_files) * validation_split))
    val_files = random.sample(h5_files, n_val)
    train_files = [f for f in h5_files if f not in val_files]
    
    print(f"Train rollouts: {len(train_files)}")
    print(f"Validation rollouts: {len(val_files)}")
    
    def load_files(files, desc):
        """Load activations from a list of files."""
        all_activations = []
        
        for file_path in tqdm(files, desc=desc):
            with h5py.File(file_path, 'r') as f:
                # Load activations: shape (n_tokens, num_layers, projections)
                acts = f['activations'][:]
                
                if projections == 3 and len(adapter_indices) > 3:
                    # Old format - only has MLP adapters
                    print("Warning: Old activation format detected. Only MLP adapters available.")
                    adapter_indices_filtered = [i for i in adapter_indices if i < 3]
                    if not adapter_indices_filtered:
                        raise ValueError("No MLP adapters requested but file only contains MLP activations")
                else:
                    adapter_indices_filtered = adapter_indices
                
                # Select only requested adapters
                acts_selected = acts[:, :, adapter_indices_filtered]
                
                # Flatten to (n_tokens, num_layers * len(adapter_types))
                acts_flat = acts_selected.reshape(-1, num_layers * len(adapter_indices_filtered))
                all_activations.append(acts_flat)
        
        # Concatenate all activations
        all_activations = np.concatenate(all_activations, axis=0)
        return all_activations, len(adapter_indices_filtered)
    
    # Load train and validation sets
    train_activations, n_adapters = load_files(train_files, "Loading train files")
    val_activations, _ = load_files(val_files, "Loading validation files")
    
    print(f"\nTrain set: {len(train_activations):,} activation vectors")
    print(f"Validation set: {len(val_activations):,} activation vectors")
    print(f"Shape: {train_activations.shape}")
    print(f"Memory usage: {(train_activations.nbytes + val_activations.nbytes) / 1e9:.2f} GB")
    
    return train_activations, val_activations, num_layers * n_adapters

# %% Simple DataLoader
class SimpleDataLoader:
    """Simple dataloader that shuffles data once per epoch and iterates in batches."""
    
    def __init__(self, activations, batch_size=1024, shuffle=True, device='cpu'):
        # Convert to float32 tensor (from float16)
        self.data = torch.from_numpy(activations).float()
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = (len(self.data) + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.data))
            data = self.data[indices]
        else:
            data = self.data
            
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size].to(self.device)
            yield batch
    
    def __len__(self):
        return self.num_batches

# %% Evaluation function
@torch.no_grad()
def evaluate_model(model, dataloader, latent_tracker=None):
    """Evaluate model on a dataset without gradient updates."""
    model.eval()
    
    total_loss = 0
    total_fvu = 0
    num_batches = 0
    
    # Reset latent tracker if provided
    if latent_tracker is not None:
        latent_tracker.reset()
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        # Forward pass
        output = model(batch)
        reconstruction = output['reconstruction']
        sparse_activations = output['sparse_latent_activations']
        
        # Compute MSE loss
        mse_loss = nn.functional.mse_loss(reconstruction, batch)
        
        # Compute FVU
        error = reconstruction - batch
        total_variance = (batch - batch.mean(0)).pow(2).sum()
        squared_error = error.pow(2)
        fvu = squared_error.sum() / total_variance
        
        # Update metrics
        total_loss += mse_loss.item()
        total_fvu += fvu.item()
        num_batches += 1
        
        # Update latent tracker if provided
        if latent_tracker is not None:
            latent_tracker.update(sparse_activations)
    
    # Get dead latent statistics
    if latent_tracker is not None:
        dead_latents, activation_rates = latent_tracker.get_dead_latents()
        num_dead = len(dead_latents)
    else:
        num_dead = 0
        activation_rates = None
    
    model.train()
    
    return {
        'loss': total_loss / num_batches,
        'fvu': total_fvu / num_batches,
        'num_dead_latents': num_dead,
        'activation_rates': activation_rates
    }

# %% Training configuration
k = 16
lr = 5e-4
batch_size = 512
expansion_factor = 8
alpha = 1/32

# Multi-epoch specific settings
num_epochs = 5
validation_split = 0.1
random_seed = 42

# Adapter types to train on (default: all)
adapter_types = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
# adapter_types = ['gate_proj', 'up_proj', 'down_proj']  # MLP only
# adapter_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Attention only

# Determine activation directory based on adapter types
if set(adapter_types) == set(['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
    # Full 7-adapter mode - check both locations
    if os.path.exists('./activations_all_adapters'):
        activation_dir = './activations_all_adapters'
    else:
        activation_dir = '../../lora-activations-dashboard/backend/activations_all_adapters'
else:
    # Custom adapter selection
    adapter_str = '-'.join([a[:1] for a in sorted(adapter_types)])
    if os.path.exists(f'./activations_{adapter_str}'):
        activation_dir = f'./activations_{adapter_str}'
    else:
        activation_dir = f'../../lora-activations-dashboard/backend/activations_{adapter_str}'

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
    'num_epochs': num_epochs,  # Number of training epochs
    'validation_split': validation_split,  # Fraction of rollouts for validation
    'save_best_model': True,  # Save model with best validation FVU
    'random_seed': random_seed,  # For reproducible train/val splits
    'use_wandb': True,  # Enable wandb logging
    'wandb_project': 'lora-interp',
    'wandb_run_name': None,  # Will be set based on adapter types
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Check if activation directory exists
if not os.path.exists(config['activation_dir']):
    print(f"\nWarning: Activation directory '{config['activation_dir']}' not found!")
    print("Please run generate_activations_multigpu.py or generate_activations_data.py first with the appropriate adapter types.")
    print(f"Example: python generate_activations_multigpu.py --adapter-types {' '.join(adapter_types)}")
    raise FileNotFoundError(f"Activation directory not found: {config['activation_dir']}")

# %% Initialize wandb
if config['use_wandb']:
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config
    )
    print("Wandb initialized")

# %% Load data with train/validation split
train_activations, val_activations, d_model = load_train_val_activations(
    config['activation_dir'], 
    config['adapter_types'],
    validation_split=config['validation_split'],
    random_seed=config['random_seed']
)

# Update config with actual d_model
config['d_model'] = d_model
config['dict_size'] = int(d_model * expansion_factor)

# Update wandb run name
adapter_str = '-'.join([a[:1] for a in sorted(config['adapter_types'])])  # e.g., 'd-g-k-o-q-u-v'
config['wandb_run_name'] = f'sae_k{k}_dict{d_model*expansion_factor}_lr{lr}_batch{batch_size}_alpha{alpha}_epochs{config["num_epochs"]}_val{config["validation_split"]}_adapters_{adapter_str}'

print(f"\nUpdated configuration:")
print(f"  d_model: {config['d_model']}")
print(f"  dict_size: {config['dict_size']}")
print(f"  adapters: {config['adapter_types']}")

# %% Create dataloaders
train_dataloader = SimpleDataLoader(
    train_activations,
    batch_size=config['batch_size'],
    shuffle=True,
    device=device
)

val_dataloader = SimpleDataLoader(
    val_activations,
    batch_size=config['batch_size'] * 2,  # Larger batch for evaluation
    shuffle=False,
    device=device
)

print(f"Created train dataloader with {len(train_dataloader)} batches")
print(f"Created validation dataloader with {len(val_dataloader)} batches")

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

# %% Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# %% Initialize latent trackers
train_latent_tracker = LatentTracker(
    n_features=config['dict_size'],
    device=device,
    dead_threshold=config['dead_threshold']
)

val_latent_tracker = LatentTracker(
    n_features=config['dict_size'],
    device=device,
    dead_threshold=config['dead_threshold']
)

# %% Multi-epoch training loop
print(f"\nStarting training for {config['num_epochs']} epochs...")

# Track best validation performance
best_val_fvu = float('inf')
best_epoch = -1

# Track metrics across epochs
epoch_metrics = []

# Total steps for learning rate scheduler (across all epochs)
total_steps = len(train_dataloader) * config['num_epochs']
scheduler = get_schedule_with_warmup_and_decay(optimizer, config['warmup_steps'], total_steps)
global_step = 0

for epoch in range(config['num_epochs']):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")
    print(f"{'='*50}")
    
    # Training phase
    model.train()
    train_latent_tracker.reset()
    
    # Track metrics for this epoch
    epoch_train_loss = 0
    epoch_train_fvu = 0
    epoch_aux_loss = 0
    epoch_num_batches = 0
    num_dead_latents_history = []
    
    # Track FVU over last 100 batches
    from collections import deque
    fvu_window = deque(maxlen=100)
    
    # Progress bar for training
    pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        # Forward pass
        output = model(batch)
        reconstruction = output['reconstruction']
        sparse_activations = output['sparse_latent_activations']
        
        # Compute MSE loss
        mse_loss = sae_loss(output, batch)
        
        # Compute FVU
        error = reconstruction - batch
        total_variance = (batch - batch.mean(0)).pow(2).sum()
        squared_error = error.pow(2)
        fvu = squared_error.sum() / total_variance
        
        # Update latent tracker
        train_latent_tracker.update(sparse_activations)
        
        # Get dead latents and compute auxiliary loss
        dead_latents, activation_rates = train_latent_tracker.get_dead_latents()
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
        epoch_train_loss += mse_loss.item()
        epoch_train_fvu += fvu.item()
        epoch_aux_loss += aux_loss.item()
        epoch_num_batches += 1
        num_dead_latents_history.append(len(dead_latents))
        
        # Add FVU to rolling window
        fvu_window.append(fvu.item())
        
        # Log to wandb
        if config['use_wandb'] and batch_idx % 10 == 0:
            wandb.log({
                'train/total_loss': total_loss.item(),
                'train/mse_loss': mse_loss.item(),
                'train/aux_loss': aux_loss.item(),
                'train/fvu': fvu.item(),
                'train/num_dead_latents': len(dead_latents),
                'train/dead_latent_rate': len(dead_latents) / config['dict_size'],
                'train/learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch,
            }, step=global_step)
        
        # Update progress bar
        if batch_idx % 10 == 0:
            avg_fvu_window = sum(fvu_window) / len(fvu_window) if fvu_window else fvu.item()
            current_lr = scheduler.get_last_lr()[0]
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'fvu': f'{avg_fvu_window:.4f}',
                'dead': len(dead_latents),
                'lr': f'{current_lr:.2e}'
            })
        
        global_step += 1
    
    # End of epoch statistics
    epoch_train_loss /= epoch_num_batches
    epoch_train_fvu /= epoch_num_batches
    epoch_aux_loss /= epoch_num_batches
    epoch_time = time.time() - epoch_start_time
    
    print(f"\nEpoch {epoch + 1} Training Summary:")
    print(f"  Time: {epoch_time:.1f}s")
    print(f"  Train Loss: {epoch_train_loss:.4f}")
    print(f"  Train FVU: {epoch_train_fvu:.4f}")
    print(f"  Aux Loss: {epoch_aux_loss:.6f}")
    print(f"  Avg Dead Latents: {np.mean(num_dead_latents_history):.1f}")
    
    # Validation phase
    print(f"\nRunning full validation...")
    val_metrics = evaluate_model(model, val_dataloader, val_latent_tracker)
    
    print(f"Validation Summary:")
    print(f"  Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Val FVU: {val_metrics['fvu']:.4f}")
    print(f"  Val Dead Latents: {val_metrics['num_dead_latents']}")
    
    # Log epoch-level metrics
    if config['use_wandb']:
        wandb.log({
            'epoch_train/loss': epoch_train_loss,
            'epoch_train/fvu': epoch_train_fvu,
            'epoch_train/aux_loss': epoch_aux_loss,
            'epoch_val/loss': val_metrics['loss'],
            'epoch_val/fvu': val_metrics['fvu'],
            'epoch_val/num_dead_latents': val_metrics['num_dead_latents'],
            'epoch': epoch,
        }, step=global_step)
    
    # Track epoch metrics
    epoch_metrics.append({
        'epoch': epoch + 1,
        'train_loss': epoch_train_loss,
        'train_fvu': epoch_train_fvu,
        'val_loss': val_metrics['loss'],
        'val_fvu': val_metrics['fvu'],
        'val_dead_latents': val_metrics['num_dead_latents']
    })
    
    # Save checkpoint
    checkpoint_path = f'checkpoint_epoch_{epoch + 1}_adapters_{adapter_str}.pt'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'epoch_metrics': epoch_metrics,
        'val_activation_rates': val_metrics['activation_rates'].cpu().numpy() if val_metrics['activation_rates'] is not None else None
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Check if this is the best model
    if config['save_best_model'] and val_metrics['fvu'] < best_val_fvu:
        best_val_fvu = val_metrics['fvu']
        best_epoch = epoch + 1
        
        # Save best model
        best_model_path = f'best_model_adapters_{adapter_str}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'config': config,
            'best_val_fvu': best_val_fvu,
            'epoch_metrics': epoch_metrics,
            'val_activation_rates': val_metrics['activation_rates'].cpu().numpy() if val_metrics['activation_rates'] is not None else None
        }, best_model_path)
        print(f"New best model! Saved to: {best_model_path}")

# %% Training complete
print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"{'='*50}")
print(f"Best validation FVU: {best_val_fvu:.4f} (epoch {best_epoch})")

# Print summary of all epochs
print("\nEpoch Summary:")
print("Epoch | Train Loss | Train FVU | Val Loss | Val FVU | Val Dead")
print("-" * 60)
for metrics in epoch_metrics:
    print(f"{metrics['epoch']:5d} | {metrics['train_loss']:10.4f} | {metrics['train_fvu']:9.4f} | "
          f"{metrics['val_loss']:8.4f} | {metrics['val_fvu']:7.4f} | {metrics['val_dead_latents']:8d}")

# %% Final model save
final_model_path = f'trained_sae_adapters_{adapter_str}_epochs{config["num_epochs"]}.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'final_epoch': config['num_epochs'],
    'best_epoch': best_epoch,
    'best_val_fvu': best_val_fvu,
    'epoch_metrics': epoch_metrics,
    'final_val_metrics': val_metrics
}, final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

# Log final metrics to wandb
if config['use_wandb']:
    wandb.log({
        'final/best_val_fvu': best_val_fvu,
        'final/best_epoch': best_epoch,
        'final/num_epochs': config['num_epochs'],
    })
    
    # Log histogram of final activation rates
    if val_metrics['activation_rates'] is not None:
        wandb.log({
            'final/val_activation_rates_histogram': wandb.Histogram(val_metrics['activation_rates'].cpu().numpy()),
        })
    
    wandb.finish()

print("\nTraining script complete!")