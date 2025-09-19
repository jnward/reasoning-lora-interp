import numpy as np
import h5py
from glob import glob
import os
import matplotlib.pyplot as plt

# Configuration
activation_dir = '../../lora-activations-dashboard/backend/activations'
n_samples = 100000  # Use 100k samples for SVD
n_components = 192  # All components since d=192

print("Loading activation data...")
h5_files = sorted(glob(os.path.join(activation_dir, 'rollout_*.h5')))
print(f'Found {len(h5_files)} H5 files')

# Load subset of activations
all_acts = []
total_loaded = 0

for i, file_path in enumerate(h5_files):
    if total_loaded >= n_samples:
        break
    with h5py.File(file_path, 'r') as f:
        acts = f['activations'][:]
        acts_flat = acts.reshape(-1, 192)
        all_acts.append(acts_flat)
        total_loaded += len(acts_flat)
    if (i + 1) % 10 == 0:
        print(f"Loaded {i+1} files, {total_loaded} samples")

# Concatenate and truncate
all_acts = np.concatenate(all_acts, axis=0)[:n_samples]
print(f'\nLoaded {len(all_acts)} activations')
print(f'Shape: {all_acts.shape}')
print(f'Memory usage: {all_acts.nbytes / 1e6:.1f} MB')

# Convert to float32 for SVD
all_acts = all_acts.astype(np.float32)

# Center the data
print("\nCentering data...")
mean_act = all_acts.mean(axis=0)
centered_acts = all_acts - mean_act

# Compute SVD
print("\nComputing SVD...")
# For full SVD, we can use numpy directly since d=192 is small
U, s, Vt = np.linalg.svd(centered_acts, full_matrices=False)

print(f"\nSingular values shape: {s.shape}")
print(f"Top 10 singular values: {s[:10]}")
print(f"Bottom 10 singular values: {s[-10:]}")

# Compute explained variance ratio
explained_variance = (s ** 2) / (n_samples - 1)
total_variance = explained_variance.sum()
explained_variance_ratio = explained_variance / total_variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Find effective rank (number of components to explain 99% variance)
n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1

print(f"\nEffective rank:")
print(f"  Components for 90% variance: {n_components_90}")
print(f"  Components for 95% variance: {n_components_95}")
print(f"  Components for 99% variance: {n_components_99}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Singular values (log scale)
ax = axes[0, 0]
ax.semilogy(s, 'b-', linewidth=2)
ax.set_xlabel('Component')
ax.set_ylabel('Singular Value')
ax.set_title('Singular Values (log scale)')
ax.grid(True, alpha=0.3)

# 2. Explained variance ratio
ax = axes[0, 1]
ax.plot(explained_variance_ratio, 'r-', linewidth=2)
ax.set_xlabel('Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Explained Variance per Component')
ax.grid(True, alpha=0.3)

# 3. Cumulative explained variance
ax = axes[1, 0]
ax.plot(cumulative_variance_ratio, 'g-', linewidth=2)
ax.axhline(y=0.90, color='k', linestyle='--', alpha=0.5, label='90%')
ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95%')
ax.axhline(y=0.99, color='k', linestyle='--', alpha=0.5, label='99%')
ax.set_xlabel('Component')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('Cumulative Explained Variance')
ax.grid(True, alpha=0.3)
ax.legend()

# 4. Condition number analysis
ax = axes[1, 1]
# Compute condition numbers for different truncations
truncations = range(10, 193, 10)
condition_numbers = []
for k in truncations:
    cond = s[0] / s[k-1]  # Ratio of largest to k-th singular value
    condition_numbers.append(cond)

ax.semilogy(truncations, condition_numbers, 'o-', linewidth=2, markersize=6)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Condition Number')
ax.set_title('Condition Number vs Truncation')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_singular_values.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to activation_singular_values.png")

# Save singular values for further analysis
np.savez('activation_svd_results.npz', 
         singular_values=s,
         explained_variance_ratio=explained_variance_ratio,
         cumulative_variance_ratio=cumulative_variance_ratio,
         mean_activation=mean_act,
         n_samples=n_samples)
print("SVD results saved to activation_svd_results.npz")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Condition number (full): {s[0] / s[-1]:.2e}")
print(f"Spectral norm: {s[0]:.2f}")
print(f"Frobenius norm: {np.linalg.norm(s):.2f}")
print(f"Nuclear norm: {np.sum(s):.2f}")
print(f"Variance explained by top component: {explained_variance_ratio[0]*100:.1f}%")
print(f"Variance explained by top 10 components: {cumulative_variance_ratio[9]*100:.1f}%")