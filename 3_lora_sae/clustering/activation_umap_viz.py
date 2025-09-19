# %% [markdown]
# # Activation UMAP Visualization
# 
# This notebook performs dimensionality reduction on LoRA activation vectors
# using PCA followed by UMAP, creating both 2D and 3D visualizations.

# %% Imports
import numpy as np
import h5py
import os
from glob import glob
from tqdm import tqdm
import pickle
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')



# %% Configuration
class Config:
    """Configuration for the visualization pipeline."""
    
    # Data paths
    activation_dir = "../2_lora_activation_interp/activations_all_adapters"
    
    # Sampling parameters
    n_samples = 100_000  # Number of samples to use
    random_seed = 42
    
    # PCA parameters
    n_pca_components = 50  # Reduce to this many dimensions first
    
    # UMAP parameters
    n_neighbors = 30
    min_dist = 0.1
    metric = 'cosine'
    
    # Visualization parameters
    figsize_2d = (15, 12)
    figsize_3d = (20, 12)
    point_size = 0.5
    alpha = 0.3
    dpi = 150
    
    # Output paths
    output_dir = "umap_results"
    
    # Cache settings
    use_cache = True
    cache_dir = "umap_cache"

config = Config()

# Create output directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

# %% Data Loading Functions
def get_file_list(activation_dir: str) -> List[str]:
    """Get sorted list of all H5 files."""
    pattern = os.path.join(activation_dir, "rollout_*.h5")
    files = sorted(glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files

def count_total_samples(files: List[str], max_files: Optional[int] = None) -> int:
    """Count total number of samples across files."""
    total = 0
    files_to_check = files[:max_files] if max_files else files
    
    for f in tqdm(files_to_check, desc="Counting samples"):
        with h5py.File(f, 'r') as hf:
            total += hf['activations'].shape[0]
    
    return total

def load_activations_with_metadata(
    files: List[str],
    n_samples: int,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load activation samples with metadata for visualization.
    
    Returns:
        activations: (n_samples, 448) flattened activation vectors
        metadata: Dict with 'layer_idx', 'adapter_type', 'file_idx'
    """
    np.random.seed(seed)
    
    # First, count total samples to determine sampling rate
    print("Counting total samples...")
    total_samples = count_total_samples(files, max_files=100)  # Check first 100 files
    estimated_total = total_samples * (len(files) / min(100, len(files)))
    
    print(f"Estimated total samples: {estimated_total:.0f}")
    sample_rate = min(1.0, n_samples / estimated_total * 1.5)  # Oversample to ensure we get enough
    
    activations_list = []
    layer_indices = []
    adapter_types = []
    file_indices = []
    
    adapter_names = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    print(f"Loading samples with rate {sample_rate:.4f}...")
    for file_idx, filepath in enumerate(tqdm(files, desc="Loading files")):
        with h5py.File(filepath, 'r') as hf:
            acts = hf['activations'][:]  # Shape: (n_tokens, 64, 7)
            n_tokens = acts.shape[0]
            
            # Sample from this file
            n_to_sample = int(n_tokens * sample_rate)
            if n_to_sample > 0:
                indices = np.random.choice(n_tokens, size=min(n_to_sample, n_tokens), replace=False)
                sampled_acts = acts[indices]  # (n_sampled, 64, 7)
                
                # Flatten to (n_sampled, 448)
                flattened = sampled_acts.reshape(len(indices), -1)
                activations_list.append(flattened)
                
                # Create metadata
                for token_idx in range(len(indices)):
                    for layer_idx in range(64):
                        layer_indices.extend([layer_idx] * 7)
                        adapter_types.extend(list(range(7)))
                    file_indices.extend([file_idx] * 448)
        
        # Check if we have enough samples
        current_total = sum(a.shape[0] for a in activations_list)
        if current_total >= n_samples:
            break
    
    # Concatenate and trim to exact number
    all_activations = np.vstack(activations_list)[:n_samples]
    
    # Create metadata arrays (one value per sample, not per dimension)
    # For visualization, we'll use the dominant layer/adapter for each sample
    metadata = {
        'layer_idx': np.array([i // 7 for i in range(0, n_samples * 448, 448)][:n_samples]),
        'adapter_type': np.array([i % 7 for i in range(0, n_samples * 448, 448)][:n_samples]),
        'file_idx': np.array(file_indices[:n_samples * 448:448][:n_samples])
    }
    
    print(f"Loaded {all_activations.shape[0]} samples with shape {all_activations.shape}")
    
    return all_activations.astype(np.float32), metadata

# %% Dimensionality Reduction Functions
def apply_pca(activations: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Apply PCA after standardization."""
    print(f"Applying StandardScaler and PCA to reduce to {n_components} dimensions...")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.2%}")
    
    return X_pca, pca, scaler

def apply_umap(X_pca: np.ndarray, n_components: int = 2, **umap_params) -> np.ndarray:
    """Apply UMAP for final dimensionality reduction."""
    print(f"Applying UMAP to reduce to {n_components}D...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=42,
        n_jobs=-1,
        low_memory=True,
        **umap_params
    )
    
    X_umap = reducer.fit_transform(X_pca)
    return X_umap

# %% Visualization Functions
def plot_2d_visualization(
    X_2d: np.ndarray,
    metadata: Dict[str, np.ndarray],
    config: Config,
    suffix: str = ""
):
    """Create 2D scatter plots with different coloring schemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=config.figsize_2d)
    
    # 1. Color by layer index
    ax = axes[0, 0]
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=metadata['layer_idx'],
        cmap='viridis',
        s=config.point_size,
        alpha=config.alpha,
        rasterized=True
    )
    ax.set_title('Colored by Layer Index')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax, label='Layer')
    
    # 2. Color by adapter type
    ax = axes[0, 1]
    adapter_names = ['gate', 'up', 'down', 'q', 'k', 'v', 'o']
    colors = plt.cm.tab10(metadata['adapter_type'])
    ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=colors,
        s=config.point_size,
        alpha=config.alpha,
        rasterized=True
    )
    ax.set_title('Colored by Adapter Type')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Add legend
    for i, name in enumerate(adapter_names):
        ax.scatter([], [], c=plt.cm.tab10(i), label=name, s=20)
    ax.legend(loc='best', fontsize=8)
    
    # 3. Density plot (hexbin)
    ax = axes[1, 0]
    hexbin = ax.hexbin(
        X_2d[:, 0], X_2d[:, 1],
        gridsize=50,
        cmap='YlOrRd',
        mincnt=1
    )
    ax.set_title('Density Plot')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.colorbar(hexbin, ax=ax, label='Count')
    
    # 4. Contour density plot
    ax = axes[1, 1]
    from scipy.stats import gaussian_kde
    
    # Subsample for KDE if too many points
    if len(X_2d) > 10000:
        idx = np.random.choice(len(X_2d), 10000, replace=False)
        X_kde = X_2d[idx]
    else:
        X_kde = X_2d
    
    try:
        xy = np.vstack([X_kde[:, 0], X_kde[:, 1]])
        z = gaussian_kde(xy)(xy)
        ax.scatter(X_kde[:, 0], X_kde[:, 1], c=z, s=1, cmap='viridis', rasterized=True)
        ax.set_title('KDE Density')
    except:
        ax.text(0.5, 0.5, 'KDE failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('KDE Density (failed)')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.suptitle(f'2D UMAP Projections{suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(config.output_dir, f'umap_2d{suffix}.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    print(f"Saved 2D visualization to {output_path}")
    plt.show()

def plot_3d_visualization(
    X_3d: np.ndarray,
    metadata: Dict[str, np.ndarray],
    config: Config,
    suffix: str = ""
):
    """Create 3D scatter plots from multiple angles."""
    
    fig = plt.figure(figsize=config.figsize_3d)
    
    # Define viewing angles
    angles = [
        (30, 45),   # Default
        (0, 0),     # Top view
        (0, 90),    # Side view
        (60, 120),  # Alternative angle
    ]
    
    for idx, (elev, azim) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 4, idx, projection='3d')
        
        # Color by layer index
        scatter = ax.scatter(
            X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
            c=metadata['layer_idx'],
            cmap='viridis',
            s=config.point_size,
            alpha=config.alpha,
            rasterized=True
        )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Layer Color\n(elev={elev}, azim={azim})')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
    
    # Same angles but colored by adapter type
    for idx, (elev, azim) in enumerate(angles, 5):
        ax = fig.add_subplot(2, 4, idx, projection='3d')
        
        colors = plt.cm.tab10(metadata['adapter_type'])
        ax.scatter(
            X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
            c=colors,
            s=config.point_size,
            alpha=config.alpha,
            rasterized=True
        )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Adapter Color\n(elev={elev}, azim={azim})')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
    
    plt.suptitle(f'3D UMAP Projections{suffix}', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(config.output_dir, f'umap_3d{suffix}.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    print(f"Saved 3D visualization to {output_path}")
    plt.show()

# %% Main Pipeline
def main():
    """Run the complete visualization pipeline."""
    
    print("=" * 50)
    print("UMAP Visualization Pipeline")
    print("=" * 50)
    
    # Get file list
    files = get_file_list(config.activation_dir)
    print(f"Found {len(files)} activation files")
    
    # Check cache
    cache_file = os.path.join(config.cache_dir, f"embeddings_{config.n_samples}.pkl")
    
    if config.use_cache and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            X_2d = cache_data['X_2d']
            X_3d = cache_data['X_3d']
            metadata = cache_data['metadata']
    else:
        # Load activations
        activations, metadata = load_activations_with_metadata(
            files, config.n_samples, config.random_seed
        )
        
        # Apply PCA
        X_pca, pca_model, scaler = apply_pca(activations, config.n_pca_components)
        
        # Apply UMAP for 2D
        X_2d = apply_umap(
            X_pca, 
            n_components=2,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric
        )
        
        # Apply UMAP for 3D
        X_3d = apply_umap(
            X_pca,
            n_components=3,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric
        )
        
        # Cache results
        if config.use_cache:
            print(f"Saving cache to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'X_2d': X_2d,
                    'X_3d': X_3d,
                    'metadata': metadata
                }, f)
    
    # Create visualizations
    print("\nCreating 2D visualizations...")
    plot_2d_visualization(X_2d, metadata, config)
    
    print("\nCreating 3D visualizations...")
    plot_3d_visualization(X_3d, metadata, config)
    
    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print(f"Results saved to {config.output_dir}/")

# %% Run Pipeline
if __name__ == "__main__":
    main()
# %%

# %%
