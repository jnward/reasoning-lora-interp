#!/usr/bin/env python3
"""Quick test script with small sample size to verify pipeline."""

import numpy as np
import h5py
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Quick test with 1000 samples
def test_pipeline():
    # Load a small sample
    files = sorted(glob("activations_all_adapters/rollout_*.h5"))[:5]  # Just first 5 files
    
    print(f"Testing with {len(files)} files")
    
    activations_list = []
    for f in files:
        with h5py.File(f, 'r') as hf:
            acts = hf['activations'][:100]  # Just 100 samples per file
            flattened = acts.reshape(acts.shape[0], -1)  # Flatten to (n, 448)
            activations_list.append(flattened)
    
    X = np.vstack(activations_list).astype(np.float32)
    print(f"Data shape: {X.shape}")
    
    # PCA
    print("Running PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA output shape: {X_pca.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # UMAP 2D
    print("Running UMAP 2D...")
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer_2d.fit_transform(X_pca)
    
    # Simple plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=1, alpha=0.5)
    plt.title('Test UMAP 2D Projection')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig('test_umap_2d.png', dpi=100)
    print("Saved test_umap_2d.png")
    plt.close()
    
    # UMAP 3D
    print("Running UMAP 3D...")
    reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    X_3d = reducer_3d.fit_transform(X_pca)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=1, alpha=0.5)
    ax.set_title('Test UMAP 3D Projection')
    plt.savefig('test_umap_3d.png', dpi=100)
    print("Saved test_umap_3d.png")
    plt.close()
    
    print("Test complete!")

if __name__ == "__main__":
    test_pipeline()