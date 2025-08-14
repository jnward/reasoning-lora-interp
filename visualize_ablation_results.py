# %%
"""
Visualization script for saved ablation results

This script loads the JSON files saved by lora_layer_ablation_kl.py
and creates customizable plots without needing to re-run the ablation experiments.
"""

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
# Find the most recent result files
def find_latest_results():
    """Find the most recent ablation result files."""
    files = {
        'layer': sorted(glob('layer_ablation_results_*.json'))[-1] if glob('layer_ablation_results_*.json') else None,
        'batch': sorted(glob('batch_ablation_results_*.json'))[-1] if glob('batch_ablation_results_*.json') else None,
        'adapter': sorted(glob('adapter_ablation_results_*.json'))[-1] if glob('adapter_ablation_results_*.json') else None
    }
    
    for key, path in files.items():
        if path:
            print(f"Found {key} results: {os.path.basename(path)}")
        else:
            print(f"No {key} results found")
    
    return files

result_files = find_latest_results()

# %%
# Load layer ablation results
if result_files['layer']:
    with open(result_files['layer'], 'r') as f:
        layer_data = json.load(f)
    
    df_layer = pd.DataFrame(layer_data['results'])
    metadata = layer_data['metadata']
    
    print(f"\nLayer Ablation Results:")
    print(f"  Model: {metadata['model']}")
    print(f"  Sequences: {metadata['n_sequences']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    print(f"\nStatistics:")
    print(df_layer[['mean_kl', 'std_kl']].describe())

# %%
# Visualize layer ablation with customizable options
if result_files['layer']:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean KL with error bars
    ax = axes[0, 0]
    ax.errorbar(df_layer['layer'], df_layer['mean_kl'], yerr=df_layer['std_kl'], 
                marker='o', markersize=4, capsize=3, alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Mean KL Divergence by Layer (with std dev)')
    ax.grid(True, alpha=0.3)
    
    # 2. Smoothed trend
    ax = axes[0, 1]
    window = 5
    smoothed = df_layer['mean_kl'].rolling(window=window, center=True).mean()
    ax.plot(df_layer['layer'], df_layer['mean_kl'], 'o', alpha=0.3, markersize=3, label='Raw')
    ax.plot(df_layer['layer'], smoothed, linewidth=2, label=f'Smoothed (window={window})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('KL Divergence Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Top and bottom layers
    ax = axes[1, 0]
    top_n = 10
    top_layers = df_layer.nlargest(top_n, 'mean_kl')
    bottom_layers = df_layer.nsmallest(top_n, 'mean_kl')
    
    x = np.arange(top_n)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, top_layers['mean_kl'].values, width, 
                   label='Most Important', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, bottom_layers['mean_kl'].values, width,
                   label='Least Important', color='blue', alpha=0.7)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title(f'Top {top_n} Most and Least Important Layers')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in range(top_n)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Layer groups analysis
    ax = axes[1, 1]
    n_layers = len(df_layer)
    early = df_layer[df_layer['layer'] < n_layers // 3]['mean_kl'].mean()
    middle = df_layer[(df_layer['layer'] >= n_layers // 3) & 
                      (df_layer['layer'] < 2 * n_layers // 3)]['mean_kl'].mean()
    late = df_layer[df_layer['layer'] >= 2 * n_layers // 3]['mean_kl'].mean()
    
    groups = ['Early\n(0-21)', 'Middle\n(22-42)', 'Late\n(43-63)']
    values = [early, middle, late]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 3))
    
    bars = ax.bar(groups, values, color=colors, alpha=0.8)
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Average Importance by Layer Group')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.suptitle(f'Layer Ablation Analysis - {metadata["n_sequences"]} sequences', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# %%
# Load and visualize batch ablation results
if result_files['batch']:
    with open(result_files['batch'], 'r') as f:
        batch_data = json.load(f)
    
    df_batch = pd.DataFrame(batch_data['results'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Mean KL by batch
    ax = axes[0]
    colors = plt.cm.coolwarm(df_batch['mean_kl'] / df_batch['mean_kl'].max())
    bars = ax.bar(df_batch['batch'], df_batch['mean_kl'], color=colors, alpha=0.8)
    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title(f'Mean KL by Layer Batch (size={batch_data["metadata"]["batch_size"]})')
    ax.set_xticks(df_batch['batch'])
    ax.set_xticklabels(df_batch['layers'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Comparison with individual layer averages (if available)
    if result_files['layer']:
        ax = axes[1]
        batch_size = batch_data['metadata']['batch_size']
        
        # Calculate average of individual layers for each batch
        individual_batch_means = []
        for batch_idx in range(len(df_batch)):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_layers = df_layer[(df_layer['layer'] >= start) & (df_layer['layer'] < end)]
            individual_batch_means.append(batch_layers['mean_kl'].mean())
        
        x = np.arange(len(df_batch))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, individual_batch_means, width, 
                      label='Avg of Individual', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, df_batch['mean_kl'], width,
                      label='Batch Ablation', alpha=0.8, color='coral')
        
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Mean KL Divergence')
        ax.set_title('Individual vs Batch Ablation Effects')
        ax.set_xticks(x)
        ax.set_xticklabels(df_batch['layers'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Batch Ablation Analysis', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

# %%
# Load and visualize adapter ablation results
if result_files['adapter']:
    with open(result_files['adapter'], 'r') as f:
        adapter_data = json.load(f)
    
    adapter_matrix = np.array(adapter_data['matrix'])
    adapter_types = adapter_data['metadata']['adapter_types']
    n_layers = adapter_data['metadata']['n_layers']
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Main heatmap
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    im = ax1.imshow(adapter_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Adapter Type', fontsize=12)
    ax1.set_title('KL Divergence from Ablating Individual Adapters', fontsize=14)
    ax1.set_yticks(range(len(adapter_types)))
    ax1.set_yticklabels(adapter_types)
    ax1.set_xticks(range(0, n_layers, 4))
    ax1.set_xticklabels(range(0, n_layers, 4))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Mean KL Divergence', fontsize=11)
    
    # Add grid
    ax1.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(adapter_types), 1), minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)
    
    # 2. Mean importance per adapter type
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    mean_per_adapter = adapter_matrix.mean(axis=1)
    colors = plt.cm.YlOrRd(mean_per_adapter / mean_per_adapter.max())
    bars = ax2.bar(range(len(adapter_types)), mean_per_adapter, color=colors, alpha=0.8)
    ax2.set_xlabel('Adapter Type', fontsize=11)
    ax2.set_ylabel('Mean KL', fontsize=11)
    ax2.set_title('Average Importance per Adapter Type', fontsize=12)
    ax2.set_xticks(range(len(adapter_types)))
    ax2.set_xticklabels(adapter_types, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_per_adapter):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Mean importance per layer
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    mean_per_layer = adapter_matrix.mean(axis=0)
    ax3.plot(mean_per_layer, linewidth=2, color='darkred')
    ax3.fill_between(range(len(mean_per_layer)), mean_per_layer, alpha=0.3, color='red')
    ax3.set_xlabel('Layer', fontsize=11)
    ax3.set_ylabel('Mean KL', fontsize=11)
    ax3.set_title('Average Importance per Layer (All Adapters)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Adapter Ablation Analysis - {adapter_data["metadata"]["n_sequences"]} sequences', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nAdapter Ablation Statistics:")
    print("-" * 40)
    
    # Rank adapters by importance
    adapter_importance = pd.DataFrame({
        'adapter': adapter_types,
        'mean_kl': mean_per_adapter,
        'max_kl': adapter_matrix.max(axis=1),
        'std_kl': adapter_matrix.std(axis=1)
    }).sort_values('mean_kl', ascending=False)
    
    print("\nAdapter types ranked by importance:")
    for _, row in adapter_importance.iterrows():
        print(f"  {row['adapter']:10s}: Mean={row['mean_kl']:.5f}, Max={row['max_kl']:.5f}, Std={row['std_kl']:.5f}")
    
    # Attention vs MLP
    attention_adapters = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_adapters = ['gate_proj', 'up_proj', 'down_proj']
    
    attention_indices = [adapter_types.index(a) for a in attention_adapters]
    mlp_indices = [adapter_types.index(a) for a in mlp_adapters]
    
    attention_mean = adapter_matrix[attention_indices, :].mean()
    mlp_mean = adapter_matrix[mlp_indices, :].mean()
    
    print(f"\nAttention vs MLP:")
    print(f"  Attention (Q,K,V,O): {attention_mean:.5f}")
    print(f"  MLP (Gate,Up,Down):  {mlp_mean:.5f}")
    print(f"  Ratio (MLP/Attention): {mlp_mean/attention_mean:.2f}x")

# %%
print("\n✓ Visualization complete!")
print("\nYou can modify this script to create custom visualizations from the saved JSON files.")