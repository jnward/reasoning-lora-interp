# %%
"""
Plot Ablation Results

This notebook loads the saved ablation results and creates figures:
1. Per-layer KL divergence bar plot
2. Adapter ablation heatmap
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
# Load the data
layer_data = None
adapter_data = None

if result_files['layer']:
    with open(result_files['layer'], 'r') as f:
        layer_data = json.load(f)
    print(f"\nLayer ablation: {layer_data['metadata']['n_sequences']} sequences, seed={layer_data['metadata']['random_seed']}")

if result_files['adapter']:
    with open(result_files['adapter'], 'r') as f:
        adapter_data = json.load(f)
    print(f"Adapter ablation: {adapter_data['metadata']['n_sequences']} sequences")

# %%
# Print basic statistics
if layer_data:
    df_layer = pd.DataFrame(layer_data['results'])
    print("\n" + "="*60)
    print("LAYER ABLATION STATISTICS")
    print("="*60)
    print(f"Mean KL across all layers: {df_layer['mean_kl'].mean():.5f}")
    print(f"Std KL across all layers: {df_layer['mean_kl'].std():.5f}")
    print(f"Max KL (most important): {df_layer['mean_kl'].max():.5f} (Layer {int(df_layer.loc[df_layer['mean_kl'].idxmax(), 'layer'])})")
    print(f"Min KL (least important): {df_layer['mean_kl'].min():.5f} (Layer {int(df_layer.loc[df_layer['mean_kl'].idxmin(), 'layer'])})")

if adapter_data:
    adapter_matrix = np.array(adapter_data['matrix'])
    adapter_types = adapter_data['metadata']['adapter_types']
    mean_per_adapter = adapter_matrix.mean(axis=1)
    
    print("\n" + "="*60)
    print("ADAPTER ABLATION STATISTICS")
    print("="*60)
    
    # Attention vs MLP comparison
    attention_adapters = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    mlp_adapters = ['gate_proj', 'up_proj', 'down_proj']
    
    attention_indices = [adapter_types.index(a) for a in attention_adapters if a in adapter_types]
    mlp_indices = [adapter_types.index(a) for a in mlp_adapters if a in adapter_types]
    
    if attention_indices and mlp_indices:
        attention_mean = adapter_matrix[attention_indices, :].mean()
        mlp_mean = adapter_matrix[mlp_indices, :].mean()
        
        print(f"Attention (Q,K,V,O): {attention_mean:.5f}")
        print(f"MLP (Gate,Up,Down):  {mlp_mean:.5f}")
        print(f"Ratio (MLP/Attention): {mlp_mean/attention_mean:.2f}x")

# %%
# Plot 3: Publication-quality combined visualization for NeurIPS paper
if layer_data and adapter_data:
    from matplotlib.gridspec import GridSpec
    
    df_layer = pd.DataFrame(layer_data['results'])
    adapter_matrix = np.array(adapter_data['matrix'])
    adapter_types = adapter_data['metadata']['adapter_types']
    n_layers = adapter_data['metadata']['n_layers']
    
    # Find global min/max for consistent color scale
    global_vmin = 0
    global_vmax = max(df_layer['mean_kl'].max(), adapter_matrix.max())
    
    # Create figure with custom gridspec for optimal sizing
    # NeurIPS single column is ~3.25 inches wide, we'll use full width
    # Make heatmap slightly shorter by adjusting height ratios
    fig = plt.figure(figsize=(7, 3.2), dpi=150)  # Higher DPI for better resolution
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.12)  # Reduced heatmap ratio
    
    # Use the same colormap for both plots
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
    
    # Top plot: Layer ablation bar chart
    ax1 = fig.add_subplot(gs[0])
    
    # Set white background for bar chart
    ax1.set_facecolor('white')
    
    # Create bars with consistent color mapping
    bar_colors = cmap(norm(df_layer['mean_kl'].values))
    
    # CRITICAL: Align bars with heatmap cells
    # Bars should be centered at integer positions 0, 1, 2, ... 63
    bars = ax1.bar(df_layer['layer'].values, df_layer['mean_kl'].values, 
                   width=1.0, edgecolor='none', color=bar_colors, align='center')
    
    # Add subtle error bars
    ax1.errorbar(df_layer['layer'].values, df_layer['mean_kl'].values, 
                 yerr=df_layer['std_kl'].values, 
                 fmt='none', ecolor='black', alpha=0.2, capsize=0, linewidth=0.5)
    
    # Formatting for bar plot
    ax1.set_ylabel('Layer-wise\nKL Divergence', fontsize=10, labelpad=19)  # Fine-tune alignment
    # Remove title - using y-label instead
    ax1.grid(True, alpha=0.15, axis='y', linewidth=0.5)  # Lighter grid
    ax1.set_axisbelow(True)
    
    # Add (a) label further to the left
    ax1.text(-0.20, 0.5, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='center')
    
    # Remove top and right spines for cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Set x-axis limits to match heatmap exactly
    ax1.set_xlim(-0.5, n_layers - 0.5)
    
    # Remove x-axis labels from top plot (will use bottom plot's labels)
    ax1.set_xticks(range(0, n_layers, 8))
    ax1.set_xticklabels([])
    
    # Make y-axis more compact
    ax1.set_ylim(0, df_layer['mean_kl'].max() * 1.1)
    y_ticks = ax1.get_yticks()
    ax1.set_yticks(y_ticks[::2])  # Show every other tick
    ax1.tick_params(axis='y', labelsize=9)
    
    # Bottom plot: Adapter heatmap
    ax2 = fig.add_subplot(gs[1])
    
    # Create heatmap with same color normalization
    im = ax2.imshow(adapter_matrix, aspect='auto', cmap=cmap, norm=norm,
                    interpolation='nearest', origin='upper')
    
    # CRITICAL: Set extent to align with bar plot
    # extent=[left, right, bottom, top]
    # We want the heatmap cells to be centered at 0, 1, 2, ... 63
    im.set_extent([-0.5, n_layers - 0.5, len(adapter_types) - 0.5, -0.5])
    
    # Set labels and formatting
    ax2.set_xlabel('Layer', fontsize=10)
    ax2.set_ylabel('Adapter Type', fontsize=10)  # Default padding for bottom
    # Remove title - using y-label instead
    
    # Add (b) label further to the left
    ax2.text(-0.20, 0.5, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='center')
    
    # Set y-axis ticks for adapter types
    ax2.set_yticks(range(len(adapter_types)))
    ax2.set_yticklabels(adapter_types, fontsize=9)
    
    # Set x-axis ticks to match bar plot
    ax2.set_xticks(range(0, n_layers, 8))
    ax2.set_xticklabels(range(0, n_layers, 8), fontsize=9)
    ax2.set_xlim(-0.5, n_layers - 0.5)
    
    # Remove gridlines from heatmap for cleaner look
    ax2.grid(False)
    
    # Add shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                       cax=cbar_ax, orientation='vertical')
    cbar.set_label('Mean KL Divergence', fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    
    # Add overall title
    fig.suptitle(f'LoRA Ablation Analysis (n={layer_data["metadata"]["n_sequences"]} sequences)', 
                 fontsize=12, y=0.98)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.17, right=0.90, top=0.92, bottom=0.12)  # Further increased left margin for labels
    
    # Save high-quality figure for paper
    plt.savefig('neurips_combined_ablation.pdf', dpi=600, bbox_inches='tight')  # Higher DPI
    plt.savefig('neurips_combined_ablation.png', dpi=600, bbox_inches='tight')  # Higher DPI
    plt.show()
    
    print("\n✓ NeurIPS figure saved as neurips_combined_ablation.pdf (600 DPI)")

# %%
# Generate multiple versions with different color scales
if layer_data and adapter_data:
    from matplotlib.gridspec import GridSpec
    
    df_layer = pd.DataFrame(layer_data['results'])
    adapter_matrix = np.array(adapter_data['matrix'])
    adapter_types = adapter_data['metadata']['adapter_types']
    n_layers = adapter_data['metadata']['n_layers']
    
    # Find global min/max for consistent color scale
    global_vmin = 0
    global_vmax = max(df_layer['mean_kl'].max(), adapter_matrix.max())
    
    # Your favorite colormaps
    favorite_colormaps = ['CMRmap', 'inferno', 'mako', 'rocket']
    
    # Different max value scalings to try
    max_scalings = [
        (1.0, 'Original (100%)'),
        (1.1, 'Extended 10%'),
        (1.2, 'Extended 20%'),
        (1.05, 'Extended 5%'),
        (1.15, 'Extended 15%'),
    ]
    
    print(f"\nTesting {len(favorite_colormaps)} colormaps with {len(max_scalings)} different max value scalings")
    print(f"Colormaps: {favorite_colormaps}")
    print(f"Original max value: {global_vmax:.4f}")
    
    for cmap_name in favorite_colormaps:
        print(f"\n{'='*60}")
        print(f"Colormap: {cmap_name}")
        print(f"{'='*60}")
        
        for scale_factor, scale_label in max_scalings:
            # Adjust the max value
            adjusted_vmax = global_vmax * scale_factor
            print(f"\n{scale_label}: max = {adjusted_vmax:.4f}")
            
            # Create figure with custom gridspec
            fig = plt.figure(figsize=(7, 3.2), dpi=100)  # Lower DPI for display
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.12)
            
            # Get the colormap
            cmap = plt.cm.get_cmap(cmap_name)
            norm = plt.Normalize(vmin=global_vmin, vmax=adjusted_vmax)
            
            # Top plot: Layer ablation bar chart
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('white')
            
            # Create bars with consistent color mapping
            bar_colors = cmap(norm(df_layer['mean_kl'].values))
            
            bars = ax1.bar(df_layer['layer'].values, df_layer['mean_kl'].values, 
                           width=1.0, edgecolor='none', color=bar_colors, align='center')
            
            # Add subtle error bars
            ax1.errorbar(df_layer['layer'].values, df_layer['mean_kl'].values, 
                         yerr=df_layer['std_kl'].values, 
                         fmt='none', ecolor='black', alpha=0.2, capsize=0, linewidth=0.5)
            
            # Formatting
            ax1.set_ylabel('Layer-wise\nKL Divergence', fontsize=10)
            ax1.grid(True, alpha=0.15, axis='y', linewidth=0.5)
            ax1.set_axisbelow(True)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_xlim(-0.5, n_layers - 0.5)
            ax1.set_xticks(range(0, n_layers, 8))
            ax1.set_xticklabels([])
            ax1.set_ylim(0, df_layer['mean_kl'].max() * 1.1)
            y_ticks = ax1.get_yticks()
            ax1.set_yticks(y_ticks[::2])
            ax1.tick_params(axis='y', labelsize=9)
            
            # Bottom plot: Adapter heatmap
            ax2 = fig.add_subplot(gs[1])
            
            im = ax2.imshow(adapter_matrix, aspect='auto', cmap=cmap, norm=norm,
                            interpolation='nearest', origin='upper')
            im.set_extent([-0.5, n_layers - 0.5, len(adapter_types) - 0.5, -0.5])
            
            # Formatting
            ax2.set_xlabel('Layer', fontsize=10)
            ax2.set_ylabel('Adapter Type', fontsize=10)
            ax2.set_yticks(range(len(adapter_types)))
            ax2.set_yticklabels(adapter_types, fontsize=9)
            ax2.set_xticks(range(0, n_layers, 8))
            ax2.set_xticklabels(range(0, n_layers, 8), fontsize=9)
            ax2.set_xlim(-0.5, n_layers - 0.5)
            ax2.grid(False)
            
            # Add shared colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                               cax=cbar_ax, orientation='vertical')
            cbar.set_label('Mean KL Divergence', fontsize=10, labelpad=8)
            cbar.ax.tick_params(labelsize=9)
            
            # Add title with colormap name and scale factor
            fig.suptitle(f'LoRA Ablation - {cmap_name} ({scale_label})', 
                         fontsize=12, y=0.98)
            
            plt.subplots_adjust(left=0.17, right=0.90, top=0.92, bottom=0.12)  # Further increased left margin for labels
            plt.show()
    
    print("\n✓ Generated all colormap variations")

# %%
# Final publication-ready version with CMRmap (extended 15%)
if layer_data and adapter_data:
    from matplotlib.gridspec import GridSpec
    
    df_layer = pd.DataFrame(layer_data['results'])
    adapter_matrix = np.array(adapter_data['matrix'])
    adapter_types = adapter_data['metadata']['adapter_types']
    n_layers = adapter_data['metadata']['n_layers']
    
    # Find global min/max for consistent color scale
    global_vmin = 0
    global_vmax = max(df_layer['mean_kl'].max(), adapter_matrix.max())
    
    # Use CMRmap with 15% extension
    adjusted_vmax = global_vmax * 1.15
    
    print(f"\n{'='*60}")
    print("FINAL PUBLICATION VERSION")
    print(f"{'='*60}")
    print(f"Colormap: CMRmap (Extended 15%)")
    print(f"Max value: {adjusted_vmax:.4f}")
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(7, 3.2), dpi=150)  # Higher DPI for publication
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.12)
    
    # Get the colormap
    cmap = plt.cm.CMRmap
    norm = plt.Normalize(vmin=global_vmin, vmax=adjusted_vmax)
    
    # Top plot: Layer ablation bar chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    
    # Create bars with consistent color mapping
    bar_colors = cmap(norm(df_layer['mean_kl'].values))
    
    bars = ax1.bar(df_layer['layer'].values, df_layer['mean_kl'].values, 
                   width=1.0, edgecolor='none', color=bar_colors, align='center')
    
    # Add subtle error bars
    ax1.errorbar(df_layer['layer'].values, df_layer['mean_kl'].values, 
                 yerr=df_layer['std_kl'].values, 
                 fmt='none', ecolor='black', alpha=0.2, capsize=0, linewidth=0.5)
    
    # Formatting
    ax1.set_ylabel('Layer-wise Mean\nKL Divergence', fontsize=10, labelpad=18.5)  # Fine-tune alignment
    ax1.grid(True, alpha=0.15, axis='y', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(-0.5, n_layers - 0.5)
    ax1.set_xticks(range(0, n_layers, 8))
    ax1.set_xticklabels([])
    ax1.set_ylim(0, df_layer['mean_kl'].max() * 1.1)
    y_ticks = ax1.get_yticks()
    ax1.set_yticks(y_ticks[::2])
    ax1.tick_params(axis='y', labelsize=9)
    
    # Add (a) label further to the left
    ax1.text(-0.20, 0.5, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='center')
    
    # Bottom plot: Adapter heatmap
    ax2 = fig.add_subplot(gs[1])
    
    im = ax2.imshow(adapter_matrix, aspect='auto', cmap=cmap, norm=norm,
                    interpolation='nearest', origin='upper')
    im.set_extent([-0.5, n_layers - 0.5, len(adapter_types) - 0.5, -0.5])
    
    # Formatting
    ax2.set_xlabel('Layer', fontsize=10)
    ax2.set_ylabel('Adapter Type', fontsize=10)  # Default padding for bottom
    
    # Add (b) label further to the left
    ax2.text(-0.20, 0.5, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='center')
    ax2.set_yticks(range(len(adapter_types)))
    ax2.set_yticklabels(adapter_types, fontsize=9)
    ax2.set_xticks(range(0, n_layers, 8))
    ax2.set_xticklabels(range(0, n_layers, 8), fontsize=9)
    ax2.set_xlim(-0.5, n_layers - 0.5)
    ax2.grid(False)
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                       cax=cbar_ax, orientation='vertical')
    cbar.set_label('Mean KL Divergence', fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    
    # Add title
    fig.suptitle('LoRA Component Ablation', fontsize=12, y=0.98)
    
    plt.subplots_adjust(left=0.17, right=0.90, top=0.92, bottom=0.12)  # Further increased left margin for labels
    
    # Save high-quality figures
    plt.savefig('lora_component_ablation_final.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('lora_component_ablation_final.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Final publication figure saved as:")
    print("  - lora_component_ablation_final.pdf (600 DPI)")
    print("  - lora_component_ablation_final.png (600 DPI)")

# %%
print("\n✓ All plots generated!")
print("\nSaved files:")
print("  - neurips_combined_ablation.pdf (600 DPI)")
print("  - neurips_combined_ablation.png (600 DPI)")

# %%