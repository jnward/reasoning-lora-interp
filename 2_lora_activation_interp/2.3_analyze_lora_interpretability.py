#!/usr/bin/env python3
"""
Analyze interpretability classifications from LoRA autointerp results.
Generates bar charts and bootstrapped confidence intervals.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse
from scipy import stats
import pandas as pd

def load_results(file_path):
    """Load interpretation results from JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['interpretations']

def bootstrap_confidence_intervals(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrapped confidence intervals for classification proportions.
    
    Returns dict with classification -> (mean, lower_ci, upper_ci)
    """
    n = len(data)
    classifications = np.array(data)
    
    # Bootstrap samples
    bootstrap_props = defaultdict(list)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(classifications, size=n, replace=True)
        
        # Calculate proportions for each classification
        counts = Counter(sample)
        for cls in [0, 1, 2]:
            prop = counts.get(cls, 0) / n
            bootstrap_props[cls].append(prop)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    results = {}
    
    for cls in [0, 1, 2]:
        props = np.array(bootstrap_props[cls])
        mean_prop = np.mean(props)
        lower = np.percentile(props, alpha/2 * 100)
        upper = np.percentile(props, (1 - alpha/2) * 100)
        results[cls] = (mean_prop, lower, upper)
    
    return results

def plot_overall_distribution(interpretations, save_path=None):
    """Plot overall distribution of interpretability classifications with confidence intervals."""
    # Extract classifications
    classifications = []
    for interp in interpretations:
        cls = interp.get('classification', -1)
        if cls != -1:  # Exclude errors
            classifications.append(cls)
    
    if not classifications:
        print("No valid classifications found!")
        return
    
    # Count classifications
    counts = Counter(classifications)
    
    # Calculate bootstrapped confidence intervals
    ci_results = bootstrap_confidence_intervals(classifications)
    
    # Prepare data for plotting
    labels = ['Monosemantic\n(0)', 'Broad but\nConsistent (1)', 'Polysemantic\n(2)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Counts
    x_pos = np.arange(3)
    counts_list = [counts.get(i, 0) for i in range(3)]
    
    bars1 = ax1.bar(x_pos, counts_list, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Interpretability Classifications', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts_list):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=11)
    
    # Right plot: Proportions with confidence intervals
    props = []
    errors_lower = []
    errors_upper = []
    
    for i in range(3):
        mean_prop, lower, upper = ci_results[i]
        props.append(mean_prop * 100)  # Convert to percentage
        errors_lower.append((mean_prop - lower) * 100)
        errors_upper.append((upper - mean_prop) * 100)
    
    bars2 = ax2.bar(x_pos, props, color=colors, edgecolor='black', linewidth=1.5)
    ax2.errorbar(x_pos, props, yerr=[errors_lower, errors_upper], 
                 fmt='none', color='black', capsize=5, capthick=2)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Proportions with 95% Bootstrap CI', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(props) * 1.3)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar, prop, err_low, err_up in zip(bars2, props, errors_lower, errors_upper):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err_up + 1,
                f'{prop:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # Add summary statistics
    total = len(classifications)
    fig.suptitle(f'Total Features Analyzed: {total}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overall distribution plot to {save_path}")
    
    plt.show()
    
    # Print confidence intervals
    print("\nBootstrapped 95% Confidence Intervals:")
    print("-" * 50)
    for i, label in enumerate(['Monosemantic (0)', 'Broad but Consistent (1)', 'Polysemantic (2)']):
        mean_prop, lower, upper = ci_results[i]
        print(f"{label:25s}: {mean_prop*100:5.1f}% [{lower*100:5.1f}%, {upper*100:5.1f}%]")

def plot_by_layer_bins(interpretations, save_path=None):
    """Plot distribution stratified by layer bins (0-7, 8-15, etc.)."""
    # Group by layer bins
    layer_bins = defaultdict(list)
    
    for interp in interpretations:
        layer = interp.get('layer', -1)
        cls = interp.get('classification', -1)
        
        if layer >= 0 and cls != -1:  # Valid layer and classification
            bin_idx = layer // 8
            layer_bins[bin_idx].append(cls)
    
    if not layer_bins:
        print("No valid layer data found!")
        return
    
    # Prepare data for stacked bar chart
    bin_labels = []
    data_matrix = []
    
    for bin_idx in sorted(layer_bins.keys()):
        start_layer = bin_idx * 8
        end_layer = min(start_layer + 7, 63)
        bin_labels.append(f'L{start_layer}-{end_layer}')
        
        # Count classifications in this bin
        counts = Counter(layer_bins[bin_idx])
        total = len(layer_bins[bin_idx])
        
        # Calculate proportions
        props = [counts.get(i, 0) / total * 100 for i in range(3)]
        data_matrix.append(props)
    
    data_matrix = np.array(data_matrix).T  # Transpose for stacking
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(bin_labels))
    width = 0.6
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    labels = ['Monosemantic (0)', 'Broad but Consistent (1)', 'Polysemantic (2)']
    
    # Left plot: Stacked percentages
    bottom = np.zeros(len(bin_labels))
    bars = []
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        bar = ax1.bar(x_pos, data_matrix[i], width, bottom=bottom, 
                     color=color, label=label, edgecolor='black', linewidth=0.5)
        bars.append(bar)
        
        # Add percentage labels
        for j, (rect, val) in enumerate(zip(bar, data_matrix[i])):
            if val > 5:  # Only show label if segment is large enough
                height = rect.get_height()
                ax1.text(rect.get_x() + rect.get_width()/2., 
                        bottom[j] + height/2,
                        f'{val:.0f}%', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=10)
        
        bottom += data_matrix[i]
    
    ax1.set_xlabel('Layer Bins', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Interpretability by Layer (Stacked)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Grouped bars with counts
    width = 0.25
    x_pos = np.arange(len(bin_labels))
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        counts = []
        for bin_idx in sorted(layer_bins.keys()):
            bin_counts = Counter(layer_bins[bin_idx])
            counts.append(bin_counts.get(i, 0))
        
        offset = (i - 1) * width
        bars = ax2.bar(x_pos + offset, counts, width, color=color, 
                      label=label, edgecolor='black', linewidth=0.5)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Layer Bins', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Interpretability by Layer (Counts)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer-stratified plot to {save_path}")
    
    plt.show()
    
    # Print statistics by layer
    print("\nStatistics by Layer Bin:")
    print("-" * 70)
    print(f"{'Layer Bin':12s} {'Total':>8s} {'Mono (0)':>12s} {'Broad (1)':>12s} {'Poly (2)':>12s}")
    print("-" * 70)
    
    for bin_idx in sorted(layer_bins.keys()):
        start_layer = bin_idx * 8
        end_layer = min(start_layer + 7, 63)
        bin_label = f'L{start_layer}-{end_layer}'
        
        counts = Counter(layer_bins[bin_idx])
        total = len(layer_bins[bin_idx])
        
        mono = counts.get(0, 0)
        broad = counts.get(1, 0) 
        poly = counts.get(2, 0)
        
        print(f"{bin_label:12s} {total:8d} {mono:5d} ({mono/total*100:5.1f}%) "
              f"{broad:5d} ({broad/total*100:5.1f}%) {poly:5d} ({poly/total*100:5.1f}%)")

def plot_by_adapter_type(interpretations, save_path=None):
    """Bonus: Plot distribution by adapter type."""
    # Group by adapter type
    adapter_groups = defaultdict(list)
    
    for interp in interpretations:
        adapter = interp.get('adapter_type', 'unknown')
        cls = interp.get('classification', -1)
        
        if cls != -1:
            adapter_groups[adapter].append(cls)
    
    if not adapter_groups:
        print("No valid adapter data found!")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    adapter_labels = sorted(adapter_groups.keys())
    x_pos = np.arange(len(adapter_labels))
    width = 0.25
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Monosemantic (0)', 'Broad but Consistent (1)', 'Polysemantic (2)']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        counts = []
        for adapter in adapter_labels:
            adapter_counts = Counter(adapter_groups[adapter])
            total = len(adapter_groups[adapter])
            prop = adapter_counts.get(i, 0) / total * 100
            counts.append(prop)
        
        offset = (i - 1) * width
        bars = ax.bar(x_pos + offset, counts, width, color=color,
                     label=label, edgecolor='black', linewidth=0.5)
        
        # Add percentage labels
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Adapter Type', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Interpretability by Adapter Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(adapter_labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved adapter-type plot to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze LoRA interpretability classifications")
    parser.add_argument("input", help="Path to interpretations JSON file")
    parser.add_argument("--save-overall", default="plots/interpretability_overall.png",
                       help="Save overall distribution plot to file (default: plots/interpretability_overall.png)")
    parser.add_argument("--save-layers", default="plots/interpretability_layers.png",
                       help="Save layer-stratified plot to file (default: plots/interpretability_layers.png)")
    parser.add_argument("--save-adapters", default="plots/interpretability_adapters.png",
                       help="Save adapter-type plot to file (default: plots/interpretability_adapters.png)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save plots to files")
    parser.add_argument("--n-bootstrap", type=int, default=10000, 
                       help="Number of bootstrap samples for CI (default: 10000)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    interpretations = load_results(args.input)
    print(f"Loaded {len(interpretations)} feature interpretations")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Determine whether to save
    save_overall = None if args.no_save else args.save_overall
    save_layers = None if args.no_save else args.save_layers
    save_adapters = None if args.no_save else args.save_adapters
    
    # Generate plots
    print("\n" + "="*50)
    print("OVERALL DISTRIBUTION")
    print("="*50)
    plot_overall_distribution(interpretations, save_overall)
    
    print("\n" + "="*50)
    print("DISTRIBUTION BY LAYER BINS")
    print("="*50)
    plot_by_layer_bins(interpretations, save_layers)
    
    print("\n" + "="*50)
    print("DISTRIBUTION BY ADAPTER TYPE")
    print("="*50)
    plot_by_adapter_type(interpretations, save_adapters)

if __name__ == "__main__":
    main()