#!/usr/bin/env python3
"""
Analyze interpretability classifications from MLP autointerp results.
Generates bar charts and bootstrapped confidence intervals.
Adapted from analyze_interpretability.py for MLP neurons.
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
    ax1.set_title('MLP Neuron Interpretability Distribution', fontsize=14, fontweight='bold')
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
    fig.suptitle(f'MLP Neurons - Total Features Analyzed: {total}', fontsize=16, fontweight='bold', y=1.02)
    
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
    ax1.set_title('MLP Neuron Interpretability by Layer (Stacked)', fontsize=14, fontweight='bold')
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
    ax2.set_title('MLP Neuron Interpretability by Layer (Counts)', fontsize=14, fontweight='bold')
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

def plot_by_neuron_index(interpretations, save_path=None):
    """Plot distribution by neuron index (0-5) instead of adapter type."""
    # Group by neuron index
    neuron_groups = defaultdict(list)
    
    for interp in interpretations:
        neuron_idx = interp.get('neuron_idx', -1)
        cls = interp.get('classification', -1)
        
        if neuron_idx >= 0 and cls != -1:
            neuron_groups[neuron_idx].append(cls)
    
    if not neuron_groups:
        print("No valid neuron index data found!")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    neuron_labels = [f'Neuron {i}' for i in sorted(neuron_groups.keys())]
    x_pos = np.arange(len(neuron_labels))
    width = 0.25
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Monosemantic (0)', 'Broad but Consistent (1)', 'Polysemantic (2)']
    
    # Left plot: Percentages
    for i, (color, label) in enumerate(zip(colors, labels)):
        percentages = []
        for neuron_idx in sorted(neuron_groups.keys()):
            neuron_counts = Counter(neuron_groups[neuron_idx])
            total = len(neuron_groups[neuron_idx])
            prop = neuron_counts.get(i, 0) / total * 100
            percentages.append(prop)
        
        offset = (i - 1) * width
        bars = ax1.bar(x_pos + offset, percentages, width, color=color,
                      label=label, edgecolor='black', linewidth=0.5)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Neuron Index', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('MLP Interpretability by Neuron Index (Percentages)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(neuron_labels)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Counts
    for i, (color, label) in enumerate(zip(colors, labels)):
        counts = []
        for neuron_idx in sorted(neuron_groups.keys()):
            neuron_counts = Counter(neuron_groups[neuron_idx])
            counts.append(neuron_counts.get(i, 0))
        
        offset = (i - 1) * width
        bars = ax2.bar(x_pos + offset, counts, width, color=color,
                      label=label, edgecolor='black', linewidth=0.5)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Neuron Index', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('MLP Interpretability by Neuron Index (Counts)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(neuron_labels)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved neuron-index plot to {save_path}")
    
    plt.show()
    
    # Print statistics by neuron index
    print("\nStatistics by Neuron Index:")
    print("-" * 70)
    print(f"{'Neuron':12s} {'Total':>8s} {'Mono (0)':>12s} {'Broad (1)':>12s} {'Poly (2)':>12s}")
    print("-" * 70)
    
    for neuron_idx in sorted(neuron_groups.keys()):
        counts = Counter(neuron_groups[neuron_idx])
        total = len(neuron_groups[neuron_idx])
        
        mono = counts.get(0, 0)
        broad = counts.get(1, 0)
        poly = counts.get(2, 0)
        
        print(f"Neuron {neuron_idx:5d} {total:8d} {mono:5d} ({mono/total*100:5.1f}%) "
              f"{broad:5d} ({broad/total*100:5.1f}%) {poly:5d} ({poly/total*100:5.1f}%)")

def plot_by_polarity(interpretations, save_path=None):
    """Bonus: Plot distribution by dominant polarity (positive vs negative)."""
    # Group by dominant polarity
    polarity_groups = defaultdict(list)
    
    for interp in interpretations:
        polarity = interp.get('dominant_polarity', 'unknown')
        cls = interp.get('classification', -1)
        
        if cls != -1 and polarity in ['positive', 'negative']:
            polarity_groups[polarity].append(cls)
    
    if not polarity_groups:
        print("No valid polarity data found!")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    polarity_labels = ['Positive', 'Negative']
    x_pos = np.arange(len(polarity_labels))
    width = 0.25
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Monosemantic (0)', 'Broad but Consistent (1)', 'Polysemantic (2)']
    
    # Left plot: Percentages
    for i, (color, label) in enumerate(zip(colors, labels)):
        percentages = []
        for polarity in ['positive', 'negative']:
            if polarity in polarity_groups:
                polarity_counts = Counter(polarity_groups[polarity])
                total = len(polarity_groups[polarity])
                prop = polarity_counts.get(i, 0) / total * 100
                percentages.append(prop)
            else:
                percentages.append(0)
        
        offset = (i - 1) * width
        bars = ax1.bar(x_pos + offset, percentages, width, color=color,
                      label=label, edgecolor='black', linewidth=0.5)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Dominant Polarity', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('MLP Interpretability by Dominant Polarity', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(polarity_labels)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Counts
    pos_total = len(polarity_groups.get('positive', []))
    neg_total = len(polarity_groups.get('negative', []))
    
    totals = [pos_total, neg_total]
    bars = ax2.bar(x_pos, totals, color=['#3498db', '#e67e22'], edgecolor='black', linewidth=1.5)
    
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{total}', ha='center', va='bottom', fontsize=11)
    
    ax2.set_xlabel('Dominant Polarity', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Dominant Polarities', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(polarity_labels)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved polarity plot to {save_path}")
    
    plt.show()
    
    # Print polarity statistics
    print("\nPolarity Distribution:")
    print("-" * 40)
    print(f"Positive dominant: {pos_total} ({pos_total/(pos_total+neg_total)*100:.1f}%)")
    print(f"Negative dominant: {neg_total} ({neg_total/(pos_total+neg_total)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze MLP neuron interpretability classifications")
    parser.add_argument("input", help="Path to MLP interpretations JSON file")
    parser.add_argument("--save-overall", default="plots/mlp_interpretability_overall.png",
                       help="Save overall distribution plot to file")
    parser.add_argument("--save-layers", default="plots/mlp_interpretability_layers.png",
                       help="Save layer-stratified plot to file")
    parser.add_argument("--save-neurons", default="plots/mlp_interpretability_neurons.png",
                       help="Save neuron-index plot to file")
    parser.add_argument("--save-polarity", default="plots/mlp_interpretability_polarity.png",
                       help="Save polarity plot to file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save plots to files")
    parser.add_argument("--n-bootstrap", type=int, default=10000, 
                       help="Number of bootstrap samples for CI (default: 10000)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    interpretations = load_results(args.input)
    print(f"Loaded {len(interpretations)} MLP neuron interpretations")
    
    # Filter out errors for analysis
    valid_interps = [i for i in interpretations if i.get('classification', -1) != -1]
    print(f"Valid interpretations (excluding errors): {len(valid_interps)}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Determine whether to save
    save_overall = None if args.no_save else args.save_overall
    save_layers = None if args.no_save else args.save_layers
    save_neurons = None if args.no_save else args.save_neurons
    save_polarity = None if args.no_save else args.save_polarity
    
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
    print("DISTRIBUTION BY NEURON INDEX")
    print("="*50)
    plot_by_neuron_index(interpretations, save_neurons)
    
    print("\n" + "="*50)
    print("DISTRIBUTION BY DOMINANT POLARITY")
    print("="*50)
    plot_by_polarity(interpretations, save_polarity)

if __name__ == "__main__":
    main()