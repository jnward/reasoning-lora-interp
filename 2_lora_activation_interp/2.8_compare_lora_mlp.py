#!/usr/bin/env python3
"""
Compare LoRA and MLP interpretability with bootstrapped confidence intervals.
Creates side-by-side bar charts with 95% CI error bars.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import argparse
from scipy import stats

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
    if n == 0:
        return {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0)}
    
    classifications = np.array(data)
    
    # Bootstrap samples
    bootstrap_props = defaultdict(list)
    
    np.random.seed(42)  # For reproducibility
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

def plot_comparison_with_ci(lora_interps, mlp_interps, save_path=None):
    """Create side-by-side comparison with confidence intervals."""
    
    # Extract valid classifications (excluding errors)
    lora_classifications = [i['classification'] for i in lora_interps 
                           if i.get('classification', -1) != -1]
    mlp_classifications = [i['classification'] for i in mlp_interps 
                          if i.get('classification', -1) != -1]
    
    # Calculate confidence intervals for both
    lora_ci = bootstrap_confidence_intervals(lora_classifications)
    mlp_ci = bootstrap_confidence_intervals(mlp_classifications)
    
    # Prepare data
    categories = ['Monosemantic\n(0)', 'Broad but\nConsistent (1)', 'Polysemantic\n(2)']
    x = np.arange(len(categories))
    width = 0.35
    
    # Colors
    lora_color = '#3498db'  # Blue
    mlp_color = '#e74c3c'   # Red
    
    # Extract proportions and CIs
    lora_props = []
    lora_errors_lower = []
    lora_errors_upper = []
    mlp_props = []
    mlp_errors_lower = []
    mlp_errors_upper = []
    
    for i in range(3):
        # LoRA
        mean, lower, upper = lora_ci[i]
        lora_props.append(mean * 100)
        lora_errors_lower.append((mean - lower) * 100)
        lora_errors_upper.append((upper - mean) * 100)
        
        # MLP
        mean, lower, upper = mlp_ci[i]
        mlp_props.append(mean * 100)
        mlp_errors_lower.append((mean - lower) * 100)
        mlp_errors_upper.append((upper - mean) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, lora_props, width, label='LoRA Features',
                   color=lora_color, edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, mlp_props, width, label='MLP Neurons',
                   color=mlp_color, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add error bars
    ax.errorbar(x - width/2, lora_props, yerr=[lora_errors_lower, lora_errors_upper],
                fmt='none', color='black', capsize=5, capthick=2, linewidth=1.5)
    ax.errorbar(x + width/2, mlp_props, yerr=[mlp_errors_lower, mlp_errors_upper],
                fmt='none', color='black', capsize=5, capthick=2, linewidth=1.5)
    
    # Add value labels
    for bar, prop in zip(bars1, lora_props):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(lora_errors_upper) + 1,
                f'{prop:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar, prop in zip(bars2, mlp_props):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(mlp_errors_upper) + 1,
                f'{prop:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Classification Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Interpretability Comparison: LoRA vs MLP\nwith 95% Bootstrapped Confidence Intervals',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.set_ylim(0, max(max(lora_props), max(mlp_props)) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample size information
    n_lora = len(lora_classifications)
    n_mlp = len(mlp_classifications)
    info_text = f'Sample sizes: LoRA n={n_lora}, MLP n={n_mlp}'
    ax.text(0.99, 0.99, info_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add statistical test (chi-square)
    if n_lora > 0 and n_mlp > 0:
        # Create contingency table
        lora_counts = Counter(lora_classifications)
        mlp_counts = Counter(mlp_classifications)
        observed = np.array([[lora_counts.get(i, 0) for i in range(3)],
                            [mlp_counts.get(i, 0) for i in range(3)]])
        
        chi2, p_value, _, _ = stats.chi2_contingency(observed)
        
        # Add significance stars
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        
        stat_text = f'χ² = {chi2:.2f}, p = {p_value:.4f} {sig_text}'
        ax.text(0.5, 0.99, stat_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("DETAILED STATISTICS WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    
    print(f"\nLoRA Features (n={n_lora}):")
    print("-" * 40)
    for i, label in enumerate(['Monosemantic', 'Broad/Consistent', 'Polysemantic']):
        mean, lower, upper = lora_ci[i]
        print(f"{label:20s}: {mean*100:5.1f}% [{lower*100:5.1f}%, {upper*100:5.1f}%]")
    
    print(f"\nMLP Neurons (n={n_mlp}):")
    print("-" * 40)
    for i, label in enumerate(['Monosemantic', 'Broad/Consistent', 'Polysemantic']):
        mean, lower, upper = mlp_ci[i]
        print(f"{label:20s}: {mean*100:5.1f}% [{lower*100:5.1f}%, {upper*100:5.1f}%]")
    
    # Calculate effect sizes (differences)
    print("\nDifferences (LoRA - MLP):")
    print("-" * 40)
    for i, label in enumerate(['Monosemantic', 'Broad/Consistent', 'Polysemantic']):
        lora_mean = lora_ci[i][0] * 100
        mlp_mean = mlp_ci[i][0] * 100
        diff = lora_mean - mlp_mean
        sign = "+" if diff > 0 else ""
        print(f"{label:20s}: {sign}{diff:5.1f}%")
    
    if n_lora > 0 and n_mlp > 0:
        print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.6f}")
        if p_value < 0.05:
            print("Result: Statistically significant difference between LoRA and MLP distributions")
        else:
            print("Result: No statistically significant difference")

def plot_layer_comparison(lora_interps, mlp_interps, save_path=None):
    """Compare LoRA vs MLP by layer bins with confidence intervals."""
    
    # Group by layer bins
    lora_bins = defaultdict(list)
    mlp_bins = defaultdict(list)
    
    for interp in lora_interps:
        layer = interp.get('layer', -1)
        cls = interp.get('classification', -1)
        if layer >= 0 and cls != -1:
            bin_idx = layer // 8
            lora_bins[bin_idx].append(cls)
    
    for interp in mlp_interps:
        layer = interp.get('layer', -1)
        cls = interp.get('classification', -1)
        if layer >= 0 and cls != -1:
            bin_idx = layer // 8
            mlp_bins[bin_idx].append(cls)
    
    # Calculate monosemantic percentage for each bin
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bin_labels = []
    lora_mono_props = []
    lora_mono_errors = []
    mlp_mono_props = []
    mlp_mono_errors = []
    
    for bin_idx in range(8):  # 0-7, 8-15, ..., 56-63
        start_layer = bin_idx * 8
        end_layer = min(start_layer + 7, 63)
        bin_labels.append(f'L{start_layer}-{end_layer}')
        
        # LoRA monosemantic percentage with CI
        if bin_idx in lora_bins and len(lora_bins[bin_idx]) > 0:
            ci_results = bootstrap_confidence_intervals(lora_bins[bin_idx], n_bootstrap=5000)
            mean, lower, upper = ci_results[0]  # Monosemantic (0)
            lora_mono_props.append(mean * 100)
            lora_mono_errors.append([(mean - lower) * 100, (upper - mean) * 100])
        else:
            lora_mono_props.append(0)
            lora_mono_errors.append([0, 0])
        
        # MLP monosemantic percentage with CI
        if bin_idx in mlp_bins and len(mlp_bins[bin_idx]) > 0:
            ci_results = bootstrap_confidence_intervals(mlp_bins[bin_idx], n_bootstrap=5000)
            mean, lower, upper = ci_results[0]  # Monosemantic (0)
            mlp_mono_props.append(mean * 100)
            mlp_mono_errors.append([(mean - lower) * 100, (upper - mean) * 100])
        else:
            mlp_mono_props.append(0)
            mlp_mono_errors.append([0, 0])
    
    # Plot
    x = np.arange(len(bin_labels))
    width = 0.35
    
    # Convert error format for matplotlib
    lora_errors_T = np.array(lora_mono_errors).T
    mlp_errors_T = np.array(mlp_mono_errors).T
    
    bars1 = ax.bar(x - width/2, lora_mono_props, width, label='LoRA Features',
                   color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, mlp_mono_props, width, label='MLP Neurons',
                   color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add error bars
    ax.errorbar(x - width/2, lora_mono_props, yerr=lora_errors_T,
                fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1)
    ax.errorbar(x + width/2, mlp_mono_props, yerr=mlp_errors_T,
                fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1)
    
    # Add value labels
    for bar, prop in zip(bars1, lora_mono_props):
        if prop > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prop:.0f}%', ha='center', va='bottom', fontsize=9)
    
    for bar, prop in zip(bars2, mlp_mono_props):
        if prop > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prop:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Layer Bins', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monosemantic Features (%)', fontsize=14, fontweight='bold')
    ax.set_title('Monosemantic Features by Layer: LoRA vs MLP\nwith 95% Confidence Intervals',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(lora_mono_props), max(mlp_mono_props)) * 1.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer comparison to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare LoRA and MLP with confidence intervals")
    parser.add_argument("--lora-results", required=True,
                       help="Path to LoRA interpretations JSON")
    parser.add_argument("--mlp-results", required=True,
                       help="Path to MLP interpretations JSON")
    parser.add_argument("--output", default="plots/lora_mlp_confidence_comparison.png",
                       help="Output file for main comparison plot")
    parser.add_argument("--output-layers", default="plots/lora_mlp_layers_comparison.png",
                       help="Output file for layer comparison plot")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                       help="Number of bootstrap samples (default: 10000)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading LoRA results from {args.lora_results}...")
    lora_interps = load_results(args.lora_results)
    
    print(f"Loading MLP results from {args.mlp_results}...")
    mlp_interps = load_results(args.mlp_results)
    
    print(f"Loaded {len(lora_interps)} LoRA features and {len(mlp_interps)} MLP neurons")
    
    # Filter valid interpretations
    lora_valid = [i for i in lora_interps if i.get('classification', -1) != -1]
    mlp_valid = [i for i in mlp_interps if i.get('classification', -1) != -1]
    print(f"Valid: {len(lora_valid)} LoRA, {len(mlp_valid)} MLP")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    
    # Generate plots
    print("\nGenerating comparison plots with confidence intervals...")
    
    # Main comparison
    plot_comparison_with_ci(lora_interps, mlp_interps, args.output)
    
    # Layer-wise comparison
    plot_layer_comparison(lora_interps, mlp_interps, args.output_layers)
    
    print(f"\nPlots saved to:")
    print(f"  - {args.output}")
    print(f"  - {args.output_layers}")

if __name__ == "__main__":
    main()