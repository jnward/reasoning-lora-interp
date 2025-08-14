"""
CLI interface for LoRA → SAE → Category flow visualization.

Usage examples:
    # Mode A: Pre-aggregated data
    python -m viz_lora_sae --mode preagg --features features.csv --out-sankey sankey.html

    # Mode B: Raw activations
    python -m viz_lora_sae --mode raw \\
        --lora-acts activations_dir/ \\
        --sae-features sae_features.json \\
        --labels categorized_features.json \\
        --categories categories.json \\
        --out-sankey sankey.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .flows import (
    load_inputs_mode_a,
    load_inputs_mode_b,
    normalize_confidences,
    aggregate_middle_nodes
)
from .figure import (
    build_sankey,
    build_category_bar,
    build_evidence_tiles,
    save_snapshot
)


def parse_categories(categories_arg: str, categories_file: Optional[str] = None) -> List[str]:
    """Parse categories from comma-separated string or JSON file."""
    if categories_file:
        with open(categories_file, 'r') as f:
            cat_data = json.load(f)
            # Extract category IDs
            if isinstance(cat_data, list):
                return [c['id'] for c in cat_data if 'id' in c]
            else:
                return list(cat_data.keys())
    elif categories_arg:
        return [c.strip() for c in categories_arg.split(',')]
    else:
        raise ValueError("Either --categories or --categories-file must be provided")


def load_sites(sites_path: Optional[str], lora_path: Optional[str] = None) -> List[str]:
    """Load site names from JSON file or infer from data."""
    if sites_path:
        with open(sites_path, 'r') as f:
            return json.load(f)
    elif lora_path:
        # Infer from H5 files
        import h5py
        from glob import glob
        
        path = Path(lora_path)
        if path.is_dir():
            h5_files = sorted(glob(str(path / "rollout_*.h5")))
            if h5_files:
                with h5py.File(h5_files[0], 'r') as f:
                    adapter_types = list(f.attrs.get('adapter_types', []))
                    num_layers = int(f.attrs.get('num_layers', 64))
                
                sites = []
                for layer in range(num_layers):
                    for adapter in adapter_types:
                        sites.append(f"L{layer}.{adapter}")
                return sites
    
    # Default fallback
    print("Warning: Could not load sites, using default configuration")
    sites = []
    for layer in range(64):
        for adapter in ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']:
            sites.append(f"L{layer}.{adapter}")
    return sites


def main():
    parser = argparse.ArgumentParser(
        description="Generate LoRA → SAE → Category flow visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['preagg', 'raw'], required=True,
                        help='Input mode: preagg (pre-aggregated) or raw (compute from activations)')
    
    # Mode A: Pre-aggregated inputs
    parser.add_argument('--features', type=str,
                        help='Path to features CSV/parquet (mode=preagg)')
    
    # Mode B: Raw activation inputs
    parser.add_argument('--lora-acts', type=str,
                        help='Path to LoRA activations directory or NPZ (mode=raw)')
    parser.add_argument('--sae-features', type=str,
                        help='Path to SAE features JSON (mode=raw)')
    parser.add_argument('--labels', type=str,
                        help='Path to feature labels/categories JSON (mode=raw)')
    
    # Common inputs
    parser.add_argument('--sites', type=str,
                        help='Path to sites.json (optional, will infer if not provided)')
    parser.add_argument('--categories', type=str,
                        help='Comma-separated list of category IDs')
    parser.add_argument('--categories-file', type=str,
                        help='Path to categories JSON file')
    
    # Processing options
    parser.add_argument('--top-k', type=int, default=6,
                        help='Number of top features to show per category (default: 6)')
    parser.add_argument('--robust-mean', action='store_true',
                        help='Use robust mean estimation (trimmed mean)')
    parser.add_argument('--max-rollouts', type=int,
                        help='Maximum number of rollouts to process (for testing)')
    
    # Output options
    parser.add_argument('--out-sankey', type=str, default='sankey.html',
                        help='Output path for Sankey diagram (default: sankey.html)')
    parser.add_argument('--out-bars', type=str, default='bars.html',
                        help='Output path for bar chart (default: bars.html)')
    parser.add_argument('--evidence', type=str,
                        help='Path to evidence CSV with feature_id,text_snippet')
    parser.add_argument('--out-evidence', type=str, default='evidence.html',
                        help='Output path for evidence tiles (if --evidence provided)')
    parser.add_argument('--snapshot', type=str, default='flows_snapshot.json',
                        help='Output path for JSON snapshot (default: flows_snapshot.json)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate mode-specific arguments
    if args.mode == 'preagg':
        if not args.features:
            parser.error("--features required for mode=preagg")
    elif args.mode == 'raw':
        if not all([args.lora_acts, args.sae_features, args.labels]):
            parser.error("--lora-acts, --sae-features, and --labels required for mode=raw")
    
    # Parse categories
    try:
        categories = parse_categories(args.categories, args.categories_file)
    except ValueError as e:
        parser.error(str(e))
    
    print(f"Processing in {args.mode} mode...")
    print(f"Categories: {categories}")
    
    # Load data based on mode
    if args.mode == 'preagg':
        print(f"Loading pre-aggregated features from {args.features}")
        df = load_inputs_mode_a(args.features)
    else:
        print(f"Loading raw activations from {args.lora_acts}")
        df = load_inputs_mode_b(
            args.lora_acts,
            args.sae_features,
            args.labels,
            robust=args.robust_mean,
            max_rollouts=args.max_rollouts
        )
    
    print(f"Loaded {len(df)} features with non-zero mass")
    
    # Normalize confidences
    df = normalize_confidences(df)
    
    # Load sites
    sites = load_sites(args.sites, args.lora_acts)
    print(f"Using {len(sites)} LoRA sites")
    
    # Aggregate nodes and build links
    print(f"Aggregating with top-{args.top_k} features per category")
    nodes, links, palette = aggregate_middle_nodes(df, categories, sites, args.top_k)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    total_mass = df['mass'].sum()
    print(f"Total mass: {total_mass:.3f}")
    print(f"Number of active features: {len(df)}")
    print(f"Number of categories: {df['category'].nunique()}")
    
    print("\nCategory breakdown:")
    for cat in categories:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            cat_mass = cat_df['mass'].sum()
            pct = (cat_mass / total_mass * 100) if total_mass > 0 else 0
            print(f"  {cat}: {cat_mass:.3f} ({pct:.1f}%), {len(cat_df)} features")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Add node colors to palette for passing to build_sankey
    palette['node_colors'] = nodes['colors']
    
    # Sankey diagram
    sankey_fig = build_sankey(nodes['labels'], links, palette)
    sankey_fig.write_html(args.out_sankey)
    print(f"Saved Sankey diagram to {args.out_sankey}")
    
    # Bar chart
    bar_fig = build_category_bar(df, categories, palette)
    bar_fig.write_html(args.out_bars)
    print(f"Saved bar chart to {args.out_bars}")
    
    # Evidence tiles (if requested)
    if args.evidence:
        print(f"Generating evidence tiles from {args.evidence}")
        evidence_fig = build_evidence_tiles(df, args.evidence, palette)
        evidence_fig.write_html(args.out_evidence)
        print(f"Saved evidence tiles to {args.out_evidence}")
    
    # Save snapshot
    params = {
        'mode': args.mode,
        'top_k': args.top_k,
        'robust_mean': args.robust_mean,
        'num_sites': len(sites),
        'num_categories': len(categories)
    }
    
    save_snapshot(nodes, links, df, params, args.snapshot)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())