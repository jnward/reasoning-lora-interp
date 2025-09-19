#!/usr/bin/env python3
"""
Compute activation densities for categorized SAE features.

This script processes cached activations through the trained SAE to compute
per-feature activation totals, then aggregates by category and subcategory.
"""

import json
import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append('/workspace/reasoning_interp/sae_interp')
from batch_topk_sae import BatchTopKSAE


def load_sae_model(model_path, device='cuda'):
    """Load the trained SAE model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters from config
    config = checkpoint['config']
    d_model = int(config['d_model'])  # Convert numpy int to Python int
    dict_size = config['dict_size']
    k = config['k']
    
    # Create model
    model = BatchTopKSAE(d_model, dict_size, k)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def process_activation_file(filepath, sae_model, device='cuda', batch_size=512):
    """Process a single H5 activation file and return feature activations."""
    feature_activations = torch.zeros(sae_model.dict_size, device=device)
    
    with h5py.File(filepath, 'r') as f:
        activations = f['activations'][:]  # Shape: [num_tokens, 64, 7]
        
        # Reshape to [num_tokens, 448] (7 adapters × 64 dims)
        num_tokens = activations.shape[0]
        activations_flat = activations.reshape(num_tokens, -1)
        
        # Process in batches
        for i in range(0, num_tokens, batch_size):
            batch = activations_flat[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            with torch.no_grad():
                # Get sparse feature activations
                sparse_features = sae_model.encode(batch_tensor)
                
                # Sum activations for each feature
                feature_activations += sparse_features.sum(dim=0)
    
    return feature_activations


def compute_all_densities(activations_dir, sae_model, max_files=None, device='cuda'):
    """Compute activation densities across all files."""
    activations_path = Path(activations_dir)
    h5_files = sorted(list(activations_path.glob('rollout_*.h5')))
    
    if max_files:
        h5_files = h5_files[:max_files]
    
    print(f"Processing {len(h5_files)} activation files...")
    
    # Accumulator for all feature activations
    total_feature_activations = torch.zeros(sae_model.dict_size, device=device)
    
    for filepath in tqdm(h5_files, desc="Processing files"):
        feature_acts = process_activation_file(filepath, sae_model, device)
        total_feature_activations += feature_acts
    
    return total_feature_activations.cpu().numpy()


def aggregate_by_category(feature_activations, categorized_features):
    """Aggregate activation densities by category and subcategory."""
    # Initialize aggregators
    category_totals = defaultdict(float)
    subcategory_totals = defaultdict(float)
    
    # Track category-subcategory relationships
    subcategory_to_category = {}
    
    # Process each categorized feature
    for feature_data in categorized_features['explanations']:
        feature_id = feature_data['feature_id']
        category = feature_data.get('category_name', 'uncategorized')
        subcategory = feature_data.get('subcategory_name', 'uncategorized')
        subcategory_id = feature_data.get('subcategory_id', -1)
        
        # Skip error categories
        if category == 'error' or category is None:
            category = 'uncategorized'
            subcategory = 'uncategorized'
        
        # Get activation value for this feature
        if feature_id < len(feature_activations):
            activation = feature_activations[feature_id]
            
            # Add to totals
            category_totals[category] += activation
            
            # Create unique subcategory key
            subcat_key = f"{category}::{subcategory}"
            subcategory_totals[subcat_key] += activation
            subcategory_to_category[subcat_key] = category
    
    return category_totals, subcategory_totals, subcategory_to_category


def normalize_densities(category_totals, subcategory_totals):
    """Normalize densities to sum to 1."""
    # Normalize categories
    cat_sum = sum(category_totals.values())
    if cat_sum > 0:
        category_normalized = {k: v/cat_sum for k, v in category_totals.items()}
    else:
        category_normalized = category_totals
    
    # Normalize subcategories within each category
    subcategory_normalized = {}
    
    # Group subcategories by category
    cat_subcats = defaultdict(list)
    for subcat_key in subcategory_totals:
        category = subcat_key.split('::')[0]
        cat_subcats[category].append(subcat_key)
    
    # Normalize within each category
    for category, subcat_keys in cat_subcats.items():
        subcat_sum = sum(subcategory_totals[k] for k in subcat_keys)
        if subcat_sum > 0:
            for subcat_key in subcat_keys:
                # Normalize relative to category total
                subcategory_normalized[subcat_key] = subcategory_totals[subcat_key] / subcat_sum
        else:
            for subcat_key in subcat_keys:
                subcategory_normalized[subcat_key] = 0
    
    return category_normalized, subcategory_normalized


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute SAE feature activation densities")
    parser.add_argument('--sae-model', type=str, 
                       default='/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt',
                       help='Path to trained SAE model')
    parser.add_argument('--categorized', type=str,
                       default='hierarchical_categorized.json',
                       help='Path to categorized features JSON')
    parser.add_argument('--activations-dir', type=str,
                       default='/workspace/reasoning_interp/2_lora_activation_interp/activations_all_adapters',
                       help='Directory containing H5 activation files')
    parser.add_argument('--output', type=str,
                       default='activation_densities.json',
                       help='Output JSON file for densities')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load SAE model
    print(f"Loading SAE model from {args.sae_model}...")
    sae_model = load_sae_model(args.sae_model, device=args.device)
    print(f"Model loaded: d_model={sae_model.d_model}, dict_size={sae_model.dict_size}, k={sae_model.k}")
    
    # Load categorized features
    print(f"Loading categorized features from {args.categorized}...")
    with open(args.categorized, 'r') as f:
        categorized_features = json.load(f)
    print(f"Loaded {len(categorized_features['explanations'])} categorized features")
    
    # Compute activation densities
    print("Computing activation densities...")
    feature_activations = compute_all_densities(
        args.activations_dir, 
        sae_model,
        max_files=args.max_files,
        device=args.device
    )
    
    # Aggregate by category
    print("Aggregating by category and subcategory...")
    category_totals, subcategory_totals, subcat_to_cat = aggregate_by_category(
        feature_activations, categorized_features
    )
    
    # Normalize densities
    print("Normalizing densities...")
    category_normalized, subcategory_normalized = normalize_densities(
        category_totals, subcategory_totals
    )
    
    # Convert all values to Python floats for JSON serialization
    category_normalized = {k: float(v) for k, v in category_normalized.items()}
    subcategory_normalized = {k: float(v) for k, v in subcategory_normalized.items()}
    
    # Prepare output data
    output_data = {
        'metadata': {
            'num_files_processed': args.max_files or 'all',
            'num_features': len(feature_activations),
            'total_activation': float(np.sum(feature_activations)),
            'device': args.device
        },
        'category_densities': category_normalized,
        'subcategory_densities': subcategory_normalized,
        'subcategory_to_category': subcat_to_cat,
        'raw_totals': {
            'categories': {k: float(v) for k, v in category_totals.items()},
            'subcategories': {k: float(v) for k, v in subcategory_totals.items()}
        }
    }
    
    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\nCategory distribution:")
    sorted_cats = sorted(category_normalized.items(), key=lambda x: x[1], reverse=True)
    for cat, density in sorted_cats:
        print(f"  {cat}: {density:.1%}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()