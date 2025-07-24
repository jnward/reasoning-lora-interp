#!/usr/bin/env python3
"""Split the large attention KL data into separate JSON files for dynamic loading."""

import json
import os
import pickle
from tqdm import tqdm
import numpy as np

def convert_numpy_to_python(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def main():
    # Load the cached data
    cache_file = "attention_kl_cache/attention_kl_67b7816b596895e32fb5fe83e5b8523f.pkl"
    
    if not os.path.exists(cache_file):
        print(f"Error: Cache file {cache_file} not found!")
        print("Please run attention_kl_dashboard_generator_fast.py first to generate the cache.")
        return
    
    print(f"Loading data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract components
    kl_divergences = data['kl_divergences']
    tokens = data['tokens']
    head_stats = data['head_stats']
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    seq_len = data['seq_len']
    attention_patterns = data.get('attention_patterns', None)
    
    # Create output directory structure
    output_dir = "attention_kl_data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "heads"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "layer_avg"), exist_ok=True)
    
    print("\nCreating metadata file...")
    # Save metadata
    metadata = {
        'tokens': tokens,
        'head_stats': head_stats,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'seq_len': seq_len,
        'has_attention_patterns': attention_patterns is not None
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    # Save individual head data
    print("\nSplitting head data...")
    for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
        for head_idx in range(n_heads):
            head_data = {
                'kl_divergences': kl_divergences[layer_idx][head_idx],
                'layer': layer_idx,
                'head': head_idx
            }
            
            # Add attention patterns if available
            if attention_patterns:
                key = f"{layer_idx}_{head_idx}"
                head_data['attention_patterns'] = {
                    'base': attention_patterns['base'].get(key, {}),
                    'lora': attention_patterns['lora'].get(key, {})
                }
            
            # Convert numpy types to Python native types
            head_data = convert_numpy_to_python(head_data)
            
            filename = os.path.join(output_dir, "heads", f"{layer_idx}_{head_idx}.json")
            with open(filename, 'w') as f:
                json.dump(head_data, f)
    
    # Save layer average data
    print("\nCreating layer average files...")
    for layer_idx in tqdm(range(n_layers), desc="Processing layer averages"):
        # Compute layer average KL divergence
        layer_kl = []
        for pos in range(seq_len):
            avg_kl = sum(kl_divergences[layer_idx][h][pos] for h in range(n_heads)) / n_heads
            layer_kl.append(avg_kl)
        
        layer_data = {
            'kl_divergences': layer_kl,
            'layer': layer_idx
        }
        
        # Add attention patterns if available
        if attention_patterns:
            layer_data['attention_patterns'] = {
                'base': attention_patterns['base_layer_avg'].get(layer_idx, {}),
                'lora': attention_patterns['lora_layer_avg'].get(layer_idx, {})
            }
        
        # Convert numpy types
        layer_data = convert_numpy_to_python(layer_data)
        
        filename = os.path.join(output_dir, "layer_avg", f"{layer_idx}.json")
        with open(filename, 'w') as f:
            json.dump(layer_data, f)
    
    # Save overall average data
    print("\nCreating overall average file...")
    overall_kl = []
    for pos in range(seq_len):
        total_kl = 0
        for l in range(n_layers):
            for h in range(n_heads):
                total_kl += kl_divergences[l][h][pos]
        overall_kl.append(total_kl / (n_layers * n_heads))
    
    overall_data = {
        'kl_divergences': overall_kl
    }
    
    # Add attention patterns if available
    if attention_patterns:
        overall_data['attention_patterns'] = {
            'base': attention_patterns.get('base_overall_avg', {}),
            'lora': attention_patterns.get('lora_overall_avg', {})
        }
    
    # Convert numpy types
    overall_data = convert_numpy_to_python(overall_data)
    
    with open(os.path.join(output_dir, "overall.json"), 'w') as f:
        json.dump(overall_data, f)
    
    # Print summary
    total_files = 1 + n_layers * n_heads + n_layers + 1  # metadata + heads + layers + overall
    print(f"\nSuccessfully created {total_files} JSON files in '{output_dir}/'")
    
    # Check file sizes
    print("\nFile size summary:")
    metadata_size = os.path.getsize(os.path.join(output_dir, "metadata.json")) / 1024
    print(f"  metadata.json: {metadata_size:.1f} KB")
    
    # Sample a few head files
    sample_files = [
        os.path.join(output_dir, "heads", "0_0.json"),
        os.path.join(output_dir, "layer_avg", "0.json"),
        os.path.join(output_dir, "overall.json")
    ]
    
    for file in sample_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"  {os.path.basename(file)}: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()