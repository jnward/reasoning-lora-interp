#!/usr/bin/env python3
"""Resume splitting the attention data from where it left off."""

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
    cache_file = "attention_kl_cache/attention_kl_22f4b57d32750951bab13a0e68a6a4b1.pkl"
    
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
    
    output_dir = "attention_kl_data"
    
    # Check what's already done
    print("\nChecking existing files...")
    existing_heads = set()
    heads_dir = os.path.join(output_dir, "heads")
    for filename in os.listdir(heads_dir):
        if filename.endswith('.json'):
            existing_heads.add(filename[:-5])  # Remove .json
    
    print(f"Found {len(existing_heads)} existing head files")
    
    # Resume head data
    print("\nResuming head data processing...")
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            key = f"{layer_idx}_{head_idx}"
            if key in existing_heads:
                continue  # Skip already processed
                
            head_data = {
                'kl_divergences': kl_divergences[layer_idx][head_idx],
                'layer': layer_idx,
                'head': head_idx
            }
            
            # Add attention patterns if available
            if attention_patterns:
                head_data['attention_patterns'] = {
                    'base': attention_patterns['base'].get(key, {}),
                    'lora': attention_patterns['lora'].get(key, {})
                }
            
            # Convert numpy types to Python native types
            head_data = convert_numpy_to_python(head_data)
            
            filename = os.path.join(output_dir, "heads", f"{key}.json")
            with open(filename, 'w') as f:
                json.dump(head_data, f)
            
            print(f"Created {key}.json")
    
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
    
    print("\nDone! All files created successfully.")

if __name__ == "__main__":
    main()