"""
Synthetic data generation for testing.
"""

import numpy as np
import pandas as pd
import json
import h5py
from pathlib import Path
from typing import Dict, List, Tuple


def generate_synthetic_activations(
    num_rollouts: int = 3,
    num_tokens: int = 100,
    num_layers: int = 3,
    num_features: int = 12,
    adapter_types: List[str] = None,
    output_dir: str = "test_data"
) -> Tuple[str, str, str]:
    """
    Generate synthetic LoRA and SAE activation data.
    
    Returns paths to:
        - LoRA activations directory
        - SAE features JSON
        - Categories JSON
    """
    if adapter_types is None:
        adapter_types = ['gate_proj', 'up_proj', 'down_proj']
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create LoRA activations directory
    lora_dir = output_path / "lora_acts"
    lora_dir.mkdir(exist_ok=True)
    
    # Generate H5 files with LoRA activations
    for rollout_idx in range(num_rollouts):
        h5_path = lora_dir / f"rollout_{rollout_idx}.h5"
        
        # Generate random activations with some structure
        # Make some sites more active than others
        activations = np.zeros((num_tokens, num_layers, len(adapter_types)))
        
        for layer in range(num_layers):
            for adapter_idx in range(len(adapter_types)):
                # Create sparse activations with varying magnitudes
                if np.random.rand() > 0.3:  # 70% chance of activity
                    active_tokens = np.random.choice(num_tokens, size=num_tokens//3, replace=False)
                    activations[active_tokens, layer, adapter_idx] = np.random.randn(len(active_tokens)) * (layer + 1) * 0.1
        
        # Save to H5
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('activations', data=activations)
            f.attrs['adapter_types'] = adapter_types
            f.attrs['num_layers'] = num_layers
            f.attrs['num_tokens'] = num_tokens
            f.attrs['rollout_idx'] = rollout_idx
    
    # Generate SAE features data
    sae_features = {'features': {}, 'metadata': {}}
    
    for feat_id in range(num_features):
        examples = []
        
        # Create sparse activations for this feature
        for rollout_idx in range(num_rollouts):
            # Each feature activates on ~10% of tokens
            num_active = max(1, num_tokens // 10)
            active_tokens = np.random.choice(num_tokens, size=num_active, replace=False)
            
            for token_idx in active_tokens:
                activation_value = np.random.exponential(2.0) * (feat_id + 1) / num_features
                
                examples.append({
                    'activation_value': float(activation_value),
                    'rollout_idx': rollout_idx,
                    'token_idx': int(token_idx),
                    'token': f'tok_{token_idx}',
                    'tokens': [f'tok_{i}' for i in range(max(0, token_idx-2), min(num_tokens, token_idx+3))],
                    'target_position': 2,
                    'activations': [float(activation_value) if i == 2 else 0.0 
                                    for i in range(5)]
                })
        
        sae_features['features'][str(feat_id)] = {
            'examples': examples,
            'stats': {'mean_activation': float(np.mean([e['activation_value'] for e in examples]))}
        }
    
    sae_path = output_path / "sae_features.json"
    with open(sae_path, 'w') as f:
        json.dump(sae_features, f)
    
    # Generate categories and labels
    categories = [
        {'id': 'math', 'label': 'Mathematics'},
        {'id': 'reasoning', 'label': 'Reasoning'},
        {'id': 'syntax', 'label': 'Syntax'},
        {'id': 'misc', 'label': 'Miscellaneous'}
    ]
    
    categories_path = output_path / "categories.json"
    with open(categories_path, 'w') as f:
        json.dump(categories, f)
    
    # Generate feature labels
    labels = {
        'explanations': [],
        'metadata': {'num_features': num_features}
    }
    
    for feat_id in range(num_features):
        # Assign features to categories with some pattern
        if feat_id % 4 == 0:
            category = 'math'
        elif feat_id % 4 == 1:
            category = 'reasoning'
        elif feat_id % 4 == 2:
            category = 'syntax'
        else:
            category = 'misc'
        
        labels['explanations'].append({
            'feature_id': feat_id,
            'category_id': category,
            'explanation': f'Test feature {feat_id}',
            'confidence': 0.8 + np.random.rand() * 0.2
        })
    
    labels_path = output_path / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    
    return str(lora_dir), str(sae_path), str(labels_path)


def generate_preaggregated_data(
    num_features: int = 20,
    num_sites: int = 12,
    categories: List[str] = None,
    output_path: str = "test_data/features.csv"
) -> str:
    """
    Generate pre-aggregated features CSV.
    """
    if categories is None:
        categories = ['cat_a', 'cat_b', 'cat_c']
    
    rows = []
    
    for feat_id in range(num_features):
        # Assign category
        category = categories[feat_id % len(categories)]
        
        # Generate site contributions
        site_contrib = {}
        active_sites = np.random.choice(num_sites, size=np.random.randint(1, min(5, num_sites)), replace=False)
        
        for site_idx in active_sites:
            site_name = f"L{site_idx // 3}.adapter_{site_idx % 3}"
            site_contrib[site_name] = float(np.random.exponential(1.0))
        
        # Compute mass as sum of contributions
        mass = sum(site_contrib.values())
        
        rows.append({
            'feature_id': f'f{feat_id}',
            'category': category,
            'conf': 0.8 + np.random.rand() * 0.2,
            'mass': mass,
            'site_contrib': json.dumps(site_contrib)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    return output_path