#!/usr/bin/env python3
"""
Collect max-activating examples for each SAE feature.
This script processes the activation data and trained SAE to find top examples.
"""

import torch
import numpy as np
import h5py
import json
from glob import glob
import os
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import heapq
import argparse
from batch_topk_sae import BatchTopKSAE


@dataclass
class ActivationExample:
    """Single activation example with context"""
    activation_value: float
    rollout_idx: int
    token_idx: int
    tokens: List[str]  # Context tokens
    activations: List[float]  # Activation values for context
    target_position: int  # Position of max-activating token in context


class TopKTracker:
    """Efficiently track top-k examples for a feature"""
    def __init__(self, k: int):
        self.k = k
        self.examples = []  # min heap
        self.counter = 0
        
    def add(self, activation: float, rollout_idx: int, token_idx: int):
        self.counter += 1
        if len(self.examples) < self.k:
            heapq.heappush(self.examples, (activation, self.counter, rollout_idx, token_idx))
        elif activation > self.examples[0][0]:
            heapq.heapreplace(self.examples, (activation, self.counter, rollout_idx, token_idx))
    
    def get_top_k(self) -> List[Tuple[float, int, int]]:
        """Return top k examples sorted by activation (highest first)"""
        return [(act, rid, tid) for act, _, rid, tid in 
                sorted(self.examples, key=lambda x: x[0], reverse=True)]


def load_sae_model(model_path: str, device: str = 'cuda') -> BatchTopKSAE:
    """Load the trained SAE model"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = BatchTopKSAE(
        d_model=config['d_model'],
        dict_size=config['dict_size'],
        k=config['k']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def process_rollout(
    rollout_path: str, 
    sae_model: BatchTopKSAE, 
    feature_trackers: Dict[int, TopKTracker],
    rollout_idx: int,
    device: str = 'cuda',
    batch_size: int = 512
):
    """Process a single rollout file and update feature trackers"""
    
    with h5py.File(rollout_path, 'r') as f:
        # Load activations
        activations = f['activations'][:]  # Shape: (n_tokens, 64, 3)
    
    # Flatten activations
    activations_flat = activations.reshape(-1, 192)
    n_tokens = len(activations_flat)
    
    # Process in batches
    for start_idx in range(0, n_tokens, batch_size):
        end_idx = min(start_idx + batch_size, n_tokens)
        batch = torch.from_numpy(activations_flat[start_idx:end_idx]).float().to(device)
        
        with torch.no_grad():
            # Get SAE features
            sparse_features = sae_model.encode(batch)  # Shape: (batch_size, dict_size)
            
            # Update trackers for each feature
            for feature_idx in range(sparse_features.shape[1]):
                feature_acts = sparse_features[:, feature_idx].cpu().numpy()
                
                for i, activation in enumerate(feature_acts):
                    if activation > 0:  # Only track non-zero activations
                        token_idx = start_idx + i
                        feature_trackers[feature_idx].add(
                            activation, rollout_idx, token_idx
                        )


def load_rollout_tokens(tokens_path: str) -> Dict[str, List[str]]:
    """Load the rollout tokens from JSON file"""
    with open(tokens_path, 'r') as f:
        return json.load(f)


def extract_context(
    rollout_idx: int,
    token_idx: int,
    rollout_files: List[str],
    sae_model: BatchTopKSAE,
    rollout_tokens: Optional[Dict[str, List[str]]] = None,
    context_window: int = 10,
    device: str = 'cuda'
) -> ActivationExample:
    """Extract context and activations around a specific token"""
    
    rollout_path = rollout_files[rollout_idx]
    # Extract actual rollout number from filename
    rollout_num = os.path.basename(rollout_path).replace('rollout_', '').replace('.h5', '')
    
    with h5py.File(rollout_path, 'r') as f:
        activations = f['activations'][:]
        
        # Get tokens from rollout_tokens if available
        if rollout_tokens and rollout_num in rollout_tokens:
            tokens = rollout_tokens[rollout_num]
        elif 'tokens' in f:
            tokens = [t.decode('utf-8') if isinstance(t, bytes) else t 
                     for t in f['tokens'][:]]
        else:
            tokens = [f"<token_{i}>" for i in range(len(activations))]
    
    # Flatten activations
    activations_flat = activations.reshape(-1, 192)
    n_activations = len(activations_flat)
    
    # Check for length mismatch
    if len(tokens) != n_activations:
        print(f"Warning: token/activation length mismatch: {len(tokens)} tokens vs {n_activations} activations")
        # Adjust tokens if needed
        if len(tokens) > n_activations:
            tokens = tokens[:n_activations]
        else:
            tokens = tokens + [f"<pad_{i}>" for i in range(n_activations - len(tokens))]
    
    # Determine context bounds
    start = max(0, token_idx - context_window)
    end = min(n_activations, token_idx + context_window + 1)
    
    # Extract context tokens
    context_tokens = tokens[start:end]
    target_position = token_idx - start
    
    # Compute SAE features for context
    context_acts = activations_flat[start:end]
    context_tensor = torch.from_numpy(context_acts).float().to(device)
    
    with torch.no_grad():
        sparse_features = sae_model.encode(context_tensor)
    
    # Debug shape issues
    if target_position >= sparse_features.shape[0]:
        print(f"Warning: target_position {target_position} >= sparse_features.shape[0] {sparse_features.shape[0]}")
        print(f"Context window: start={start}, end={end}, token_idx={token_idx}")
        print(f"Context length: {len(context_tokens)}, sparse_features shape: {sparse_features.shape}")
    
    # Get the specific feature's activation value
    feature_acts = sparse_features.cpu().numpy()
    
    # We'll store the activation of the target token for all features
    # The specific feature will be extracted later
    target_activation = float(sparse_features[target_position, :].max()) if target_position < sparse_features.shape[0] else 0.0
    
    return ActivationExample(
        activation_value=target_activation,
        rollout_idx=rollout_idx,
        token_idx=token_idx,
        tokens=context_tokens,
        activations=feature_acts[target_position].tolist(),  # Store all feature activations
        target_position=target_position
    ), sparse_features


def main():
    parser = argparse.ArgumentParser(description='Collect SAE feature max-activating examples')
    parser.add_argument('--model-path', type=str, default='trained_sae.pt',
                       help='Path to trained SAE model')
    parser.add_argument('--data-dir', type=str, 
                       default='../lora-activations-dashboard/backend/activations',
                       help='Directory containing activation H5 files')
    parser.add_argument('--tokens-path', type=str,
                       default='../lora-activations-dashboard/backend/rollout_tokens.json',
                       help='Path to rollout_tokens.json')
    parser.add_argument('--output-path', type=str, default='sae_features_data.json',
                       help='Output JSON file path')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top examples per feature')
    parser.add_argument('--context-window', type=int, default=10,
                       help='Context tokens on each side')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--max-rollouts', type=int, default=None,
                       help='Maximum number of rollouts to process')
    
    args = parser.parse_args()
    
    # Load SAE model
    print("Loading SAE model...")
    sae_model, config = load_sae_model(args.model_path, args.device)
    n_features = config['dict_size']
    print(f"Loaded SAE with {n_features} features")
    
    # Initialize trackers for each feature
    feature_trackers = {i: TopKTracker(args.top_k) for i in range(n_features)}
    
    # Load rollout tokens
    print("Loading rollout tokens...")
    try:
        rollout_tokens = load_rollout_tokens(args.tokens_path)
        print(f"Loaded tokens for {len(rollout_tokens)} rollouts")
    except:
        print("Warning: Could not load rollout_tokens.json, will use placeholder tokens")
        rollout_tokens = None
    
    # Get rollout files
    rollout_files = sorted(glob(os.path.join(args.data_dir, 'rollout_*.h5')))
    if args.max_rollouts:
        rollout_files = rollout_files[:args.max_rollouts]
    print(f"Processing {len(rollout_files)} rollout files...")
    
    # Phase 1: Find top activating tokens
    for rollout_idx, rollout_path in enumerate(tqdm(rollout_files, desc="Finding top activations")):
        process_rollout(
            rollout_path, sae_model, feature_trackers, 
            rollout_idx, args.device
        )
    
    # Phase 2: Extract contexts for top examples
    print("\nExtracting contexts for top examples...")
    features_data = {}
    
    # Count features with examples
    active_features = [i for i in range(n_features) 
                      if len(feature_trackers[i].get_top_k()) > 0]
    print(f"Found {len(active_features)} active features out of {n_features} total")
    
    for feature_idx in tqdm(active_features, desc="Extracting contexts"):
        top_examples = feature_trackers[feature_idx].get_top_k()
        
        if not top_examples:
            continue
        
        feature_examples = []
        for activation_value, rollout_idx, token_idx in top_examples:
            example, sparse_features = extract_context(
                rollout_idx, token_idx, rollout_files,
                sae_model, rollout_tokens, args.context_window, args.device
            )
            
            # Update with correct activation value for this specific feature
            example.activation_value = float(sparse_features[example.target_position, feature_idx])
            
            # Only store activations for this specific feature across context
            example.activations = sparse_features[:, feature_idx].cpu().numpy().tolist()
            
            feature_examples.append(asdict(example))
        
        features_data[str(feature_idx)] = {
            'examples': feature_examples,
            'stats': {
                'max_activation': max(ex['activation_value'] for ex in feature_examples) if feature_examples else 0,
                'n_examples': len(feature_examples)
            }
        }
    
    # Add metadata
    output_data = {
        'metadata': {
            'n_features': n_features,
            'top_k': args.top_k,
            'context_window': args.context_window,
            'n_rollouts_processed': len(rollout_files),
            'sae_config': config
        },
        'features': features_data
    }
    
    # Save to JSON
    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")


if __name__ == '__main__':
    main()