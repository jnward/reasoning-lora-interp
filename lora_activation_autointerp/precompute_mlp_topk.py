#!/usr/bin/env python3
"""
Precomputation of top-k examples for MLP neurons with correct polarity handling.
Each neuron is treated as a single feature that can activate in either direction.
"""

import json
import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc


@dataclass
class FeatureExample:
    """Single example for a feature."""
    rollout_idx: int
    token_idx: int
    activation_value: float  # Always stored as absolute value
    original_value: float   # Original value with sign
    context_start: int
    context_end: int
    tokens: List[str]
    activations: List[float]  # Original activations with sign


class DualPolarityTracker:
    """Tracks both positive and negative activations for neurons."""
    
    def __init__(self, k: int, n_features: int):
        """
        Initialize tracker for both polarities.
        
        Args:
            k: Number of top examples to track per polarity
            n_features: Total number of neurons
        """
        self.k = k
        self.n_features = n_features
        
        # Track positive activations
        self.pos_values = np.full((n_features, k), -np.inf, dtype=np.float32)
        self.pos_rollout_idx = np.zeros((n_features, k), dtype=np.int32)
        self.pos_token_idx = np.zeros((n_features, k), dtype=np.int32)
        self.pos_min = np.full(n_features, -np.inf, dtype=np.float32)
        
        # Track negative activations (by absolute value)
        self.neg_values = np.full((n_features, k), -np.inf, dtype=np.float32)
        self.neg_rollout_idx = np.zeros((n_features, k), dtype=np.int32)
        self.neg_token_idx = np.zeros((n_features, k), dtype=np.int32)
        self.neg_min = np.full(n_features, -np.inf, dtype=np.float32)
    
    def update_positive(self, feature_idx: int, values: np.ndarray, 
                       token_indices: np.ndarray, rollout_idx: int):
        """Update positive activations for a feature."""
        if np.max(values) <= self.pos_min[feature_idx]:
            return
        
        combined_values = np.concatenate([self.pos_values[feature_idx], values])
        combined_rollouts = np.concatenate([
            self.pos_rollout_idx[feature_idx],
            np.full(len(values), rollout_idx, dtype=np.int32)
        ])
        combined_tokens = np.concatenate([
            self.pos_token_idx[feature_idx],
            token_indices.astype(np.int32)
        ])
        
        if len(combined_values) > self.k:
            top_k_idx = np.argpartition(combined_values, -self.k)[-self.k:]
            top_k_idx = top_k_idx[np.argsort(combined_values[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(combined_values)[::-1]
        
        self.pos_values[feature_idx] = combined_values[top_k_idx[:self.k]]
        self.pos_rollout_idx[feature_idx] = combined_rollouts[top_k_idx[:self.k]]
        self.pos_token_idx[feature_idx] = combined_tokens[top_k_idx[:self.k]]
        self.pos_min[feature_idx] = self.pos_values[feature_idx][-1]
    
    def update_negative(self, feature_idx: int, abs_values: np.ndarray, 
                        token_indices: np.ndarray, rollout_idx: int):
        """Update negative activations for a feature (using absolute values)."""
        if np.max(abs_values) <= self.neg_min[feature_idx]:
            return
        
        combined_values = np.concatenate([self.neg_values[feature_idx], abs_values])
        combined_rollouts = np.concatenate([
            self.neg_rollout_idx[feature_idx],
            np.full(len(abs_values), rollout_idx, dtype=np.int32)
        ])
        combined_tokens = np.concatenate([
            self.neg_token_idx[feature_idx],
            token_indices.astype(np.int32)
        ])
        
        if len(combined_values) > self.k:
            top_k_idx = np.argpartition(combined_values, -self.k)[-self.k:]
            top_k_idx = top_k_idx[np.argsort(combined_values[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(combined_values)[::-1]
        
        self.neg_values[feature_idx] = combined_values[top_k_idx[:self.k]]
        self.neg_rollout_idx[feature_idx] = combined_rollouts[top_k_idx[:self.k]]
        self.neg_token_idx[feature_idx] = combined_tokens[top_k_idx[:self.k]]
        self.neg_min[feature_idx] = self.neg_values[feature_idx][-1]
    
    def get_dominant_polarity(self, feature_idx: int) -> Tuple[str, List[Tuple[int, int, float]]]:
        """
        Determine dominant polarity and return examples from that polarity.
        
        Returns:
            (polarity, examples) where polarity is 'positive' or 'negative'
            and examples is list of (rollout_idx, token_idx, abs_value)
        """
        # Get max values
        max_pos = self.pos_values[feature_idx, 0] if self.pos_values[feature_idx, 0] > -np.inf else 0
        max_neg = self.neg_values[feature_idx, 0] if self.neg_values[feature_idx, 0] > -np.inf else 0
        
        if max_pos >= max_neg:
            # Positive dominant
            valid_mask = self.pos_values[feature_idx] > -np.inf
            valid_indices = np.where(valid_mask)[0]
            examples = [(int(self.pos_rollout_idx[feature_idx, i]),
                        int(self.pos_token_idx[feature_idx, i]),
                        float(self.pos_values[feature_idx, i]))
                       for i in valid_indices]
            return 'positive', examples
        else:
            # Negative dominant
            valid_mask = self.neg_values[feature_idx] > -np.inf
            valid_indices = np.where(valid_mask)[0]
            examples = [(int(self.neg_rollout_idx[feature_idx, i]),
                        int(self.neg_token_idx[feature_idx, i]),
                        float(self.neg_values[feature_idx, i]))  # Already absolute
                       for i in valid_indices]
            return 'negative', examples


class MLPTopKCollector:
    """Collector for top-k MLP neuron activations with polarity-aware handling."""
    
    def __init__(self,
                 activations_dir: str,
                 top_k: int = 64,
                 context_window: int = 20,
                 min_activation_threshold: float = 0.01,
                 num_workers: int = 4):
        """Initialize collector."""
        self.activations_dir = activations_dir
        self.top_k = top_k
        self.context_window = context_window
        self.min_activation_threshold = min_activation_threshold
        self.num_workers = num_workers
        
        # Load tokens
        tokens_file = os.path.join(activations_dir, "tokens.json")
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
        
        print("Loading tokens...")
        with open(tokens_file, 'r') as f:
            self.all_tokens = json.load(f)
        
        # Convert string keys to int
        self.all_tokens = {int(k): v for k, v in self.all_tokens.items()}
    
    def _load_h5_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load H5 file and return activations with rollout index."""
        rollout_idx = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        with h5py.File(file_path, 'r') as f:
            activations = np.array(f['activations'][:], dtype=np.float16)
        return activations, rollout_idx
    
    def _process_rollout(self, activations: np.ndarray, rollout_idx: int,
                        layers: List[int], neuron_indices: List[int],
                        tracker: DualPolarityTracker):
        """Process a single rollout."""
        for neuron_idx in neuron_indices:
            neuron_acts = activations[:, layers, neuron_idx].astype(np.float32)
            
            for i, layer in enumerate(layers):
                layer_acts = neuron_acts[:, i]
                feature_idx = layer * len(neuron_indices) + neuron_indices.index(neuron_idx)
                
                # Find positive activations
                pos_mask = layer_acts > self.min_activation_threshold
                if np.any(pos_mask):
                    pos_indices = np.where(pos_mask)[0]
                    pos_values = layer_acts[pos_indices]
                    tracker.update_positive(feature_idx, pos_values, pos_indices, rollout_idx)
                
                # Find negative activations
                neg_mask = layer_acts < -self.min_activation_threshold
                if np.any(neg_mask):
                    neg_indices = np.where(neg_mask)[0]
                    neg_abs_values = np.abs(layer_acts[neg_indices])
                    tracker.update_negative(feature_idx, neg_abs_values, neg_indices, rollout_idx)
    
    def _extract_contexts(self, examples_by_rollout: Dict, layers: List[int], 
                         neuron_indices: List[int], polarities: Dict) -> Dict:
        """Extract contexts for all examples."""
        features_dict = {}
        
        for rollout_idx, examples in tqdm(examples_by_rollout.items(), 
                                         desc="Extracting contexts", ncols=100):
            h5_path = os.path.join(self.activations_dir, f"rollout_{rollout_idx}.h5")
            
            with h5py.File(h5_path, 'r') as f:
                rollout_activations = np.array(f['activations'][:], dtype=np.float16)
            
            rollout_tokens = self.all_tokens.get(rollout_idx, [])
            
            for layer, neuron_idx, token_idx, abs_value in examples:
                feature_id = f"L{layer}_neuron{neuron_idx}"
                polarity = polarities[feature_id]
                
                if feature_id not in features_dict:
                    features_dict[feature_id] = {
                        'layer': layer,
                        'neuron_idx': neuron_idx,
                        'dominant_polarity': polarity,
                        'examples': [],
                        'stats': {
                            'num_examples': 0,
                            'max_activation': -np.inf,
                            'mean_abs_activation': 0
                        }
                    }
                
                # Extract context window
                context_start = max(0, token_idx - self.context_window)
                context_end = min(len(rollout_tokens), token_idx + self.context_window + 1)
                context_tokens = rollout_tokens[context_start:context_end]
                
                # Get activations for this context window
                window_acts = rollout_activations[context_start:context_end, layer, neuron_idx]
                
                # Get original value (with sign)
                original_value = rollout_activations[token_idx, layer, neuron_idx]
                
                # For negative polarity, convert all values to positive for display
                if polarity == 'negative':
                    display_acts = np.abs(window_acts).tolist()
                else:
                    display_acts = window_acts.tolist()
                
                # Create example
                example_dict = asdict(FeatureExample(
                    rollout_idx=rollout_idx,
                    token_idx=token_idx,
                    activation_value=float(abs_value),  # Always positive for display
                    original_value=float(original_value),  # Keep original for reference
                    context_start=context_start,
                    context_end=context_end,
                    tokens=context_tokens,
                    activations=display_acts  # Positive values for autointerp
                ))
                
                features_dict[feature_id]['examples'].append(example_dict)
                
                # Update stats
                stats = features_dict[feature_id]['stats']
                stats['num_examples'] += 1
                stats['max_activation'] = max(stats['max_activation'], abs_value)
                stats['mean_abs_activation'] += abs_value
        
        # Finalize stats
        for feature_data in features_dict.values():
            stats = feature_data['stats']
            if stats['num_examples'] > 0:
                stats['mean_abs_activation'] /= stats['num_examples']
        
        return features_dict
    
    def collect_all_features(self, layers: Optional[List[int]] = None,
                            neuron_indices: Optional[List[int]] = None) -> Dict:
        """Collect top-k examples for all specified features."""
        if layers is None:
            layers = list(range(64))
        if neuron_indices is None:
            neuron_indices = list(range(6))
        
        n_features = len(layers) * len(neuron_indices)
        
        print(f"Collecting top-{self.top_k} examples for {len(layers)} layers × "
              f"{len(neuron_indices)} neurons = {n_features} features")
        
        # Initialize tracker
        tracker = DualPolarityTracker(self.top_k, n_features)
        
        # Get all rollout files
        rollout_files = []
        for h5_file in sorted(glob.glob(os.path.join(self.activations_dir, "rollout_*.h5"))):
            rollout_files.append(h5_file)
        
        print(f"\nProcessing {len(rollout_files)} rollout files...")
        
        # Process files
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            all_futures = []
            for file_path in rollout_files:
                future = executor.submit(self._load_h5_file, file_path)
                all_futures.append(future)
            
            for future in tqdm(as_completed(all_futures), total=len(all_futures), 
                             desc="Rollouts", ncols=100):
                activations, rollout_idx = future.result()
                self._process_rollout(activations, rollout_idx, layers, neuron_indices, tracker)
                del activations
                
                if len(all_futures) > 100 and all_futures.index(future) % 100 == 0:
                    gc.collect()
        
        # Determine dominant polarity for each feature and collect examples
        print("\nDetermining dominant polarities and extracting examples...")
        
        examples_by_rollout = {}
        polarities = {}
        
        for layer in layers:
            for neuron_idx in neuron_indices:
                feature_idx = layer * len(neuron_indices) + neuron_indices.index(neuron_idx)
                feature_id = f"L{layer}_neuron{neuron_idx}"
                
                # Get dominant polarity and examples
                polarity, examples = tracker.get_dominant_polarity(feature_idx)
                polarities[feature_id] = polarity
                
                # Group by rollout
                for rollout_idx, token_idx, value in examples:
                    if rollout_idx not in examples_by_rollout:
                        examples_by_rollout[rollout_idx] = []
                    examples_by_rollout[rollout_idx].append(
                        (layer, neuron_idx, token_idx, value)
                    )
        
        # Extract contexts
        features_dict = self._extract_contexts(examples_by_rollout, layers, 
                                              neuron_indices, polarities)
        
        return features_dict


def main():
    parser = argparse.ArgumentParser(description="Precompute top-k examples for MLP neurons")
    parser.add_argument("--activations-dir", 
                       default="data/mlp_activations",
                       help="Directory containing activation H5 files")
    parser.add_argument("--output", 
                       default="data/mlp_topk_384_features.json",
                       help="Output JSON file")
    parser.add_argument("--layers", nargs="+", type=int, 
                       help="Specific layers to process (default: all)")
    parser.add_argument("--neurons", nargs="+", type=int,
                       help="Specific neuron indices (default: 0-5)")
    parser.add_argument("--top-k", type=int, default=64,
                       help="Number of top examples per feature")
    parser.add_argument("--context-window", type=int, default=20,
                       help="Context window size")
    parser.add_argument("--min-activation", type=float, default=0.01,
                       help="Minimum activation threshold")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    if args.neurons is None:
        args.neurons = list(range(6))
    
    # Initialize collector
    collector = MLPTopKCollector(
        activations_dir=args.activations_dir,
        top_k=args.top_k,
        context_window=args.context_window,
        min_activation_threshold=args.min_activation,
        num_workers=args.num_workers
    )
    
    # Collect features
    start_time = time.time()
    results = collector.collect_all_features(
        layers=args.layers,
        neuron_indices=args.neurons
    )
    elapsed = time.time() - start_time
    
    # Add metadata
    output_data = {
        'metadata': {
            'top_k': args.top_k,
            'context_window': args.context_window,
            'min_activation_threshold': args.min_activation,
            'layers': args.layers if args.layers else list(range(64)),
            'neuron_indices': args.neurons,
            'processing_time': elapsed,
            'num_features': len(results),
            'num_rollouts': len(collector.all_tokens)
        },
        'features': results
    }
    
    # Save to JSON
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Saved {len(results)} features to {args.output}")
    
    # Print polarity statistics
    pos_count = sum(1 for f in results.values() if f['dominant_polarity'] == 'positive')
    neg_count = len(results) - pos_count
    print(f"\nPolarity distribution:")
    print(f"  Positive dominant: {pos_count} ({100*pos_count/len(results):.1f}%)")
    print(f"  Negative dominant: {neg_count} ({100*neg_count/len(results):.1f}%)")


if __name__ == "__main__":
    import glob
    main()