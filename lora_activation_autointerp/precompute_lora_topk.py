#!/usr/bin/env python3
"""
Optimized precomputation of top-k examples for LoRA features.
Implements performance optimizations from SAE collection script.
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
    activation_value: float
    context_start: int
    context_end: int
    tokens: List[str]
    activations: List[float]


class FastTopKTracker:
    """Optimized top-k tracker using pre-allocated numpy arrays."""
    
    def __init__(self, k: int, n_features: int):
        """
        Initialize tracker with pre-allocated arrays.
        
        Args:
            k: Number of top examples to track
            n_features: Total number of features
        """
        self.k = k
        self.n_features = n_features
        
        # Pre-allocate arrays for better memory efficiency
        # We track both positive and negative separately
        self.top_values = np.full((n_features, k), -np.inf, dtype=np.float32)
        self.top_rollout_idx = np.zeros((n_features, k), dtype=np.int32)
        self.top_token_idx = np.zeros((n_features, k), dtype=np.int32)
        self.min_values = np.full(n_features, -np.inf, dtype=np.float32)
        
    def batch_update(self, 
                     feature_idx: int,
                     activations: np.ndarray,
                     indices: np.ndarray,
                     rollout_idx: int):
        """
        Update tracker for a single feature with multiple activation values.
        
        Args:
            feature_idx: Feature index
            activations: Activation values
            indices: Token indices
            rollout_idx: Rollout index
        """
        if len(activations) == 0:
            return
            
        # Sort activations in descending order
        sorted_idx = np.argsort(activations)[::-1]
        
        # Take top-k from this batch
        n_to_add = min(self.k, len(activations))
        
        for i in range(n_to_add):
            act_val = activations[sorted_idx[i]]
            tok_idx = indices[sorted_idx[i]]
            
            # Check if this should be inserted
            if act_val > self.min_values[feature_idx]:
                # Find insertion position
                insert_pos = np.searchsorted(self.top_values[feature_idx], act_val)
                
                if insert_pos > 0:  # Should be inserted
                    # Shift arrays
                    if insert_pos < self.k:
                        self.top_values[feature_idx, :insert_pos] = self.top_values[feature_idx, 1:insert_pos+1]
                        self.top_rollout_idx[feature_idx, :insert_pos] = self.top_rollout_idx[feature_idx, 1:insert_pos+1]
                        self.top_token_idx[feature_idx, :insert_pos] = self.top_token_idx[feature_idx, 1:insert_pos+1]
                    
                    # Insert new value
                    pos = min(insert_pos, self.k-1)
                    self.top_values[feature_idx, pos] = act_val
                    self.top_rollout_idx[feature_idx, pos] = rollout_idx
                    self.top_token_idx[feature_idx, pos] = tok_idx
                    
                    # Update minimum
                    self.min_values[feature_idx] = self.top_values[feature_idx, 0]
    
    def get_top_k(self, feature_idx: int) -> List[Tuple[float, int, int]]:
        """Get top-k examples for a feature."""
        valid_mask = self.top_values[feature_idx] > -np.inf
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Return in descending order
        results = []
        for i in reversed(valid_indices):
            results.append((
                float(self.top_values[feature_idx, i]),
                int(self.top_rollout_idx[feature_idx, i]),
                int(self.top_token_idx[feature_idx, i])
            ))
        return results


class OptimizedTopKCollector:
    """Optimized collector for top-k LoRA examples."""
    
    ADAPTER_INDICES = {
        'gate_proj': 0,
        'up_proj': 1,
        'down_proj': 2,
        'q_proj': 3,
        'k_proj': 4,
        'v_proj': 5,
        'o_proj': 6
    }
    
    def __init__(self,
                 activation_dir: str = "../sae_interp/activations_all_adapters",
                 top_k: int = 50,
                 context_window: int = 30,
                 min_activation_threshold: float = 0.01,
                 max_rollouts: Optional[int] = None,
                 batch_size: int = 4,
                 num_workers: int = 4):
        """
        Initialize optimized collector.
        
        Args:
            activation_dir: Directory with activation files
            top_k: Number of top examples to keep per feature
            context_window: Tokens on each side of max activation
            min_activation_threshold: Minimum absolute activation to consider
            max_rollouts: Maximum number of rollouts to process
            batch_size: Number of rollout files to process in parallel
            num_workers: Number of parallel workers for file loading
        """
        self.activation_dir = activation_dir
        self.top_k = top_k
        self.context_window = context_window
        self.min_activation_threshold = min_activation_threshold
        self.max_rollouts = max_rollouts
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load tokens
        print("Loading tokens...")
        self.rollout_tokens = self._load_tokens()
        
        # Limit rollouts if specified
        if self.max_rollouts is not None:
            available_rollouts = sorted(list(self.rollout_tokens.keys()))[:self.max_rollouts]
            self.rollout_tokens = {k: self.rollout_tokens[k] for k in available_rollouts}
            print(f"Limiting to {self.max_rollouts} rollouts")
        
        self.num_rollouts = len(self.rollout_tokens)
        
    def _load_tokens(self) -> Dict[int, List[str]]:
        """Load tokens from JSON file."""
        tokens_path = os.path.join(self.activation_dir, "rollout_tokens.json")
        with open(tokens_path, 'r') as f:
            tokens_dict = json.load(f)
        return {int(k): v for k, v in tokens_dict.items()}
    
    def _load_h5_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load H5 file and return activations with rollout index."""
        rollout_idx = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        with h5py.File(file_path, 'r') as f:
            activations = np.array(f['activations'][:], dtype=np.float16)
        return activations, rollout_idx
    
    def _process_rollout_vectorized(self,
                                    activations: np.ndarray,
                                    rollout_idx: int,
                                    layers: List[int],
                                    adapter_types: List[str],
                                    trackers: Dict[str, FastTopKTracker]):
        """
        Process a single rollout with vectorized operations.
        
        Args:
            activations: Shape (num_tokens, 64, 7)
            rollout_idx: Rollout index
            layers: Layers to process
            adapter_types: Adapter types to process
            trackers: Dictionary of trackers for positive/negative
        """
        num_tokens = activations.shape[0]
        
        # Process each adapter type
        for adapter_name in adapter_types:
            adapter_idx = self.ADAPTER_INDICES[adapter_name]
            
            # Extract activations for all layers at once
            # Shape: (num_tokens, num_layers)
            adapter_acts = activations[:, layers, adapter_idx].astype(np.float32)
            
            # Process each layer
            for i, layer in enumerate(layers):
                layer_acts = adapter_acts[:, i]
                
                # Find positive activations (vectorized)
                pos_mask = layer_acts > self.min_activation_threshold
                if np.any(pos_mask):
                    pos_indices = np.where(pos_mask)[0]
                    pos_values = layer_acts[pos_indices]
                    
                    # Update tracker
                    feature_idx = self._get_feature_index(layer, adapter_name, 'positive', 
                                                         layers, adapter_types)
                    trackers['positive'].batch_update(feature_idx, pos_values, 
                                                     pos_indices, rollout_idx)
                
                # Find negative activations (vectorized)
                neg_mask = layer_acts < -self.min_activation_threshold
                if np.any(neg_mask):
                    neg_indices = np.where(neg_mask)[0]
                    neg_values = -layer_acts[neg_indices]  # Use absolute value for sorting
                    
                    # Update tracker
                    feature_idx = self._get_feature_index(layer, adapter_name, 'negative',
                                                         layers, adapter_types)
                    trackers['negative'].batch_update(feature_idx, neg_values,
                                                     neg_indices, rollout_idx)
    
    def _get_feature_index(self, layer: int, adapter: str, polarity: str,
                          layers: List[int], adapter_types: List[str]) -> int:
        """Convert feature specification to index."""
        layer_idx = layers.index(layer)
        adapter_idx = adapter_types.index(adapter)
        
        # For each polarity tracker, the index is just based on layer and adapter
        # since we have separate trackers for positive and negative
        return layer_idx * len(adapter_types) + adapter_idx
    
    def _extract_contexts_batch(self,
                               examples_by_rollout: Dict[int, List[Tuple]],
                               layers: List[int],
                               adapter_types: List[str]) -> Dict[str, List[dict]]:
        """
        Extract contexts for multiple examples grouped by rollout.
        
        Args:
            examples_by_rollout: Examples grouped by rollout index
            layers: Layers being processed
            adapter_types: Adapter types being processed
            
        Returns:
            Dictionary mapping feature_id to list of examples
        """
        results = {}
        
        for rollout_idx, examples in tqdm(examples_by_rollout.items(), 
                                         desc="Extracting contexts", ncols=100):
            # Load activations once for this rollout
            h5_path = os.path.join(self.activation_dir, f"rollout_{rollout_idx}.h5")
            with h5py.File(h5_path, 'r') as f:
                all_acts = f['activations'][:]
            
            tokens = self.rollout_tokens[rollout_idx]
            
            # Process each example
            for feature_id, activation_value, token_idx in examples:
                # Parse feature ID
                parts = feature_id.split('_')
                layer = int(parts[0][1:])
                polarity = parts[-1]
                adapter_name = '_'.join(parts[1:-1])
                adapter_idx = self.ADAPTER_INDICES[adapter_name]
                
                # Get feature activations
                feature_acts = all_acts[:, layer, adapter_idx].astype(np.float32)
                
                # Define context window
                context_start = max(0, token_idx - self.context_window)
                context_end = min(len(tokens), token_idx + self.context_window + 1)
                
                # Extract context
                context_tokens = tokens[context_start:context_end]
                context_activations = feature_acts[context_start:context_end].tolist()
                
                example = asdict(FeatureExample(
                    rollout_idx=rollout_idx,
                    token_idx=token_idx,
                    activation_value=float(activation_value) if polarity == 'positive' 
                                   else -float(activation_value),  # Restore sign
                    context_start=context_start,
                    context_end=context_end,
                    tokens=context_tokens,
                    activations=context_activations
                ))
                
                if feature_id not in results:
                    results[feature_id] = []
                results[feature_id].append(example)
        
        return results
    
    def collect_all_features(self,
                             layers: Optional[List[int]] = None,
                             adapter_types: Optional[List[str]] = None) -> Dict:
        """
        Collect top-k examples for all specified features with optimizations.
        """
        if layers is None:
            layers = list(range(64))
        if adapter_types is None:
            adapter_types = ['gate_proj', 'up_proj', 'down_proj']
        
        n_features = len(layers) * len(adapter_types) * 2  # *2 for positive/negative
        
        print(f"Collecting top-{self.top_k} examples for {len(layers)} layers × "
              f"{len(adapter_types)} adapters × 2 polarities = {n_features} features")
        
        # Initialize trackers
        trackers = {
            'positive': FastTopKTracker(self.top_k, n_features // 2),
            'negative': FastTopKTracker(self.top_k, n_features // 2)
        }
        
        # Process rollouts in batches with parallel loading
        rollout_files = []
        for rollout_idx in self.rollout_tokens.keys():
            h5_path = os.path.join(self.activation_dir, f"rollout_{rollout_idx}.h5")
            if os.path.exists(h5_path):
                rollout_files.append(h5_path)
        
        print(f"\nProcessing {len(rollout_files)} rollout files...")
        
        # Process with thread pool for parallel I/O
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all file loading tasks
            all_futures = []
            for file_path in rollout_files:
                future = executor.submit(self._load_h5_file, file_path)
                all_futures.append(future)
            
            # Process completed files with a single progress bar
            for future in tqdm(as_completed(all_futures), total=len(all_futures), 
                             desc="Rollouts", ncols=100):
                activations, rollout_idx = future.result()
                self._process_rollout_vectorized(activations, rollout_idx,
                                                layers, adapter_types, trackers)
                
                # Free memory
                del activations
                
                # Garbage collect periodically
                if len(all_futures) > 100 and all_futures.index(future) % 100 == 0:
                    gc.collect()
        
        # Extract final top-k and build contexts
        print("\nExtracting top-k examples with context...")
        
        # Collect all examples grouped by rollout for efficient context extraction
        examples_by_rollout = {}
        
        for layer in layers:
            for adapter_name in adapter_types:
                for polarity in ['positive', 'negative']:
                    feature_id = f"L{layer}_{adapter_name}_{polarity}"
                    feature_idx = self._get_feature_index(layer, adapter_name, polarity,
                                                         layers, adapter_types)
                    
                    # Get tracker based on polarity
                    tracker = trackers[polarity]
                    tracker_idx = feature_idx  # Don't divide by 2 - feature_idx is already correct!
                    
                    # Get top-k examples
                    top_examples = tracker.get_top_k(tracker_idx)
                    
                    for activation_value, rollout_idx, token_idx in top_examples:
                        if rollout_idx not in examples_by_rollout:
                            examples_by_rollout[rollout_idx] = []
                        examples_by_rollout[rollout_idx].append(
                            (feature_id, activation_value, token_idx)
                        )
        
        # Extract contexts in batch
        feature_examples = self._extract_contexts_batch(examples_by_rollout, 
                                                       layers, adapter_types)
        
        # Format results
        results = {}
        for layer in layers:
            for adapter_name in adapter_types:
                for polarity in ['positive', 'negative']:
                    feature_id = f"L{layer}_{adapter_name}_{polarity}"
                    
                    if feature_id in feature_examples:
                        examples = feature_examples[feature_id]
                        all_values = [e['activation_value'] for e in examples]
                        stats = {
                            "num_examples": len(examples),
                            "max_activation": max(all_values) if all_values else 0.0,
                            "min_activation": min(all_values) if all_values else 0.0,
                            "mean_abs_activation": float(np.mean(np.abs(all_values))) if all_values else 0.0
                        }
                    else:
                        examples = []
                        stats = {
                            "num_examples": 0,
                            "max_activation": 0.0,
                            "min_activation": 0.0,
                            "mean_abs_activation": 0.0
                        }
                    
                    results[feature_id] = {
                        "examples": examples,
                        "stats": stats
                    }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        print(f"\nSaving to {output_path}...")
        
        # Add metadata
        output_data = {
            "metadata": {
                "top_k": self.top_k,
                "context_window": self.context_window,
                "min_activation_threshold": self.min_activation_threshold,
                "num_features": len(results),
                "num_rollouts": self.num_rollouts
            },
            "features": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f)
        
        # Print summary
        total_examples = sum(r["stats"]["num_examples"] for r in results.values())
        features_with_examples = sum(1 for r in results.values() 
                                    if r["stats"]["num_examples"] > 0)
        
        print(f"\nSummary:")
        print(f"  Total features: {len(results)}")
        print(f"  Features with examples: {features_with_examples}")
        print(f"  Total examples: {total_examples}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Optimized precomputation for LoRA features")
    parser.add_argument("--layers", nargs="+", type=int, help="Layers to process")
    parser.add_argument("--adapters", nargs="+", help="Adapter types")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top examples")
    parser.add_argument("--context-window", type=int, default=30, help="Context window size")
    parser.add_argument("--output", default="data/lora_topk_optimized.json", help="Output file")
    parser.add_argument("--min-threshold", type=float, default=0.01, help="Min activation threshold")
    parser.add_argument("--max-rollouts", type=int, help="Max rollouts to process")
    parser.add_argument("--batch-size", type=int, default=4, help="Rollout batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = OptimizedTopKCollector(
        top_k=args.top_k,
        context_window=args.context_window,
        min_activation_threshold=args.min_threshold,
        max_rollouts=args.max_rollouts,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Collect examples
    start_time = time.time()
    results = collector.collect_all_features(
        layers=args.layers,
        adapter_types=args.adapters
    )
    
    # Save results
    collector.save_results(results, args.output)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Speed: {collector.num_rollouts / elapsed:.1f} rollouts/second")


if __name__ == "__main__":
    main()