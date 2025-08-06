#!/usr/bin/env python3
"""
Optimized SAE feature collection with performance improvements.
Processes activation data to find max-activating examples for each feature.
"""

import torch
import torch.nn.functional as F
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
import time
from concurrent.futures import ThreadPoolExecutor
import gc


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass
class ActivationExample:
    """Single activation example with context"""
    activation_value: float
    rollout_idx: int
    token_idx: int
    token: str
    tokens: List[str]  # Changed from 'context' to match dashboard
    target_position: int
    activations: List[float]  # Changed from 'context_activations' to match dashboard


class FastTopKTracker:
    """Optimized top-k tracker using numpy arrays"""
    def __init__(self, k: int, n_features: int):
        self.k = k
        self.n_features = n_features
        
        # Pre-allocate arrays for better memory efficiency
        self.top_values = np.full((n_features, k), -np.inf, dtype=np.float32)
        self.top_rollout_idx = np.zeros((n_features, k), dtype=np.int32)
        self.top_token_idx = np.zeros((n_features, k), dtype=np.int32)
        self.min_values = np.full(n_features, -np.inf, dtype=np.float32)
        
    def batch_add(self, activations: np.ndarray, rollout_idx: int, start_token_idx: int):
        """Add a batch of activations efficiently"""
        # activations shape: [batch_size, n_features]
        batch_size = activations.shape[0]
        
        # Find features with activations > current minimum
        for token_offset in range(batch_size):
            token_activations = activations[token_offset]
            token_idx = start_token_idx + token_offset
            
            # Vectorized comparison - only track positive activations
            mask = (token_activations > self.min_values) & (token_activations > 0)
            active_features = np.where(mask)[0]
            
            for feat_idx in active_features:
                activation = token_activations[feat_idx]
                
                # Find position to insert
                insert_pos = np.searchsorted(self.top_values[feat_idx], activation)
                
                if insert_pos > 0:  # Should be inserted
                    # Shift arrays
                    if insert_pos < self.k:
                        self.top_values[feat_idx, :insert_pos] = self.top_values[feat_idx, 1:insert_pos+1]
                        self.top_rollout_idx[feat_idx, :insert_pos] = self.top_rollout_idx[feat_idx, 1:insert_pos+1]
                        self.top_token_idx[feat_idx, :insert_pos] = self.top_token_idx[feat_idx, 1:insert_pos+1]
                    
                    # Insert new value
                    self.top_values[feat_idx, min(insert_pos, self.k-1)] = activation
                    self.top_rollout_idx[feat_idx, min(insert_pos, self.k-1)] = rollout_idx
                    self.top_token_idx[feat_idx, min(insert_pos, self.k-1)] = token_idx
                    
                    # Update minimum
                    self.min_values[feat_idx] = self.top_values[feat_idx, 0]
    
    def get_top_k_for_feature(self, feature_idx: int) -> List[Tuple[float, int, int]]:
        """Get top k examples for a specific feature"""
        valid_mask = self.top_values[feature_idx] > -np.inf
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Return in descending order, filtering out non-positive activations
        results = []
        for i in reversed(valid_indices):
            activation_value = float(self.top_values[feature_idx, i])
            if activation_value > 0:  # Only include positive activations
                results.append((
                    activation_value,
                    int(self.top_rollout_idx[feature_idx, i].item()),
                    int(self.top_token_idx[feature_idx, i].item())
                ))
        return results


def load_sae_model_optimized(model_path: str, device: str) -> Tuple[BatchTopKSAE, dict]:
    """Load SAE model with optimizations"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = BatchTopKSAE(
        d_model=config['d_model'],
        dict_size=config['dict_size'],
        k=config['k']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert to half precision for faster computation
    model = model.half()
    
    # Skip torch.compile to avoid CUDA graph issues with concatenation
    # The performance benefit is not worth the complexity for this use case
    # if hasattr(torch, 'compile'):
    #     model.encode = torch.compile(model.encode, mode='reduce-overhead')
    
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    return model, config


def load_h5_file_async(file_path: str) -> Tuple[np.ndarray, dict]:
    """Load H5 file in thread"""
    with h5py.File(file_path, 'r') as f:
        activations = np.array(f['activations'][:], dtype=np.float16)
        attrs = dict(f.attrs)
    return activations, attrs


def process_files_batch(
    file_batch: List[Tuple[str, int]],  # (file_path, global_rollout_idx)
    sae_model: BatchTopKSAE,
    config: dict,
    tracker: FastTopKTracker,
    device: str,
    batch_size: int = 2048,  # Larger batch size
    pbar: Optional[tqdm] = None
) -> int:
    """Process a batch of files efficiently"""
    
    total_tokens = 0
    
    # Get adapter configuration
    adapter_types = config.get('adapter_types', ['gate_proj', 'up_proj', 'down_proj'])
    all_adapters = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    for file_path, global_rollout_idx in file_batch:
        # Load file
        activations, attrs = load_h5_file_async(file_path)
        
        # Extract dimensions
        projections = attrs.get('projections', 3)
        num_layers = attrs['num_layers']
        
        # Map adapter types to indices
        adapter_indices = [all_adapters.index(a) for a in adapter_types if all_adapters.index(a) < projections]
        
        # Select only the adapters used in training
        activations_selected = activations[:, :, adapter_indices]
        
        # Flatten activations
        activations_flat = activations_selected.reshape(-1, num_layers * len(adapter_indices))
        n_tokens = len(activations_flat)
        total_tokens += n_tokens
        
        # Process in large batches
        for start_idx in range(0, n_tokens, batch_size):
            end_idx = min(start_idx + batch_size, n_tokens)
            batch = torch.from_numpy(activations_flat[start_idx:end_idx]).to(device, dtype=torch.float16)
            
            with torch.no_grad():
                # Get SAE features
                sparse_features = sae_model.encode(batch)  # Shape: (batch_size, dict_size)
                
                # Convert to numpy for faster processing
                sparse_np = sparse_features.cpu().float().numpy()
                
                # Batch update tracker
                tracker.batch_add(sparse_np, global_rollout_idx, start_idx)
            
            # Clear GPU cache periodically
            if start_idx % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        if pbar:
            pbar.update(1)
    
    return total_tokens


def extract_context_batch(
    examples: List[Tuple[int, float, int, int]],  # (feature_idx, activation, rollout_idx, token_idx)
    data_dir: str,
    sae_model: BatchTopKSAE,
    config: dict,
    rollout_tokens: Optional[Dict[str, List[str]]],
    context_window: int,
    device: str
) -> Dict[int, List[dict]]:
    """Extract contexts for multiple examples efficiently with optimized batching"""
    
    # Group by rollout file
    rollout_groups = {}
    for feature_idx, activation, rollout_idx, token_idx in examples:
        if rollout_idx not in rollout_groups:
            rollout_groups[rollout_idx] = []
        rollout_groups[rollout_idx].append((feature_idx, activation, token_idx))
    
    results = {feat_idx: [] for feat_idx, _, _, _ in examples}
    
    # Prepare for parallel file loading
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Process each rollout file once
        for rollout_idx, group in rollout_groups.items():
            rollout_path = os.path.join(data_dir, f'rollout_{rollout_idx}.h5')
            
            # Load file
            activations, attrs = load_h5_file_async(rollout_path)
            projections = attrs.get('projections', 3)
            num_layers = attrs['num_layers']
            
            # Get tokens
            if rollout_tokens and str(rollout_idx) in rollout_tokens:
                tokens = rollout_tokens[str(rollout_idx)]
            else:
                tokens = [f"<token_{i}>" for i in range(len(activations))]
            
            # Process activations
            adapter_types = config.get('adapter_types', ['gate_proj', 'up_proj', 'down_proj'])
            all_adapters = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
            adapter_indices = [all_adapters.index(a) for a in adapter_types if all_adapters.index(a) < projections]
            
            activations_selected = activations[:, :, adapter_indices]
            activations_flat = activations_selected.reshape(-1, num_layers * len(adapter_indices))
            
            # Find unique token range needed for all examples in this rollout
            min_start = min(max(0, token_idx - context_window) for _, _, token_idx in group)
            max_end = max(min(len(tokens), token_idx + context_window + 1) for _, _, token_idx in group)
            
            # Encode all needed tokens in one batch
            if max_end > min_start:
                batch_acts = activations_flat[min_start:max_end]
                batch_tensor = torch.from_numpy(batch_acts).to(device, dtype=torch.float16)
                
                with torch.no_grad():
                    # Process in chunks if batch is too large
                    max_batch_size = 4096
                    if len(batch_tensor) > max_batch_size:
                        sparse_features_list = []
                        for i in range(0, len(batch_tensor), max_batch_size):
                            chunk = batch_tensor[i:i+max_batch_size]
                            sparse_chunk = sae_model.encode(chunk)
                            # Clone to prevent CUDA graphs from overwriting
                            sparse_features_list.append(sparse_chunk.clone())
                        sparse_features_full = torch.cat(sparse_features_list, dim=0)
                    else:
                        sparse_features_full = sae_model.encode(batch_tensor)
                
                # Keep on GPU for slicing
                for feature_idx, activation_value, token_idx in group:
                    # Get context bounds relative to full encoding
                    start = max(0, token_idx - context_window)
                    end = min(len(tokens), token_idx + context_window + 1)
                    
                    # Extract context
                    context_tokens = tokens[start:end]
                    target_position = token_idx - start
                    
                    # Slice from pre-computed features
                    relative_start = start - min_start
                    relative_end = end - min_start
                    context_features = sparse_features_full[relative_start:relative_end, feature_idx]
                    
                    # Convert to list
                    feature_acts = context_features.cpu().float().numpy().tolist()
                    
                    example = {
                        'activation_value': float(context_features[target_position].item()),
                        'rollout_idx': int(rollout_idx),
                        'token_idx': int(token_idx),
                        'token': tokens[token_idx],
                        'tokens': context_tokens,  # Dashboard expects 'tokens' not 'context'
                        'target_position': int(target_position),
                        'activations': feature_acts  # Dashboard expects 'activations' not 'context_activations'
                    }
                    
                    results[feature_idx].append(example)
            
            # Clear GPU cache after each rollout
            torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fast SAE feature collection')
    parser.add_argument('--model-path', type=str, default='trained_sae_adapters_g-u-d-q-k-v-o.pt',
                       help='Path to trained SAE model')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing activation H5 files')
    parser.add_argument('--tokens-path', type=str, default=None,
                       help='Path to rollout_tokens.json')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top examples per feature')
    parser.add_argument('--context-window', type=int, default=10,
                       help='Context tokens on each side')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--max-rollouts', type=int, default=None,
                       help='Maximum number of rollouts to process')
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size for SAE encoding')
    parser.add_argument('--file-batch-size', type=int, default=10,
                       help='Number of files to process in parallel')
    parser.add_argument('--context-batch-size', type=int, default=1000,
                       help='Number of examples to process at once during context extraction')
    
    args = parser.parse_args()
    
    print("=== Fast SAE Feature Collection ===")
    start_time = time.time()
    
    # Load SAE model
    print("Loading SAE model...")
    sae_model, config = load_sae_model_optimized(args.model_path, args.device)
    n_features = config['dict_size']
    print(f"Loaded SAE with {n_features} features (using float16)")
    
    # Auto-detect data directory
    if args.data_dir is None:
        adapter_types = config.get('adapter_types', ['gate_proj', 'up_proj', 'down_proj'])
        if set(adapter_types) == set(['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            if os.path.exists('./activations_all_adapters'):
                args.data_dir = './activations_all_adapters'
            else:
                args.data_dir = '../../lora-activations-dashboard/backend/activations_all_adapters'
        else:
            adapter_str = '-'.join([a[:1] for a in sorted(adapter_types)])
            if os.path.exists(f'./activations_{adapter_str}'):
                args.data_dir = f'./activations_{adapter_str}'
            else:
                args.data_dir = f'../../lora-activations-dashboard/backend/activations_{adapter_str}'
    
    print(f"Using activation directory: {args.data_dir}")
    
    # Load tokens if available
    if args.tokens_path is None:
        tokens_path = os.path.join(args.data_dir, 'rollout_tokens.json')
        if os.path.exists(tokens_path):
            args.tokens_path = tokens_path
    
    rollout_tokens = None
    if args.tokens_path and os.path.exists(args.tokens_path):
        print(f"Loading tokens from {args.tokens_path}")
        with open(args.tokens_path, 'r') as f:
            rollout_tokens = json.load(f)
    
    # Get rollout files
    rollout_files = sorted(glob(os.path.join(args.data_dir, 'rollout_*.h5')))
    if args.max_rollouts:
        rollout_files = rollout_files[:args.max_rollouts]
    print(f"Found {len(rollout_files)} rollout files")
    
    # Initialize optimized tracker
    print("Initializing feature tracker...")
    tracker = FastTopKTracker(args.top_k, n_features)
    
    # Phase 1: Find top activations
    print("\nPhase 1: Finding top activations...")
    phase1_start = time.time()
    
    # Process files in batches
    file_batches = []
    for i, file_path in enumerate(rollout_files):
        global_idx = int(os.path.basename(file_path).replace('rollout_', '').replace('.h5', ''))
        file_batches.append((file_path, global_idx))
    
    # Process with progress bar
    total_tokens = 0
    with tqdm(total=len(rollout_files), desc="Processing files") as pbar:
        for i in range(0, len(file_batches), args.file_batch_size):
            batch = file_batches[i:i+args.file_batch_size]
            tokens = process_files_batch(
                batch, sae_model, config, tracker, 
                args.device, args.batch_size, pbar
            )
            total_tokens += tokens
            
            # Show throughput
            elapsed = time.time() - phase1_start
            if elapsed > 0:
                throughput = total_tokens / elapsed
                pbar.set_postfix({'tokens/s': f'{throughput:.0f}'})
    
    phase1_time = time.time() - phase1_start
    print(f"Phase 1 complete: {phase1_time:.1f}s ({total_tokens/phase1_time:.0f} tokens/s)")
    
    # Phase 2: Extract contexts
    print("\nPhase 2: Extracting contexts for top examples...")
    phase2_start = time.time()
    
    # Collect all examples to process
    all_examples = []
    active_features = []
    
    for feature_idx in range(n_features):
        top_k = tracker.get_top_k_for_feature(feature_idx)
        if top_k:
            active_features.append(feature_idx)
            for activation, rollout_idx, token_idx in top_k:
                all_examples.append((feature_idx, activation, rollout_idx, token_idx))
    
    print(f"Found {len(active_features)} active features with {len(all_examples)} total examples")
    
    # Process contexts in larger batches
    features_data = {}
    batch_size = args.context_batch_size  # Use configurable batch size
    
    with tqdm(total=len(all_examples), desc="Extracting contexts") as pbar:
        for i in range(0, len(all_examples), batch_size):
            batch = all_examples[i:i+batch_size]
            
            results = extract_context_batch(
                batch, args.data_dir, sae_model, config,
                rollout_tokens, args.context_window, args.device
            )
            
            # Merge results
            for feature_idx, examples in results.items():
                if str(feature_idx) not in features_data:
                    features_data[str(feature_idx)] = {
                        'examples': [],
                        'stats': {'max_activation': 0, 'n_examples': 0}
                    }
                
                features_data[str(feature_idx)]['examples'].extend(examples)
                if examples:
                    max_act = max(ex['activation_value'] for ex in examples)
                    features_data[str(feature_idx)]['stats']['max_activation'] = max(
                        features_data[str(feature_idx)]['stats']['max_activation'],
                        max_act
                    )
            
            pbar.update(len(batch))
    
    # Update stats
    for feature_str in features_data:
        features_data[feature_str]['stats']['n_examples'] = len(features_data[feature_str]['examples'])
    
    phase2_time = time.time() - phase2_start
    print(f"Phase 2 complete: {phase2_time:.1f}s")
    
    # Build final output
    output_data = {
        'metadata': {
            'n_features': n_features,
            'top_k': args.top_k,
            'context_window': args.context_window,
            'n_rollouts_processed': len(rollout_files),
            'sae_config': config,
            'adapter_types': config.get('adapter_types', ['gate_proj', 'up_proj', 'down_proj'])
        },
        'features': features_data
    }
    
    # Save output
    if args.output_path is None:
        model_name = os.path.basename(args.model_path).replace('.pt', '')
        args.output_path = f'sae_features_data_{model_name}.json'
    
    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\n=== Complete ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Phase 1: {phase1_time:.1f}s ({phase1_time/total_time*100:.1f}%)")
    print(f"Phase 2: {phase2_time:.1f}s ({phase2_time/total_time*100:.1f}%)")
    print(f"Average throughput: {total_tokens/phase1_time:.0f} tokens/s")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()