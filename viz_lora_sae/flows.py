"""
Flow computation module for LoRA → SAE → Category visualization.

This module handles data loading and flow weight computation for both pre-aggregated
and raw activation modes.
"""

import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from glob import glob
import warnings
from tqdm import tqdm


def load_inputs_mode_a(features_path: str) -> pd.DataFrame:
    """
    Load pre-aggregated features data.
    
    Args:
        features_path: Path to CSV or parquet file with pre-computed flows
        
    Returns:
        DataFrame with columns: feature_id, category, conf, mass, site_contrib
    """
    path = Path(features_path)
    
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Parse JSON site_contrib if stored as string
    if 'site_contrib' in df.columns and df['site_contrib'].dtype == 'object':
        df['site_contrib'] = df['site_contrib'].apply(json.loads)
    
    required_cols = ['feature_id', 'category', 'conf', 'mass', 'site_contrib']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def load_inputs_mode_b(
    lora_path: str,
    sae_features_path: str, 
    labels_path: str,
    robust: bool = False,
    max_rollouts: Optional[int] = None
) -> pd.DataFrame:
    """
    Load raw activations and compute flows.
    
    Args:
        lora_path: Path to directory with H5 files or NPZ file with LoRA activations
        sae_features_path: Path to JSON file with SAE feature activations
        labels_path: Path to JSON file with feature→category mappings
        robust: Use robust mean estimation (trimmed mean)
        max_rollouts: Maximum number of rollouts to process (for testing)
        
    Returns:
        DataFrame with columns: feature_id, category, conf, mass, site_contrib
    """
    # Load SAE features and labels
    with open(sae_features_path, 'r') as f:
        sae_data = json.load(f)
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    # Create feature→category mapping
    feature_to_category = {}
    feature_to_conf = {}
    
    for item in labels_data.get('explanations', []):
        fid = str(item['feature_id'])
        feature_to_category[fid] = item.get('category_id', 'uncategorized')
        # Confidence defaults to 1.0 if not specified
        feature_to_conf[fid] = item.get('confidence', 1.0)
    
    # Extract adapter types and layer info from first H5 file
    lora_path = Path(lora_path)
    if lora_path.is_dir():
        h5_files = sorted(glob(str(lora_path / "rollout_*.h5")))
        if not h5_files:
            raise ValueError(f"No H5 files found in {lora_path}")
        
        with h5py.File(h5_files[0], 'r') as f:
            adapter_types = list(f.attrs.get('adapter_types', []))
            num_layers = int(f.attrs.get('num_layers', 64))
    else:
        # NPZ format support
        data = np.load(lora_path, allow_pickle=True)
        adapter_types = data.get('adapter_types', ['gate_proj', 'up_proj', 'down_proj'])
        num_layers = data.get('num_layers', 64)
        h5_files = None
    
    # Build site names: L{layer}.{adapter_type}
    sites = []
    site_to_idx = {}
    idx = 0
    for layer in range(num_layers):
        for adapter in adapter_types:
            site = f"L{layer}.{adapter}"
            sites.append(site)
            site_to_idx[site] = (layer, adapter_types.index(adapter))
            idx += 1
    
    # Initialize flow accumulator: w_{s→f} for each site-feature pair
    # Using dict of dicts for sparse representation
    flow_accumulator = {site: {} for site in sites}
    token_count = 0
    
    # Process activations
    if h5_files:
        # Process H5 files
        files_to_process = h5_files[:max_rollouts] if max_rollouts else h5_files
        
        for h5_file in tqdm(files_to_process, desc="Processing rollouts"):
            with h5py.File(h5_file, 'r') as f:
                # Shape: [num_tokens, num_layers, num_adapters]
                lora_acts = np.array(f['activations'])
                rollout_idx = int(f.attrs.get('rollout_idx', 0))
                num_tokens = lora_acts.shape[0]
                
                # Get SAE activations for this rollout
                sae_acts_rollout = _get_sae_acts_for_rollout(
                    sae_data, rollout_idx, num_tokens
                )
                
                # Compute flows: w_{s→f} += E_t[|a_s(t)| * z_f^+(t)]
                _accumulate_flows(
                    flow_accumulator, lora_acts, sae_acts_rollout,
                    sites, site_to_idx, robust
                )
                
                token_count += num_tokens
    else:
        # NPZ format - simplified version
        warnings.warn("NPZ format support is simplified - using mock data")
        token_count = 1000  # Mock value
    
    # Normalize flows by token count to get expectation
    for site in flow_accumulator:
        for fid in flow_accumulator[site]:
            flow_accumulator[site][fid] /= max(1, token_count)
    
    # Build DataFrame
    rows = []
    for fid in feature_to_category:
        # Compute feature mass: m_f = sum_s w_{s→f}
        site_contrib = {}
        mass = 0.0
        
        for site in sites:
            if fid in flow_accumulator[site]:
                contrib = flow_accumulator[site][fid]
                site_contrib[site] = float(contrib)
                mass += contrib
        
        if mass > 0:  # Only include features with non-zero mass
            rows.append({
                'feature_id': fid,
                'category': feature_to_category[fid],
                'conf': feature_to_conf[fid],
                'mass': mass,
                'site_contrib': site_contrib
            })
    
    df = pd.DataFrame(rows)
    return df


def _get_sae_acts_for_rollout(
    sae_data: dict, rollout_idx: int, num_tokens: int
) -> Dict[str, np.ndarray]:
    """
    Extract SAE activations for a specific rollout.
    
    Returns dict mapping feature_id → activation array
    """
    sae_acts = {}
    
    # Iterate through features
    for fid, feature_data in sae_data.get('features', {}).items():
        # Initialize activation array
        acts = np.zeros(num_tokens, dtype=np.float32)
        
        # Fill in activations from examples
        for example in feature_data.get('examples', []):
            if example['rollout_idx'] == rollout_idx:
                token_idx = example['token_idx']
                if 0 <= token_idx < num_tokens:
                    acts[token_idx] = example['activation_value']
        
        if acts.any():  # Only store if non-zero
            sae_acts[str(fid)] = acts
    
    return sae_acts


def _accumulate_flows(
    flow_accumulator: dict,
    lora_acts: np.ndarray,
    sae_acts: Dict[str, np.ndarray],
    sites: List[str],
    site_to_idx: dict,
    robust: bool = False
) -> None:
    """
    Accumulate flow weights w_{s→f} = E_t[|a_s(t)| * z_f^+(t)].
    
    Modifies flow_accumulator in place.
    """
    num_tokens = lora_acts.shape[0]
    
    for site in sites:
        layer_idx, adapter_idx = site_to_idx[site]
        
        # Get LoRA activations for this site
        # Shape: [num_tokens]
        a_s = np.abs(lora_acts[:, layer_idx, adapter_idx])
        
        # Compute flows to each feature
        for fid, z_f in sae_acts.items():
            # Compute product |a_s(t)| * z_f^+(t)
            # z_f should already be positive (post-ReLU)
            product = a_s * np.maximum(z_f, 0)
            
            if robust:
                # Trimmed mean: remove top/bottom 5%
                flow = _trimmed_mean(product, trim_percent=0.05)
            else:
                # Standard mean
                flow = np.mean(product)
            
            # Accumulate
            if fid not in flow_accumulator[site]:
                flow_accumulator[site][fid] = 0.0
            flow_accumulator[site][fid] += flow * num_tokens  # Will divide by total later


def _trimmed_mean(arr: np.ndarray, trim_percent: float = 0.05) -> float:
    """Compute trimmed mean removing top/bottom percentiles."""
    if len(arr) == 0:
        return 0.0
    
    lower = int(len(arr) * trim_percent)
    upper = int(len(arr) * (1 - trim_percent))
    
    if lower >= upper:
        return np.mean(arr)
    
    sorted_arr = np.sort(arr)
    return np.mean(sorted_arr[lower:upper])


def normalize_confidences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize confidences to sum to 1 per feature (for multi-label support).
    
    For single-label case, this is a no-op.
    """
    df = df.copy()
    
    # Group by feature and normalize confidences
    for fid in df['feature_id'].unique():
        mask = df['feature_id'] == fid
        total_conf = df.loc[mask, 'conf'].sum()
        if total_conf > 0:
            df.loc[mask, 'conf'] = df.loc[mask, 'conf'] / total_conf
    
    return df


def aggregate_middle_nodes(
    df: pd.DataFrame,
    categories: List[str],
    sites: List[str],
    top_k: int = 6
) -> Tuple[dict, dict, dict]:
    """
    Build Sankey nodes and links with top-K aggregation.
    
    Args:
        df: Features DataFrame
        categories: Ordered list of category IDs
        sites: Ordered list of site names
        top_k: Number of top features to keep per category
        
    Returns:
        Tuple of (nodes, links, palette) where:
        - nodes: {labels: [...], types: [...], colors: [...]}
        - links: {source: [...], target: [...], value: [...], color: [...]}
        - palette: {category_id: hex_color}
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors
    
    # Build color palette
    palette = {}
    for i, cat in enumerate(categories):
        color = plt.cm.tab20(i % 20)
        palette[cat] = matplotlib.colors.to_hex(color)
    
    # Initialize nodes
    node_labels = []
    node_types = []
    node_colors = []
    node_to_idx = {}
    
    # Add site nodes (left column)
    for site in sites:
        node_labels.append(f"LoRA:{site}")
        node_types.append("site")
        node_colors.append("#999999")  # Gray for sites
        node_to_idx[f"site:{site}"] = len(node_labels) - 1
    
    # Process each category
    feature_to_node = {}  # Map feature_id to node index
    
    for cat in categories:
        cat_df = df[df['category'] == cat].copy()
        if len(cat_df) == 0:
            continue
        
        # Sort by mass descending
        cat_df = cat_df.sort_values('mass', ascending=False)
        
        # Top-K features as individual nodes
        top_features = cat_df.head(top_k)
        for _, row in top_features.iterrows():
            fid = row['feature_id']
            node_labels.append(f"F{fid}")
            node_types.append("feature")
            node_colors.append(palette[cat])
            idx = len(node_labels) - 1
            node_to_idx[f"feature:{fid}"] = idx
            feature_to_node[fid] = idx
        
        # Aggregate remaining features
        remaining = cat_df.iloc[top_k:]
        if len(remaining) > 0:
            node_labels.append(f"Other ({cat})")
            node_types.append("feature_agg")
            node_colors.append(palette[cat])
            idx = len(node_labels) - 1
            node_to_idx[f"other:{cat}"] = idx
            
            # Map remaining features to this node
            for _, row in remaining.iterrows():
                feature_to_node[row['feature_id']] = idx
    
    # Add category nodes (right column)
    for cat in categories:
        # Get readable label from categories.json if available
        node_labels.append(f"Cat:{cat}")
        node_types.append("category")
        node_colors.append(palette[cat])
        node_to_idx[f"category:{cat}"] = len(node_labels) - 1
    
    # Build links
    links = {
        'source': [],
        'target': [],
        'value': [],
        'color': []
    }
    
    # Site → Feature links
    for _, row in df.iterrows():
        fid = row['feature_id']
        if fid not in feature_to_node:
            continue
        
        target_idx = feature_to_node[fid]
        site_contrib = row['site_contrib']
        cat = row['category']
        
        for site, weight in site_contrib.items():
            if weight > 0 and f"site:{site}" in node_to_idx:
                links['source'].append(node_to_idx[f"site:{site}"])
                links['target'].append(target_idx)
                links['value'].append(weight)
                # Translucent category color using rgba
                hex_color = palette[cat]
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                links['color'].append(f"rgba({r},{g},{b},0.5)")  # 50% alpha
    
    # Feature → Category links
    for cat in categories:
        cat_df = df[df['category'] == cat]
        cat_idx = node_to_idx.get(f"category:{cat}")
        
        if cat_idx is None:
            continue
        
        # Sum masses for all features in this category
        for _, row in cat_df.iterrows():
            fid = row['feature_id']
            if fid in feature_to_node:
                source_idx = feature_to_node[fid]
                # Feature→Category flow: mass * confidence
                weight = row['mass'] * row['conf']
                
                if weight > 0:
                    links['source'].append(source_idx)
                    links['target'].append(cat_idx)
                    links['value'].append(weight)
                    links['color'].append(palette[cat])  # Solid color
    
    nodes = {
        'labels': node_labels,
        'types': node_types,
        'colors': node_colors
    }
    
    return nodes, links, palette