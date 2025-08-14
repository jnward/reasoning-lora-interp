#!/usr/bin/env python3
"""
Analyze which features are clustering by index in the UMAP projection.

This helps identify if certain feature ranges have unusual clustering patterns.
"""

import json
import numpy as np
import argparse
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt


def load_data(embeddings_path, projections_path, interpretations_path):
    """Load embeddings, projections, and interpretations."""
    embeddings = np.load(embeddings_path) if embeddings_path else None
    projections = np.load(projections_path)
    
    with open(interpretations_path, 'r') as f:
        data = json.load(f)
    
    explanations = data['explanations']
    feature_ids = [e['feature_id'] for e in explanations]
    texts = [e['explanation'] for e in explanations]
    
    return embeddings, projections, feature_ids, texts


def find_index_clusters(projections, feature_ids, k=10, index_threshold=50):
    """
    Find features whose nearest neighbors have similar indices.
    
    Args:
        projections: UMAP projections (n_samples, n_dims)
        feature_ids: List of feature IDs
        k: Number of nearest neighbors to check
        index_threshold: Max index difference to consider "nearby"
    
    Returns:
        Dictionary mapping feature_id to clustering score
    """
    # Build nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(projections)
    
    # For each point, check if neighbors have similar indices
    index_clustering_scores = {}
    suspicious_features = []
    
    for i, fid in enumerate(feature_ids):
        # Find k nearest neighbors (excluding self)
        distances, indices = nn.kneighbors([projections[i]])
        neighbor_indices = indices[0][1:]  # Exclude self
        
        # Get feature IDs of neighbors
        neighbor_ids = [feature_ids[idx] for idx in neighbor_indices]
        
        # Calculate how many neighbors have similar feature IDs
        similar_id_neighbors = sum(1 for nid in neighbor_ids 
                                  if abs(nid - fid) <= index_threshold)
        
        clustering_score = similar_id_neighbors / k
        index_clustering_scores[fid] = clustering_score
        
        # Flag if majority of neighbors have similar indices
        if clustering_score >= 0.5:
            suspicious_features.append({
                'feature_id': fid,
                'score': clustering_score,
                'neighbor_ids': neighbor_ids,
                'avg_id_distance': np.mean([abs(nid - fid) for nid in neighbor_ids])
            })
    
    return index_clustering_scores, suspicious_features


def analyze_suspicious_features(suspicious_features, texts, feature_ids):
    """Analyze what's special about features that cluster by index."""
    if not suspicious_features:
        print("No suspicious index-based clustering found!")
        return
    
    print(f"\nFound {len(suspicious_features)} features with index-based clustering:")
    print("="*60)
    
    # Group by ranges
    ranges = {}
    for feat in suspicious_features:
        range_key = (feat['feature_id'] // 100) * 100
        if range_key not in ranges:
            ranges[range_key] = []
        ranges[range_key].append(feat)
    
    # Analyze each range
    for range_start in sorted(ranges.keys()):
        range_features = ranges[range_start]
        print(f"\nRange {range_start}-{range_start+99}: {len(range_features)} features")
        
        # Show a few examples
        for feat in range_features[:3]:
            fid = feat['feature_id']
            idx = feature_ids.index(fid)
            text = texts[idx]
            print(f"  Feature {fid} (score={feat['score']:.2f}): {text[:60]}...")
            print(f"    Neighbors: {feat['neighbor_ids'][:5]}...")
    
    # Check for common patterns in explanations
    print("\n" + "="*60)
    print("Analyzing explanation patterns in suspicious features...")
    
    suspicious_ids = [f['feature_id'] for f in suspicious_features]
    suspicious_texts = [texts[feature_ids.index(fid)] for fid in suspicious_ids]
    all_other_texts = [texts[i] for i, fid in enumerate(feature_ids) 
                       if fid not in suspicious_ids]
    
    # Compare text statistics
    print(f"\nExplanation statistics:")
    print(f"  Suspicious features (n={len(suspicious_texts)}):")
    print(f"    Avg length: {np.mean([len(t) for t in suspicious_texts]):.1f}")
    
    # Check for common words/patterns
    suspicious_words = ' '.join(suspicious_texts).lower().split()
    other_words = ' '.join(all_other_texts).lower().split()
    
    # Find overrepresented words in suspicious features
    from collections import Counter
    suspicious_counter = Counter(suspicious_words)
    other_counter = Counter(other_words)
    
    # Normalize by total word count
    total_suspicious = sum(suspicious_counter.values())
    total_other = sum(other_counter.values())
    
    print(f"\nMost common words in suspicious features:")
    for word, count in suspicious_counter.most_common(10):
        freq_suspicious = count / total_suspicious
        freq_other = other_counter.get(word, 0) / total_other
        ratio = freq_suspicious / (freq_other + 1e-6)
        if ratio > 1.5:  # Overrepresented
            print(f"  '{word}': {ratio:.1f}x more common")


def plot_index_clustering(projections, feature_ids, index_clustering_scores):
    """Create visualization of index-based clustering."""
    scores = [index_clustering_scores[fid] for fid in feature_ids]
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: UMAP colored by clustering score
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(projections[:, 0], projections[:, 1], 
                         c=scores, cmap='RdYlBu_r', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Index Clustering Score')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Projection Colored by Index Clustering')
    
    # Plot 2: Histogram of clustering scores
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=20, edgecolor='black')
    plt.xlabel('Index Clustering Score')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Index Clustering Scores')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Suspicious threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('index_clustering_analysis.png', dpi=150)
    print(f"\nSaved visualization to index_clustering_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze index-based clustering in UMAP")
    parser.add_argument("--embeddings", type=str, help="Path to embeddings numpy file")
    parser.add_argument("--projections", type=str, default="umap_projections_2d.npy",
                       help="Path to UMAP projections")
    parser.add_argument("--interpretations", type=str, 
                       default="all_interpretations_o3.json",
                       help="Path to interpretations JSON")
    parser.add_argument("--k-neighbors", type=int, default=10,
                       help="Number of nearest neighbors to check")
    parser.add_argument("--index-threshold", type=int, default=50,
                       help="Max index difference to consider 'nearby'")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    embeddings, projections, feature_ids, texts = load_data(
        args.embeddings, args.projections, args.interpretations
    )
    
    # Find index-based clusters
    print(f"\nAnalyzing index clustering (k={args.k_neighbors}, threshold={args.index_threshold})...")
    index_clustering_scores, suspicious_features = find_index_clusters(
        projections, feature_ids, args.k_neighbors, args.index_threshold
    )
    
    # Analyze suspicious features
    analyze_suspicious_features(suspicious_features, texts, feature_ids)
    
    # Create visualization
    if len(projections.shape) == 2 and projections.shape[1] >= 2:
        plot_index_clustering(projections, feature_ids, index_clustering_scores)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY:")
    scores = list(index_clustering_scores.values())
    print(f"  Total features: {len(feature_ids)}")
    print(f"  Features with >50% index clustering: {len(suspicious_features)}")
    print(f"  Mean clustering score: {np.mean(scores):.3f}")
    print(f"  Max clustering score: {np.max(scores):.3f}")
    
    # List feature ID ranges with high clustering
    if suspicious_features:
        suspicious_ids = [f['feature_id'] for f in suspicious_features]
        print(f"\n  Feature ID ranges with clustering:")
        for i in range(0, max(suspicious_ids), 100):
            count = sum(1 for fid in suspicious_ids if i <= fid < i+100)
            if count > 0:
                print(f"    {i:4d}-{i+99:4d}: {count:3d} features")


if __name__ == "__main__":
    main()