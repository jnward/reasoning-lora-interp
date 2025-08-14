#!/usr/bin/env python3
"""
Check if the suspicious clustering features have abnormal embeddings.
"""

import numpy as np
import json
from scipy.spatial.distance import cosine
from collections import defaultdict

# Load embeddings
embeddings = np.load('feature_embeddings.npy')
print(f"Loaded embeddings: {embeddings.shape}")

# Load interpretations to get feature IDs
with open('all_interpretations_o3.json', 'r') as f:
    data = json.load(f)
    
feature_ids = [e['feature_id'] for e in data['explanations']]
explanations = [e['explanation'] for e in data['explanations']]

# Create mapping from feature_id to index
fid_to_idx = {fid: idx for idx, fid in enumerate(feature_ids)}

# Define suspicious ranges (from clustering analysis)
suspicious_ranges = [
    (458, 499, 15),   # 15 features clustered
    (501, 599, 38),   # 38 features clustered
    (606, 699, 16),   # 16 features clustered
    (1524, 1599, 26), # 26 features clustered
    (1600, 1699, 42), # 42 features clustered
    (1701, 1799, 20), # 20 features clustered
    (3473, 3499, 13), # 13 features clustered
    (3508, 3599, 15), # 15 features clustered
]

print("\nAnalyzing embeddings for suspicious features...")
print("="*60)

# Collect all suspicious features
all_suspicious = []
for start, end, count in suspicious_ranges:
    for fid in range(start, end + 1):
        if fid in fid_to_idx:
            all_suspicious.append(fid)

print(f"Total suspicious features: {len(all_suspicious)}")

# Check for empty explanations
empty_features = []
for fid in all_suspicious:
    idx = fid_to_idx[fid]
    if explanations[idx] == '' or explanations[idx].strip() == '':
        empty_features.append(fid)
        
print(f"Features with empty explanations: {empty_features}")

# Analyze embedding properties
print("\n" + "-"*40)
print("Embedding statistics by range:")
print("-"*40)

for start, end, expected_count in suspicious_ranges:
    range_embeddings = []
    range_features = []
    
    for fid in range(start, end + 1):
        if fid in fid_to_idx:
            idx = fid_to_idx[fid]
            range_embeddings.append(embeddings[idx])
            range_features.append(fid)
    
    if len(range_embeddings) > 0:
        range_embeddings = np.array(range_embeddings)
        
        # Check various statistics
        means = np.mean(range_embeddings, axis=1)
        stds = np.std(range_embeddings, axis=1)
        norms = np.linalg.norm(range_embeddings, axis=1)
        
        print(f"\nRange {start}-{end} ({len(range_features)} features):")
        print(f"  Embedding norms: mean={np.mean(norms):.3f}, std={np.std(norms):.3f}")
        print(f"  Embedding means: mean={np.mean(means):.6f}, std={np.std(means):.6f}")
        print(f"  Embedding stds:  mean={np.mean(stds):.3f}, std={np.std(stds):.3f}")
        
        # Check if embeddings are identical or very similar
        if len(range_embeddings) > 1:
            # Compute pairwise similarities
            similarities = []
            for i in range(len(range_embeddings)):
                for j in range(i+1, len(range_embeddings)):
                    sim = 1 - cosine(range_embeddings[i], range_embeddings[j])
                    similarities.append(sim)
            
            print(f"  Pairwise cosine similarities: mean={np.mean(similarities):.3f}, max={np.max(similarities):.3f}")
            
            # Check for near-identical embeddings
            very_similar = sum(1 for s in similarities if s > 0.99)
            if very_similar > 0:
                print(f"  WARNING: {very_similar} pairs with >0.99 similarity!")
        
        # Check for embeddings that might be defaults/errors
        zero_embeddings = [fid for i, fid in enumerate(range_features) 
                          if np.allclose(range_embeddings[i], 0)]
        if zero_embeddings:
            print(f"  WARNING: Zero embeddings for features: {zero_embeddings}")

# Compare suspicious vs normal features
print("\n" + "="*60)
print("Comparing suspicious vs normal features:")
print("="*60)

normal_features = [fid for fid in feature_ids if fid not in all_suspicious]
suspicious_embeddings = np.array([embeddings[fid_to_idx[fid]] for fid in all_suspicious])
normal_embeddings = np.array([embeddings[fid_to_idx[fid]] for fid in normal_features])

print(f"\nSuspicious features (n={len(suspicious_embeddings)}):")
print(f"  Norm: mean={np.mean(np.linalg.norm(suspicious_embeddings, axis=1)):.3f}")
print(f"  Mean: mean={np.mean(np.mean(suspicious_embeddings, axis=1)):.6f}")
print(f"  Std:  mean={np.mean(np.std(suspicious_embeddings, axis=1)):.3f}")

print(f"\nNormal features (n={len(normal_embeddings)}):")
print(f"  Norm: mean={np.mean(np.linalg.norm(normal_embeddings, axis=1)):.3f}")
print(f"  Mean: mean={np.mean(np.mean(normal_embeddings, axis=1)):.6f}")
print(f"  Std:  mean={np.mean(np.std(normal_embeddings, axis=1)):.3f}")

# Check specifically for features with empty explanations
print("\n" + "="*60)
print("Analyzing features with empty explanations:")
print("="*60)

empty_embeddings = []
for fid in empty_features:
    idx = fid_to_idx[fid]
    empty_embeddings.append(embeddings[idx])
    print(f"\nFeature {fid} (empty explanation):")
    print(f"  Norm: {np.linalg.norm(embeddings[idx]):.3f}")
    print(f"  Mean: {np.mean(embeddings[idx]):.6f}")
    print(f"  Std:  {np.std(embeddings[idx]):.3f}")
    print(f"  First 10 values: {embeddings[idx][:10]}")

# Check if empty explanations have identical embeddings
if len(empty_embeddings) > 1:
    print("\nComparing embeddings of empty explanations:")
    for i in range(len(empty_embeddings)):
        for j in range(i+1, len(empty_embeddings)):
            sim = 1 - cosine(empty_embeddings[i], empty_embeddings[j])
            print(f"  Features {empty_features[i]} vs {empty_features[j]}: similarity = {sim:.6f}")
            if np.allclose(empty_embeddings[i], empty_embeddings[j]):
                print(f"    WARNING: Embeddings are identical!")