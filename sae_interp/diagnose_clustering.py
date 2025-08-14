# %%
"""
Diagnose clustering issues with SAE features
"""

import torch
import numpy as np
import json
# import matplotlib.pyplot as plt  # Not available in this environment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import umap
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('/workspace/reasoning_interp/sae_interp')
from batch_topk_sae import BatchTopKSAE

# %%
# Load the SAE model
SAE_PATH = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
print("Loading SAE model...")

checkpoint = torch.load(SAE_PATH, map_location='cpu')
if 'config' in checkpoint:
    config = checkpoint['config']
    d_model = config['d_model']
    dict_size = config['dict_size']
    k = config['k']
else:
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    decoder_shape = state_dict['W_dec'].shape
    dict_size, d_model = decoder_shape
    k = 50

print(f"Model config - d_model: {d_model}, dict_size: {dict_size}, k: {k}")

sae = BatchTopKSAE(d_model=d_model, dict_size=dict_size, k=k)
if 'model_state_dict' in checkpoint:
    sae.load_state_dict(checkpoint['model_state_dict'])
else:
    sae.load_state_dict(checkpoint)

# %%
# Extract decoder matrix and check its properties
decoder_matrix = sae.W_dec.detach().cpu().numpy()
print(f"Decoder matrix shape: {decoder_matrix.shape}")
print(f"Shape interpretation: [{dict_size} features, {d_model} dimensions]")

# Check if vectors are already normalized
norms = np.linalg.norm(decoder_matrix, axis=1)
print(f"\nDecoder row norms (first 10): {norms[:10]}")
print(f"Mean norm: {norms.mean():.4f}, Std: {norms.std():.4f}")
print(f"Min norm: {norms.min():.4f}, Max norm: {norms.max():.4f}")

# %%
# Check similarity structure
print("\nChecking similarity structure...")

# Take a sample of features
sample_size = 100
sample_indices = np.random.choice(dict_size, sample_size, replace=False)
sample_features = decoder_matrix[sample_indices]

# Normalize the sample
sample_normalized = normalize(sample_features, norm='l2', axis=1)

# Compute cosine similarity matrix
sim_matrix = cosine_similarity(sample_normalized)

print(f"Similarity matrix shape: {sim_matrix.shape}")
print(f"Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
print(f"Mean similarity (excluding diagonal): {sim_matrix[~np.eye(sample_size, dtype=bool)].mean():.4f}")

# Plot similarity distribution
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Distribution of Cosine Similarities', 
                                    'Similarity Matrix (100 random features)'))

# Histogram
fig.add_trace(
    go.Histogram(x=sim_matrix[~np.eye(sample_size, dtype=bool)].flatten(), nbinsx=50),
    row=1, col=1
)

# Heatmap
fig.add_trace(
    go.Heatmap(z=sim_matrix, colorscale='RdBu', zmin=-1, zmax=1),
    row=1, col=2
)

fig.update_xaxes(title_text='Cosine Similarity', row=1, col=1)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_layout(title='Similarity Analysis', showlegend=False)
fig.write_html('similarity_analysis.html')
print("Saved similarity analysis to similarity_analysis.html")

# %%
# Load categories to check if similar features have similar categories
CATEGORIES_PATH = "/workspace/reasoning_interp/sae_interp/autointerp/categorized_features.json"
with open(CATEGORIES_PATH, 'r') as f:
    cat_data = json.load(f)

feature_to_category = {}
for item in cat_data['explanations']:
    feature_to_category[item['feature_id']] = item.get('category_id', 'uncategorized')

# Check if features with same category are similar
print("\nAnalyzing within-category similarity...")
from collections import defaultdict

category_features = defaultdict(list)
for fid, cat in feature_to_category.items():
    category_features[cat].append(fid)

# For each category with multiple features, compute average similarity
category_sims = {}
for cat, fids in category_features.items():
    if len(fids) > 5:  # Only look at categories with enough features
        cat_features = decoder_matrix[fids]
        cat_features_norm = normalize(cat_features, norm='l2', axis=1)
        cat_sim = cosine_similarity(cat_features_norm)
        # Get average similarity (excluding diagonal)
        mask = ~np.eye(len(fids), dtype=bool)
        avg_sim = cat_sim[mask].mean() if mask.any() else 0
        category_sims[cat] = avg_sim
        print(f"  {cat[:40]:40s}: {avg_sim:.4f} (n={len(fids)})")

# %%
# Try different UMAP parameters
print("\nTrying different UMAP configurations...")

# Normalize decoder matrix
decoder_normalized = normalize(decoder_matrix, norm='l2', axis=1)

# Only use active features
active_feature_ids = [i for i in range(dict_size) if i in feature_to_category]
decoder_active = decoder_normalized[active_feature_ids]
print(f"Using {len(active_feature_ids)} active features")

configs = [
    {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'cosine'},
    {'n_neighbors': 30, 'min_dist': 0.1, 'metric': 'cosine'},
    {'n_neighbors': 15, 'min_dist': 0.01, 'metric': 'cosine'},
    {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'},
    {'n_neighbors': 5, 'min_dist': 0.05, 'metric': 'cosine'},
]

fig = make_subplots(rows=2, cols=3,
                    subplot_titles=[f"n={c['n_neighbors']}, d={c['min_dist']}, m={c['metric']}" 
                                   for c in configs] + [''])

# Get categories for active features
categories = [feature_to_category[fid] for fid in active_feature_ids]

# Create a simple color map for top categories
from collections import Counter
top_categories = [cat for cat, _ in Counter(categories).most_common(10)]
colors = []
for cat in categories:
    if cat in top_categories:
        colors.append(top_categories.index(cat))
    else:
        colors.append(10)  # "Other" category

for idx, config in enumerate(configs):
    print(f"\nConfig {idx+1}: {config}")
    
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        **config
    )
    
    embedding = reducer.fit_transform(decoder_active)
    
    # Determine subplot position
    row = idx // 3 + 1
    col = idx % 3 + 1
    
    fig.add_trace(
        go.Scatter(x=embedding[:, 0], y=embedding[:, 1],
                  mode='markers',
                  marker=dict(size=2, color=colors, colorscale='Viridis', opacity=0.5),
                  showlegend=False),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text='UMAP 1', row=row, col=col)
    fig.update_yaxes(title_text='UMAP 2', row=row, col=col)

fig.update_layout(title='UMAP Parameter Comparison', height=800, width=1200)
fig.write_html('umap_comparison.html')
print("Saved UMAP comparison to umap_comparison.html")

print("\nSaved diagnostic plots to similarity_analysis.html and umap_comparison.html")

# %%
# Check if transposing helps (maybe features should be columns?)
print("\nTrying with transposed decoder matrix...")

# Transpose: now each column is a feature
decoder_transposed = decoder_matrix.T  # Shape: [d_model, dict_size]
print(f"Transposed shape: {decoder_transposed.shape}")

# Normalize columns (each feature)
decoder_t_norm = normalize(decoder_transposed, norm='l2', axis=0)

# For UMAP, we need features as rows, so transpose back
features_for_umap = decoder_t_norm.T[active_feature_ids]

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    n_components=2,
    random_state=42
)

embedding_transposed = reducer.fit_transform(features_for_umap)

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=embedding_transposed[:, 0], y=embedding_transposed[:, 1],
              mode='markers',
              marker=dict(size=3, color=colors, colorscale='Viridis', 
                         opacity=0.6, showscale=True,
                         colorbar=dict(title='Category')),
              text=[f"Cat: {cat}" for cat in categories],
              hovertemplate='%{text}<extra></extra>')
)
fig.update_layout(title='UMAP with Transposed Decoder (features as columns)',
                 xaxis_title='UMAP 1', yaxis_title='UMAP 2',
                 width=800, height=600)
fig.write_html('umap_transposed.html')
print("Saved transposed UMAP to umap_transposed.html")

print("Diagnostic complete!")