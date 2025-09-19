# %%
"""
Feature Clustering Visualization with UMAP

This notebook loads a trained SAE, extracts decoder features,
and visualizes them in 2D using UMAP with category coloring.
"""

# %%
# Imports
import torch
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import umap
from sklearn.preprocessing import normalize
import sys
sys.path.append('/workspace/reasoning_interp/sae_interp')
from batch_topk_sae import BatchTopKSAE

print("Libraries imported successfully")

# %%
# Configuration
SAE_PATH = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
CATEGORIES_PATH = "/workspace/reasoning_interp/sae_interp/autointerp/categorized_features.json"
OUTPUT_HTML = "feature_clusters_visualization.html"

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'
RANDOM_STATE = 42

print(f"SAE path: {SAE_PATH}")
print(f"Categories path: {CATEGORIES_PATH}")

# %%
# Load the SAE model and extract decoder matrix
print("Loading SAE model...")

# Load the saved model state
checkpoint = torch.load(SAE_PATH, map_location='cpu')
print(f"Checkpoint keys: {checkpoint.keys()}")

# Extract model configuration from checkpoint
if 'config' in checkpoint:
    config = checkpoint['config']
    d_model = config['d_model']
    dict_size = config['dict_size']
    k = config['k']
else:
    # Infer from state dict shapes
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    decoder_shape = state_dict['W_dec'].shape
    dict_size, d_model = decoder_shape
    k = 50  # Default value

print(f"Model config - d_model: {d_model}, dict_size: {dict_size}, k: {k}")

# Create model and load weights
sae = BatchTopKSAE(d_model=d_model, dict_size=dict_size, k=k)
if 'model_state_dict' in checkpoint:
    sae.load_state_dict(checkpoint['model_state_dict'])
else:
    sae.load_state_dict(checkpoint)

# Extract decoder matrix
decoder_matrix = sae.W_dec.detach().cpu().numpy()
print(f"Decoder matrix shape: {decoder_matrix.shape}")

# %%
# Apply row-wise normalization for cosine similarity
print("Normalizing decoder features...")

# Normalize each row (feature) to unit length
decoder_normalized = normalize(decoder_matrix, norm='l2', axis=1)
print(f"Normalized decoder shape: {decoder_normalized.shape}")
print(f"Sample norms after normalization: {np.linalg.norm(decoder_normalized[:5], axis=1)}")

# %%
# Compute UMAP embedding
print("Computing UMAP embedding...")
print(f"Using {UMAP_N_NEIGHBORS} neighbors, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}")

# Initialize UMAP
reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric=UMAP_METRIC,
    n_components=2,
    random_state=RANDOM_STATE,
    verbose=True
)

# Fit and transform
embedding = reducer.fit_transform(decoder_normalized)
print(f"UMAP embedding shape: {embedding.shape}")

# %%
# Load categorization data
print("Loading categorization data...")

with open(CATEGORIES_PATH, 'r') as f:
    cat_data = json.load(f)

# Extract explanations and categories
explanations = cat_data['explanations']
print(f"Loaded {len(explanations)} feature explanations")

# Create lookup dictionaries
feature_to_category = {}
feature_to_explanation = {}

for item in explanations:
    fid = item['feature_id']
    feature_to_category[fid] = item.get('category_id', 'uncategorized')
    feature_to_explanation[fid] = item.get('explanation', 'No explanation')

# Get all unique categories from the actual data
unique_categories = list(set(feature_to_category.values()))

# Also check if we need to add 'uncategorized' for features not in the categorization file
all_features_covered = all(i in feature_to_category for i in range(dict_size))
if not all_features_covered:
    if 'uncategorized' not in unique_categories:
        unique_categories.append('uncategorized')

print(f"Found {len(unique_categories)} unique categories")

# %%
# Prepare data for visualization
print("Preparing visualization data...")

# Filter out dead features (those not in the categorization file)
# Only include features that have explanations
active_feature_ids = [i for i in range(dict_size) if i in feature_to_explanation]
print(f"Found {len(active_feature_ids)} active features out of {dict_size} total features")
print(f"Excluding {dict_size - len(active_feature_ids)} dead features")

# Create arrays for plotting (only for active features)
feature_ids = active_feature_ids
categories = [feature_to_category[i] for i in feature_ids]
explanations_list = [feature_to_explanation[i] for i in feature_ids]

# Filter the UMAP embedding to only include active features
embedding_filtered = embedding[active_feature_ids]

# Create hover text
hover_texts = [
    f"Feature {fid}<br>"
    f"Category: {cat}<br>"
    f"Explanation: {exp}"
    for fid, cat, exp in zip(feature_ids, categories, explanations_list)
]

# Assign colors to categories
color_map = {}
colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Set2
for i, cat in enumerate(unique_categories):
    color_map[cat] = colors[i % len(colors)]

# Get colors for each point
point_colors = [color_map[cat] for cat in categories]

print(f"Prepared data for {len(feature_ids)} features")

# %%S
# Create interactive Plotly visualization
print("Creating interactive visualization...")

# Create the figure
fig = go.Figure()

# Add a trace for each category (for legend grouping)
for category in unique_categories:
    # Find indices for this category
    indices = [i for i, cat in enumerate(categories) if cat == category]
    
    if indices:
        fig.add_trace(go.Scatter(
            x=embedding_filtered[indices, 0],
            y=embedding_filtered[indices, 1],
            mode='markers',
            name=category,
            text=[hover_texts[i] for i in indices],
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                color=color_map[category],
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            )
        ))

# Update layout
fig.update_layout(
    title={
        'text': 'SAE Feature Clustering with UMAP (Colored by Category)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    width=1200,
    height=800,
    hovermode='closest',
    legend=dict(
        title='Categories',
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.01,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=1
    ),
    plot_bgcolor='#f8f9fa',
    paper_bgcolor='white'
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

print("Visualization created successfully")

# %%
# Save the visualization
print(f"Saving visualization to {OUTPUT_HTML}...")

fig.write_html(
    OUTPUT_HTML,
    config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'feature_clusters',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
)

print(f"Visualization saved to {OUTPUT_HTML}")
print("You can open this file in a browser to interact with the plot")

# %%
# Display the figure (if in notebook environment)
fig.show()

# %%
# Print category distribution statistics
print("\n" + "="*50)
print("Category Distribution Statistics")
print("="*50)

from collections import Counter
category_counts = Counter(categories)

# Sort by count
sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 10 most common categories:")
for cat, count in sorted_categories[:10]:
    percentage = (count / len(categories)) * 100
    print(f"  {cat:40s}: {count:4d} ({percentage:5.1f}%)")

print(f"\nTotal features: {len(categories)}")
print(f"Unique categories: {len(unique_categories)}")

# %%
# Optional: Save embedding data for later use
print("\nSaving embedding data...")

embedding_data = {
    'umap_coordinates': embedding_filtered.tolist(),
    'feature_ids': feature_ids,
    'categories': categories,
    'explanations': explanations_list,
    'umap_params': {
        'n_neighbors': UMAP_N_NEIGHBORS,
        'min_dist': UMAP_MIN_DIST,
        'metric': UMAP_METRIC,
        'random_state': RANDOM_STATE
    }
}

with open('feature_embedding_data.json', 'w') as f:
    json.dump(embedding_data, f, indent=2)

print("Embedding data saved to feature_embedding_data.json")
print("\nDone!")
# %%
