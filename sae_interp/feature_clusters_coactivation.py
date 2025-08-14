# %%
"""
Feature Clustering Visualization using Coactivation Statistics

This notebook computes coactivation patterns from SAE features 
and visualizes them using UMAP with category coloring.
"""

# %%
# Imports
import torch
import numpy as np
import json
import h5py
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import umap
from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import sys
from tqdm import tqdm
sys.path.append('/workspace/reasoning_interp/sae_interp')
from batch_topk_sae import BatchTopKSAE

print("Libraries imported successfully")

# %%
# Configuration
SAE_PATH = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
ACTIVATIONS_DIR = "/workspace/reasoning_interp/sae_interp/activations_all_adapters"
CATEGORIES_PATH = "/workspace/reasoning_interp/sae_interp/autointerp/categorized_features.json"
OUTPUT_HTML = "feature_clusters_coactivation.html"

# Processing parameters
NUM_ROLLOUTS = 256  # Process all rollouts
BATCH_SIZE = 512    # Batch size for SAE processing
ACTIVATION_THRESHOLD = 0.01  # Threshold for considering a feature "active"

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'precomputed'  # We'll provide a distance matrix
RANDOM_STATE = 42

print(f"SAE path: {SAE_PATH}")
print(f"Activations dir: {ACTIVATIONS_DIR}")
print(f"Will process {NUM_ROLLOUTS} rollouts")

# %%
# Load the SAE model
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

# Create model and load weights
sae = BatchTopKSAE(d_model=d_model, dict_size=dict_size, k=k)
if 'model_state_dict' in checkpoint:
    sae.load_state_dict(checkpoint['model_state_dict'])
else:
    sae.load_state_dict(checkpoint)

sae.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae = sae.to(device)
print(f"SAE loaded and moved to {device}")

# %%
# Process activations through SAE to get feature activations
print("Processing activations to get SAE features...")

# Initialize storage for feature activations
all_feature_activations = []

# Process rollouts
for rollout_idx in tqdm(range(NUM_ROLLOUTS), desc="Processing rollouts"):
    h5_path = Path(ACTIVATIONS_DIR) / f"rollout_{rollout_idx}.h5"
    
    if not h5_path.exists():
        continue
    
    # Load activations from H5 file
    with h5py.File(h5_path, 'r') as f:
        # Shape: (num_tokens, 64_layers, 7_adapters)
        activations = f['activations'][:]
    
    # Flatten the last two dimensions to get (num_tokens, 448)
    num_tokens = activations.shape[0]
    activations_flat = activations.reshape(num_tokens, -1)
    
    # Process in batches through SAE
    rollout_features = []
    for i in range(0, num_tokens, BATCH_SIZE):
        batch = activations_flat[i:i+BATCH_SIZE]
        batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Get sparse feature activations
            sparse_features = sae.encode(batch_tensor)
            rollout_features.append(sparse_features.cpu().numpy())
    
    # Concatenate all batches for this rollout
    if rollout_features:
        rollout_features = np.vstack(rollout_features)
        all_feature_activations.append(rollout_features)

# Concatenate all rollouts
all_feature_activations = np.vstack(all_feature_activations)
print(f"Total feature activation shape: {all_feature_activations.shape}")

# %%
# Compute coactivation statistics
print("Computing coactivation matrix...")

# Binarize activations (active vs inactive)
binary_activations = (all_feature_activations > ACTIVATION_THRESHOLD).astype(np.float32)

# Compute coactivation frequencies
# For efficiency, we'll compute this in chunks
n_features = dict_size
coactivation_matrix = np.zeros((n_features, n_features), dtype=np.float32)

# Compute co-occurrence counts
print("Computing co-occurrence counts...")
for i in tqdm(range(n_features), desc="Features"):
    # Get activation pattern for feature i
    feature_i_active = binary_activations[:, i]
    
    # Compute co-occurrence with all other features
    # This is essentially a dot product
    coactivations = binary_activations.T @ feature_i_active
    coactivation_matrix[i, :] = coactivations

# Normalize to get Jaccard similarity
print("Computing Jaccard similarity...")
# Compute individual activation counts
activation_counts = binary_activations.sum(axis=0)

# Jaccard similarity: intersection / union
# Union = count_i + count_j - intersection
jaccard_matrix = np.zeros_like(coactivation_matrix)
for i in range(n_features):
    for j in range(i, n_features):
        intersection = coactivation_matrix[i, j]
        union = activation_counts[i] + activation_counts[j] - intersection
        if union > 0:
            jaccard_matrix[i, j] = intersection / union
            jaccard_matrix[j, i] = jaccard_matrix[i, j]
        else:
            jaccard_matrix[i, j] = 0
            jaccard_matrix[j, i] = 0

# Set diagonal to 1
np.fill_diagonal(jaccard_matrix, 1.0)

print(f"Jaccard similarity matrix shape: {jaccard_matrix.shape}")
print(f"Similarity range: [{jaccard_matrix.min():.4f}, {jaccard_matrix.max():.4f}]")
print(f"Mean similarity (excluding diagonal): {jaccard_matrix[~np.eye(n_features, dtype=bool)].mean():.4f}")

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

# Filter to only active features (those with explanations)
active_feature_ids = [i for i in range(dict_size) if i in feature_to_explanation]
print(f"Found {len(active_feature_ids)} active features out of {dict_size} total")

# Filter Jaccard matrix to only active features
jaccard_active = jaccard_matrix[np.ix_(active_feature_ids, active_feature_ids)]

# %%
# Convert similarity to distance for UMAP
print("Converting similarity to distance...")

# Convert Jaccard similarity to distance (1 - similarity)
# Ensure it's symmetric and has zero diagonal
distance_matrix = 1.0 - jaccard_active
np.fill_diagonal(distance_matrix, 0.0)

# Check for any numerical issues
distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

print(f"Distance matrix shape: {distance_matrix.shape}")
print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")

# %%
# Compute UMAP embedding
print("Computing UMAP embedding from coactivation distances...")

# Initialize UMAP with precomputed distances
reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric='precomputed',
    n_components=2,
    random_state=RANDOM_STATE,
    verbose=True
)

# Fit and transform
embedding = reducer.fit_transform(distance_matrix)
print(f"UMAP embedding shape: {embedding.shape}")

# %%
# Prepare data for visualization
print("Preparing visualization data...")

# Get categories and explanations for active features
feature_ids = active_feature_ids
categories = [feature_to_category[i] for i in feature_ids]
explanations_list = [feature_to_explanation[i] for i in feature_ids]

# Get activation statistics for hover text
mean_activations = all_feature_activations[:, active_feature_ids].mean(axis=0)
activation_frequencies = (binary_activations[:, active_feature_ids].mean(axis=0) * 100)

# Create hover text with more information
hover_texts = [
    f"Feature {fid}<br>"
    f"Category: {cat}<br>"
    f"Explanation: {exp}<br>"
    f"Mean activation: {mean_act:.3f}<br>"
    f"Active in {freq:.1f}% of tokens"
    for fid, cat, exp, mean_act, freq in zip(
        feature_ids, categories, explanations_list, 
        mean_activations, activation_frequencies
    )
]

# Get unique categories and assign colors
unique_categories = list(set(categories))
print(f"Found {len(unique_categories)} unique categories")

# Assign colors to categories
color_map = {}
colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Set2
for i, cat in enumerate(unique_categories):
    color_map[cat] = colors[i % len(colors)]

# %%
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
            x=embedding[indices, 0],
            y=embedding[indices, 1],
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
        'text': 'SAE Feature Clustering with Coactivation Statistics',
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
    paper_bgcolor='white',
    annotations=[
        dict(
            text=f"Based on {NUM_ROLLOUTS} rollouts, {all_feature_activations.shape[0]} total tokens",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor='left', yanchor='bottom',
            font=dict(size=10, color='gray')
        )
    ]
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
            'filename': 'feature_clusters_coactivation',
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
# Print category distribution and coactivation statistics
print("\n" + "="*50)
print("Category Distribution & Coactivation Statistics")
print("="*50)

from collections import Counter
category_counts = Counter(categories)

# Sort by count
sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 10 most common categories:")
for cat, count in sorted_categories[:10]:
    percentage = (count / len(categories)) * 100
    print(f"  {cat:40s}: {count:4d} ({percentage:5.1f}%)")

# Analyze within-category coactivation
print("\nWithin-category coactivation strength:")
category_coactivations = {}
for cat in unique_categories[:10]:  # Top 10 categories
    cat_indices = [active_feature_ids.index(fid) for fid in active_feature_ids 
                   if feature_to_category.get(fid) == cat]
    if len(cat_indices) > 1:
        # Get mean Jaccard similarity within category
        cat_similarities = []
        for i, idx_i in enumerate(cat_indices):
            for idx_j in cat_indices[i+1:]:
                cat_similarities.append(jaccard_active[idx_i, idx_j])
        if cat_similarities:
            mean_sim = np.mean(cat_similarities)
            category_coactivations[cat] = mean_sim
            print(f"  {cat[:40]:40s}: {mean_sim:.4f}")

print(f"\nTotal active features: {len(categories)}")
print(f"Unique categories: {len(unique_categories)}")
print(f"Total tokens processed: {all_feature_activations.shape[0]}")

# %%
# Save coactivation data for later analysis
print("\nSaving coactivation data...")

coactivation_data = {
    'jaccard_matrix': jaccard_active.tolist(),
    'umap_coordinates': embedding.tolist(),
    'feature_ids': feature_ids,
    'categories': categories,
    'explanations': explanations_list,
    'activation_frequencies': activation_frequencies.tolist(),
    'mean_activations': mean_activations.tolist(),
    'processing_info': {
        'num_rollouts': NUM_ROLLOUTS,
        'num_tokens': all_feature_activations.shape[0],
        'activation_threshold': ACTIVATION_THRESHOLD,
        'umap_n_neighbors': UMAP_N_NEIGHBORS,
        'umap_min_dist': UMAP_MIN_DIST,
        'random_state': RANDOM_STATE
    }
}

with open('feature_coactivation_data.json', 'w') as f:
    json.dump(coactivation_data, f, indent=2)

print("Coactivation data saved to feature_coactivation_data.json")
print("\nDone!")
# %%
