# %%
"""
Feature Clustering Visualization using Coactivation Statistics - GPU Optimized Version

This notebook computes coactivation patterns from SAE features 
and visualizes them using UMAP with category coloring.
Optimized version using PyTorch GPU operations for massive speedup.
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
from scipy.stats import pearsonr, gaussian_kde
import sys
from tqdm import tqdm
import os
import pickle
sys.path.append('/workspace/reasoning_interp/sae_interp')
from batch_topk_sae import BatchTopKSAE

print("Libraries imported successfully")

# %%
# Configuration
SAE_PATH = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
ACTIVATIONS_DIR = "/workspace/reasoning_interp/2_lora_activation_interp/activations_all_adapters"
CATEGORIES_PATH = "/workspace/reasoning_interp/sae_interp/autointerp/categorized_features.json"
OUTPUT_HTML = "feature_clusters_coactivation.html"

# Processing parameters
NUM_ROLLOUTS = 1000  # Process all rollouts
BATCH_SIZE = 512    # Batch size for SAE processing
ACTIVATION_THRESHOLD = 0.01  # Threshold for considering a feature "active"
MIN_ACTIVATIONS = 16  # Minimum number of activations required to include feature

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
# Check for cached coactivation statistics
CACHE_FILE = "coactivation_cache.pt"
cache_valid = False

if os.path.exists(CACHE_FILE):
    print("Found cache file, checking validity...")
    cache = torch.load(CACHE_FILE, map_location=device)
    
    # Check if cache parameters match current settings
    if (cache.get('num_rollouts') == NUM_ROLLOUTS and 
        cache.get('activation_threshold') == ACTIVATION_THRESHOLD and
        cache.get('dict_size') == dict_size and
        cache.get('activations_dir') == str(ACTIVATIONS_DIR)):
        
        print("Cache is valid, loading statistics...")
        coactivation_accumulator = cache['coactivation_matrix']
        activation_count_accumulator = cache['activation_counts']
        activation_sum_accumulator = cache['activation_sums']
        total_tokens = cache['total_tokens']
        cache_valid = True
        print(f"Loaded cached statistics for {total_tokens} tokens")
    else:
        print("Cache parameters don't match, will recompute...")

# %%
# Process activations through SAE to get feature activations (if not cached)
if not cache_valid:
    print("Processing activations to get SAE features...")
    
    # We'll compute coactivation statistics incrementally to avoid memory issues
    # Initialize accumulators for statistics we need
    n_features = dict_size
    coactivation_accumulator = torch.zeros((n_features, n_features), dtype=torch.float32, device=device)
    activation_count_accumulator = torch.zeros(n_features, dtype=torch.float32, device=device)
    total_tokens = 0
    
    # For storing mean activations and other stats
    activation_sum_accumulator = torch.zeros(n_features, dtype=torch.float32, device=device)

    # Process rollouts
    for rollout_idx in tqdm(range(NUM_ROLLOUTS), desc="Processing rollouts", disable=cache_valid):
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
        rollout_features_list = []
        for i in range(0, num_tokens, BATCH_SIZE):
            batch = activations_flat[i:i+BATCH_SIZE]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # Get sparse feature activations
                sparse_features = sae.encode(batch_tensor)
                rollout_features_list.append(sparse_features)
        
        # Concatenate this rollout's features
        if rollout_features_list:
            rollout_features = torch.cat(rollout_features_list, dim=0)
            
            # Update statistics incrementally
            # Binarize for coactivation
            binary_features = (rollout_features > ACTIVATION_THRESHOLD).float()
            
            # Update coactivation matrix (accumulate outer products)
            coactivation_accumulator += torch.matmul(binary_features.T, binary_features)
            
            # Update activation counts
            activation_count_accumulator += binary_features.sum(dim=0)
            
            # Update sum for mean calculation
            activation_sum_accumulator += rollout_features.sum(dim=0)
            
            total_tokens += rollout_features.shape[0]
            
            # Clear GPU memory after each rollout
            del rollout_features, binary_features
            torch.cuda.empty_cache()
    
    print(f"Processed {total_tokens} total tokens")
    
    # Save cache
    print("Saving coactivation statistics to cache...")
    cache_data = {
        'coactivation_matrix': coactivation_accumulator,
        'activation_counts': activation_count_accumulator,
        'activation_sums': activation_sum_accumulator,
        'total_tokens': total_tokens,
        'num_rollouts': NUM_ROLLOUTS,
        'activation_threshold': ACTIVATION_THRESHOLD,
        'dict_size': dict_size,
        'activations_dir': str(ACTIVATIONS_DIR)
    }
    torch.save(cache_data, CACHE_FILE)
    print(f"Cache saved to {CACHE_FILE}")

# %%
# Compute Jaccard similarity from accumulated statistics
print("Computing Jaccard similarity matrix from accumulated statistics...")

# We already have the coactivation matrix and activation counts from accumulation
coactivation_matrix = coactivation_accumulator
activation_counts = activation_count_accumulator

print("Computing Jaccard similarity matrix (vectorized)...")
# Vectorized Jaccard similarity computation using broadcasting
# Create matrices for pairwise unions
counts_i = activation_counts.unsqueeze(1)  # Shape: (n_features, 1)
counts_j = activation_counts.unsqueeze(0)  # Shape: (1, n_features)

# Union = count_i + count_j - intersection
union_matrix = counts_i + counts_j - coactivation_matrix

# Avoid division by zero
union_matrix = torch.clamp(union_matrix, min=1e-10)

# Jaccard similarity
jaccard_matrix = coactivation_matrix / union_matrix

# Set diagonal to 1
jaccard_matrix.fill_diagonal_(1.0)

# Convert to CPU numpy for further processing
jaccard_matrix_np = jaccard_matrix.cpu().numpy()

print(f"Jaccard similarity matrix shape: {jaccard_matrix_np.shape}")
print(f"Similarity range: [{jaccard_matrix_np.min():.4f}, {jaccard_matrix_np.max():.4f}]")
print(f"Mean similarity (excluding diagonal): {jaccard_matrix_np[~np.eye(dict_size, dtype=bool)].mean():.4f}")

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

# Filter to only active features (those with explanations AND sufficient activations)
# First get activation counts for filtering
total_activations = activation_counts.cpu().numpy()

# Apply both filters: has explanation AND meets minimum activation threshold
active_feature_ids = [
    i for i in range(dict_size) 
    if i in feature_to_explanation and total_activations[i] >= MIN_ACTIVATIONS
]
print(f"Found {len(active_feature_ids)} active features out of {dict_size} total")
print(f"  - Features with explanations: {len([i for i in range(dict_size) if i in feature_to_explanation])}")
print(f"  - After filtering for >= {MIN_ACTIVATIONS} activations: {len(active_feature_ids)}")

# Filter Jaccard matrix to only active features
jaccard_active = jaccard_matrix_np[np.ix_(active_feature_ids, active_feature_ids)]

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

# Get activation statistics for hover text - use accumulated statistics
# Create index tensor for active features
active_indices = torch.tensor(active_feature_ids, device=device)

# Calculate mean activations from accumulated sums
mean_activations_all = (activation_sum_accumulator / total_tokens)
mean_activations = mean_activations_all[active_indices].cpu().numpy()

# Calculate activation frequencies from counts
activation_frequencies_all = (activation_count_accumulator / total_tokens * 100)
activation_frequencies = activation_frequencies_all[active_indices].cpu().numpy()

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

# Use vibrant colors that pop against black background
vibrant_colors = [
    '#FF006E',  # Hot pink
    '#00F5FF',  # Cyan
    '#FFBE0B',  # Gold
    '#FB5607',  # Orange red
    '#8338EC',  # Purple
    '#3A86FF',  # Bright blue
    '#06FFB4',  # Mint green
    '#FF4365',  # Coral
    '#00D9FF',  # Sky blue
    '#FFC600',  # Yellow
    '#C77DFF',  # Lavender
    '#7CFF00',  # Lime
    '#FF006E',  # Magenta
    '#00FFC8',  # Aqua
    '#FFD60A',  # Bright yellow
    '#FF8500',  # Dark orange
    '#B91C8C',  # Deep pink
    '#4CC9F0',  # Light blue
    '#F72585',  # Pink
    '#4361EE',  # Royal blue
]

# Extend with more colors if needed
more_colors = px.colors.qualitative.Light24
vibrant_colors.extend(more_colors)

# Assign colors to categories
color_map = {}
for i, cat in enumerate(unique_categories):
    color_map[cat] = vibrant_colors[i % len(vibrant_colors)]

# %%
# Create interactive Plotly visualization
print("Creating interactive visualization...")

# Create the figure
fig = go.Figure()

# First, add overall density heatmap (as background)
print("Computing overall density heatmap...")

# Create a grid for density evaluation
x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

# Add some padding
padding = 0.1 * max(x_max - x_min, y_max - y_min)
x_min, x_max = x_min - padding, x_max + padding
y_min, y_max = y_min - padding, y_max + padding

# Create grid
grid_size = 150  # Higher resolution for smoother heatmap
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()])

# Compute overall KDE for all points
print(f"Computing KDE for {len(embedding)} points...")
kde = gaussian_kde(embedding.T, bw_method='scott')  # Use Scott's rule for bandwidth

# Evaluate KDE on grid
Z = kde(grid_points).reshape(X_grid.shape)

# Normalize Z for better visualization
Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

# Add density heatmap
fig.add_trace(go.Heatmap(
    x=x_grid,
    y=y_grid,
    z=Z_normalized,
    colorscale=[
        [0, 'rgba(0,0,0,0)'],      # Transparent black for low density
        [0.2, 'rgba(30,0,50,0.3)'], # Deep purple
        [0.4, 'rgba(60,0,100,0.5)'], # Purple
        [0.6, 'rgba(100,0,150,0.7)'], # Bright purple
        [0.8, 'rgba(150,50,200,0.8)'], # Violet
        [1.0, 'rgba(255,150,255,0.9)']  # Bright magenta for highest density
    ],
    showscale=False,
    hoverinfo='skip',
    zsmooth='best'  # Smooth interpolation
))

# Now add scatter points for each category (on top of contours)
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
                size=7,
                opacity=0.85,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            )
        ))

# Update layout with black background
fig.update_layout(
    title={
        'text': 'SAE Feature Clustering with Coactivation Statistics',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24, 'color': 'white'}
    },
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    width=1200,
    height=800,
    hovermode='closest',
    legend=dict(
        title=dict(text='Categories', font=dict(color='white')),
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.01,
        bgcolor='rgba(20, 20, 20, 0.8)',
        bordercolor='#444',
        borderwidth=1,
        font=dict(color='white')
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    annotations=[
        dict(
            text=f"Based on {NUM_ROLLOUTS} rollouts, {total_tokens} total tokens | Min activations: {MIN_ACTIVATIONS}",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor='left', yanchor='bottom',
            font=dict(size=10, color='#888')
        )
    ]
)

# Update axes with subtle grid
fig.update_xaxes(
    showgrid=True, 
    gridwidth=0.5, 
    gridcolor='#333',
    zeroline=True,
    zerolinewidth=0.5,
    zerolinecolor='#444',
    showline=True,
    linewidth=1,
    linecolor='#444',
    title_font=dict(color='white'),
    tickfont=dict(color='#999')
)
fig.update_yaxes(
    showgrid=True, 
    gridwidth=0.5, 
    gridcolor='#333',
    zeroline=True,
    zerolinewidth=0.5,
    zerolinecolor='#444',
    showline=True,
    linewidth=1,
    linecolor='#444',
    title_font=dict(color='white'),
    tickfont=dict(color='#999')
)

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
print(f"Total tokens processed: {total_tokens}")

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
        'num_tokens': total_tokens,
        'activation_threshold': ACTIVATION_THRESHOLD,
        'min_activations': MIN_ACTIVATIONS,
        'umap_n_neighbors': UMAP_N_NEIGHBORS,
        'umap_min_dist': UMAP_MIN_DIST,
        'random_state': RANDOM_STATE
    }
}

with open('feature_coactivation_data.json', 'w') as f:
    json.dump(coactivation_data, f, indent=2)

print("Coactivation data saved to feature_coactivation_data.json")
print("\nOptimized version completed successfully!")
print("Performance improvements:")
print("  - Incremental accumulation to avoid memory issues")
print("  - Process rollouts one at a time with GPU memory clearing")
print("  - Vectorized Jaccard similarity computation")
print("  - All matrix operations on GPU")
print(f"  - Filtered to {len(active_feature_ids)} features with >= {MIN_ACTIVATIONS} activations")
# %%