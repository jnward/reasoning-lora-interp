# %%
"""
UMAP Visualization of SAE Features by Category

This notebook visualizes SAE features using UMAP projections of their explanation embeddings,
colored by semantic categories rather than clustering results.
"""

# %%
# Imports
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.stats import gaussian_kde
import pandas as pd

print("Libraries imported successfully")

# %%
# Configuration
BASE_DIR = Path("/workspace/reasoning_interp/sae_interp/autointerp")
EMBEDDINGS_PATH = BASE_DIR / "feature_embeddings.npy"
UMAP_2D_PATH = BASE_DIR / "umap_projections_2d.npy"
UMAP_3D_PATH = BASE_DIR / "umap_projections_3d.npy"
UMAP_METADATA_PATH = BASE_DIR / "umap_projections_metadata.json"
CATEGORIZED_FEATURES_PATH = BASE_DIR / "coarse_categorized_features.json"
CATEGORIES_PATH = BASE_DIR / "coarse_categories.json"

# Output paths
OUTPUT_2D_HTML = BASE_DIR / "umap_category_visualization_2d.html"
OUTPUT_3D_HTML = BASE_DIR / "umap_category_visualization_3d.html"

# Visualization parameters
USE_BLACK_BACKGROUND = False
POINT_SIZE_2D = 6
POINT_SIZE_3D = 4
POINT_OPACITY = 1.0
ADD_DENSITY_OVERLAY = False  # No density heatmap

print(f"Base directory: {BASE_DIR}")
print(f"Will save outputs to: {OUTPUT_2D_HTML} and {OUTPUT_3D_HTML}")

# %%
# Load all data
print("Loading data...")

# Load UMAP projections
umap_2d = np.load(UMAP_2D_PATH)
umap_3d = np.load(UMAP_3D_PATH)
print(f"UMAP 2D shape: {umap_2d.shape}")
print(f"UMAP 3D shape: {umap_3d.shape}")

# Load UMAP metadata
with open(UMAP_METADATA_PATH, 'r') as f:
    umap_metadata = json.load(f)
print(f"Number of features in UMAP: {umap_metadata['num_features']}")

# Load categorized features
with open(CATEGORIZED_FEATURES_PATH, 'r') as f:
    cat_data = json.load(f)
explanations_data = cat_data['explanations']
print(f"Number of categorized features: {len(explanations_data)}")

# Load category definitions
with open(CATEGORIES_PATH, 'r') as f:
    categories_list = json.load(f)
category_map = {cat['id']: cat['label'] for cat in categories_list}
print(f"Number of categories: {len(category_map)}")

# %%
# Prepare data for visualization
print("Preparing visualization data...")

# Create lookup dictionaries from categorized features
feature_to_category = {}
feature_to_explanation = {}
feature_to_category_label = {}

for item in explanations_data:
    fid = item['feature_id']
    cat_id = item.get('category_id', 'uncategorized')
    feature_to_category[fid] = cat_id
    feature_to_explanation[fid] = item.get('explanation', 'No explanation')
    feature_to_category_label[fid] = category_map.get(cat_id, cat_id)

# Get feature IDs from UMAP metadata (these are in order)
feature_ids = umap_metadata['feature_ids']

# Create arrays aligned with UMAP projections
categories = []
category_labels = []
explanations = []

for fid in feature_ids:
    categories.append(feature_to_category.get(fid, 'uncategorized'))
    category_labels.append(feature_to_category_label.get(fid, 'Uncategorized'))
    explanations.append(feature_to_explanation.get(fid, f'Feature {fid}'))

print(f"Aligned {len(feature_ids)} features with categories")

# Get unique categories and their counts
unique_categories = list(set(categories))
category_counts = pd.Series(categories).value_counts()
print(f"\nCategory distribution:")
for cat_id in category_counts.index[:10]:  # Show top 10
    label = category_map.get(cat_id, cat_id)
    count = category_counts[cat_id]
    print(f"  {label}: {count} features")

# %%
# Define systematic HSL color palette with equal spacing
print("Setting up systematically spaced color palette...")

import colorsys

# Sort categories by frequency to ensure consistent ordering
category_order = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
ordered_categories = [cat_id for cat_id, _ in category_order]

# Generate colors with equal hue spacing, constant saturation and lightness
num_categories = len(ordered_categories)
# Adjust for white background - use high saturation and medium lightness
SATURATION = 0.9   # High saturation for vibrant colors
LIGHTNESS = 0.45   # Medium lightness for good contrast on white

# Create evenly spaced hues
hues = [i / num_categories for i in range(num_categories)]

# Convert HSL to RGB and then to hex
distinct_colors_systematic = []
for hue in hues:
    # Convert HSL to RGB
    # Note: colorsys uses HLS order (Hue, Lightness, Saturation)
    rgb = colorsys.hls_to_rgb(hue, LIGHTNESS, SATURATION)
    # Convert to hex
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    distinct_colors_systematic.append(hex_color)

# Create color mapping based on ordered categories
color_map = {}
for i, cat_id in enumerate(ordered_categories):
    color_map[cat_id] = distinct_colors_systematic[i]
    cat_label = category_map.get(cat_id, cat_id)
    print(f"  {cat_label}: {distinct_colors_systematic[i]}")

# For the visualization loop, we'll use the ordered categories
unique_categories = ordered_categories

print(f"Assigned colors to {len(unique_categories)} categories")

# %%
# Create 2D visualization
print("Creating 2D visualization...")

# Create hover text
hover_texts = []
for fid, exp, cat_label in zip(feature_ids, explanations, category_labels):
    # Wrap long explanations
    if len(exp) > 100:
        exp_wrapped = exp[:100] + "..."
    else:
        exp_wrapped = exp
    hover_texts.append(
        f"Feature {fid}<br>"
        f"Category: {cat_label}<br>"
        f"Explanation: {exp_wrapped}"
    )

# Create the figure
fig_2d = go.Figure()

# Optional: Add density overlay
if ADD_DENSITY_OVERLAY and len(umap_2d) > 100:
    print("Computing density overlay...")
    
    # Create grid for density
    x_min, x_max = umap_2d[:, 0].min(), umap_2d[:, 0].max()
    y_min, y_max = umap_2d[:, 1].min(), umap_2d[:, 1].max()
    
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding
    
    grid_size = 150
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    
    # Compute KDE
    kde = gaussian_kde(umap_2d.T, bw_method='scott')
    Z = kde(grid_points).reshape(X_grid.shape)
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
    
    # Add heatmap
    fig_2d.add_trace(go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=Z_normalized,
        colorscale=[
            [0, 'rgba(0,0,0,0)'],
            [0.2, 'rgba(30,0,50,0.3)'],
            [0.4, 'rgba(60,0,100,0.5)'],
            [0.6, 'rgba(100,0,150,0.7)'],
            [0.8, 'rgba(150,50,200,0.8)'],
            [1.0, 'rgba(255,150,255,0.9)']
        ],
        showscale=False,
        hoverinfo='skip',
        zsmooth='best'
    ))

# Add scatter points for each category
for cat_id in unique_categories:
    # Get indices for this category
    indices = [i for i, cat in enumerate(categories) if cat == cat_id]
    
    if indices:
        cat_label = category_map.get(cat_id, cat_id)
        
        fig_2d.add_trace(go.Scatter(
            x=umap_2d[indices, 0],
            y=umap_2d[indices, 1],
            mode='markers',
            name=cat_label,
            text=[hover_texts[i] for i in indices],
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                color=color_map[cat_id],
                size=POINT_SIZE_2D,
                opacity=POINT_OPACITY,
                line=dict(width=0.5, color='rgba(0,0,0,0.2)')  # Dark border for white background
            )
        ))

# Update layout
if USE_BLACK_BACKGROUND:
    fig_2d.update_layout(
        title=dict(
            text='SAE Features - UMAP of Explanation Embeddings (Colored by Category)',
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='white')
        ),
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        width=1400,
        height=900,
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            title=dict(text='Categories', font=dict(color='white', size=14)),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(20, 20, 20, 0.8)',
            bordercolor='#444',
            borderwidth=1,
            font=dict(color='white', size=11)
        )
    )
    
    # Update axes
    fig_2d.update_xaxes(
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
    fig_2d.update_yaxes(
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
else:
    fig_2d.update_layout(
        title='SAE Features - UMAP of Explanation Embeddings (Colored by Category)',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        width=1400,
        height=900,
        hovermode='closest',
        template='plotly_white'
    )

print("2D visualization created")

# %%
# Save 2D visualization
fig_2d.write_html(
    str(OUTPUT_2D_HTML),
    config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'umap_categories_2d',
            'height': 900,
            'width': 1400,
            'scale': 2
        }
    }
)
print(f"Saved 2D visualization to {OUTPUT_2D_HTML}")

# %%
# Create 3D visualization
print("Creating 3D visualization...")

# Create the 3D figure
fig_3d = go.Figure()

# Add scatter points for each category
for cat_id in unique_categories:
    # Get indices for this category
    indices = [i for i, cat in enumerate(categories) if cat == cat_id]
    
    if indices:
        cat_label = category_map.get(cat_id, cat_id)
        
        fig_3d.add_trace(go.Scatter3d(
            x=umap_3d[indices, 0],
            y=umap_3d[indices, 1],
            z=umap_3d[indices, 2],
            mode='markers',
            name=cat_label,
            text=[hover_texts[i] for i in indices],
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                color=color_map[cat_id],
                size=POINT_SIZE_3D,
                opacity=POINT_OPACITY,
                line=dict(width=0.5, color='rgba(0,0,0,0.2)')  # Dark border for white background
            )
        ))

# Update layout
if USE_BLACK_BACKGROUND:
    fig_3d.update_layout(
        title=dict(
            text='SAE Features - 3D UMAP of Explanation Embeddings (Colored by Category)',
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='UMAP 1', font=dict(color='white')),
                backgroundcolor='black',
                gridcolor='#333',
                showbackground=True,
                zerolinecolor='#444',
                tickfont=dict(color='#999')
            ),
            yaxis=dict(
                title=dict(text='UMAP 2', font=dict(color='white')),
                backgroundcolor='black',
                gridcolor='#333',
                showbackground=True,
                zerolinecolor='#444',
                tickfont=dict(color='#999')
            ),
            zaxis=dict(
                title=dict(text='UMAP 3', font=dict(color='white')),
                backgroundcolor='black',
                gridcolor='#333',
                showbackground=True,
                zerolinecolor='#444',
                tickfont=dict(color='#999')
            ),
            bgcolor='black'
        ),
        width=1400,
        height=900,
        hovermode='closest',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            title=dict(text='Categories', font=dict(color='white', size=14)),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(20, 20, 20, 0.8)',
            bordercolor='#444',
            borderwidth=1,
            font=dict(color='white', size=11)
        )
    )
else:
    fig_3d.update_layout(
        title='SAE Features - 3D UMAP of Explanation Embeddings (Colored by Category)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1400,
        height=900,
        hovermode='closest',
        template='plotly_white'
    )

print("3D visualization created")

# %%
# Save 3D visualization
fig_3d.write_html(
    str(OUTPUT_3D_HTML),
    config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'umap_categories_3d',
            'height': 900,
            'width': 1400,
            'scale': 2
        }
    }
)
print(f"Saved 3D visualization to {OUTPUT_3D_HTML}")

# %%
# Display figures in notebook (if running interactively)
fig_2d.show()

# %%
fig_3d.show()

# %%
# Print summary statistics
print("\n" + "="*60)
print("Visualization Summary")
print("="*60)
print(f"Total features visualized: {len(feature_ids)}")
print(f"Number of categories: {len(unique_categories)}")
print(f"\nTop 5 categories by feature count:")

for i, (cat_id, count) in enumerate(category_counts.items()[:5]):
    label = category_map.get(cat_id, cat_id)
    percentage = (count / len(feature_ids)) * 100
    print(f"  {i+1}. {label}: {count} features ({percentage:.1f}%)")

print(f"\nOutput files:")
print(f"  2D: {OUTPUT_2D_HTML}")
print(f"  3D: {OUTPUT_3D_HTML}")
print("\nOpen these HTML files in a browser to explore the interactive visualizations.")
print("="*60)

# %%
# Optional: Create a combined dashboard with both views
print("\nCreating combined dashboard...")

from plotly.subplots import make_subplots

# Create subplot with 2D view
fig_combined = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'scatter'}]],
    subplot_titles=('UMAP 2D Projection by Category',)
)

# Add all category traces to the combined figure
for cat_id in unique_categories:
    indices = [i for i, cat in enumerate(categories) if cat == cat_id]
    if indices:
        cat_label = category_map.get(cat_id, cat_id)
        fig_combined.add_trace(
            go.Scatter(
                x=umap_2d[indices, 0],
                y=umap_2d[indices, 1],
                mode='markers',
                name=cat_label,
                text=[hover_texts[i] for i in indices],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    color=color_map[cat_id],
                    size=POINT_SIZE_2D-1,
                    opacity=POINT_OPACITY,
                    line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                ),
                showlegend=True
            ),
            row=1, col=1
        )

# Update layout for combined view
fig_combined.update_layout(
    title=dict(
        text='SAE Features UMAP - Semantic Categories',
        x=0.5,
        xanchor='center',
        font=dict(size=20, color='white' if USE_BLACK_BACKGROUND else 'black')
    ),
    height=800,
    width=1200,
    showlegend=True,
    hovermode='closest',
    plot_bgcolor='black' if USE_BLACK_BACKGROUND else 'white',
    paper_bgcolor='black' if USE_BLACK_BACKGROUND else 'white',
    font=dict(color='white' if USE_BLACK_BACKGROUND else 'black'),
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.02,
        bgcolor='rgba(20, 20, 20, 0.8)' if USE_BLACK_BACKGROUND else 'rgba(255, 255, 255, 0.9)',
        bordercolor='#444' if USE_BLACK_BACKGROUND else '#ccc',
        borderwidth=1,
        font=dict(size=10)
    )
)

# Save combined dashboard
OUTPUT_COMBINED = BASE_DIR / "umap_category_dashboard.html"
fig_combined.write_html(
    str(OUTPUT_COMBINED),
    config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'umap_category_dashboard',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
)
print(f"Saved combined dashboard to {OUTPUT_COMBINED}")

print("\nAll visualizations complete!")
# %%