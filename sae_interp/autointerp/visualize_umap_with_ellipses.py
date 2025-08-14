# %%
"""
UMAP Visualization with Confidence Ellipses

This notebook creates a 2D UMAP visualization with ellipses showing the main mass
of each category's distribution (e.g., 95% confidence ellipses).
"""

# %%
# Imports
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
import colorsys
import pandas as pd

print("Libraries imported successfully")

# %%
# Configuration
BASE_DIR = Path("/workspace/reasoning_interp/sae_interp/autointerp")
UMAP_2D_PATH = BASE_DIR / "umap_projections_2d.npy"
UMAP_METADATA_PATH = BASE_DIR / "umap_projections_metadata.json"
CATEGORIZED_FEATURES_PATH = BASE_DIR / "coarse_categorized_features.json"
CATEGORIES_PATH = BASE_DIR / "coarse_categories.json"

# Output path
OUTPUT_HTML = BASE_DIR / "umap_with_ellipses_2d.html"

# Visualization parameters
POINT_SIZE = 6
POINT_OPACITY = 0.8
ELLIPSE_CONFIDENCE = 0.50  # 50% confidence ellipse (captures core mass only)
MIN_POINTS_FOR_ELLIPSE = 10  # Need at least this many points to draw an ellipse

print(f"Will save output to: {OUTPUT_HTML}")

# %%
# Load all data
print("Loading data...")

# Load UMAP projections
umap_2d = np.load(UMAP_2D_PATH)
print(f"UMAP 2D shape: {umap_2d.shape}")

# Load UMAP metadata
with open(UMAP_METADATA_PATH, 'r') as f:
    umap_metadata = json.load(f)

# Load categorized features
with open(CATEGORIZED_FEATURES_PATH, 'r') as f:
    cat_data = json.load(f)
explanations_data = cat_data['explanations']

# Load category definitions
with open(CATEGORIES_PATH, 'r') as f:
    categories_list = json.load(f)
category_map = {cat['id']: cat['label'] for cat in categories_list}

# %%
# Prepare data
print("Preparing visualization data...")

# Create lookup dictionaries
feature_to_category = {}
feature_to_explanation = {}
feature_to_category_label = {}

for item in explanations_data:
    fid = item['feature_id']
    cat_id = item.get('category_id', 'uncategorized')
    feature_to_category[fid] = cat_id
    feature_to_explanation[fid] = item.get('explanation', 'No explanation')
    feature_to_category_label[fid] = category_map.get(cat_id, cat_id)

# Get feature IDs from UMAP metadata
feature_ids = umap_metadata['feature_ids']

# Create arrays aligned with UMAP projections
categories = []
category_labels = []
explanations = []

for fid in feature_ids:
    categories.append(feature_to_category.get(fid, 'uncategorized'))
    category_labels.append(feature_to_category_label.get(fid, 'Uncategorized'))
    explanations.append(feature_to_explanation.get(fid, f'Feature {fid}'))

# Get category counts and order
category_counts = pd.Series(categories).value_counts()
category_order = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
ordered_categories = [cat_id for cat_id, _ in category_order]

print(f"Found {len(ordered_categories)} categories")

# %%
# Create systematic color palette
print("Setting up color palette...")

num_categories = len(ordered_categories)
SATURATION = 0.9
LIGHTNESS = 0.45

# Create evenly spaced hues
hues = [i / num_categories for i in range(num_categories)]

# Convert HSL to RGB and then to hex
color_map = {}
for i, cat_id in enumerate(ordered_categories):
    hue = hues[i]
    rgb = colorsys.hls_to_rgb(hue, LIGHTNESS, SATURATION)
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    color_map[cat_id] = hex_color

# %%
def calculate_confidence_ellipse(points, confidence=0.95, n_points=100):
    """
    Calculate confidence ellipse parameters for 2D points.
    
    Returns x and y coordinates for plotting the ellipse boundary.
    """
    if len(points) < 3:  # Need at least 3 points to define an ellipse
        return None, None
    
    # Calculate mean and covariance
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Calculate angle of rotation
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Calculate width and height (semi-axes lengths)
    # Chi-square value for confidence level
    chi2_val = stats.chi2.ppf(confidence, df=2)
    width = 2 * np.sqrt(chi2_val * eigenvalues[0])
    height = 2 * np.sqrt(chi2_val * eigenvalues[1])
    
    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse_x = width/2 * np.cos(theta)
    ellipse_y = height/2 * np.sin(theta)
    
    # Rotate ellipse
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    ellipse_points = np.dot(rotation_matrix, np.array([ellipse_x, ellipse_y]))
    
    # Translate to mean
    ellipse_x = ellipse_points[0, :] + mean[0]
    ellipse_y = ellipse_points[1, :] + mean[1]
    
    return ellipse_x, ellipse_y

# %%
# Create the figure
print("Creating visualization with ellipses...")

fig = go.Figure()

# First, add ellipses for each category (as background)
for cat_id in ordered_categories:
    # Get indices for this category
    indices = [i for i, cat in enumerate(categories) if cat == cat_id]
    
    if len(indices) >= MIN_POINTS_FOR_ELLIPSE:
        # Get points for this category
        cat_points = umap_2d[indices]
        
        # Calculate confidence ellipse
        ellipse_x, ellipse_y = calculate_confidence_ellipse(
            cat_points, 
            confidence=ELLIPSE_CONFIDENCE
        )
        
        if ellipse_x is not None:
            cat_label = category_map.get(cat_id, cat_id)
            
            # Add ellipse as a filled shape
            fig.add_trace(go.Scatter(
                x=ellipse_x,
                y=ellipse_y,
                mode='lines',
                fill='toself',
                fillcolor=color_map[cat_id],
                opacity=0.15,  # Very transparent fill
                line=dict(
                    color=color_map[cat_id],
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip',
                name=f'{cat_label} ellipse'
            ))

# Then add scatter points for each category (on top)
for cat_id in ordered_categories:
    # Get indices for this category
    indices = [i for i, cat in enumerate(categories) if cat == cat_id]
    
    if indices:
        cat_label = category_map.get(cat_id, cat_id)
        cat_count = len(indices)
        
        # Create hover text
        hover_texts = []
        for idx in indices:
            fid = feature_ids[idx]
            exp = explanations[idx]
            if len(exp) > 100:
                exp = exp[:100] + "..."
            hover_texts.append(
                f"Feature {fid}<br>"
                f"Category: {cat_label}<br>"
                f"Explanation: {exp}"
            )
        
        fig.add_trace(go.Scatter(
            x=umap_2d[indices, 0],
            y=umap_2d[indices, 1],
            mode='markers',
            name=f'{cat_label} ({cat_count})',
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                color=color_map[cat_id],
                size=POINT_SIZE,
                opacity=POINT_OPACITY,
                line=dict(width=0.5, color='rgba(0,0,0,0.2)')
            )
        ))

# Update layout
fig.update_layout(
    title={
        'text': f'SAE Features UMAP with {int(ELLIPSE_CONFIDENCE*100)}% Confidence Ellipses',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    xaxis_title='UMAP 1',
    yaxis_title='UMAP 2',
    width=1400,
    height=900,
    hovermode='closest',
    template='plotly_white',
    legend=dict(
        title=dict(text='Categories (count)', font=dict(size=12)),
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.01,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(size=10)
    ),
    annotations=[
        dict(
            text=f"Ellipses show {int(ELLIPSE_CONFIDENCE*100)}% confidence regions (min {MIN_POINTS_FOR_ELLIPSE} points)",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=-0.05,
            xanchor='left', yanchor='top',
            font=dict(size=10, color='gray')
        )
    ]
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#e0e0e0')
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#e0e0e0')

print("Visualization created")

# %%
# Save visualization
fig.write_html(
    str(OUTPUT_HTML),
    config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'umap_with_ellipses',
            'height': 900,
            'width': 1400,
            'scale': 2
        }
    }
)
print(f"Saved visualization to {OUTPUT_HTML}")

# %%
# Display the figure
fig.show()

# %%
# Print statistics about the ellipses
print("\n" + "="*60)
print("Ellipse Statistics")
print("="*60)

for cat_id in ordered_categories:
    indices = [i for i, cat in enumerate(categories) if cat == cat_id]
    cat_label = category_map.get(cat_id, cat_id)
    
    if len(indices) >= MIN_POINTS_FOR_ELLIPSE:
        cat_points = umap_2d[indices]
        
        # Calculate statistics
        mean = np.mean(cat_points, axis=0)
        std = np.std(cat_points, axis=0)
        
        print(f"\n{cat_label}:")
        print(f"  Points: {len(indices)}")
        print(f"  Center: ({mean[0]:.2f}, {mean[1]:.2f})")
        print(f"  Std Dev: (±{std[0]:.2f}, ±{std[1]:.2f})")
        
        # Calculate how many points are within the ellipse
        ellipse_x, ellipse_y = calculate_confidence_ellipse(
            cat_points, 
            confidence=ELLIPSE_CONFIDENCE
        )
        if ellipse_x is not None:
            # Approximate: check how many points would be in confidence region
            cov = np.cov(cat_points.T)
            inv_cov = np.linalg.inv(cov)
            mean_point = mean
            
            distances = []
            for point in cat_points:
                diff = point - mean_point
                mahalanobis = np.sqrt(diff @ inv_cov @ diff)
                distances.append(mahalanobis)
            
            chi2_val = np.sqrt(stats.chi2.ppf(ELLIPSE_CONFIDENCE, df=2))
            points_in_ellipse = sum(d <= chi2_val for d in distances)
            
            print(f"  Points in {int(ELLIPSE_CONFIDENCE*100)}% ellipse: {points_in_ellipse}/{len(indices)} "
                  f"({100*points_in_ellipse/len(indices):.1f}%)")
    else:
        print(f"\n{cat_label}:")
        print(f"  Points: {len(indices)} (too few for ellipse)")

print("\n" + "="*60)
print(f"Total features: {len(feature_ids)}")
print(f"Categories with ellipses: {sum(1 for cat in ordered_categories if sum(1 for c in categories if c == cat) >= MIN_POINTS_FOR_ELLIPSE)}/{len(ordered_categories)}")
print("="*60)
# %%