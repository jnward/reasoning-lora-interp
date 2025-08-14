#!/usr/bin/env python3
"""
Create interactive UMAP visualizations of SAE feature embeddings.

This script:
1. Loads embeddings from numpy file
2. Applies UMAP dimensionality reduction
3. Creates interactive Plotly visualizations
4. Saves HTML files for exploration
"""

import json
import numpy as np
import os
import argparse
from typing import List, Dict, Optional, Tuple
import umap
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_embeddings_and_metadata(
    embeddings_path: str,
    metadata_path: Optional[str] = None,
    interpretations_path: Optional[str] = None
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Load embeddings and associated metadata.
    
    Args:
        embeddings_path: Path to numpy file with embeddings
        metadata_path: Optional path to metadata JSON
        interpretations_path: Optional path to original interpretations JSON
        
    Returns:
        Tuple of (embeddings, feature_ids, explanations)
    """
    # Load embeddings
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    feature_ids = list(range(len(embeddings)))
    explanations = [f"Feature {i}" for i in feature_ids]
    
    # Try to load metadata if available
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_ids = metadata.get('feature_ids', feature_ids)
        explanations = metadata.get('explanations', explanations)
        print(f"Loaded metadata from {metadata_path}")
    
    # Or load from original interpretations file
    elif interpretations_path and os.path.exists(interpretations_path):
        with open(interpretations_path, 'r') as f:
            data = json.load(f)
        interps = data.get('explanations', [])
        if interps:
            feature_ids = [item['feature_id'] for item in interps]
            explanations = [item['explanation'] for item in interps]
            print(f"Loaded interpretations from {interpretations_path}")
    
    return embeddings, feature_ids, explanations


def compute_umap_projection(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42,
    scale_data: bool = True
) -> np.ndarray:
    """
    Compute UMAP projection of embeddings.
    
    Args:
        embeddings: Input embeddings array
        n_components: Number of dimensions to reduce to (2 or 3)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric to use
        random_state: Random seed for reproducibility
        scale_data: Whether to standardize data before UMAP
        
    Returns:
        UMAP projection array
    """
    print(f"\nComputing {n_components}D UMAP projection...")
    print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    
    # Optionally scale the data
    if scale_data:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        print("Data standardized (zero mean, unit variance)")
    else:
        embeddings_scaled = embeddings
    
    # Create and fit UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    
    projection = reducer.fit_transform(embeddings_scaled)
    print(f"UMAP projection complete. Shape: {projection.shape}")
    
    return projection


def create_2d_visualization(
    projection: np.ndarray,
    feature_ids: List[int],
    explanations: List[str],
    output_path: str = "umap_2d_plot.html",
    color_by: str = "feature_id",
    title: str = "SAE Feature Embeddings - UMAP 2D Projection"
) -> None:
    """
    Create interactive 2D UMAP visualization.
    
    Args:
        projection: 2D UMAP projection
        feature_ids: List of feature IDs
        explanations: List of feature explanations
        output_path: Path to save HTML file
        color_by: How to color points ("feature_id", "explanation_length", etc.)
        title: Plot title
    """
    print(f"\nCreating 2D visualization...")
    
    # Prepare color values
    if color_by == "feature_id":
        colors = feature_ids
        colorbar_title = "Feature ID"
    elif color_by == "explanation_length":
        colors = [len(exp) for exp in explanations]
        colorbar_title = "Explanation Length"
    else:
        colors = feature_ids
        colorbar_title = "Feature ID"
    
    # Create hover text with wrapped explanations
    hover_texts = []
    for fid, exp in zip(feature_ids, explanations):
        # Wrap long explanations
        if len(exp) > 60:
            wrapped = exp[:60] + "..."
        else:
            wrapped = exp
        hover_texts.append(f"Feature {fid}: {wrapped}")
    
    # Create the scatter plot
    fig = go.Figure(data=[
        go.Scatter(
            x=projection[:, 0],
            y=projection[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                line=dict(width=0.5, color='white')
            ),
            text=hover_texts,
            hovertemplate='%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=1000,
        height=800,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Saved 2D visualization to {output_path}")


def create_3d_visualization(
    projection: np.ndarray,
    feature_ids: List[int],
    explanations: List[str],
    output_path: str = "umap_3d_plot.html",
    color_by: str = "feature_id",
    title: str = "SAE Feature Embeddings - UMAP 3D Projection"
) -> None:
    """
    Create interactive 3D UMAP visualization.
    
    Args:
        projection: 3D UMAP projection
        feature_ids: List of feature IDs
        explanations: List of feature explanations
        output_path: Path to save HTML file
        color_by: How to color points
        title: Plot title
    """
    print(f"\nCreating 3D visualization...")
    
    # Prepare color values
    if color_by == "feature_id":
        colors = feature_ids
        colorbar_title = "Feature ID"
    elif color_by == "explanation_length":
        colors = [len(exp) for exp in explanations]
        colorbar_title = "Explanation Length"
    else:
        colors = feature_ids
        colorbar_title = "Feature ID"
    
    # Create hover text
    hover_texts = []
    for fid, exp in zip(feature_ids, explanations):
        if len(exp) > 60:
            wrapped = exp[:60] + "..."
        else:
            wrapped = exp
        hover_texts.append(f"Feature {fid}: {wrapped}")
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=projection[:, 0],
            y=projection[:, 1],
            z=projection[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                line=dict(width=0.5, color='white')
            ),
            text=hover_texts,
            hovertemplate='%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3"
        ),
        width=1000,
        height=800,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Save to HTML
    fig.write_html(output_path)
    print(f"Saved 3D visualization to {output_path}")


def save_projections(
    projection_2d: np.ndarray,
    projection_3d: np.ndarray,
    feature_ids: List[int],
    explanations: List[str],
    output_prefix: str = "umap_projections"
) -> None:
    """
    Save UMAP projections and metadata for later use.
    
    Args:
        projection_2d: 2D UMAP projection
        projection_3d: 3D UMAP projection
        feature_ids: List of feature IDs
        explanations: List of explanations
        output_prefix: Prefix for output files
    """
    # Save projections as numpy arrays
    np.save(f"{output_prefix}_2d.npy", projection_2d)
    np.save(f"{output_prefix}_3d.npy", projection_3d)
    
    # Save metadata
    metadata = {
        "num_features": len(feature_ids),
        "feature_ids": feature_ids,
        "explanations": explanations,
        "projection_2d_shape": projection_2d.shape,
        "projection_3d_shape": projection_3d.shape
    }
    
    with open(f"{output_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved projections and metadata with prefix: {output_prefix}")


def main():
    parser = argparse.ArgumentParser(description="Create UMAP visualizations of feature embeddings")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="feature_embeddings.npy",
        help="Path to numpy file with embeddings"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional path to metadata JSON file"
    )
    parser.add_argument(
        "--interpretations",
        type=str,
        default="all_interpretations_o3.json",
        help="Path to original interpretations JSON"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="umap",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric for UMAP"
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Don't standardize data before UMAP"
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="feature_id",
        choices=["feature_id", "explanation_length"],
        help="How to color points in visualization"
    )
    parser.add_argument(
        "--save-projections",
        action="store_true",
        help="Save UMAP projections for later use"
    )
    
    args = parser.parse_args()
    
    # Load embeddings and metadata
    if not args.metadata:
        # Try to find metadata file automatically
        potential_metadata = args.embeddings.replace('.npy', '_metadata.json')
        if os.path.exists(potential_metadata):
            args.metadata = potential_metadata
    
    embeddings, feature_ids, explanations = load_embeddings_and_metadata(
        args.embeddings,
        args.metadata,
        args.interpretations
    )
    
    # Print summary
    print(f"\nData summary:")
    print(f"  Number of features: {len(feature_ids)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Sample explanations:")
    for i in range(min(5, len(explanations))):
        print(f"    Feature {feature_ids[i]}: {explanations[i]}")
    
    # Compute 2D UMAP projection
    projection_2d = compute_umap_projection(
        embeddings,
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        scale_data=not args.no_scale
    )
    
    # Create 2D visualization
    create_2d_visualization(
        projection_2d,
        feature_ids,
        explanations,
        output_path=f"{args.output_prefix}_2d_plot.html",
        color_by=args.color_by,
        title="SAE Feature Embeddings - UMAP 2D Projection"
    )
    
    # Compute 3D UMAP projection
    projection_3d = compute_umap_projection(
        embeddings,
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        scale_data=not args.no_scale
    )
    
    # Create 3D visualization
    create_3d_visualization(
        projection_3d,
        feature_ids,
        explanations,
        output_path=f"{args.output_prefix}_3d_plot.html",
        color_by=args.color_by,
        title="SAE Feature Embeddings - UMAP 3D Projection"
    )
    
    # Optionally save projections
    if args.save_projections:
        save_projections(
            projection_2d,
            projection_3d,
            feature_ids,
            explanations,
            output_prefix=f"{args.output_prefix}_projections"
        )
    
    # Print final summary
    print("\n" + "="*50)
    print("Visualization complete!")
    print(f"2D plot: {args.output_prefix}_2d_plot.html")
    print(f"3D plot: {args.output_prefix}_3d_plot.html")
    if args.save_projections:
        print(f"Projections saved: {args.output_prefix}_projections_*.npy")
    print("\nOpen the HTML files in a web browser to explore the visualizations.")
    print("Hover over points to see feature IDs and explanations.")
    print("="*50)


if __name__ == "__main__":
    main()