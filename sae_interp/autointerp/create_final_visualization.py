#!/usr/bin/env python3
"""
Create final annotated visualization of SAE feature clusters.

Combines UMAP projections with GPT-4o cluster annotations for
an interactive, interpretable visualization.
"""

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
from typing import Dict, List
import random


def load_all_data(projections_path: str, interpretations_path: str, 
                  clustering_path: str, annotations_path: str):
    """Load all necessary data for visualization."""
    
    # Load UMAP projections
    projections = np.load(projections_path)
    
    # Load original interpretations
    with open(interpretations_path, 'r') as f:
        interp_data = json.load(f)
    
    # Load clustering results
    with open(clustering_path, 'r') as f:
        cluster_data = json.load(f)
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    return projections, interp_data, cluster_data, annotations


def create_cluster_color_map(annotations: Dict) -> Dict:
    """Create a consistent color mapping for clusters."""
    random.seed(42)
    
    # Extract clusters sorted by size
    clusters = sorted(annotations['clusters'], key=lambda x: x['size'], reverse=True)
    
    # Define a color palette - use distinct colors for top clusters
    top_colors = [
        '#e74c3c',  # Red
        '#3498db',  # Blue  
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#e67e22',  # Carrot
        '#34495e',  # Dark gray-blue
        '#16a085',  # Green sea
        '#d35400',  # Pumpkin
        '#8e44ad',  # Wisteria
        '#c0392b',  # Pomegranate
        '#27ae60',  # Nephritis
        '#2980b9',  # Belize hole
        '#f1c40f',  # Sunflower
    ]
    
    color_map = {}
    for i, cluster in enumerate(clusters):
        if i < len(top_colors):
            color_map[cluster['id']] = top_colors[i]
        else:
            # Generate random color for remaining clusters
            color_map[cluster['id']] = '#{:06x}'.format(random.randint(0x333333, 0xCCCCCC))
    
    # Noise points get gray
    color_map[-1] = '#808080'
    
    return color_map


def create_main_visualization(projections: np.ndarray, interp_data: Dict,
                             cluster_data: Dict, annotations: Dict,
                             output_path: str):
    """Create the main annotated cluster visualization."""
    
    # Extract data
    feature_ids = cluster_data['feature_ids']
    graph_labels = cluster_data['graph']['labels']
    
    # Create feature ID to explanation mapping
    explanation_map = {e['feature_id']: e['explanation'] for e in interp_data['explanations']}
    
    # Get cluster summaries
    cluster_summaries = {c['id']: c['summary'] for c in annotations['clusters']}
    
    # Create color map
    color_map = create_cluster_color_map(annotations)
    
    # Prepare data for plotting
    colors = [color_map[label] for label in graph_labels]
    
    # Create hover text with cluster info
    hover_texts = []
    for fid, label in zip(feature_ids, graph_labels):
        exp = explanation_map.get(fid, "")
        if len(exp) > 60:
            exp = exp[:60] + "..."
        
        cluster_summary = cluster_summaries.get(label, "Unknown cluster")
        hover_text = (
            f"<b>Feature {fid}</b><br>"
            f"{exp}<br><br>"
            f"<b>Cluster {label}:</b><br>"
            f"{cluster_summary}"
        )
        hover_texts.append(hover_text)
    
    # Create the main scatter plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=projections[:, 0],
        y=projections[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            line=dict(width=0.5, color='white')
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    # Add cluster labels for major clusters
    clusters_by_size = sorted(annotations['clusters'], key=lambda x: x['size'], reverse=True)
    
    for cluster in clusters_by_size[:15]:  # Label top 15 clusters
        cluster_id = cluster['id']
        cluster_points = projections[np.array(graph_labels) == cluster_id]
        
        if len(cluster_points) > 0:
            # Calculate centroid
            centroid = cluster_points.mean(axis=0)
            
            # Create short label
            summary = cluster['summary']
            if len(summary) > 30:
                # Take first few key words
                words = summary.split()[:4]
                short_label = ' '.join(words) + '...'
            else:
                short_label = summary
            
            # Add annotation
            fig.add_annotation(
                x=centroid[0],
                y=centroid[1],
                text=f"<b>C{cluster_id}</b><br>{short_label}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#999",
                ax=20,
                ay=-30,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=color_map[cluster_id],
                borderwidth=2,
                font=dict(size=9)
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="SAE Feature Clusters with LLM Annotations<br><sub>Graph-based clustering (35 communities) on 1650 features</sub>",
            font=dict(size=20)
        ),
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        width=1400,
        height=900,
        hovermode='closest',
        template='plotly_white',
        font=dict(family="Arial, sans-serif")
    )
    
    # Save
    fig.write_html(output_path)
    print(f"Saved main visualization to {output_path}")


def create_cluster_summary_panel(annotations: Dict, output_path: str):
    """Create a summary panel showing all clusters with their descriptions."""
    
    clusters = sorted(annotations['clusters'], key=lambda x: x['size'], reverse=True)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.7],
        specs=[[{"type": "bar"}, {"type": "table"}]],
        subplot_titles=("Cluster Sizes", "Cluster Descriptions")
    )
    
    # Bar chart of cluster sizes
    cluster_ids = [f"C{c['id']}" for c in clusters]
    sizes = [c['size'] for c in clusters]
    
    fig.add_trace(
        go.Bar(
            x=cluster_ids[:20],  # Top 20
            y=sizes[:20],
            marker=dict(
                color=sizes[:20],
                colorscale='Viridis',
                showscale=False
            ),
            text=sizes[:20],
            textposition='auto',
            hovertemplate='Cluster %{x}<br>Size: %{y} features<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Table of cluster descriptions
    table_data = []
    for c in clusters[:20]:  # Top 20
        table_data.append([
            f"C{c['id']}",
            str(c['size']),
            c['summary'][:60] + "..." if len(c['summary']) > 60 else c['summary']
        ])
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Cluster", "Size", "Description"],
                font=dict(size=12, color='white'),
                fill_color='#3498db',
                align=['left', 'center', 'left']
            ),
            cells=dict(
                values=list(zip(*table_data)),
                font=dict(size=11),
                align=['left', 'center', 'left'],
                fill_color=['#f0f0f0', '#ffffff'],
                height=25
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Cluster Summary Statistics",
        height=600,
        showlegend=False,
        font=dict(family="Arial, sans-serif")
    )
    
    fig.update_xaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="Number of Features", row=1, col=1)
    
    # Save
    fig.write_html(output_path)
    print(f"Saved summary panel to {output_path}")


def create_combined_dashboard(main_viz_path: str, summary_path: str, output_path: str):
    """Create a combined HTML dashboard with both visualizations."""
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAE Feature Clustering Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }}
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .viz-frame {{
                width: 100%;
                height: 900px;
                border: none;
                margin-bottom: 20px;
            }}
            .summary-frame {{
                width: 100%;
                height: 650px;
                border: none;
            }}
            .section-title {{
                color: #34495e;
                margin: 20px 0 10px 0;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
            }}
            .description {{
                color: #7f8c8d;
                margin-bottom: 20px;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 SAE Feature Clustering Analysis</h1>
            <div class="subtitle">
                Interactive visualization of 1650 Sparse Autoencoder features clustered into semantic groups
            </div>
            
            <h2 class="section-title">Interactive Cluster Visualization</h2>
            <div class="description">
                Hover over points to see feature explanations and cluster assignments. 
                Clusters are annotated with GPT-4o-generated summaries.
                The 15 largest clusters are labeled on the plot.
            </div>
            <iframe src="{main_viz}" class="viz-frame"></iframe>
            
            <h2 class="section-title">Cluster Statistics & Descriptions</h2>
            <div class="description">
                Summary of the top 20 clusters by size, showing the distribution of features
                and their semantic descriptions.
            </div>
            <iframe src="{summary}" class="summary-frame"></iframe>
        </div>
    </body>
    </html>
    """
    
    # Read the visualization files to embed them
    with open(main_viz_path, 'r') as f:
        main_viz_content = f.read()
    
    with open(summary_path, 'r') as f:
        summary_content = f.read()
    
    # Write combined dashboard
    with open(output_path, 'w') as f:
        f.write(html_template.format(
            main_viz=main_viz_path.split('/')[-1],
            summary=summary_path.split('/')[-1]
        ))
    
    print(f"Created combined dashboard at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create final annotated visualization")
    parser.add_argument("--projections", type=str, default="umap_projections_2d.npy",
                       help="Path to UMAP projections")
    parser.add_argument("--interpretations", type=str, default="all_interpretations_o3.json",
                       help="Path to original interpretations")
    parser.add_argument("--clustering", type=str, default="clustering_colored_results.json",
                       help="Path to clustering results")
    parser.add_argument("--annotations", type=str, default="cluster_annotations_gpt4o.json",
                       help="Path to GPT-4o annotations")
    parser.add_argument("--output-prefix", type=str, default="final",
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    print("Loading data...")
    projections, interp_data, cluster_data, annotations = load_all_data(
        args.projections, args.interpretations, 
        args.clustering, args.annotations
    )
    
    print(f"Creating visualizations for {len(projections)} features...")
    
    # Create main visualization
    main_viz_path = f"{args.output_prefix}_cluster_viz.html"
    create_main_visualization(
        projections, interp_data, cluster_data, annotations,
        main_viz_path
    )
    
    # Create summary panel
    summary_path = f"{args.output_prefix}_cluster_summary.html"
    create_cluster_summary_panel(annotations, summary_path)
    
    # Create combined dashboard
    dashboard_path = f"{args.output_prefix}_dashboard.html"
    create_combined_dashboard(main_viz_path, summary_path, dashboard_path)
    
    print("\n" + "="*60)
    print("✨ Final visualizations created!")
    print("="*60)
    print(f"Main visualization: {main_viz_path}")
    print(f"Summary panel: {summary_path}")
    print(f"Combined dashboard: {dashboard_path}")
    print("\nOpen the dashboard in a web browser to explore your clustered features!")
    print("="*60)


if __name__ == "__main__":
    main()