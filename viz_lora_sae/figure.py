"""
Figure generation module for LoRA → SAE → Category visualization.

This module creates Plotly figures for Sankey diagrams, bar charts, and evidence tiles.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np


def build_sankey(
    node_labels: List[str],
    links: Dict,
    palette: Dict[str, str],
    width: int = 1100,
    height: int = 520,
    title: Optional[str] = None
) -> go.Figure:
    """
    Build a Sankey diagram visualization.
    
    Args:
        node_labels: List of node labels
        links: Dict with 'source', 'target', 'value', 'color' lists
        palette: Category color palette
        width: Figure width in pixels
        height: Figure height in pixels
        title: Optional figure title
        
    Returns:
        Plotly Figure object
    """
    # Get node colors from nodes dict if available
    node_colors = palette.get('node_colors', ['#999999'] * len(node_labels))
    
    # Create Sankey figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate='%{label}<br>Flow: %{value:.3f}<extra></extra>'
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color'],
            hovertemplate='%{source.label} → %{target.label}<br>Weight: %{value:.3f}<extra></extra>'
        ),
        arrangement='snap',
        orientation='h'
    )])
    
    # Update layout
    fig.update_layout(
        title=title or "LoRA → SAE Features → Categories Flow",
        width=width,
        height=height,
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def build_category_bar(
    df: pd.DataFrame,
    categories: List[str],
    palette: Dict[str, str],
    width: int = 500,
    height: int = 380,
    title: Optional[str] = None
) -> go.Figure:
    """
    Build a bar chart of total activation mass per category.
    
    Args:
        df: Features DataFrame with 'category' and 'mass' columns
        categories: Ordered list of category IDs
        palette: Category color palette
        width: Figure width in pixels
        height: Figure height in pixels
        title: Optional figure title
        
    Returns:
        Plotly Figure object
    """
    # Compute total mass per category
    category_masses = df.groupby('category')['mass'].sum()
    
    # Ensure all categories are present (even with 0 mass)
    category_masses = category_masses.reindex(categories, fill_value=0)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=category_masses.values,
            marker_color=[palette.get(cat, '#999') for cat in categories],
            text=[f'{v:.2f}' for v in category_masses.values],
            textposition='auto',
            hovertemplate='%{x}<br>Total Mass: %{y:.3f}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title or "Total Activation Mass by Category",
        xaxis_title="Category",
        yaxis_title="Total Mass",
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=40, b=100),
        xaxis=dict(tickangle=45),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='#e0e0e0', gridwidth=0.5)
    )
    
    return fig


def build_evidence_tiles(
    df: pd.DataFrame,
    examples_path: str,
    palette: Dict[str, str],
    max_tiles: int = 10,
    width: int = 1100,
    height: int = 200
) -> go.Figure:
    """
    Build evidence tiles showing example text snippets.
    
    Args:
        df: Features DataFrame
        examples_path: Path to CSV with feature_id and text_snippet columns
        palette: Category color palette
        max_tiles: Maximum number of tiles to show
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    import pandas as pd
    
    # Load examples
    examples_df = pd.read_csv(examples_path)
    
    # Join with features to get categories
    examples_df = examples_df.merge(
        df[['feature_id', 'category', 'mass']],
        on='feature_id',
        how='inner'
    )
    
    # Sort by mass and take top examples
    examples_df = examples_df.sort_values('mass', ascending=False).head(max_tiles)
    
    # Create subplot figure with tiles
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=min(len(examples_df), max_tiles),
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        subplot_titles=[f"F{row['feature_id']}" for _, row in examples_df.iterrows()]
    )
    
    # Add tiles
    for i, (_, row) in enumerate(examples_df.iterrows()):
        col = i + 1
        
        # Create a colored rectangle as background
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[0, 0, 1, 1, 0],
                fill='toself',
                fillcolor=palette.get(row['category'], '#999'),
                opacity=0.3,
                line=dict(color=palette.get(row['category'], '#999'), width=2),
                mode='lines',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=col
        )
        
        # Add text annotation
        text = row.get('text_snippet', '')
        if len(text) > 120:
            text = text[:117] + "..."
        
        fig.add_annotation(
            text=f"<b>{text}</b>",
            x=0.5, y=0.5,
            xref=f"x{col}", yref=f"y{col}",
            showarrow=False,
            font=dict(size=10, color='black'),
            align='center',
            xanchor='center',
            yanchor='middle'
        )
    
    # Update layout
    fig.update_layout(
        title="Top Feature Activation Examples",
        width=width,
        height=height,
        showlegend=False,
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Hide axes
    for i in range(1, min(len(examples_df), max_tiles) + 1):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i)
    
    return fig


def build_combined_dashboard(
    sankey_fig: go.Figure,
    bar_fig: go.Figure,
    evidence_fig: Optional[go.Figure] = None,
    width: int = 1600,
    height: int = 800
) -> go.Figure:
    """
    Combine multiple figures into a single dashboard.
    
    Args:
        sankey_fig: Sankey diagram figure
        bar_fig: Bar chart figure
        evidence_fig: Optional evidence tiles figure
        width: Total dashboard width
        height: Total dashboard height
        
    Returns:
        Combined Plotly Figure
    """
    from plotly.subplots import make_subplots
    
    # Determine layout based on whether evidence is included
    if evidence_fig:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.7, 0.3],
            column_widths=[0.7, 0.3],
            specs=[
                [{"type": "sankey", "colspan": 1}, {"type": "bar"}],
                [{"colspan": 2}, None]
            ],
            subplot_titles=["Information Flow", "Category Totals", "Example Activations"]
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[
                [{"type": "sankey"}, {"type": "bar"}]
            ],
            subplot_titles=["Information Flow", "Category Totals"]
        )
    
    # Add Sankey
    for trace in sankey_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add bar chart
    for trace in bar_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Add evidence if provided
    if evidence_fig:
        for trace in evidence_fig.data:
            fig.add_trace(trace, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="LoRA → SAE → Category Analysis Dashboard",
        width=width,
        height=height,
        showlegend=False,
        paper_bgcolor='white'
    )
    
    return fig


def save_snapshot(
    nodes: dict,
    links: dict,
    df: pd.DataFrame,
    params: dict,
    output_path: str
) -> None:
    """
    Save analysis snapshot to JSON for reproducibility.
    
    Args:
        nodes: Sankey nodes data
        links: Sankey links data
        df: Features DataFrame
        params: Analysis parameters
        output_path: Path to save JSON snapshot
    """
    import json
    
    # Compute summary statistics
    total_mass = df['mass'].sum()
    category_masses = df.groupby('category')['mass'].sum().to_dict()
    
    # Count features per category
    category_counts = df.groupby('category').size().to_dict()
    
    # Compute coverage for top-K features
    coverage_stats = {}
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat].sort_values('mass', ascending=False)
        top_k = params.get('top_k', 6)
        top_mass = cat_df.head(top_k)['mass'].sum()
        total_cat_mass = cat_df['mass'].sum()
        coverage = (top_mass / total_cat_mass * 100) if total_cat_mass > 0 else 0
        coverage_stats[cat] = {
            'top_k_coverage_percent': round(coverage, 2),
            'num_features': len(cat_df),
            'num_top_k': min(top_k, len(cat_df))
        }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj
    
    snapshot = {
        'params': params,
        'nodes': convert_to_python_types(nodes),
        'links': {
            'source': [int(x) for x in links['source']],
            'target': [int(x) for x in links['target']],
            'value': [float(x) for x in links['value']],
            'num_links': len(links['source'])
        },
        'summary': {
            'total_mass': float(total_mass),
            'num_features': len(df),
            'num_categories': int(df['category'].nunique()),
            'category_masses': {k: float(v) for k, v in category_masses.items()},
            'category_counts': {k: int(v) for k, v in category_counts.items()},
            'coverage_stats': convert_to_python_types(coverage_stats)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"Saved analysis snapshot to {output_path}")