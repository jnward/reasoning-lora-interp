#!/usr/bin/env python3
"""
Create a publication-ready hierarchical donut plot with a compact legend.
"""

import json
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import argparse


# Define color palette for categories
CATEGORY_COLORS = {
    'symbolic_manipulation': '#FF6B6B',      # Red
    'variable_value_tracking': '#4ECDC4',    # Teal
    'computational_operations': '#45B7D1',   # Blue
    'reasoning_flow_control': '#96CEB4',     # Green
    'domain_specific_patterns': '#FFEAA7',   # Yellow
    'relational_logical_operators': '#DDA0DD', # Plum
    'result_formulation': '#FFA07A',         # Light Salmon
}


def get_subcategory_color(category: str, index: int, total: int) -> str:
    """Generate a shade of the category color for subcategories."""
    base_color = CATEGORY_COLORS.get(category, '#CCCCCC')
    
    # Convert hex to RGB
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Create shade variation (darker to lighter)
    factor = 0.7 + (0.3 * index / max(total - 1, 1))
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def prepare_donut_data(densities_data: Dict) -> Tuple[List, List, List, List, Dict]:
    """Prepare data for the hierarchical donut plot."""
    category_densities = densities_data['category_densities']
    subcategory_densities = densities_data['subcategory_densities']
    
    # Sort categories by density for better visualization
    sorted_categories = sorted(category_densities.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare inner ring (categories)
    inner_labels = []
    inner_values = []
    inner_colors = []
    
    for cat, density in sorted_categories:
        inner_labels.append(cat.replace('_', ' ').title())
        inner_values.append(density)
        inner_colors.append(CATEGORY_COLORS.get(cat, '#CCCCCC'))
    
    # Prepare outer ring (subcategories) and legend mapping
    outer_labels = []
    outer_values = []
    outer_colors = []
    legend_mapping = {}  # Maps subcategory to parent category
    
    # Process subcategories grouped by category
    for cat, cat_density in sorted_categories:
        cat_title = cat.replace('_', ' ').title()
        
        # Find all subcategories for this category
        cat_subcats = []
        for subcat_key, subcat_density in subcategory_densities.items():
            if subcat_key.startswith(f"{cat}::"):
                subcat_name = subcat_key.split('::')[1]
                cat_subcats.append((subcat_name, subcat_density))
        
        # Sort subcategories by density
        cat_subcats.sort(key=lambda x: x[1], reverse=True)
        
        # Add to outer ring with proportional sizing
        for i, (subcat, subcat_density) in enumerate(cat_subcats):
            # Scale subcategory value by category density
            scaled_value = subcat_density * cat_density
            
            subcat_title = subcat.replace('_', ' ').title()
            outer_labels.append(subcat_title)
            outer_values.append(scaled_value)
            outer_colors.append(get_subcategory_color(cat, i, len(cat_subcats)))
            legend_mapping[subcat_title] = cat_title
    
    return inner_labels, inner_values, inner_colors, outer_labels, outer_values, outer_colors, legend_mapping


def create_publication_plot(densities_data: Dict, title: str = "SAE Feature Activation Distribution") -> go.Figure:
    """Create a publication-ready hierarchical donut plot."""
    # Prepare data
    (inner_labels, inner_values, inner_colors, 
     outer_labels, outer_values, outer_colors, legend_mapping) = prepare_donut_data(densities_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add outer ring (subcategories)
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.67,  # Equal width rings with center hole
        marker=dict(colors=outer_colors, line=dict(color='white', width=0.5)),
        textposition='inside',  # Force text inside the slices
        textinfo='percent',  # Just show percentage, no labels
        textfont=dict(size=7),
        hovertemplate='<b>%{label}</b><br>%{percent}<br><extra></extra>',
        sort=False,
        direction='clockwise',
        rotation=90,
        showlegend=False,  # Don't show subcategories in legend
        name='Subcategories'
    ))
    
    # Add inner ring (categories)
    fig.add_trace(go.Pie(
        labels=inner_labels,
        values=inner_values,
        hole=0.33,  # Center hole for equal width rings
        marker=dict(colors=inner_colors, line=dict(color='white', width=1)),
        textposition='inside',
        textinfo='label+percent',  # Show both label and percentage
        textfont=dict(size=9, color='white'),
        hovertemplate='<b>%{label}</b><br>%{percent}<br><extra></extra>',
        sort=False,
        direction='clockwise',
        rotation=90,
        domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),  # 80% of plot area
        showlegend=False,  # Don't show categories in legend either
        name='Categories'
    ))
    
    # Create a simplified legend showing just categories with their colors
    # This helps readers understand the color scheme
    for i, (label, color) in enumerate(zip(inner_labels, inner_colors)):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            showlegend=True,
            name=label
        ))
    
    # Update layout for publication
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            itemsizing='constant',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#CCCCCC',
            borderwidth=0.5
        ),
        margin=dict(l=20, r=180, t=50, b=20),  # Right margin for legend
        height=500,  # Reasonable height for publication
        width=700,   # Width to accommodate legend
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=[]  # No center text
    )
    
    # Hide axes for scatter traces
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Create publication-ready hierarchical donut plot")
    parser.add_argument('--input', type=str, default='activation_densities_complete.json',
                       help='Input JSON file with activation densities')
    parser.add_argument('--output', type=str, default='publication_plot',
                       help='Output base name (will create .html and .png)')
    parser.add_argument('--title', type=str, default='SAE Feature Activation Distribution',
                       help='Title for the visualization')
    parser.add_argument('--width', type=int, default=700,
                       help='Plot width in pixels')
    parser.add_argument('--height', type=int, default=500,
                       help='Plot height in pixels')
    
    args = parser.parse_args()
    
    # Load densities data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        densities_data = json.load(f)
    
    # Create visualization
    print("Creating publication-ready donut plot...")
    fig = create_publication_plot(densities_data, title=args.title)
    
    # Update dimensions if specified
    if args.width != 700 or args.height != 500:
        fig.update_layout(width=args.width, height=args.height)
    
    # Save to HTML
    html_output = f"{args.output}.html"
    print(f"Saving HTML visualization to {html_output}...")
    fig.write_html(
        html_output,
        include_plotlyjs='cdn',
        config={'displayModeBar': False, 'displaylogo': False, 'responsive': True}
    )
    
    # Save to PNG
    png_output = f"{args.output}.png"
    try:
        print(f"Saving PNG to {png_output}...")
        fig.write_image(png_output, width=args.width, height=args.height)
        print(f"Successfully saved PNG: {png_output}")
    except Exception as e:
        print(f"Could not save PNG: {e}")
    
    # Print summary
    print("\nVisualization created successfully!")
    print(f"Files created:")
    print(f"  - HTML: {html_output}")
    print(f"  - PNG: {png_output}")
    
    # Print category distribution
    category_densities = densities_data['category_densities']
    print("\nCategory Distribution:")
    sorted_cats = sorted(category_densities.items(), key=lambda x: x[1], reverse=True)
    for cat, density in sorted_cats:
        print(f"  {cat.replace('_', ' ').title()}: {density:.1%}")


if __name__ == "__main__":
    main()