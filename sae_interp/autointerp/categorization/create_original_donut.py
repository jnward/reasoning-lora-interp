#!/usr/bin/env python3
"""
Create the original hierarchical donut plot visualization of SAE feature activation densities.
Going back to the original version that was working.
"""

import json
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Define warm color palette for categories (yellow to magenta gradient)
# Ordered by density from highest to lowest
CATEGORY_COLORS_WARM = {
    'computational_operations': '#FFD700',      # Gold/Yellow (highest density)
    'symbolic_manipulation': '#FF8C00',         # Dark Orange
    'variable_value_tracking': '#FF4500',       # Orange-Red
    'reasoning_flow_control': '#DC143C',        # Crimson Red
    'domain_specific_patterns': '#C71585',      # Medium Violet Red
    'relational_logical_operators': '#8B008B',  # Dark Magenta
    'result_formulation': '#4B0082',            # Indigo (lowest density)
}


def get_subcategory_color(category: str, index: int, total: int) -> str:
    """Generate a shade variation of the category color for subcategories."""
    base_color = CATEGORY_COLORS_WARM.get(category, '#808080')
    
    # Convert hex to RGB
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Create lightness variation (lighter to darker as index increases)
    # Use a wider range for better differentiation
    factor = 0.5 + (0.5 * (1 - index / max(total - 1, 1)))
    
    # Apply factor with some saturation preservation
    r = int(r * factor + (255 * (1 - factor) * 0.3))
    g = int(g * factor + (255 * (1 - factor) * 0.3))
    b = int(b * factor + (255 * (1 - factor) * 0.3))
    
    # Clamp values
    r = min(255, max(0, r))
    g = min(255, max(0, g))
    b = min(255, max(0, b))
    
    return f'#{r:02x}{g:02x}{b:02x}'


def prepare_donut_data(densities_data: Dict) -> Tuple[List, List, List, List]:
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
        # Use warm color palette
        inner_colors.append(CATEGORY_COLORS_WARM.get(cat, '#808080'))
    
    # Prepare outer ring (subcategories)
    outer_labels = []
    outer_values = []
    outer_colors = []
    
    # Process subcategories grouped by category
    for cat, cat_density in sorted_categories:
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
            
            outer_labels.append(subcat.replace('_', ' ').title())
            outer_values.append(scaled_value)
            # Use hierarchical coloring - same hue as parent, different lightness
            outer_colors.append(get_subcategory_color(cat, i, len(cat_subcats)))
    
    return inner_labels, inner_values, inner_colors, outer_labels, outer_values, outer_colors


def create_hierarchical_donut(densities_data: Dict, title: str = "SAE Feature Activation Densities") -> go.Figure:
    """Create the hierarchical donut plot."""
    # Prepare data
    (inner_labels, inner_values, inner_colors, 
     outer_labels, outer_values, outer_colors) = prepare_donut_data(densities_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add outer ring (subcategories)
    # Show all labels without percentages
    outer_text = outer_labels  # Use all labels
    
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.65,
        marker=dict(colors=outer_colors, line=dict(color='white', width=1)),
        textposition='auto',  # Let Plotly decide inside vs outside
        text=outer_text,  # Use custom text
        textinfo='text',  # Only show our custom text, not labels or percent
        textfont=dict(size=11),  # Good size for readability
        hovertemplate='<b>%{label}</b><br>Density: %{percent}<br><extra></extra>',
        name='Subcategories',
        sort=False,
        direction='clockwise',
        rotation=90
    ))
    
    # Add inner ring (categories)
    fig.add_trace(go.Pie(
        labels=inner_labels,
        values=inner_values,
        hole=0.3,
        marker=dict(colors=inner_colors, line=dict(color='white', width=2)),
        textposition='inside',
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),  # Increased from 12
        hovertemplate='<b>%{label}</b><br>Density: %{percent}<br><extra></extra>',
        name='Categories',
        sort=False,
        direction='clockwise',
        rotation=90,
        domain=dict(x=[0.15, 0.85], y=[0.15, 0.85])
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15,  # Moved legend even further right
            font=dict(size=10)
        ),
        margin=dict(l=0, r=300, t=10, b=0),  # Increased right margin more
        height=700,
        width=1100,  # Increased width to accommodate legend
        annotations=[]  # No center text
    )
    
    return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Create hierarchical donut plot visualization")
    parser.add_argument('--input', type=str, default='activation_densities_complete.json',
                       help='Input JSON file with activation densities')
    parser.add_argument('--output', type=str, default='hierarchical_donut.html',
                       help='Output HTML file for visualization')
    parser.add_argument('--title', type=str, default='SAE Feature Activation Densities (All Features Categorized)',
                       help='Title for the visualization')
    
    args = parser.parse_args()
    
    # Load densities data
    print(f"Loading activation densities from {args.input}...")
    with open(args.input, 'r') as f:
        densities_data = json.load(f)
    
    # Create visualization
    print("Creating hierarchical donut plot...")
    fig = create_hierarchical_donut(densities_data, title=args.title)
    
    # Save to HTML
    print(f"Saving visualization to {args.output}...")
    fig.write_html(
        args.output,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}
    )
    
    # Try to save as PNG
    try:
        png_output = args.output.replace('.html', '.png')
        fig.write_image(png_output, width=1000, height=700)
        print(f"Also saved as PNG: {png_output}")
    except Exception as e:
        print(f"Could not save PNG: {e}")
    
    # Print summary
    print("\nVisualization created successfully!")
    print(f"Open {args.output} in a web browser to view the interactive plot.")
    
    # Print category summary
    category_densities = densities_data['category_densities']
    print("\nCategory Distribution Summary:")
    sorted_cats = sorted(category_densities.items(), key=lambda x: x[1], reverse=True)
    for cat, density in sorted_cats:
        print(f"  {cat.replace('_', ' ').title()}: {density:.1%}")


if __name__ == "__main__":
    main()