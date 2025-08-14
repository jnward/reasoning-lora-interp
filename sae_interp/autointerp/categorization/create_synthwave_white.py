#!/usr/bin/env python3
"""
Create synthwave version with white background.
"""

import json
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import colorsys


# Synthwave colors (retrowave aesthetic) - no green, no coral
def get_synthwave_colors():
    return {
        'computational_operations': '#FFBE0B',      # Yellow
        'symbolic_manipulation': '#FB5607',         # Orange  
        'variable_value_tracking': '#FF0040',       # Bright Red
        'reasoning_flow_control': '#9B2EBF',        # Medium Purple (lightened for better legibility)
        'domain_specific_patterns': '#8338EC',      # Purple (moved from reasoning_flow_control)
        'relational_logical_operators': '#3A86FF',  # Blue (moved from domain_specific_patterns)
        'result_formulation': '#00D9FF',            # Cyan (moved from relational_logical_operators)
    }


def get_subcategory_color(category: str, index: int, total: int, category_colors: Dict) -> str:
    """Generate a shade variation of the category color for subcategories."""
    base_color = category_colors.get(category, '#808080')
    
    # Convert hex to RGB
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # For synthwave on white, create lighter variations
    # Convert to HSL for better control
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    
    # Make subcategories lighter than parent
    # First subcategory is lightest
    lightness_boost = 0.3 * (1 - index / max(total - 1, 1))
    new_l = min(0.95, l + lightness_boost)
    
    # Also reduce saturation slightly for lighter colors
    new_s = s * (0.6 + 0.4 * index / max(total - 1, 1))
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, new_l, new_s)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def prepare_donut_data(densities_data: Dict, category_colors: Dict) -> Tuple[List, List, List, List]:
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
        inner_colors.append(category_colors.get(cat, '#808080'))
    
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
            outer_colors.append(get_subcategory_color(cat, i, len(cat_subcats), category_colors))
    
    return inner_labels, inner_values, inner_colors, outer_labels, outer_values, outer_colors


def create_hierarchical_donut(densities_data: Dict, category_colors: Dict, title: str = "") -> go.Figure:
    """Create the hierarchical donut plot."""
    # Prepare data
    category_densities = densities_data['category_densities']
    subcategory_densities = densities_data['subcategory_densities']
    
    # Sort categories by density for better visualization
    sorted_categories = sorted(category_densities.items(), key=lambda x: x[1], reverse=True)
    
    # Create figure
    fig = go.Figure()
    
    # First, add all subcategories as a single trace for the outer ring
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
        
        # Add to outer ring
        for i, (subcat, subcat_density) in enumerate(cat_subcats):
            scaled_value = subcat_density * cat_density
            outer_labels.append(subcat.replace('_', ' ').title())
            outer_values.append(scaled_value)
            outer_colors.append(get_subcategory_color(cat, i, len(cat_subcats), category_colors))
    
    # Add outer ring first
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.75,
        marker=dict(colors=outer_colors, line=dict(color='white', width=1)),
        textposition='outside',
        text=outer_labels,
        textinfo='text',
        textfont=dict(size=14, color='black'),
        hovertemplate='<b>%{label}</b><br>Density: %{percent}<br><extra></extra>',
        showlegend=False,  # Don't show in legend
        sort=False,
        direction='clockwise',
        rotation=90,
        domain=dict(x=[0.0, 0.48], y=[0.0, 1.0])  # Position on left, shifted left
    ))
    
    # Add inner ring last (renders on top) with smaller radius
    inner_labels = []
    inner_values = []
    inner_colors = []
    inner_text = []
    
    for cat, density in sorted_categories:
        label = cat.replace('_', ' ').title()
        inner_labels.append(label)
        inner_values.append(density)
        inner_colors.append(category_colors.get(cat, '#808080'))
        if density >= 0.05:
            inner_text.append(f"{label}<br>{density:.1%}")
        else:
            inner_text.append(label)
    
    fig.add_trace(go.Pie(
        labels=inner_labels,
        values=inner_values,
        hole=0.20,
        marker=dict(colors=inner_colors, line=dict(color='white', width=2)),
        textposition='inside',
        text=inner_text,
        textinfo='text',
        textfont=dict(size=18, color='black'),
        hovertemplate='<b>%{label}</b><br>Density: %{percent}<br><extra></extra>',
        showlegend=False,  # Don't show in legend
        sort=False,
        direction='clockwise',
        rotation=90,
        insidetextorientation='radial',
        domain=dict(x=[0.0399, 0.4401], y=[0.083, 0.917])  # Smaller and positioned on left, shifted left
    ))
    
    # Now add traces for the legend in hierarchical order
    # Split into two groups for two-column legend
    for idx, (cat, cat_density) in enumerate(sorted_categories):
        cat_label = cat.replace('_', ' ').title()
        cat_color = category_colors.get(cat, '#808080')
        
        # Determine which legend group (left or right)
        legend_num = "legend" if idx < 3 else "legend2"
        
        # Add category to legend with percentage
        cat_percentage = cat_density * 100
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=cat_color),
            showlegend=True,
            legendgroup=cat,
            legendgrouptitle_text=None,
            legend=legend_num,
            name=f"<b>{cat_label}</b> ({cat_percentage:.1f}%)",
            hoverinfo='skip'
        ))
        
        # Find all subcategories for this category
        cat_subcats = []
        for subcat_key, subcat_density in subcategory_densities.items():
            if subcat_key.startswith(f"{cat}::"):
                subcat_name = subcat_key.split('::')[1]
                cat_subcats.append((subcat_name, subcat_density))
        
        # Sort subcategories by density
        cat_subcats.sort(key=lambda x: x[1], reverse=True)
        
        # Add subcategories to legend with percentages
        for i, (subcat, subcat_density) in enumerate(cat_subcats):
            subcat_label = subcat.replace('_', ' ').title()
            subcat_color = get_subcategory_color(cat, i, len(cat_subcats), category_colors)
            # Calculate actual percentage of total (scaled by category density)
            subcat_percentage = subcat_density * cat_density * 100
            
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=8, color=subcat_color),
                showlegend=True,
                legendgroup=cat,
                legend=legend_num,
                name=f"  {subcat_label} ({subcat_percentage:.1f}%)",  # Indent subcategories with percentage
                hoverinfo='skip'
            ))
    
    # Update layout with two legends side by side
    fig.update_layout(
        title=title if title else "",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.65,
            font=dict(size=14),
            itemsizing='constant',
            tracegroupgap=20
        ),
        legend2=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.98,
            font=dict(size=14),
            itemsizing='constant',
            tracegroupgap=20
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        width=1470,  # 21:10 aspect ratio
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, visible=False),
        annotations=[]
    )
    
    return fig


def main():
    """Generate synthwave version with white background."""
    
    # Load densities data
    print("Loading activation densities...")
    with open('activation_densities_complete.json', 'r') as f:
        densities_data = json.load(f)
    
    # Get synthwave colors
    colors = get_synthwave_colors()
    
    # Generate the plot
    print("Generating Synthwave (White BG) version...")
    fig = create_hierarchical_donut(
        densities_data, 
        colors, 
        title=""
    )
    
    # Save files
    html_file = "donut_synthwave_white.html"
    png_file = "donut_synthwave_white.png"
    
    fig.write_html(
        html_file,
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}
    )
    print(f"Saved HTML: {html_file}")
    
    try:
        fig.write_image(png_file, width=1470, height=700)
        print(f"Saved PNG: {png_file}")
    except Exception as e:
        print(f"Could not save PNG: {e}")
    
    print("\nSynthwave with white background generated successfully!")


if __name__ == "__main__":
    main()