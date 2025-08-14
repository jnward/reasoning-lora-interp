#!/usr/bin/env python3
"""
Create multiple versions of the hierarchical donut plot with different color schemes.
"""

import json
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Version 1: Using matplotlib's Set2 colormap (professional, distinct)
def get_set2_colors():
    colors = plt.cm.Set2(np.linspace(0, 1, 7))
    return {
        'computational_operations': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[0][:3]]),
        'symbolic_manipulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[1][:3]]),
        'variable_value_tracking': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[2][:3]]),
        'reasoning_flow_control': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[3][:3]]),
        'domain_specific_patterns': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[4][:3]]),
        'relational_logical_operators': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[5][:3]]),
        'result_formulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[6][:3]]),
    }


# Version 2: Using matplotlib's Set3 colormap (pastel, soft)
def get_set3_colors():
    colors = plt.cm.Set3(np.linspace(0, 1, 7))
    return {
        'computational_operations': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[0][:3]]),
        'symbolic_manipulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[1][:3]]),
        'variable_value_tracking': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[2][:3]]),
        'reasoning_flow_control': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[3][:3]]),
        'domain_specific_patterns': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[4][:3]]),
        'relational_logical_operators': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[5][:3]]),
        'result_formulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[6][:3]]),
    }


# Version 3: Using Plotly's default qualitative colors
def get_plotly_colors():
    return {
        'computational_operations': '#636EFA',      # Blue
        'symbolic_manipulation': '#EF553B',         # Red
        'variable_value_tracking': '#00CC96',       # Green
        'reasoning_flow_control': '#AB63FA',        # Purple
        'domain_specific_patterns': '#FFA15A',      # Orange
        'relational_logical_operators': '#19D3F3',  # Cyan
        'result_formulation': '#FF6692',            # Pink
    }


# Version 4: Using Tab10 colormap (bright, distinct)
def get_tab10_colors():
    colors = plt.cm.tab10(np.linspace(0, 0.7, 7))  # Avoid too similar colors at the end
    return {
        'computational_operations': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[0][:3]]),
        'symbolic_manipulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[1][:3]]),
        'variable_value_tracking': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[2][:3]]),
        'reasoning_flow_control': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[3][:3]]),
        'domain_specific_patterns': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[4][:3]]),
        'relational_logical_operators': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[5][:3]]),
        'result_formulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[6][:3]]),
    }


# Version 5: Using viridis-inspired discrete colors (scientific)
def get_viridis_discrete():
    # Sample viridis at specific points for good contrast
    viridis = plt.cm.viridis(np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]))
    return {
        'computational_operations': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[6][:3]]),  # Brightest
        'symbolic_manipulation': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[5][:3]]),
        'variable_value_tracking': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[4][:3]]),
        'reasoning_flow_control': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[3][:3]]),
        'domain_specific_patterns': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[2][:3]]),
        'relational_logical_operators': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[1][:3]]),
        'result_formulation': '#%02x%02x%02x' % tuple([int(255*c) for c in viridis[0][:3]]),
    }


def get_subcategory_color(category: str, index: int, total: int, category_colors: Dict) -> str:
    """Generate a shade variation of the category color for subcategories."""
    base_color = category_colors.get(category, '#808080')
    
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
    (inner_labels, inner_values, inner_colors, 
     outer_labels, outer_values, outer_colors) = prepare_donut_data(densities_data, category_colors)
    
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
        title=title,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15,  # Moved legend even further right
            font=dict(size=10)
        ),
        margin=dict(l=0, r=300, t=50 if title else 10, b=0),
        height=700,
        width=1100,  # Increased width to accommodate legend
        annotations=[]  # No center text
    )
    
    return fig


def main():
    """Generate multiple versions of the donut plot with different color schemes."""
    
    # Load densities data
    print("Loading activation densities...")
    with open('activation_densities_complete.json', 'r') as f:
        densities_data = json.load(f)
    
    # Define color schemes
    color_schemes = [
        ("Set2 (Professional)", get_set2_colors()),
        ("Set3 (Pastel)", get_set3_colors()),
        ("Plotly Default", get_plotly_colors()),
        ("Tab10 (Bright)", get_tab10_colors()),
        ("Viridis Discrete", get_viridis_discrete()),
    ]
    
    # Generate each version
    for name, colors in color_schemes:
        print(f"\nGenerating {name} version...")
        fig = create_hierarchical_donut(densities_data, colors, title=f"SAE Feature Categories - {name}")
        
        # Save files
        filename_base = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        html_file = f"donut_{filename_base}.html"
        png_file = f"donut_{filename_base}.png"
        
        fig.write_html(
            html_file,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}
        )
        print(f"  Saved HTML: {html_file}")
        
        try:
            fig.write_image(png_file, width=1100, height=700)
            print(f"  Saved PNG: {png_file}")
        except Exception as e:
            print(f"  Could not save PNG: {e}")
    
    print("\nAll versions generated successfully!")
    print("\nYou can compare them by opening the HTML files in a browser.")


if __name__ == "__main__":
    main()