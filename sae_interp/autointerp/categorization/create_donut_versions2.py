#!/usr/bin/env python3
"""
Create multiple versions of the hierarchical donut plot with different color schemes.
Second batch with more refined color choices.
"""

import json
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import colorsys


# Version 1: Tableau 20 inspired colors (refined professional)
def get_tableau_colors():
    return {
        'computational_operations': '#1f77b4',      # Strong Blue
        'symbolic_manipulation': '#ff7f0e',         # Orange
        'variable_value_tracking': '#2ca02c',       # Green
        'reasoning_flow_control': '#d62728',        # Red
        'domain_specific_patterns': '#9467bd',      # Purple
        'relational_logical_operators': '#8c564b',  # Brown
        'result_formulation': '#e377c2',            # Pink
    }


# Version 2: D3 Category20b subset (muted but distinct)
def get_d3_colors():
    return {
        'computational_operations': '#393b79',      # Dark Blue
        'symbolic_manipulation': '#e7969c',         # Light Red/Pink
        'variable_value_tracking': '#6b6ecf',       # Purple Blue
        'reasoning_flow_control': '#637939',        # Olive Green
        'domain_specific_patterns': '#b5cf6b',      # Light Green
        'relational_logical_operators': '#ce6dbd',  # Light Purple
        'result_formulation': '#9c9ede',            # Light Blue Purple
    }


# Version 3: Color Brewer Dark2 (good for print)
def get_dark2_colors():
    colors = plt.cm.Dark2(np.linspace(0, 0.9, 7))
    return {
        'computational_operations': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[0][:3]]),
        'symbolic_manipulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[1][:3]]),
        'variable_value_tracking': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[2][:3]]),
        'reasoning_flow_control': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[3][:3]]),
        'domain_specific_patterns': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[4][:3]]),
        'relational_logical_operators': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[5][:3]]),
        'result_formulation': '#%02x%02x%02x' % tuple([int(255*c) for c in colors[6][:3]]),
    }


# Version 4: Seaborn Deep palette equivalent
def get_deep_colors():
    # These are the seaborn "deep" palette colors
    return {
        'computational_operations': '#4C72B0',      # Blue
        'symbolic_manipulation': '#DD8452',         # Orange
        'variable_value_tracking': '#55A868',       # Green
        'reasoning_flow_control': '#C44E52',        # Red
        'domain_specific_patterns': '#8172B3',      # Purple
        'relational_logical_operators': '#937860',  # Brown
        'result_formulation': '#DA8BC3',            # Pink
    }


# Version 5: Custom gradient from teal to coral (modern)
def get_modern_gradient():
    # Create a custom gradient with good contrast
    return {
        'computational_operations': '#20B2AA',      # Light Sea Green (highest)
        'symbolic_manipulation': '#48D1CC',         # Medium Turquoise
        'variable_value_tracking': '#40E0D0',       # Turquoise
        'reasoning_flow_control': '#FFA07A',        # Light Salmon
        'domain_specific_patterns': '#FA8072',      # Salmon
        'relational_logical_operators': '#F08080',  # Light Coral
        'result_formulation': '#CD5C5C',            # Indian Red (lowest)
    }


# Version 6: Material Design inspired
def get_material_colors():
    return {
        'computational_operations': '#2196F3',      # Blue
        'symbolic_manipulation': '#FF9800',         # Orange
        'variable_value_tracking': '#4CAF50',       # Green
        'reasoning_flow_control': '#F44336',        # Red
        'domain_specific_patterns': '#9C27B0',      # Purple
        'relational_logical_operators': '#00BCD4',  # Cyan
        'result_formulation': '#795548',            # Brown
    }


# Version 7: Nature-inspired palette
def get_nature_colors():
    return {
        'computational_operations': '#2E7D32',      # Forest Green
        'symbolic_manipulation': '#F57C00',         # Deep Orange
        'variable_value_tracking': '#0288D1',       # Light Blue
        'reasoning_flow_control': '#C62828',        # Deep Red
        'domain_specific_patterns': '#6A1B9A',      # Deep Purple
        'relational_logical_operators': '#00796B',  # Teal
        'result_formulation': '#5D4037',            # Brown
    }


def get_subcategory_color(category: str, index: int, total: int, category_colors: Dict) -> str:
    """Generate a shade variation of the category color for subcategories."""
    base_color = category_colors.get(category, '#808080')
    
    # Convert hex to RGB
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert to HSL for better lightness control
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    
    # Vary lightness - make subcategories lighter than parent
    # First subcategory is lightest, gradually getting closer to parent color
    lightness_range = 0.3  # How much to vary the lightness
    new_l = min(0.9, l + lightness_range * (1 - index / max(total - 1, 1)))
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, new_l, s)
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
        title=title if title else "",
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
        ("Tableau", get_tableau_colors()),
        ("D3 Muted", get_d3_colors()),
        ("Dark2", get_dark2_colors()),
        ("Deep", get_deep_colors()),
        ("Modern Gradient", get_modern_gradient()),
        ("Material", get_material_colors()),
        ("Nature", get_nature_colors()),
    ]
    
    # Generate each version
    for name, colors in color_schemes:
        print(f"\nGenerating {name} version...")
        fig = create_hierarchical_donut(densities_data, colors, title=f"SAE Feature Categories - {name}")
        
        # Save files
        filename_base = name.lower().replace(' ', '_')
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