#!/usr/bin/env python3
"""
Create cool, modern versions of the hierarchical donut plot with trendy color schemes.
"""

import json
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import colorsys


# Version 1: Cyberpunk Neon (dark with bright accents)
def get_cyberpunk_colors():
    return {
        'computational_operations': '#00FFFF',      # Cyan
        'symbolic_manipulation': '#FF00FF',         # Magenta
        'variable_value_tracking': '#00FF00',       # Lime
        'reasoning_flow_control': '#FF1493',        # Deep Pink
        'domain_specific_patterns': '#FFD700',      # Gold
        'relational_logical_operators': '#FF4500',  # Orange Red
        'result_formulation': '#9400D3',            # Violet
    }


# Version 2: Sunset Gradient (warm sunset colors)
def get_sunset_colors():
    return {
        'computational_operations': '#FF6B35',      # Sunset Orange
        'symbolic_manipulation': '#F77FBE',         # Pink
        'variable_value_tracking': '#FCE77D',       # Pale Yellow
        'reasoning_flow_control': '#965FD4',        # Purple
        'domain_specific_patterns': '#734B8F',      # Deep Purple
        'relational_logical_operators': '#4A3C6B',  # Dark Purple
        'result_formulation': '#2C2451',            # Midnight
    }


# Version 3: Ocean Depths (blues and teals)
def get_ocean_colors():
    return {
        'computational_operations': '#05F4FF',      # Bright Cyan
        'symbolic_manipulation': '#0597F2',         # Bright Blue
        'variable_value_tracking': '#054A91',       # Deep Blue
        'reasoning_flow_control': '#3E7CB1',        # Steel Blue
        'domain_specific_patterns': '#81A4CD',      # Light Steel
        'relational_logical_operators': '#2A628F',  # Dark Teal
        'result_formulation': '#13293D',            # Navy
    }


# Version 4: Retro Miami (80s inspired)
def get_miami_colors():
    return {
        'computational_operations': '#F72585',      # Hot Pink
        'symbolic_manipulation': '#B5179E',         # Purple Pink
        'variable_value_tracking': '#7209B7',       # Purple
        'reasoning_flow_control': '#560BAD',        # Deep Purple
        'domain_specific_patterns': '#480CA8',      # Blue Purple
        'relational_logical_operators': '#3A0CA3',  # Royal Blue
        'result_formulation': '#3F37C9',            # Blue
    }


# Version 5: Nordic Aurora (northern lights inspired)
def get_aurora_colors():
    return {
        'computational_operations': '#A8E6CF',      # Mint Green
        'symbolic_manipulation': '#7FD1B9',         # Teal Green
        'variable_value_tracking': '#FF8B94',       # Salmon Pink
        'reasoning_flow_control': '#A8DADC',        # Powder Blue
        'domain_specific_patterns': '#457B9D',      # Steel Blue
        'relational_logical_operators': '#1D3557',  # Dark Blue
        'result_formulation': '#F1FAEE',            # Off White
    }


# Version 6: Synthwave (retrowave aesthetic)
def get_synthwave_colors():
    return {
        'computational_operations': '#FF006E',      # Hot Pink
        'symbolic_manipulation': '#FB5607',         # Orange
        'variable_value_tracking': '#FFBE0B',       # Yellow
        'reasoning_flow_control': '#8338EC',        # Purple
        'domain_specific_patterns': '#3A86FF',      # Blue
        'relational_logical_operators': '#06FFB4',  # Mint
        'result_formulation': '#FF4365',            # Coral
    }


# Version 7: Pastel Dream (soft aesthetic colors)
def get_pastel_dream_colors():
    return {
        'computational_operations': '#FFB3BA',      # Light Pink
        'symbolic_manipulation': '#FFDFBA',         # Peach
        'variable_value_tracking': '#FFFFBA',       # Light Yellow
        'reasoning_flow_control': '#BAFFC9',        # Mint
        'domain_specific_patterns': '#BAE1FF',      # Baby Blue
        'relational_logical_operators': '#C9B6FF',  # Lavender
        'result_formulation': '#FFB3F0',            # Light Magenta
    }


# Version 8: Tokyo Night (inspired by code editor theme)
def get_tokyo_night_colors():
    return {
        'computational_operations': '#7AA2F7',      # Blue
        'symbolic_manipulation': '#BB9AF7',         # Purple
        'variable_value_tracking': '#9ECE6A',       # Green
        'reasoning_flow_control': '#F7768E',        # Red
        'domain_specific_patterns': '#FF9E64',      # Orange
        'relational_logical_operators': '#73DACA',  # Cyan
        'result_formulation': '#E0AF68',            # Yellow
    }


# Version 9: Gradient Fire (yellow to red to purple)
def get_fire_colors():
    return {
        'computational_operations': '#FFEB3B',      # Yellow
        'symbolic_manipulation': '#FFC107',         # Amber
        'variable_value_tracking': '#FF9800',       # Orange
        'reasoning_flow_control': '#FF5722',        # Deep Orange
        'domain_specific_patterns': '#F44336',      # Red
        'relational_logical_operators': '#E91E63',  # Pink
        'result_formulation': '#9C27B0',            # Purple
    }


def get_subcategory_color(category: str, index: int, total: int, category_colors: Dict, style: str = "default") -> str:
    """Generate a shade variation of the category color for subcategories."""
    base_color = category_colors.get(category, '#808080')
    
    # Convert hex to RGB
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    if style in ["cyberpunk", "synthwave", "miami"]:
        # For neon styles, vary the brightness while keeping the vivid color
        factor = 0.4 + (0.6 * (1 - index / max(total - 1, 1)))
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
    elif style == "pastel_dream":
        # Keep it light and vary slightly
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        new_l = max(0.7, min(0.95, l + 0.1 * (1 - index / max(total - 1, 1))))
        r, g, b = colorsys.hls_to_rgb(h, new_l, s * 0.8)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
    else:
        # Default: vary lightness
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        lightness_range = 0.3
        new_l = min(0.9, l + lightness_range * (1 - index / max(total - 1, 1)))
        r, g, b = colorsys.hls_to_rgb(h, new_l, s)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def prepare_donut_data(densities_data: Dict, category_colors: Dict, style: str = "default") -> Tuple[List, List, List, List]:
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
            outer_colors.append(get_subcategory_color(cat, i, len(cat_subcats), category_colors, style))
    
    return inner_labels, inner_values, inner_colors, outer_labels, outer_values, outer_colors


def create_hierarchical_donut(densities_data: Dict, category_colors: Dict, title: str = "", style: str = "default", 
                             bg_color: str = "white", text_color: str = "white") -> go.Figure:
    """Create the hierarchical donut plot."""
    # Prepare data
    (inner_labels, inner_values, inner_colors, 
     outer_labels, outer_values, outer_colors) = prepare_donut_data(densities_data, category_colors, style)
    
    # Create figure
    fig = go.Figure()
    
    # Add outer ring (subcategories)
    # Show all labels without percentages
    outer_text = outer_labels  # Use all labels
    
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.65,
        marker=dict(colors=outer_colors, line=dict(color=bg_color, width=1)),
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
        marker=dict(colors=inner_colors, line=dict(color=bg_color, width=2)),
        textposition='inside',
        textinfo='label+percent',
        textfont=dict(size=14, color=text_color),
        hovertemplate='<b>%{label}</b><br>Density: %{percent}<br><extra></extra>',
        name='Categories',
        sort=False,
        direction='clockwise',
        rotation=90,
        domain=dict(x=[0.15, 0.85], y=[0.15, 0.85])
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title if title else "", font=dict(color='#333' if bg_color == 'white' else '#FFF')),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15,
            font=dict(size=10, color='#333' if bg_color == 'white' else '#FFF')
        ),
        margin=dict(l=0, r=300, t=50 if title else 10, b=0),
        height=700,
        width=1100,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        annotations=[]
    )
    
    return fig


def main():
    """Generate multiple cool versions of the donut plot."""
    
    # Load densities data
    print("Loading activation densities...")
    with open('activation_densities_complete.json', 'r') as f:
        densities_data = json.load(f)
    
    # Define color schemes with their styles and backgrounds
    color_schemes = [
        ("Cyberpunk", get_cyberpunk_colors(), "cyberpunk", "#0a0a0a", "white"),
        ("Sunset", get_sunset_colors(), "default", "white", "white"),
        ("Ocean", get_ocean_colors(), "default", "white", "white"),
        ("Miami", get_miami_colors(), "miami", "#1a1a2e", "white"),
        ("Aurora", get_aurora_colors(), "default", "white", "black"),
        ("Synthwave", get_synthwave_colors(), "synthwave", "#0f0f23", "white"),
        ("Pastel Dream", get_pastel_dream_colors(), "pastel_dream", "white", "black"),
        ("Tokyo Night", get_tokyo_night_colors(), "default", "#1a1b26", "white"),
        ("Fire", get_fire_colors(), "default", "white", "white"),
    ]
    
    # Generate each version
    for name, colors, style, bg_color, text_color in color_schemes:
        print(f"\nGenerating {name} version...")
        fig = create_hierarchical_donut(
            densities_data, 
            colors, 
            title=f"SAE Feature Categories - {name}",
            style=style,
            bg_color=bg_color,
            text_color=text_color
        )
        
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
    
    print("\nAll cool versions generated successfully!")
    print("\nThese include:")
    print("- Cyberpunk: Neon colors on dark background")
    print("- Sunset: Warm gradient colors")
    print("- Ocean: Blues and teals")
    print("- Miami: 80s retro colors")
    print("- Aurora: Northern lights inspired")
    print("- Synthwave: Retrowave aesthetic")
    print("- Pastel Dream: Soft pastel colors")
    print("- Tokyo Night: Code editor theme inspired")
    print("- Fire: Yellow to red to purple gradient")


if __name__ == "__main__":
    main()