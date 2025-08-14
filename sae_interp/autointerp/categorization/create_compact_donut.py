#!/usr/bin/env python3
"""
Create a compact hierarchical donut plot for publication with custom HTML legend.
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


def create_compact_donut(densities_data: Dict, title: str = "Feature Distribution") -> str:
    """Create a compact hierarchical donut plot with custom HTML layout."""
    # Prepare data
    (inner_labels, inner_values, inner_colors, 
     outer_labels, outer_values, outer_colors, legend_mapping) = prepare_donut_data(densities_data)
    
    # Create figure with NO built-in legend
    fig = go.Figure()
    
    # Add outer ring (subcategories)
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.67,  # Equal width rings with center hole
        marker=dict(colors=outer_colors, line=dict(color='white', width=0.5)),
        textposition='auto',
        textinfo='label+percent',  # Show both label and percentage
        textfont=dict(size=8),
        hovertemplate='<b>%{label}</b><br>%{percent}<br><extra></extra>',
        sort=False,
        direction='clockwise',
        rotation=90,
        showlegend=False
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
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),  # Use full space for the donut
        showlegend=False
    ))
    
    # Update layout - no legend, compact
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        showlegend=False,  # No built-in legend
        margin=dict(l=10, r=10, t=40, b=10),  # Minimal margins
        height=400,
        width=400,  # Square plot for the donut
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=[]
    )
    
    # Convert figure to HTML div
    plot_html = fig.to_html(
        include_plotlyjs='cdn',
        div_id="donut-plot",
        config={'displayModeBar': False, 'displaylogo': False, 'responsive': True}
    )
    
    # Create custom HTML legend
    legend_html = create_custom_legend(inner_labels, inner_colors, outer_labels, outer_colors, legend_mapping)
    
    # Combine into full HTML with side-by-side layout
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            background: white;
        }}
        .container {{
            display: flex;
            align-items: flex-start;
            max-width: 800px;
            margin: 0 auto;
        }}
        .plot-container {{
            flex: 0 0 400px;
        }}
        .legend-container {{
            flex: 1;
            padding-left: 20px;
            font-size: 11px;
            columns: 2;
            column-gap: 15px;
        }}
        .legend-category {{
            break-inside: avoid;
            margin-bottom: 8px;
        }}
        .legend-header {{
            font-weight: bold;
            display: flex;
            align-items: center;
            margin-bottom: 3px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-left: 15px;
            margin-bottom: 2px;
        }}
        .color-box {{
            width: 10px;
            height: 10px;
            margin-right: 5px;
            border: 0.5px solid #ccc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="plot-container">
            {plot_html}
        </div>
        <div class="legend-container">
            {legend_html}
        </div>
    </div>
</body>
</html>
"""
    
    return full_html


def create_custom_legend(inner_labels, inner_colors, outer_labels, outer_colors, legend_mapping):
    """Create a custom HTML legend with two columns."""
    legend_html = ""
    
    # Group subcategories by category
    category_groups = {}
    for i, (outer_label, outer_color) in enumerate(zip(outer_labels, outer_colors)):
        parent = legend_mapping.get(outer_label)
        if parent:
            if parent not in category_groups:
                category_groups[parent] = []
            category_groups[parent].append((outer_label, outer_color))
    
    # Create legend entries
    for cat_label, cat_color in zip(inner_labels, inner_colors):
        legend_html += f'''
        <div class="legend-category">
            <div class="legend-header">
                <span class="color-box" style="background-color: {cat_color};"></span>
                {cat_label}
            </div>'''
        
        # Add subcategories if they exist
        if cat_label in category_groups:
            for subcat_label, subcat_color in category_groups[cat_label]:
                # Shorten long labels
                display_label = subcat_label
                if len(display_label) > 20:
                    display_label = display_label[:17] + "..."
                    
                legend_html += f'''
            <div class="legend-item">
                <span class="color-box" style="background-color: {subcat_color};"></span>
                {display_label}
            </div>'''
        
        legend_html += '</div>'
    
    return legend_html


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Create compact hierarchical donut plot")
    parser.add_argument('--input', type=str, default='activation_densities_complete.json',
                       help='Input JSON file with activation densities')
    parser.add_argument('--output', type=str, default='compact_donut.html',
                       help='Output HTML file for visualization')
    parser.add_argument('--title', type=str, default='SAE Feature Activation Distribution',
                       help='Title for the visualization')
    
    args = parser.parse_args()
    
    # Load densities data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        densities_data = json.load(f)
    
    # Create visualization
    print("Creating compact donut plot with custom legend...")
    html_content = create_compact_donut(densities_data, title=args.title)
    
    # Save to HTML
    print(f"Saving visualization to {args.output}...")
    with open(args.output, 'w') as f:
        f.write(html_content)
    
    # Print summary
    print("\nVisualization created successfully!")
    print(f"Open {args.output} in a web browser to view the plot.")
    print("\nFeatures:")
    print("- Donut plot: 400x400px")
    print("- Two-column legend to the right")
    print("- Total width: ~800px (suitable for single column)")
    print("- Compact height: 400px")


if __name__ == "__main__":
    main()