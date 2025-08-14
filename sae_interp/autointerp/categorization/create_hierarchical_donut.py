#!/usr/bin/env python3
"""
Create a compact hierarchical donut plot for publication in a single-column paper.
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


def create_publication_donut(densities_data: Dict, title: str = "Feature Distribution") -> go.Figure:
    """Create a compact hierarchical donut plot for publication."""
    # Prepare data
    (inner_labels, inner_values, inner_colors, 
     outer_labels, outer_values, outer_colors, legend_mapping) = prepare_donut_data(densities_data)
    
    # Create figure with compact height
    fig = go.Figure()
    
    # Add outer ring (subcategories) - equal width rings
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.67,  # Equal width rings with center hole
        marker=dict(colors=outer_colors, line=dict(color='white', width=0.5)),
        textposition='auto',
        textinfo='label+percent',  # Show both label and percentage
        textfont=dict(size=9),
        hovertemplate='<b>%{label}</b><br>%{percent}<br><extra></extra>',
        name='',  # Empty name for legend
        sort=False,
        direction='clockwise',
        rotation=90,
        showlegend=False  # Don't show in legend
    ))
    
    # Add inner ring (categories) - equal width rings
    fig.add_trace(go.Pie(
        labels=inner_labels,
        values=inner_values,
        hole=0.33,  # Center hole for equal width rings
        marker=dict(colors=inner_colors, line=dict(color='white', width=1)),
        textposition='inside',
        textinfo='label+percent',  # Show both label and percentage
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{label}</b><br>%{percent}<br><extra></extra>',
        name='',
        sort=False,
        direction='clockwise',
        rotation=90,
        domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),  # Larger donut
        showlegend=False  # Don't show in legend
    ))
    
    # Create full hierarchical legend with categories and subcategories
    from plotly.subplots import make_subplots
    
    # We'll add dummy scatter traces for the legend
    # Group subcategories by their parent category
    category_groups = {}
    for i, (subcat, parent_cat) in enumerate(legend_mapping.items()):
        if parent_cat not in category_groups:
            category_groups[parent_cat] = []
        subcat_idx = outer_labels.index(subcat)
        category_groups[parent_cat].append((subcat, outer_colors[subcat_idx]))
    
    # Calculate total legend items for splitting into columns
    total_items = 0
    for cat in inner_labels:
        total_items += 1  # Category header
        if cat in category_groups:
            total_items += len(category_groups[cat])  # Subcategories
    
    # Split into two columns
    items_per_column = (total_items + 1) // 2
    current_items = 0
    column = 1
    
    # Add legend entries with column assignment
    for cat_idx, cat in enumerate(inner_labels):
        cat_color = inner_colors[cat_idx]
        
        # Check if we should switch to column 2
        if current_items >= items_per_column and column == 1:
            column = 2
        
        # Add category header
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=cat_color, symbol='square'),
            legendgroup=f"col{column}_{cat}",
            legendgrouptitle_text=None,
            name=f"<b>{cat}</b>",
            showlegend=True,
            legendrank=column * 1000 + current_items  # Control ordering
        ))
        current_items += 1
        
        # Add subcategories
        if cat in category_groups:
            for subcat, subcat_color in category_groups[cat]:
                # Check if we should switch to column 2
                if current_items >= items_per_column and column == 1:
                    column = 2
                
                # Shorten long names
                display_name = subcat
                if len(display_name) > 18:
                    display_name = display_name[:15] + "..."
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=8, color=subcat_color, symbol='circle'),
                    legendgroup=f"col{column}_{cat}",
                    name=f"  {display_name}",  # Indent
                    showlegend=True,
                    legendrank=column * 1000 + current_items  # Control ordering
                ))
                current_items += 1
    
    # Update layout for publication
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14),  # Smaller title
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left", 
            x=1.02,  # Move legend to the right
            font=dict(size=8),  # Small but readable font size
            itemsizing='constant',
            tracegroupgap=2,  # Small gap between items
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#CCCCCC',
            borderwidth=0.5
        ),
        margin=dict(l=20, r=300, t=40, b=20),  # More right margin for two-column legend
        height=450,  # Reasonable height
        width=800,   # Wider to accommodate legend columns
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=[]  # No center text
    )
    
    # Update xaxis and yaxis to be invisible (for legend traces)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Create publication-ready hierarchical donut plot")
    parser.add_argument('--input', type=str, default='activation_densities_complete.json',
                       help='Input JSON file with activation densities')
    parser.add_argument('--output', type=str, default='publication_donut.html',
                       help='Output HTML file for visualization')
    parser.add_argument('--title', type=str, default='SAE Feature Activation Distribution',
                       help='Title for the visualization')
    parser.add_argument('--counts', action='store_true',
                       help='Use feature counts instead of activation densities')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.counts:
        input_file = 'feature_counts_fixed.json'
        default_title = 'SAE Feature Count Distribution'
    else:
        input_file = args.input
        default_title = args.title
    
    # Load densities data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        densities_data = json.load(f)
    
    # Create visualization
    print("Creating publication-ready donut plot...")
    fig = create_publication_donut(densities_data, title=default_title)
    
    # Save to HTML
    print(f"Saving visualization to {args.output}...")
    fig.write_html(
        args.output,
        include_plotlyjs='cdn',
        config={'displayModeBar': False, 'displaylogo': False, 'responsive': True}
    )
    
    # Also save as PNG for preview
    png_output = args.output.replace('.html', '.png')
    try:
        fig.write_image(png_output, width=800, height=450)
        print(f"Also saved as PNG: {png_output}")
    except Exception as e:
        print(f"Could not save PNG (kaleido may not be installed): {e}")
    
    # Print summary
    print("\nVisualization created successfully!")
    print(f"Open {args.output} in a web browser to view the plot.")
    
    # Print category summary
    category_densities = densities_data['category_densities']
    print("\nCategory Distribution:")
    sorted_cats = sorted(category_densities.items(), key=lambda x: x[1], reverse=True)
    for cat, density in sorted_cats:
        print(f"  {cat.replace('_', ' ').title()}: {density:.1%}")


if __name__ == "__main__":
    main()