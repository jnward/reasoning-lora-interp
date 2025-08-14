#!/usr/bin/env python3
"""Test script to generate PNG from plotly figure."""

import json
import plotly.graph_objects as go
from create_hierarchical_donut import prepare_donut_data, CATEGORY_COLORS

# Load data
with open('activation_densities_complete.json', 'r') as f:
    densities_data = json.load(f)

# Prepare data
(inner_labels, inner_values, inner_colors, 
 outer_labels, outer_values, outer_colors, legend_mapping) = prepare_donut_data(densities_data)

# Create figure
fig = go.Figure()

# Add outer ring
fig.add_trace(go.Pie(
    labels=outer_labels,
    values=outer_values,
    hole=0.67,
    marker=dict(colors=outer_colors, line=dict(color='white', width=0.5)),
    textposition='auto',
    textinfo='label+percent',
    textfont=dict(size=9),
    sort=False,
    direction='clockwise',
    rotation=90,
    showlegend=False
))

# Add inner ring
fig.add_trace(go.Pie(
    labels=inner_labels,
    values=inner_values,
    hole=0.33,
    marker=dict(colors=inner_colors, line=dict(color='white', width=1)),
    textposition='inside',
    textinfo='label+percent',
    textfont=dict(size=10, color='white'),
    sort=False,
    direction='clockwise',
    rotation=90,
    domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),
    showlegend=False
))

# Update layout
fig.update_layout(
    title=dict(text="SAE Feature Activation Distribution", font=dict(size=14)),
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20),
    height=450,
    width=450,
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Save as PNG
fig.write_image("test_donut.png", width=450, height=450)
print("Saved test_donut.png")

# Also save with larger size
fig.write_image("test_donut_large.png", width=600, height=600)
print("Saved test_donut_large.png")