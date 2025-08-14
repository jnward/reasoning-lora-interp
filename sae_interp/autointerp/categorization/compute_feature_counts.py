#!/usr/bin/env python3
"""
Compute feature counts by category and subcategory for SAE features.

This script counts the number of features in each category/subcategory
rather than their activation densities.
"""

import json
from collections import defaultdict
import argparse


def count_features_by_category(categorized_features):
    """Count features by category and subcategory."""
    # Initialize counters
    category_counts = defaultdict(int)
    subcategory_counts = defaultdict(int)
    
    # Track category-subcategory relationships
    subcategory_to_category = {}
    
    # Count successful categorizations
    successful_count = 0
    total_count = 0
    
    # Process each categorized feature
    for feature_data in categorized_features['explanations']:
        total_count += 1
        category = feature_data.get('category_name', 'uncategorized')
        subcategory = feature_data.get('subcategory_name', 'uncategorized')
        
        # Skip error categories
        if category == 'error' or category is None:
            category = 'uncategorized'
            subcategory = 'uncategorized'
        else:
            successful_count += 1
        
        # Count the feature
        category_counts[category] += 1
        
        # Create unique subcategory key
        subcat_key = f"{category}::{subcategory}"
        subcategory_counts[subcat_key] += 1
        subcategory_to_category[subcat_key] = category
    
    return category_counts, subcategory_counts, subcategory_to_category, successful_count, total_count


def normalize_counts(category_counts, subcategory_counts):
    """Normalize counts to percentages."""
    # Normalize categories
    total_features = sum(category_counts.values())
    if total_features > 0:
        category_normalized = {k: v/total_features for k, v in category_counts.items()}
    else:
        category_normalized = category_counts
    
    # Normalize subcategories within each category
    subcategory_normalized = {}
    
    # Group subcategories by category
    cat_subcats = defaultdict(list)
    for subcat_key in subcategory_counts:
        category = subcat_key.split('::')[0]
        cat_subcats[category].append(subcat_key)
    
    # Normalize within each category
    for category, subcat_keys in cat_subcats.items():
        subcat_sum = sum(subcategory_counts[k] for k in subcat_keys)
        if subcat_sum > 0:
            for subcat_key in subcat_keys:
                # Normalize relative to category total
                subcategory_normalized[subcat_key] = subcategory_counts[subcat_key] / subcat_sum
        else:
            for subcat_key in subcat_keys:
                subcategory_normalized[subcat_key] = 0
    
    return category_normalized, subcategory_normalized


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compute feature counts by category")
    parser.add_argument('--categorized', type=str,
                       default='hierarchical_categorized.json',
                       help='Path to categorized features JSON')
    parser.add_argument('--output', type=str,
                       default='feature_counts.json',
                       help='Output JSON file for counts')
    
    args = parser.parse_args()
    
    # Load categorized features
    print(f"Loading categorized features from {args.categorized}...")
    with open(args.categorized, 'r') as f:
        categorized_features = json.load(f)
    print(f"Loaded {len(categorized_features['explanations'])} categorized features")
    
    # Count features by category
    print("Counting features by category and subcategory...")
    category_counts, subcategory_counts, subcat_to_cat, successful, total = count_features_by_category(
        categorized_features
    )
    
    # Normalize counts
    print("Normalizing counts...")
    category_normalized, subcategory_normalized = normalize_counts(
        category_counts, subcategory_counts
    )
    
    # Convert all values to Python floats for JSON serialization
    category_normalized = {k: float(v) for k, v in category_normalized.items()}
    subcategory_normalized = {k: float(v) for k, v in subcategory_normalized.items()}
    
    # Prepare output data
    output_data = {
        'metadata': {
            'total_features': total,
            'successfully_categorized': successful,
            'failed_categorization': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'data_type': 'feature_counts'
        },
        'category_densities': category_normalized,  # Keep same key name for compatibility
        'subcategory_densities': subcategory_normalized,  # Keep same key name for compatibility
        'subcategory_to_category': subcat_to_cat,
        'raw_totals': {
            'categories': {k: int(v) for k, v in category_counts.items()},
            'subcategories': {k: int(v) for k, v in subcategory_counts.items()}
        }
    }
    
    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\nCategory distribution (by feature count):")
    sorted_cats = sorted(category_normalized.items(), key=lambda x: x[1], reverse=True)
    for cat, percentage in sorted_cats:
        count = category_counts[cat]
        print(f"  {cat}: {count} features ({percentage:.1%})")
    
    print(f"\nTotal features: {total}")
    print(f"Successfully categorized: {successful} ({successful/total:.1%})")
    print(f"Failed categorization: {total - successful} ({(total - successful)/total:.1%})")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()