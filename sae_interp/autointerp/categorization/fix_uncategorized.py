#!/usr/bin/env python3
"""
Fix uncategorized features by manually assigning appropriate categories.
"""

import json

# Manual categorization based on feature explanations
manual_categorizations = {
    55: {
        "explanation": "Detects formatted system prompts and LaTeX/math expressions.",
        "category_name": "symbolic_manipulation",
        "subcategory_name": "latex_commands",
        "subcategory_id": 1.1
    },
    294: {
        "explanation": "Numeric enumeration tokens and parameter labels in lists.",
        "category_name": "reasoning_flow_control",
        "subcategory_name": "step_organization",
        "subcategory_id": 4.3
    },
    663: {
        "explanation": "Metrizable subspace definitions in mathematical context.",
        "category_name": "domain_specific_patterns",
        "subcategory_name": "abstract_algebra",
        "subcategory_id": 5.4
    },
    1334: {
        "explanation": "Math formulas—numbers, operators, variables in arithmetic computations.",
        "category_name": "computational_operations",
        "subcategory_name": "arithmetic_ops",
        "subcategory_id": 3.1
    },
    1920: {
        "explanation": "Tokens labeling numeric parts, axes, and complex components.",
        "category_name": "variable_value_tracking",
        "subcategory_name": "indexed_variables",
        "subcategory_id": 2.3
    },
    2298: {
        "explanation": "Math notation tokens (numbers, variables, LaTeX command fragments).",
        "category_name": "symbolic_manipulation",
        "subcategory_name": "math_operators",
        "subcategory_id": 1.3
    },
    2994: {
        "explanation": "Technical expressions: chemical reactions and math formulas; non‐monosemantic.",
        "category_name": "domain_specific_patterns",
        "subcategory_name": "chemistry_notation",
        "subcategory_id": 5.2
    },
    3033: {
        "explanation": "LaTeX math notation tokens.",
        "category_name": "symbolic_manipulation",
        "subcategory_name": "latex_commands",
        "subcategory_id": 1.1
    }
}

def fix_categorizations():
    """Fix the uncategorized features in the JSON file."""
    
    # Load the categorized features
    print("Loading hierarchical_categorized.json...")
    with open('hierarchical_categorized.json', 'r') as f:
        data = json.load(f)
    
    # Track fixes
    fixed_count = 0
    
    # Process each feature
    for feature in data['explanations']:
        feature_id = feature['feature_id']
        
        # Check if this is one of our uncategorized features
        if feature_id in manual_categorizations:
            old_category = feature.get('category_name', 'None')
            
            # Apply manual categorization
            manual_cat = manual_categorizations[feature_id]
            feature['category_name'] = manual_cat['category_name']
            feature['subcategory_name'] = manual_cat['subcategory_name']
            feature['subcategory_id'] = manual_cat['subcategory_id']
            
            print(f"Fixed feature {feature_id}: {manual_cat['explanation'][:50]}...")
            print(f"  Old: {old_category}")
            print(f"  New: {manual_cat['category_name']} -> {manual_cat['subcategory_name']}")
            fixed_count += 1
    
    # Save the fixed version
    output_file = 'hierarchical_categorized_fixed.json'
    print(f"\nSaving fixed categorizations to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nFixed {fixed_count} features")
    print(f"Saved to {output_file}")
    
    # Verify no uncategorized features remain
    uncategorized_count = sum(1 for f in data['explanations'] 
                             if f.get('category_name') in ['error', 'uncategorized', None])
    print(f"Remaining uncategorized features: {uncategorized_count}")
    
    return output_file

if __name__ == "__main__":
    fix_categorizations()