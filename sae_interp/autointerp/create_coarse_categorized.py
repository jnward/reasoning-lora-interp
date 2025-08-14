#!/usr/bin/env python3
"""
Remap fine-grained categories to coarse categories.
"""

import json

# Mapping from fine to coarse categories
FINE_TO_COARSE = {
    "latex_tex_markup": "math_notation_formatting",
    "delimiters_layout": "math_notation_formatting",
    "equality_assignment": "arithmetic_algebra",
    "arithmetic_ops": "arithmetic_algebra",
    "powers_indices": "arithmetic_algebra",
    "numerals_literals": "arithmetic_algebra",
    "numeric_formats_approx": "arithmetic_algebra",
    "variables_identifiers": "arithmetic_algebra",
    "elementary_functions_ops": "functions_calculus",
    "calculus_markers": "functions_calculus",
    "linear_algebra_matrices": "geometry_linear_algebra",
    "sequences_recurrences_algorithms": "discrete_structures",
    "combinatorics_factorials": "discrete_structures",
    "number_theory_modular": "discrete_structures",
    "geometry_coordinates": "geometry_linear_algebra",
    "abstract_algebra_structures": "discrete_structures",
    "problem_scaffolding_directives": "problem_scaffolding_interaction",
    "discourse_metacognition_markers": "discourse_metacognition",
    "final_answer_boxing": "problem_scaffolding_interaction",
    "physics_variables_units": "physical_sciences_notation",
    "chemistry_nomenclature_mechanisms": "chemistry_nomenclature_mechanisms",
    "quantum_dirac_notation": "physical_sciences_notation",
    "sets_logic_implication": "discrete_structures",
    "measurement_data_scores": "physical_sciences_notation",
    "meta_prompt_conversation_tokens": "problem_scaffolding_interaction",
    "puzzles_wordplay": "problem_scaffolding_interaction",
    "biology_genomics_notation": "life_sciences_notation",
    "non_english_unicode_technical_tokens": "math_notation_formatting",
    "named_theorems_identities_canonical": "discrete_structures",
    "noise_mistokenization": "math_notation_formatting"
}

def main():
    # Load the fine-grained categorized features
    with open('categorized_features.json', 'r') as f:
        fine_data = json.load(f)
    
    # Create coarse categorized version
    coarse_data = {
        "metadata": fine_data["metadata"].copy()
    }
    
    # Update metadata
    coarse_data["metadata"]["categorization"]["type"] = "coarse"
    coarse_data["metadata"]["categorization"]["num_categories"] = 10
    coarse_data["metadata"]["categorization"]["fine_to_coarse_mapping"] = FINE_TO_COARSE
    
    # Remap explanations to coarse categories
    coarse_explanations = []
    remapped_count = 0
    unchanged_count = 0
    
    for item in fine_data["explanations"]:
        new_item = item.copy()
        old_category = item.get("category_id", "uncategorized")
        
        if old_category in FINE_TO_COARSE:
            new_item["category_id"] = FINE_TO_COARSE[old_category]
            new_item["fine_category_id"] = old_category  # Keep track of original
            remapped_count += 1
        else:
            # Keep as is if not in mapping (e.g., "uncategorized")
            unchanged_count += 1
        
        coarse_explanations.append(new_item)
    
    coarse_data["explanations"] = coarse_explanations
    
    # Save the coarse categorized features
    with open('coarse_categorized_features.json', 'w') as f:
        json.dump(coarse_data, f, indent=2)
    
    print(f"Created coarse_categorized_features.json")
    print(f"  - Remapped {remapped_count} features to coarse categories")
    print(f"  - Left {unchanged_count} features unchanged")
    
    # Print category distribution
    category_counts = {}
    for item in coarse_explanations:
        cat = item.get("category_id", "uncategorized")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCoarse category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} features")

if __name__ == "__main__":
    main()