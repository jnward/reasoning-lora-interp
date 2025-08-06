#!/usr/bin/env python3
"""
Estimate costs for running autointerp on your features.
"""

import json
import argparse

# Approximate costs per 1M tokens (as of late 2024)
# Check https://openrouter.ai/models for latest pricing
MODEL_COSTS = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    "openai/o3-mini": {"input": 1.10, "output": 4.40},  # o3-mini reasoning model
    "openai/o3-mini-high": {"input": 6.60, "output": 26.40},  # o3-mini high reasoning
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "google/gemini-pro-1.5": {"input": 1.25, "output": 5.00},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.59, "output": 0.79},
}

def estimate_costs(json_path: str, model: str = "openai/o3-mini", num_features: int = None):
    """
    Estimate costs for interpreting features.
    """
    # Load data to count tokens
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Count features
    total_features = len(data['features'])
    features_with_examples = sum(1 for f in data['features'].values() if f.get('examples'))
    
    if num_features is None:
        num_features = features_with_examples
    else:
        num_features = min(num_features, features_with_examples)
    
    # Estimate tokens per feature
    # Prompt is ~250 tokens + 12 examples * ~50 tokens each = ~850 tokens
    avg_input_tokens_per_feature = 850
    avg_output_tokens_per_feature = 30  # Brief phrase explanation
    
    # For reasoning models, add extra tokens for reasoning process
    if "o3" in model:
        # o3-mini uses ~1200 tokens per response in our tests
        avg_output_tokens_per_feature = 1200  # Includes reasoning tokens
    
    total_input_tokens = num_features * avg_input_tokens_per_feature
    total_output_tokens = num_features * avg_output_tokens_per_feature
    
    # Calculate costs
    if model in MODEL_COSTS:
        input_cost = (total_input_tokens / 1_000_000) * MODEL_COSTS[model]["input"]
        output_cost = (total_output_tokens / 1_000_000) * MODEL_COSTS[model]["output"]
        total_cost = input_cost + output_cost
        
        print(f"\n=== Cost Estimate for {model} ===")
        print(f"Features to interpret: {num_features}")
        print(f"Estimated input tokens: {total_input_tokens:,}")
        print(f"Estimated output tokens: {total_output_tokens:,}")
        print(f"Input cost: ${input_cost:.2f}")
        print(f"Output cost: ${output_cost:.2f}")
        print(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
        print(f"Cost per feature: ${total_cost/num_features:.4f}")
    else:
        print(f"\nNo pricing data for {model}")
        print(f"Features to interpret: {num_features}")
        print(f"Estimated total tokens: {total_input_tokens + total_output_tokens:,}")
    
    print(f"\n=== Dataset Info ===")
    print(f"Total features in dataset: {total_features}")
    print(f"Features with examples: {features_with_examples}")

def main():
    parser = argparse.ArgumentParser(description="Estimate autointerp costs")
    parser.add_argument("--input", required=True, help="Path to SAE features JSON")
    parser.add_argument("--model", default="openai/gpt-4o", help="Model to use")
    parser.add_argument("--num-features", type=int, help="Number of features (default: all)")
    
    args = parser.parse_args()
    
    print("\n=== Available Models and Pricing (per 1M tokens) ===")
    for model, costs in MODEL_COSTS.items():
        print(f"{model}: ${costs['input']:.2f} input, ${costs['output']:.2f} output")
    
    estimate_costs(args.input, args.model, args.num_features)

if __name__ == "__main__":
    main()
