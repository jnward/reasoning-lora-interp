#!/usr/bin/env python3
"""
Demo script for context-aware autointerp format.

This demonstrates the format that shows full text context first,
then lists activating tokens with their values below.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import argparse


def load_sae_features(json_path: str) -> Dict:
    """Load SAE features from our collection format."""
    with open(json_path, 'r') as f:
        return json.load(f)


def normalize_to_10_scale(activations: List[float], max_activation: float) -> List[float]:
    """
    Normalize activations to 0-10 range.
    """
    if max_activation == 0:
        return [0.0] * len(activations)
    
    normalized = [(act / max_activation) * 10.0 for act in activations]
    return [min(10.0, max(0.0, act)) for act in normalized]  # Clip to 0-10


def format_with_inline_annotations(tokens: List[str], 
                                   activations: List[float],
                                   max_activation: float,
                                   threshold: float = 0.1,
                                   context_window: int = 16) -> str:
    """
    Format text showing full context first, then list activating tokens below.
    Shows context window around the max activation.
    
    Example output:
    cat and mouse ran around the tree. They quickly
    tree 7.81
    ran 2.30
    around 1.01
    """
    normalized = normalize_to_10_scale(activations, max_activation)
    
    # Find the position of maximum activation
    max_position = activations.index(max(activations))
    
    # Define context window
    start = max(0, max_position - context_window)
    end = min(len(tokens), max_position + context_window + 1)
    
    # Extract tokens and activations in window
    window_tokens = tokens[start:end]
    window_activations = activations[start:end]
    window_normalized = normalized[start:end]
    
    # Build the full text (no annotations)
    text_parts = []
    for i, token in enumerate(window_tokens):
        if i > 0:
            # Add space before tokens that need it
            if not any(token.startswith(p) for p in [".", ",", "!", "?", ":", ";", ")", "]", "}", "'", '"']):
                text_parts.append(" ")
        text_parts.append(token)
    
    full_text = "".join(text_parts)
    
    # Add ellipsis if we truncated
    if start > 0:
        full_text = "... " + full_text
    if end < len(tokens):
        full_text = full_text + " ..."
    
    # Collect activating tokens with their values
    activating_tokens = []
    for token, act, norm in zip(window_tokens, window_activations, window_normalized):
        if norm > threshold:
            activating_tokens.append((token, norm))
    
    # Sort by activation value (highest first)
    activating_tokens.sort(key=lambda x: x[1], reverse=True)
    
    # Format the result
    result_parts = [full_text]
    for token, activation in activating_tokens:
        result_parts.append(f"{token} {activation:.2f}")
    
    return "\n".join(result_parts)


def generate_context_prompt(feature_data: Dict, 
                           feature_id: int,
                           max_examples: int = 12,  # Limit to top 12 examples
                           threshold: float = 0.1,
                           context_window: int = 16) -> str:
    """
    Generate a prompt using the context-aware format.
    """
    # Context-aware prompt template
    PROMPT_TEMPLATE = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document where the neuron activates and describe what it's looking for.

Your explanation should be just a few words or a short phrase. Don't write complete sentences. The neuron might be responding to:
- Individual tokens or specific words
- Common phrases or expressions  
- Abstract concepts or behaviors
- Broader context or topics

The activation format shows the full text first, then lists tokens where the neuron fired along with their activation strengths (0-10 scale). Higher values mean stronger activation.

For example:
cat and mouse ran around the tree. They quickly
tree 7.81
ran 2.30
around 1.01

<neuron_activations>
{activations_str}
</neuron_activations>

Explanation (just a few words):"""
    
    feature_info = feature_data['features'].get(str(feature_id))
    if not feature_info or not feature_info['examples']:
        return None
    
    # Calculate max activation across all examples
    max_activation = max(
        max(example['activations'])
        for example in feature_info['examples']
    )
    
    if max_activation == 0:
        return None
    
    # Format activation examples
    formatted_examples = []
    
    for i, example in enumerate(feature_info['examples'][:max_examples], 1):
        formatted = format_with_inline_annotations(
            example['tokens'],
            example['activations'],
            max_activation,
            threshold,
            context_window
        )
        formatted_examples.append(f"Example {i}:\n{formatted}")
    
    # Join examples with double newlines (OpenAI style)
    activations_str = "\n\n".join(formatted_examples)
    
    # Generate final prompt
    prompt = PROMPT_TEMPLATE.format(activations_str=activations_str)
    
    return prompt


def compare_formats(feature_data: Dict, feature_id: int):
    """
    Show both SAELens format and context format side by side.
    """
    feature_info = feature_data['features'].get(str(feature_id))
    if not feature_info or not feature_info['examples']:
        print(f"No examples for feature {feature_id}")
        return
    
    # Get first example
    example = feature_info['examples'][0]
    tokens = example['tokens']
    activations = example['activations']
    
    # Calculate max for normalization
    max_activation = max(
        max(ex['activations'])
        for ex in feature_info['examples']
    )
    
    if max_activation == 0:
        print(f"No activations for feature {feature_id}")
        return
    
    # Normalize
    normalized = normalize_to_10_scale(activations, max_activation)
    
    print(f"\n{'='*80}")
    print(f"Feature {feature_id} - Format Comparison")
    print(f"{'='*80}\n")
    
    # SAELens format
    print("SAELens Format (tab-separated):\n")
    for token, act in zip(tokens[:20], normalized[:20]):  # First 20 tokens
        print(f"{token}\t{act:.1f}")
    if len(tokens) > 20:
        print("...")
    
    print("\n" + "-"*40 + "\n")
    
    # Context format
    print("Context-Aware Format (inline annotations):\n")
    formatted = format_with_inline_annotations(tokens, activations, max_activation, threshold=0.1, context_window=16)
    # Wrap long lines for display
    words = formatted.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 80:
            print(line)
            line = word
        else:
            line = line + " " + word if line else word
    if line:
        print(line)
    
    print("\n" + "="*80)


def demonstrate_context_format(json_path: str, feature_ids: List[int] = None):
    """
    Demonstrate context-aware prompt format for features.
    """
    print("Loading SAE features...")
    data = load_sae_features(json_path)
    
    if feature_ids is None:
        # Get features with non-zero activations
        feature_ids = []
        for fid_str, finfo in data['features'].items():
            if finfo.get('stats', {}).get('max_activation', 0) > 0:
                feature_ids.append(int(fid_str))
                if len(feature_ids) >= 3:
                    break
    
    print(f"\nDemonstrating context format for features: {feature_ids}\n")
    
    # First show format comparison
    if feature_ids:
        compare_formats(data, feature_ids[0])
    
    # Then show full prompts
    for feature_id in feature_ids:
        print(f"\n\n{'='*80}")
        print(f"Feature {feature_id} - Full Prompt")
        print(f"{'='*80}\n")
        
        prompt = generate_context_prompt(data, feature_id)
        
        if prompt:
            print(prompt)
            print("\n" + "-"*40)
            print("Expected response: A single sentence explanation of what patterns trigger this neuron.")
        else:
            print(f"No valid activations for feature {feature_id}")


def save_context_prompts(json_path: str, output_path: str, max_features: int = None):
    """
    Save prompts in context format for batch processing.
    """
    print("Loading SAE features...")
    data = load_sae_features(json_path)
    
    prompts = []
    feature_count = 0
    
    for fid_str, finfo in data['features'].items():
        if max_features and feature_count >= max_features:
            break
            
        if not finfo.get('examples') or finfo.get('stats', {}).get('max_activation', 0) == 0:
            continue
            
        feature_id = int(fid_str)
        prompt = generate_context_prompt(data, feature_id)
        
        if prompt:
            prompts.append({
                'feature_id': feature_id,
                'prompt': prompt,
                'metadata': {
                    'num_examples': len(finfo['examples']),
                    'max_activation': finfo['stats']['max_activation'],
                    'format': 'context-aware'
                }
            })
            feature_count += 1
    
    print(f"\nGenerated prompts for {len(prompts)} features")
    
    with open(output_path, 'w') as f:
        json.dump({
            'prompts': prompts,
            'format_info': {
                'prompt_type': 'context-aware',
                'activation_range': '0-10',
                'annotation_format': '<<token:activation>>',
                'expected_response': 'single sentence explanation'
            }
        }, f, indent=2)
    
    print(f"Saved prompts to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo context-aware autointerp format")
    parser.add_argument("--input", required=True, help="Path to SAE features JSON")
    parser.add_argument("--features", nargs="+", type=int, help="Specific features to show")
    parser.add_argument("--save-prompts", help="Save all prompts to this file")
    parser.add_argument("--max-features", type=int, help="Max features for batch save")
    parser.add_argument("--threshold", type=float, default=0.1, help="Min activation to annotate")
    
    args = parser.parse_args()
    
    if args.save_prompts:
        save_context_prompts(args.input, args.save_prompts, args.max_features)
    else:
        demonstrate_context_format(args.input, args.features)


if __name__ == "__main__":
    main()
