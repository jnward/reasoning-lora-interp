#!/usr/bin/env python3
"""
Enhanced categorization of SAE features using hierarchical categories with activation examples.

This script takes feature explanations AND activation examples to provide more accurate categorization.
"""

import json
import asyncio
import aiohttp
import os
import sys
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import argparse
from tqdm.asyncio import tqdm as async_tqdm
import backoff


@dataclass
class ActivationExample:
    """Represents an activation example for a feature."""
    tokens: List[str]
    activations: List[float]
    activation_value: float
    token: str  # The max-activating token
    token_idx: int
    rollout_idx: int


@dataclass
class Feature:
    """Represents a feature to be categorized."""
    feature_id: int
    explanation: str
    activation_examples: List[ActivationExample]
    category_name: Optional[str] = None
    subcategory_name: Optional[str] = None
    subcategory_id: Optional[float] = None


@dataclass
class Subcategory:
    """Represents a subcategory."""
    id: float
    name: str
    description: str


@dataclass
class Category:
    """Represents a category with subcategories."""
    id: int
    name: str
    description: str
    subcategories: List[Subcategory]


class HierarchicalCategorizerWithExamples:
    """Categorizes features using hierarchical categories with activation examples."""
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-5",
                 temperature: float = 0.0,
                 max_concurrent: int = 20,
                 max_retries: int = 3,
                 context_window: int = 10):
        """
        Initialize the categorizer.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5)
            temperature: Temperature for generation
            max_concurrent: Maximum concurrent API requests
            max_retries: Maximum retries per request
            context_window: Number of tokens on each side to show for context
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.context_window = context_window
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
        # API endpoint - use responses endpoint for GPT-5 with reasoning
        if "gpt-5" in model:
            self.api_url = "https://api.openai.com/v1/responses"
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
        
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    def format_activation_example(self, example: ActivationExample, max_activation: float) -> str:
        """
        Format a single activation example showing context and activations.
        
        Args:
            example: Activation example to format
            max_activation: Maximum activation value for normalization
            
        Returns:
            Formatted string showing the context and activations
        """
        # Get the index of the max-activating token
        max_idx = example.token_idx % len(example.tokens)
        
        # Calculate context window
        start_idx = max(0, max_idx - self.context_window)
        end_idx = min(len(example.tokens), max_idx + self.context_window + 1)
        
        # Extract context tokens and their activations
        context_tokens = example.tokens[start_idx:end_idx]
        context_activations = example.activations[start_idx:end_idx] if example.activations else []
        
        # Build the context string
        full_text = "".join(context_tokens)
        
        # Find activating tokens (above threshold)
        threshold = max_activation * 0.1  # 10% of max
        activating_tokens = []
        
        if context_activations:
            for i, (token, activation) in enumerate(zip(context_tokens, context_activations)):
                if activation > threshold:
                    # Normalize to 0-10 scale
                    normalized = (activation / max_activation) * 10
                    activating_tokens.append((token.strip(), normalized))
        
        # Sort by activation value (highest first)
        activating_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Format the result
        result_parts = [full_text]
        for token, activation in activating_tokens[:5]:  # Show top 5 activations
            result_parts.append(f"{token} {activation:.2f}")
        
        return "\n".join(result_parts)
    
    def create_prompt(self, feature: Feature, categories: List[Category]) -> str:
        """
        Create categorization prompt for the LLM including activation examples.
        
        Args:
            feature: Feature to categorize
            categories: List of available categories with subcategories
            
        Returns:
            Formatted prompt string
        """
        category_descriptions = []
        for cat in categories:
            subcats = "\n".join([
                f"    - {subcat.name} ({subcat.id}): {subcat.description}"
                for subcat in cat.subcategories
            ])
            category_descriptions.append(
                f"Category: {cat.name}\n"
                f"Description: {cat.description}\n"
                f"Subcategories:\n{subcats}"
            )
        
        categories_text = "\n\n".join(category_descriptions)
        
        # Format activation examples if available
        examples_text = ""
        if feature.activation_examples:
            # Find max activation across all examples
            max_activation = max(ex.activation_value for ex in feature.activation_examples)
            
            # Format up to 5 examples
            formatted_examples = []
            for i, example in enumerate(feature.activation_examples[:5], 1):
                formatted = self.format_activation_example(example, max_activation)
                formatted_examples.append(f"Example {i}:\n{formatted}")
            
            examples_text = "\n\n".join(formatted_examples)
        
        prompt = f"""Categorize this neural network feature into exactly one category and one subcategory.

Feature explanation: "{feature.explanation}"

{f'''Activation examples showing where the feature fires (format: text followed by activating tokens with strengths 0-10):

{examples_text}

These examples show the actual text contexts where this feature activates. The tokens listed below each example are the ones that triggered the feature, with their activation strengths. Use these concrete examples to better understand what the feature is detecting.
''' if examples_text else ''}

Available categories and subcategories:
{categories_text}

Analyze the feature explanation{' and activation examples' if examples_text else ''} to determine which category and subcategory best match its function. The activation examples provide concrete evidence of what triggers this feature.

Reply with ONLY a JSON object in this exact format:
{{"category": "category_name", "subcategory": "subcategory_name", "subcategory_id": subcategory_id_number}}

Your response:"""
        
        return prompt
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def categorize_single(self, 
                                feature: Feature, 
                                categories: List[Category]) -> Tuple[str, str, float]:
        """
        Categorize a single feature using the API.
        
        Args:
            feature: Feature to categorize
            categories: List of available categories with subcategories
            
        Returns:
            Tuple of (category_name, subcategory_name, subcategory_id)
        """
        async with self.semaphore:
            prompt = self.create_prompt(feature, categories)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Different payload structure for GPT-5 responses endpoint
            if "gpt-5" in self.model:
                # Combine system and user messages into a single input for responses endpoint
                full_prompt = f"""You are an expert at categorizing neural network features based on their explanations and activation patterns. You understand mathematical and scientific terminology. Always respond with valid JSON only.

{prompt}"""
                
                payload = {
                    "model": self.model,
                    "input": full_prompt,
                    "reasoning": {
                        "effort": "high"
                    }
                }
            else:
                # Standard chat completions format for other models
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert at categorizing neural network features based on their explanations and activation patterns. You understand mathematical and scientific terminology. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": 2000,  # More tokens for processing examples
                    "n": 1
                }
                
                # Only add temperature if it's not the default (gpt-5-mini only supports default)
                if self.temperature != 1.0 and self.model != "gpt-5-mini":
                    payload["temperature"] = self.temperature
            
            async with self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for complex prompts
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract content from response - different format for responses endpoint
                if "gpt-5" in self.model:
                    # Responses endpoint format - returns list with reasoning and message objects
                    output = data.get("output", [])
                    content = ""
                    
                    # Parse the list of output objects
                    if isinstance(output, list):
                        for item in output:
                            if isinstance(item, dict) and item.get("type") == "message":
                                # Extract text from message content
                                msg_content = item.get("content", [])
                                for content_item in msg_content:
                                    if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                                        content = content_item.get("text", "").strip()
                                        break
                                if content:
                                    break
                    
                    if not content:
                        # Fallback if structure is different
                        print(f"Unexpected response structure: {data}")
                        raise ValueError("Could not extract content from GPT-5 response")
                else:
                    # Chat completions format
                    content = data["choices"][0]["message"]["content"].strip()
                
                try:
                    # Parse JSON response
                    result = json.loads(content)
                    
                    # Validate response format
                    if not all(k in result for k in ["category", "subcategory", "subcategory_id"]):
                        raise ValueError(f"Missing required fields in response: {result}")
                    
                    # Validate that the category and subcategory exist
                    valid = False
                    for cat in categories:
                        if cat.name == result["category"]:
                            for subcat in cat.subcategories:
                                if subcat.name == result["subcategory"] and subcat.id == result["subcategory_id"]:
                                    valid = True
                                    break
                            break
                    
                    if not valid:
                        print(f"\nInvalid categorization for feature {feature.feature_id}")
                        print(f"Feature: {feature.explanation}")
                        print(f"Response: {result}")
                        raise ValueError(f"Invalid category/subcategory combination: {result}")
                    
                    return result["category"], result["subcategory"], result["subcategory_id"]
                    
                except json.JSONDecodeError as e:
                    print(f"\nJSON decode error for feature {feature.feature_id}")
                    print(f"Feature: {feature.explanation}")
                    print(f"Model response: '{content}'")
                    raise ValueError(f"Invalid JSON response: {e}")
    
    async def categorize_all(self, 
                            features: List[Feature], 
                            categories: List[Category]) -> List[Feature]:
        """
        Categorize all features in parallel.
        
        Args:
            features: List of features to categorize
            categories: List of available categories with subcategories
            
        Returns:
            List of features with categorization filled in
        """
        tasks = []
        
        # Create progress bar
        pbar = async_tqdm(total=len(features), desc="Categorizing features")
        
        async def categorize_with_progress(feature: Feature) -> Feature:
            """Categorize and update progress."""
            try:
                cat_name, subcat_name, subcat_id = await self.categorize_single(feature, categories)
                feature.category_name = cat_name
                feature.subcategory_name = subcat_name
                feature.subcategory_id = subcat_id
            except Exception as e:
                print(f"\nError categorizing feature {feature.feature_id}: {e}")
                feature.category_name = "error"
                feature.subcategory_name = "error"
                feature.subcategory_id = -1
            finally:
                pbar.update(1)
            return feature
        
        # Create all tasks
        for feature in features:
            task = asyncio.create_task(categorize_with_progress(feature))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        pbar.close()
        
        return results


def load_hierarchical_categories(filepath: str) -> List[Category]:
    """Load hierarchical categories from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    categories = []
    for cat_data in data["categories"]:
        subcategories = [
            Subcategory(
                id=subcat["id"],
                name=subcat["name"],
                description=subcat["description"]
            )
            for subcat in cat_data["subcategories"]
        ]
        
        categories.append(Category(
            id=cat_data["id"],
            name=cat_data["name"],
            description=cat_data["description"],
            subcategories=subcategories
        ))
    
    return categories


def load_features_with_examples(
    explanations_file: str,
    examples_file: str,
    limit: Optional[int] = None
) -> List[Feature]:
    """
    Load features with their explanations and activation examples.
    
    Args:
        explanations_file: Path to file with feature explanations
        examples_file: Path to file with activation examples
        limit: Optional limit on number of features to load
        
    Returns:
        List of Feature objects with explanations and examples
    """
    # Load explanations
    with open(explanations_file, "r") as f:
        explanations_data = json.load(f)
    
    # Load activation examples
    with open(examples_file, "r") as f:
        examples_data = json.load(f)
    
    features = []
    
    # Process each explanation
    for item in explanations_data["explanations"][:limit]:
        feature_id = item["feature_id"]
        
        # Get activation examples for this feature
        activation_examples = []
        if str(feature_id) in examples_data.get("features", {}):
            feature_data = examples_data["features"][str(feature_id)]
            for ex in feature_data.get("examples", [])[:5]:  # Limit to 5 examples
                activation_examples.append(ActivationExample(
                    tokens=ex.get("tokens", []),
                    activations=ex.get("activations", []),
                    activation_value=ex.get("activation_value", 0),
                    token=ex.get("token", ""),
                    token_idx=ex.get("token_idx", 0),
                    rollout_idx=ex.get("rollout_idx", 0)
                ))
        
        features.append(Feature(
            feature_id=feature_id,
            explanation=item["explanation"],
            activation_examples=activation_examples
        ))
    
    return features


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Categorize features using hierarchical categories with activation examples")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--explanations", type=str, default="../all_interpretations_o3.json",
                       help="Input JSON file with feature explanations")
    parser.add_argument("--examples", type=str, default="../../sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json",
                       help="Input JSON file with activation examples")
    parser.add_argument("--categories", type=str, default="hierarchical_categories.json",
                       help="JSON file with hierarchical category definitions")
    parser.add_argument("--output", type=str, default="hierarchical_categorized_with_examples.json",
                       help="Output JSON file with categorized features")
    parser.add_argument("--model", type=str, default="gpt-5",
                       help="OpenAI model to use")
    parser.add_argument("--max-concurrent", type=int, default=20,
                       help="Maximum concurrent API requests")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for generation")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of features to process (for testing)")
    parser.add_argument("--context-window", type=int, default=10,
                       help="Number of tokens on each side of max activation to show")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Load features with examples
    print(f"Loading features from {args.explanations} and {args.examples}...")
    features = load_features_with_examples(args.explanations, args.examples, args.limit)
    
    if args.limit:
        print(f"Limited to {len(features)} features for testing")
    else:
        print(f"Loaded {len(features)} features")
    
    # Count features with examples
    features_with_examples = sum(1 for f in features if f.activation_examples)
    print(f"  Features with activation examples: {features_with_examples}")
    
    # Load categories
    print(f"Loading categories from {args.categories}...")
    categories = load_hierarchical_categories(args.categories)
    print(f"Loaded {len(categories)} categories with subcategories")
    
    # Categorize features
    print(f"\nStarting categorization with {args.max_concurrent} concurrent requests...")
    start_time = time.time()
    
    async with HierarchicalCategorizerWithExamples(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        context_window=args.context_window
    ) as categorizer:
        categorized_features = await categorizer.categorize_all(features, categories)
    
    elapsed_time = time.time() - start_time
    
    # Load original explanations data to preserve all fields
    with open(args.explanations, "r") as f:
        original_data = json.load(f)
    
    # Prepare output - maintain same format as input with added fields
    output_data = {
        "metadata": {
            **original_data.get("metadata", {}),
            "hierarchical_categorization_with_examples": {
                "model": args.model,
                "num_categories": len(categories),
                "total_subcategories": sum(len(cat.subcategories) for cat in categories),
                "features_with_examples": features_with_examples,
                "context_window": args.context_window,
                "processing_time_seconds": elapsed_time,
                "features_per_second": len(features) / elapsed_time if elapsed_time > 0 else 0
            }
        },
        "explanations": []
    }
    
    # Build output with all original fields plus new categorization
    for i, f in enumerate(categorized_features):
        # Get original item data
        original_item = original_data["explanations"][i] if i < len(original_data["explanations"]) else {
            "feature_id": f.feature_id,
            "explanation": f.explanation
        }
        
        # Add categorization fields
        output_item = {
            **original_item,
            "category_name": f.category_name,
            "subcategory_name": f.subcategory_name,
            "subcategory_id": f.subcategory_id,
            "had_activation_examples": len(f.activation_examples) > 0
        }
        
        output_data["explanations"].append(output_item)
    
    # Count successful categorizations
    successful = sum(1 for f in categorized_features if f.category_name and f.category_name != "error")
    
    # Save output
    print(f"\nSaving categorized features to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\nCategorization complete!")
    print(f"  Total features: {len(features)}")
    print(f"  Successfully categorized: {successful}")
    print(f"  Failed: {len(features) - successful}")
    print(f"  Time taken: {elapsed_time:.2f} seconds")
    print(f"  Features per second: {len(features) / elapsed_time:.2f}" if elapsed_time > 0 else "N/A")
    
    # Show category distribution
    if successful > 0:
        print("\nCategory distribution:")
        category_counts = {}
        subcategory_counts = {}
        
        for f in categorized_features:
            if f.category_name and f.category_name != "error":
                category_counts[f.category_name] = category_counts.get(f.category_name, 0) + 1
                subcat_key = f"{f.category_name}::{f.subcategory_name}"
                subcategory_counts[subcat_key] = subcategory_counts.get(subcat_key, 0) + 1
        
        # Sort by count
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cat_name, count in sorted_cats:
            print(f"\n  {cat_name}: {count} features")
            # Show subcategories for this category
            cat_subcats = [(k.split("::")[1], v) for k, v in subcategory_counts.items() if k.startswith(f"{cat_name}::")]
            cat_subcats.sort(key=lambda x: x[1], reverse=True)
            for subcat_name, subcount in cat_subcats[:5]:  # Show top 5 subcategories
                print(f"    - {subcat_name}: {subcount}")
            if len(cat_subcats) > 5:
                print(f"    ... and {len(cat_subcats) - 5} more subcategories")


if __name__ == "__main__":
    asyncio.run(main())