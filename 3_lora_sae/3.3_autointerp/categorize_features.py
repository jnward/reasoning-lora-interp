#!/usr/bin/env python3
"""
Categorize SAE feature explanations using GPT-5-mini.

This script takes feature explanations from all_interpretations_o3.json
and categorizes them using the categories defined in categories.json.
"""

import json
import asyncio
import aiohttp
import os
import sys
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import argparse
from tqdm.asyncio import tqdm as async_tqdm
import backoff


@dataclass
class Feature:
    """Represents a feature to be categorized."""
    feature_id: int
    explanation: str
    category_id: Optional[str] = None


@dataclass
class Category:
    """Represents a category."""
    id: str
    label: str


class FeatureCategorizer:
    """Categorizes features using GPT-5-mini."""
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-5-mini",
                 temperature: float = 0.0,
                 max_concurrent: int = 20,
                 max_retries: int = 3):
        """
        Initialize the categorizer.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5-mini)
            temperature: Temperature for generation
            max_concurrent: Maximum concurrent API requests
            max_retries: Maximum retries per request
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
        # API endpoint
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    def create_prompt(self, feature: Feature, categories: List[Category]) -> str:
        """
        Create categorization prompt for GPT-5-mini.
        
        Args:
            feature: Feature to categorize
            categories: List of available categories
            
        Returns:
            Formatted prompt string
        """
        category_list = "\n".join([
            f"- {cat.id}: {cat.label}"
            for cat in categories
        ])
        
        prompt = f"""Categorize this neural network feature into one of the given categories.

Feature: "{feature.explanation}"

Categories:
{category_list}

Reply with ONLY the category ID string, nothing else. Examples of valid responses:
- latex_tex_markup
- numerals_literals
- geometry_coordinates

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
                                categories: List[Category]) -> str:
        """
        Categorize a single feature using the API.
        
        Args:
            feature: Feature to categorize
            categories: List of available categories
            
        Returns:
            Category ID string
        """
        async with self.semaphore:
            prompt = self.create_prompt(feature, categories)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You categorize features. Respond with EXACTLY one category ID from the provided list. No explanations, no punctuation, just the category ID string."},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 1000,  # Increased to account for reasoning tokens
                "n": 1
            }
            
            # Only add temperature if it's not the default (GPT-5-mini only supports default)
            if self.temperature != 1.0 and self.model != "gpt-5-mini":
                payload["temperature"] = self.temperature
            
            async with self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract category ID from response
                content = data["choices"][0]["message"]["content"].strip()
                
                # Debug logging for empty responses
                if not content:
                    print(f"\nEmpty response for feature {feature.feature_id}: {feature.explanation[:50]}...")
                    print(f"Full API response: {data}")
                
                # Validate that it's a valid category ID
                valid_ids = {cat.id for cat in categories}
                if content not in valid_ids:
                    # Try to extract a valid ID if there's extra text
                    found = False
                    for cat_id in valid_ids:
                        if cat_id in content:
                            content = cat_id
                            found = True
                            break
                    
                    if not found:
                        # Log the problematic response for debugging
                        print(f"\nInvalid response for feature {feature.feature_id}")
                        print(f"Feature explanation: {feature.explanation}")
                        print(f"Model response: '{content}'")
                        raise ValueError(f"Invalid category ID returned: {content}")
                
                return content
    
    async def categorize_all(self, 
                            features: List[Feature], 
                            categories: List[Category]) -> List[Feature]:
        """
        Categorize all features in parallel.
        
        Args:
            features: List of features to categorize
            categories: List of available categories
            
        Returns:
            List of features with category_id filled in
        """
        tasks = []
        
        # Create progress bar
        pbar = async_tqdm(total=len(features), desc="Categorizing features")
        
        async def categorize_with_progress(feature: Feature) -> Feature:
            """Categorize and update progress."""
            try:
                category_id = await self.categorize_single(feature, categories)
                feature.category_id = category_id
            except Exception as e:
                print(f"\nError categorizing feature {feature.feature_id}: {e}")
                feature.category_id = "error"
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


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Categorize features using GPT-5-mini")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--input", type=str, default="all_interpretations_o3.json",
                       help="Input JSON file with feature explanations")
    parser.add_argument("--categories", type=str, default="categories.json",
                       help="JSON file with category definitions")
    parser.add_argument("--output", type=str, default="categorized_features.json",
                       help="Output JSON file with categorized features")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                       help="OpenAI model to use")
    parser.add_argument("--max-concurrent", type=int, default=20,
                       help="Maximum concurrent API requests")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Load input data
    print(f"Loading features from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Extract features
    features = []
    for item in data["explanations"]:
        features.append(Feature(
            feature_id=item["feature_id"],
            explanation=item["explanation"]
        ))
    
    print(f"Loaded {len(features)} features")
    
    # Load categories
    print(f"Loading categories from {args.categories}...")
    with open(args.categories, "r") as f:
        category_data = json.load(f)
    
    categories = [Category(id=cat["id"], label=cat["label"]) for cat in category_data]
    print(f"Loaded {len(categories)} categories")
    
    # Categorize features
    print(f"\nStarting categorization with {args.max_concurrent} concurrent requests...")
    start_time = time.time()
    
    async with FeatureCategorizer(
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent
    ) as categorizer:
        categorized_features = await categorizer.categorize_all(features, categories)
    
    elapsed_time = time.time() - start_time
    
    # Prepare output
    output_data = {
        "metadata": {
            **data.get("metadata", {}),
            "categorization": {
                "model": args.model,
                "num_categories": len(categories),
                "processing_time_seconds": elapsed_time,
                "features_per_second": len(features) / elapsed_time
            }
        },
        "explanations": [
            {
                "feature_id": f.feature_id,
                "explanation": f.explanation,
                "category_id": f.category_id
            }
            for f in categorized_features
        ]
    }
    
    # Count successful categorizations
    successful = sum(1 for f in categorized_features if f.category_id and f.category_id != "error")
    
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
    print(f"  Features per second: {len(features) / elapsed_time:.2f}")
    
    # Show category distribution
    if successful > 0:
        print("\nCategory distribution:")
        category_counts = {}
        for f in categorized_features:
            if f.category_id and f.category_id != "error":
                category_counts[f.category_id] = category_counts.get(f.category_id, 0) + 1
        
        # Sort by count
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get category labels
        cat_labels = {cat.id: cat.label for cat in categories}
        
        for cat_id, count in sorted_counts[:10]:  # Show top 10
            label = cat_labels.get(cat_id, "Unknown")
            print(f"  {cat_id} ({label}): {count}")
        
        if len(sorted_counts) > 10:
            print(f"  ... and {len(sorted_counts) - 10} more categories")


if __name__ == "__main__":
    asyncio.run(main())