#!/usr/bin/env python3
"""
Automated interpretation for LoRA activations using precomputed examples.
Uses shared utilities from autointerp_utils.py.
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, List, Optional
from autointerp_utils import (
    BaseAutoInterp, 
    generate_explanation,
    CLASSIFICATION_LABELS
)


@dataclass
class LoRAFeature:
    """Represents a LoRA feature."""
    layer: int
    adapter_type: str
    polarity: str
    
    def to_id(self) -> str:
        return f"L{self.layer}_{self.adapter_type}_{self.polarity}"


class LoRAAutoInterp(BaseAutoInterp):
    """Automated interpretation for LoRA features."""
    
    async def process_feature(self,
                             feature_id: str,
                             feature_data: Dict,
                             session: aiohttp.ClientSession) -> Dict:
        """Process a single LoRA feature."""
        examples = feature_data['examples']
        stats = feature_data['stats']
        
        if not examples:
            return {
                "feature_id": feature_id,
                "explanation": "No significant activations found",
                "num_examples": 0,
                "stats": stats
            }
        
        # Generate explanation using shared function
        result = await generate_explanation(
            feature_id=feature_id,
            examples=examples,
            session=session,
            api_key=self.api_key,
            model_name=self.model_name,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens_per_explanation=self.max_tokens_per_explanation,
            max_examples_for_interp=self.max_examples_for_interp,
            activation_threshold=self.activation_threshold,
            request_semaphore=self.request_semaphore,
            random_seed=self.random_seed
        )
        
        # Parse feature ID for structured info
        # Format: L{layer}_{adapter_type}_{polarity}
        parts = feature_id.split('_')
        layer = int(parts[0][1:])  # Remove 'L' prefix
        polarity = parts[-1]  # Last part is always polarity
        adapter_type = '_'.join(parts[1:-1])  # Everything in between is adapter name
        
        # Handle both dict and string responses (for backwards compatibility)
        if isinstance(result, dict):
            explanation = result.get("explanation", "Error")
            classification = result.get("classification", -1)
            classification_reasoning = result.get("classification_reasoning", "")
        else:
            # Fallback for string responses
            explanation = result
            classification = -1
            classification_reasoning = "Legacy string response"
        
        return {
            "feature_id": feature_id,
            "layer": layer,
            "adapter_type": adapter_type,
            "polarity": polarity,
            "explanation": explanation,
            "num_examples": len(examples),
            "classification": classification,
            "classification_label": CLASSIFICATION_LABELS.get(classification, "unknown"),
            "classification_reasoning": classification_reasoning,
            "stats": stats
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA interpretation using precomputed examples")
    parser.add_argument("--examples", default="data/lora_topk_all_features.json", 
                       help="Path to precomputed examples")
    parser.add_argument("--features", nargs="+", help="Specific feature IDs to process")
    parser.add_argument("--output", default="data/lora_interpretations.json", help="Output file")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--model", default="gpt-5-mini", help="Model for explanations")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--max-examples", type=int, default=10, help="Max examples per interpretation")
    
    args = parser.parse_args()
    
    # Initialize interpreter
    interpreter = LoRAAutoInterp(
        examples_file=args.examples,
        api_key=args.api_key,
        model_name=args.model,
        max_concurrent_requests=args.max_concurrent,
        max_examples_for_interp=args.max_examples
    )
    
    # Run interpretation
    asyncio.run(interpreter.run(
        feature_filter=args.features,
        output_path=args.output
    ))


if __name__ == "__main__":
    main()