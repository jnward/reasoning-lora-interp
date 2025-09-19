#!/usr/bin/env python3
"""
Context-aware automated interpretation for SAE features.

This implementation shows the full context first, then lists activating tokens
with their activation values below, rather than using inline annotations.

Format:
cat and mouse ran around the tree. They quickly
tree 7.81
ran 2.30
around 1.01
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
from tqdm import tqdm
import os


@dataclass
class ActivationRecord:
    """Represents tokens and their activations for a feature."""
    tokens: List[str]
    activations: List[float]
    rollout_idx: Optional[int] = None
    activation_value: Optional[float] = None


@dataclass
class ExplanationResult:
    """Result of automated interpretation."""
    feature_id: int
    explanation: str
    score: Optional[float] = None
    activation_records: Optional[List[ActivationRecord]] = None


class AutoInterpContext:
    """
    Context-aware automated interpretation for SAE features.
    
    This shows the full text context first, then lists activating tokens
    with their activation strengths below, making it easier to see patterns
    in what the feature responds to.
    """
    
    # Modified prompt template with context-aware format
    EXPLANATION_PROMPT_TEMPLATE = """\
We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document where the neuron activates and describe what it's firing for.

Some activations will be noisy, in these cases you'll have to look for common phrases or concepts in the examples.

If a feature always activates for the same token, you should note this in your explanation, and also state whether that feature represents that token in some specific context. You may need to look at words surrounding activating tokens in order to understand why a feature is firing.
Features should have a clear, singular explanation and should be monosemantic. If there isn't a clear monosemantic explanation, note this.

Your explanation should not exceed ten words. Don't write complete sentences. The neuron might be responding to:
- Individual tokens or specific words
- Phrases or expressions  
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

What is this neuron looking for? Answer in just a few words:"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "openai/o3-mini",  # Default to o3-mini reasoning model
                 api_base: str = "https://openrouter.ai/api/v1",
                 temperature: float = 0.0,
                 max_tokens_per_explanation: int = 60,
                 num_samples: int = 20,
                 max_explanation_activation_records: int = 12,  # Limit to top 12 examples
                 activation_threshold: float = 0.1,  # Only show activations above this
                 context_window: int = 16,  # Tokens on each side of max activation
                 use_reasoning_mode: bool = True,  # Enable reasoning mode for CoT models
                 max_examples_per_rollout: int = 4):  # Max examples from same rollout
        """
        Initialize the context-aware interpretation system.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model_name: Model to use for generating explanations (OpenRouter format)
            api_base: API base URL (default: OpenRouter)
            temperature: Temperature for generation (default: 0.0 for deterministic)
            max_tokens_per_explanation: Max tokens in explanation
            num_samples: Number of activation samples to use
            max_explanation_activation_records: Max records for explanation
            activation_threshold: Minimum activation to annotate (normalized to 0-10)
            context_window: Tokens on each side of max activation
            use_reasoning_mode: Enable reasoning mode for CoT models like o3-mini
            max_examples_per_rollout: Maximum examples from same rollout for diversity
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required (set OPENROUTER_API_KEY or OPENAI_API_KEY)")
            
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens_per_explanation = max_tokens_per_explanation
        self.num_samples = num_samples
        self.max_explanation_activation_records = max_explanation_activation_records
        self.activation_threshold = activation_threshold
        self.context_window = context_window
        self.use_reasoning_mode = use_reasoning_mode
        self.max_examples_per_rollout = max_examples_per_rollout
        
    def normalize_activations(self, activations: np.ndarray, max_activation: float) -> np.ndarray:
        """
        Normalize activations to 0-10 range.
        
        Args:
            activations: Raw activation values
            max_activation: Maximum activation value for normalization
            
        Returns:
            Normalized activations in 0-10 range
        """
        if max_activation == 0:
            return np.zeros_like(activations)
            
        # Normalize to 0-10 range
        normalized = (activations / max_activation) * 10.0
        
        # Clip to ensure we're in 0-10 range
        normalized = np.clip(normalized, 0, 10)
        
        return normalized
    
    def format_activation_context(self, 
                                  tokens: List[str], 
                                  activations: List[float],
                                  max_activation: float) -> str:
        """
        Format tokens showing full context first, then list activating tokens below.
        Shows context window around the max activation.
        
        Args:
            tokens: List of tokens
            activations: Activation values for each token
            max_activation: Maximum activation for normalization
            
        Returns:
            String with full text followed by activating tokens and their values
        """
        normalized = self.normalize_activations(np.array(activations), max_activation)
        
        # Find the position of maximum activation
        max_position = np.argmax(activations)
        
        # Define context window
        start = max(0, max_position - self.context_window)
        end = min(len(tokens), max_position + self.context_window + 1)
        
        # Extract tokens and activations in window
        window_tokens = tokens[start:end]
        window_activations = activations[start:end]
        window_normalized = normalized[start:end]
        
        # Build the full text (no annotations)
        text_parts = []
        for i, token in enumerate(window_tokens):
            if i > 0:
                # Add space before tokens that need it
                if not any(token.startswith(p) for p in [".", ",", "!", "?", ":", ";", ")", "]", "}"]):
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
            if norm > self.activation_threshold:
                activating_tokens.append((token, norm))
        
        # Sort by activation value (highest first)
        activating_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Format the result
        result_parts = [full_text]
        for token, activation in activating_tokens:
            result_parts.append(f"{token} {activation:.2f}")
        
        return "\n".join(result_parts)
    
    def format_activations_for_prompt(self, 
                                      activation_records: List[ActivationRecord],
                                      max_activation: float) -> str:
        """
        Format activation records with context-aware inline annotations.
        Uses numbered examples format from OpenAI's implementation.
        
        Args:
            activation_records: List of activation records
            max_activation: Maximum activation for normalization
            
        Returns:
            Formatted string for the prompt
        """
        formatted_examples = []
        
        for i, record in enumerate(activation_records[:self.max_explanation_activation_records], 1):
            # Format this example with inline annotations
            formatted = self.format_activation_context(
                record.tokens,
                record.activations,
                max_activation
            )
            formatted_examples.append(f"Example {i}:\n{formatted}")
        
        # Join examples with double newlines
        return "\n\n".join(formatted_examples)
    
    async def generate_explanation(self, 
                                   activation_records: List[ActivationRecord],
                                   max_activation: float,
                                   feature_id: Optional[int] = None) -> str:
        """
        Generate explanation for a feature based on activation records.
        
        Args:
            activation_records: List of activation records for the feature
            max_activation: Maximum activation value
            
        Returns:
            Generated explanation
        """
        # Format activations for prompt
        activations_str = self.format_activations_for_prompt(
            activation_records, 
            max_activation
        )
        
        # Build prompt
        prompt = self.EXPLANATION_PROMPT_TEMPLATE.format(
            activations_str=activations_str
        )
        
        # Call API (OpenRouter or OpenAI compatible)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add OpenRouter specific headers if using OpenRouter
        if "openrouter" in self.api_base:
            headers["HTTP-Referer"] = "https://github.com/your-repo"  # Optional but recommended
            headers["X-Title"] = "SAE Feature Interpretation"  # Optional
        
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens_per_explanation,
            "n": 1
        }
        
        # Add parameters for o3 models - they REQUIRE include_reasoning on OpenRouter
        if "o3" in self.model_name:
            # o3 models on OpenRouter require include_reasoning=true to work properly
            payload["include_reasoning"] = True
            # o3-mini needs at least 1000 tokens to work with our prompt
            payload["max_tokens"] = 4000  # Higher limit for more complex reasoning
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                
        if "error" in result:
            raise Exception(f"API error: {result['error']}")
            
        # Debug: Print raw response for o3 models
        if "o3" in self.model_name and self.use_reasoning_mode:
            print(f"\nDEBUG - Raw API response for feature {feature_id}:")
            print(f"Model: {self.model_name}")
            print(f"Response keys: {result.keys()}")
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                print(f"Choice keys: {choice.keys()}")
                if "message" in choice:
                    print(f"Message keys: {choice['message'].keys()}")
                    print(f"Content preview: {choice['message'].get('content', '')[:200]}...")
                    if "reasoning_content" in choice["message"]:
                        print(f"Reasoning content present: {len(choice['message']['reasoning_content'])} chars")
        
        # Extract explanation - handle potential reasoning content
        if "choices" not in result or not result["choices"]:
            print(f"WARNING: No choices in response for feature")
            return "No explanation generated"
            
        choice = result["choices"][0]
        if "message" not in choice:
            print(f"WARNING: No message in choice for feature")
            return "No explanation generated"
            
        # For o3 models with reasoning, the actual answer might be in a different field
        message = choice["message"]
        content = message.get("content", "").strip()
        
        # Check if there's a reasoning field
        reasoning = message.get("reasoning", "")
        
        # Try to extract explanation from content
        explanation = ""
        
        # Method 1: Direct content
        if content:
            # Sometimes the model returns the full prompt + answer
            prompts_to_check = [
                "What is this neuron looking for?",
                "What is this neuron looking for? Answer in just a few words:",
                "Explanation (just a few words):",
                "Answer in just a few words:"
            ]
            
            for prompt_end in prompts_to_check:
                if prompt_end in content:
                    parts = content.split(prompt_end)
                    if len(parts) > 1:
                        explanation = parts[-1].strip()
                        break
            
            if not explanation:
                # Use the full content as explanation
                explanation = content
        
        # Method 2: Check reasoning field if no explanation yet
        if not explanation and reasoning:
            for prompt_end in prompts_to_check:
                if prompt_end in reasoning:
                    parts = reasoning.split(prompt_end)
                    if len(parts) > 1:
                        explanation = parts[-1].strip()
                        break
        
        # Clean up the explanation - remove any trailing prompt artifacts
        if explanation:
            # Remove common endings that might be included
            for suffix in ["</neuron_activations>", "Give a short explanation", "Explanation:"]:
                if suffix in explanation:
                    explanation = explanation.split(suffix)[0].strip()
        
        if not explanation:
            print(f"WARNING: Empty explanation for feature {feature_id}")
            print(f"Full content was: {content[:500]}...")
            
        return explanation
    
    def load_activation_data(self, json_path: str) -> Dict[int, List[ActivationRecord]]:
        """
        Load activation data from our SAE feature collection format.
        
        Args:
            json_path: Path to JSON file with feature data
            
        Returns:
            Dictionary mapping feature IDs to activation records
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        features_data = {}
        
        for feature_id_str, feature_info in data['features'].items():
            feature_id = int(feature_id_str)
            activation_records = []
            
            # Convert our format to ActivationRecord format
            for example in feature_info['examples']:
                record = ActivationRecord(
                    tokens=example['tokens'],
                    activations=example['activations'],
                    rollout_idx=example.get('rollout_idx'),
                    activation_value=example.get('activation_value')
                )
                activation_records.append(record)
                
            features_data[feature_id] = activation_records
            
        return features_data
    
    def ensure_rollout_diversity(self, 
                                activation_records: List[ActivationRecord], 
                                max_per_rollout: int = 4) -> List[ActivationRecord]:
        """
        Ensure diversity by limiting examples from the same rollout.
        
        Args:
            activation_records: List of activation records
            max_per_rollout: Maximum examples from same rollout (default: 4)
            
        Returns:
            Filtered list with rollout diversity
        """
        # If no rollout info, return as is
        if not activation_records or activation_records[0].rollout_idx is None:
            return activation_records
        
        # Count examples per rollout
        rollout_counts = {}
        diverse_records = []
        
        # Sort by activation value (highest first) to keep best examples
        sorted_records = sorted(activation_records, 
                              key=lambda r: r.activation_value or 0, 
                              reverse=True)
        
        for record in sorted_records:
            rollout_id = record.rollout_idx
            
            # Track how many we've seen from this rollout
            if rollout_id not in rollout_counts:
                rollout_counts[rollout_id] = 0
            
            # Add if under limit
            if rollout_counts[rollout_id] < max_per_rollout:
                diverse_records.append(record)
                rollout_counts[rollout_id] += 1
        
        # Log if we filtered anything
        if len(diverse_records) < len(activation_records):
            num_rollouts = len(rollout_counts)
            print(f"  Diversity filter: {len(activation_records)} → {len(diverse_records)} examples "
                  f"(from {num_rollouts} rollouts)")
        
        return diverse_records
    
    async def autointerp_features(self,
                                  activation_data_path: str,
                                  output_path: str,
                                  feature_ids: Optional[List[int]] = None):
        """
        Run automated interpretation on features.
        
        Args:
            activation_data_path: Path to activation data JSON
            output_path: Path to save results
            feature_ids: Specific features to interpret (None = all)
        """
        # Load activation data
        print("Loading activation data...")
        features_data = self.load_activation_data(activation_data_path)
        
        # Filter to requested features
        if feature_ids is not None:
            features_data = {fid: features_data[fid] 
                            for fid in feature_ids if fid in features_data}
        
        print(f"Interpreting {len(features_data)} features...")
        
        results = []
        
        # Process each feature
        for feature_id, activation_records in tqdm(features_data.items()):
            if not activation_records:
                continue
                
            # Calculate max activation for normalization
            max_activation = max(
                max(record.activations) 
                for record in activation_records
            )
            
            if max_activation == 0:
                continue
            
            # Apply rollout diversity filter first
            diverse_records = self.ensure_rollout_diversity(activation_records, 
                                                           max_per_rollout=self.max_examples_per_rollout)
            
            # Then limit number of records used for explanation
            records_for_explanation = diverse_records[:self.max_explanation_activation_records]
            
            try:
                # Generate explanation
                explanation = await self.generate_explanation(
                    records_for_explanation,
                    max_activation,
                    feature_id=feature_id
                )
                
                result = ExplanationResult(
                    feature_id=feature_id,
                    explanation=explanation,
                    activation_records=activation_records
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing feature {feature_id}: {e}")
                continue
        
        # Save results
        print(f"Saving results to {output_path}...")
        output_data = {
            "metadata": {
                "format": "context-aware",
                "model_name": self.model_name,
                "api_base": self.api_base,
                "num_features": len(results)
            },
            "explanations": [
                {
                    "feature_id": r.feature_id,
                    "explanation": r.explanation
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Done! Interpreted {len(results)} features.")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context-aware automated interpretation")
    parser.add_argument("--input", required=True, help="Path to activation data JSON")
    parser.add_argument("--output", required=True, help="Path to save explanations")
    parser.add_argument("--features", nargs="+", type=int, help="Specific features to interpret")
    parser.add_argument("--api-key", help="API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="openai/o3-mini", help="Model for explanations (OpenRouter format)")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1", help="API base URL")
    parser.add_argument("--threshold", type=float, default=0.1, help="Minimum activation to show")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning mode for CoT models")
    parser.add_argument("--max-per-rollout", type=int, default=4, help="Max examples from same rollout")
    
    args = parser.parse_args()
    
    # Initialize autointerp
    autointerp = AutoInterpContext(
        api_key=args.api_key,
        model_name=args.model,
        api_base=args.api_base,
        activation_threshold=args.threshold,
        use_reasoning_mode=not args.no_reasoning,
        max_examples_per_rollout=args.max_per_rollout
    )
    
    # Run interpretation
    asyncio.run(autointerp.autointerp_features(
        activation_data_path=args.input,
        output_path=args.output,
        feature_ids=args.features
    ))


if __name__ == "__main__":
    main()
