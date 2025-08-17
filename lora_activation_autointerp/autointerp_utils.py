#!/usr/bin/env python3
"""
Shared utilities for automated interpretation of neural network features.
Contains common logic used by both LoRA and MLP autointerp scripts.
"""

import json
import numpy as np
import os
from typing import List, Dict, Optional
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm
import time
import re
import random


# Shared prompt template used by both LoRA and MLP
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

Look at every example, and then generate an explanation. After generating an explanation, assess how monosemantic the feature is.

<neuron_activations>
{activations_str}
</neuron_activations>

Classify the feature's interpretability:
0: The feature is specific, clear, and monosemantic. All given examples clearly adhere to the explanation. The explanation is not broad, and the examples are not noisy. This feature has a clear and obvious interpretation.
1: The feature may be broad or noisy, but ALL given examples still adhere to the generated explanation. The explanation may not be obvious.
2: The feature appears polysemantic. Some examples do not clearly adhere to the generated explanation. The explanation does not cleanly explain the given examples.

Respond with JSON in exactly this format:
{{
  "explanation": "your concise explanation in just a few words",
  "classification": <0, 1, or 2>,
  "classification_reasoning": "brief justification for your classification"
}}"""


def normalize_activations(activations: List[float], max_abs: float) -> np.ndarray:
    """Normalize activations to -10 to 10 range."""
    if max_abs == 0:
        return np.zeros(len(activations))
    normalized = (np.array(activations) / max_abs) * 10.0
    return np.clip(normalized, -10, 10)


def format_activation_context(tokens: List[str],
                             activations: List[float],
                             max_abs: float,
                             token_idx_in_context: int,
                             activation_threshold: float = 0.5) -> str:
    """Format tokens with context and activation values."""
    # Limit to 8 tokens before and after the target token
    start_offset = max(0, token_idx_in_context - 8)
    end_offset = min(len(tokens), token_idx_in_context + 9)  # +9 to include 8 after
    
    # Extract the window
    window_tokens = tokens[start_offset:end_offset]
    window_activations = activations[start_offset:end_offset]
    
    # Adjust token index for the window
    window_token_idx = token_idx_in_context - start_offset
    
    # Normalize activations for the window
    normalized = normalize_activations(window_activations, max_abs)
    
    # Build the full text - concatenate without spaces
    full_text = "".join(window_tokens)
    
    # Add ellipsis if truncated
    if start_offset > 0:
        full_text = "..." + full_text
    if end_offset < len(tokens):
        full_text = full_text + "..."
    
    # Collect activating tokens with their values
    # Show all significant activations as absolute values
    activating_tokens = []
    for i, (token, act, norm) in enumerate(zip(window_tokens, window_activations, normalized)):
        # Include all tokens above threshold (use absolute value)
        if abs(norm) > activation_threshold:
            activating_tokens.append((token, abs(norm)))
    
    # Sort by absolute activation value
    activating_tokens.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Format result
    result_parts = [full_text]
    for token, activation in activating_tokens[:10]:  # Limit to top 10 tokens
        result_parts.append(f"{token} {activation:.2f}")
    
    return "\n".join(result_parts)


def format_activations_for_prompt(examples: List[Dict],
                                  max_abs: float,
                                  max_examples_for_interp: int = 10,
                                  activation_threshold: float = 0.5,
                                  random_seed: Optional[int] = None) -> str:
    """Format multiple examples for the prompt with random shuffling."""
    # Randomly shuffle the examples to avoid order bias
    shuffled_examples = examples[:max_examples_for_interp].copy()
    
    # Use seed if provided for reproducibility
    if random_seed is not None:
        random.Random(random_seed).shuffle(shuffled_examples)
    else:
        random.shuffle(shuffled_examples)
    
    formatted_examples = []
    
    for i, example in enumerate(shuffled_examples, 1):
        # Calculate token index in context
        token_idx_in_context = example['token_idx'] - example['context_start']
        
        formatted = format_activation_context(
            example['tokens'],
            example['activations'],
            max_abs,
            token_idx_in_context,
            activation_threshold
        )
        formatted_examples.append(f"Example {i}:\n{formatted}")
    
    return "\n\n".join(formatted_examples)


async def generate_explanation(feature_id: str,
                               examples: List[Dict],
                               session: aiohttp.ClientSession,
                               api_key: str,
                               model_name: str,
                               api_base: str,
                               temperature: float,
                               max_tokens_per_explanation: int,
                               max_examples_for_interp: int,
                               activation_threshold: float,
                               request_semaphore: asyncio.Semaphore,
                               random_seed: Optional[int] = None,
                               max_retries: int = 5) -> Dict:
    """
    Generate explanation for a feature using the LLM API with retry logic.
    
    Will retry up to max_retries times if the API returns an error or 
    the response cannot be parsed as valid JSON.
    """
    if not examples:
        return {
            "explanation": "No activating examples found",
            "classification": -1,
            "classification_reasoning": "No examples to analyze"
        }
    
    # Find max absolute activation
    max_abs = max(abs(e.get('activation_value', e.get('original_value', 0))) for e in examples)
    
    # Format activations
    activations_str = format_activations_for_prompt(
        examples, max_abs, max_examples_for_interp, 
        activation_threshold, random_seed
    )
    
    # Build prompt
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        activations_str=activations_str
    )
    
    # Call API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Build payload for OpenAI API with system message to ensure JSON output
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that analyzes neural network activations. Always respond with valid JSON in the exact format requested."},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": max_tokens_per_explanation
    }
    
    # Only add temperature if not using gpt-5-mini
    if "gpt-5" not in model_name.lower():
        payload["temperature"] = temperature
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            async with request_semaphore:
                async with session.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    # Check response status
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Attempt {attempt + 1}/{max_retries} - API returned status {response.status}: {error_text[:200]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return {
                            "explanation": f"Error: API returned status {response.status} after {max_retries} attempts",
                            "classification": -1,
                            "classification_reasoning": "API error after retries"
                        }
                    
                    result = await response.json()
                    
                    if "error" in result:
                        print(f"Attempt {attempt + 1}/{max_retries} - API error for {feature_id}: {result['error']}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return {
                            "explanation": "API error after retries",
                            "classification": -1,
                            "classification_reasoning": "API error after retries"
                        }
                    
                    if not result.get("choices") or not result["choices"][0].get("message"):
                        print(f"Attempt {attempt + 1}/{max_retries} - Unexpected response format")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {
                            "explanation": "Error: Unexpected response format after retries",
                            "classification": -1,
                            "classification_reasoning": "Response format error after retries"
                        }
                    
                    explanation = result["choices"][0]["message"]["content"]
                    if not explanation:
                        print(f"Attempt {attempt + 1}/{max_retries} - Empty explanation received")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {
                            "explanation": "Error: Empty explanation after retries",
                            "classification": -1,
                            "classification_reasoning": "Empty response after retries"
                        }
                    
                    # Check if there's a finish_reason indicating truncation
                    finish_reason = result["choices"][0].get("finish_reason")
                    if finish_reason == "length":
                        print(f"Warning: Response was truncated (hit token limit). Partial response: {explanation[:200]}...")
                    
                    # Parse JSON response
                    try:
                        # Try to find JSON block - look for opening and closing braces
                        json_match = re.search(r'\{[^}]*"explanation"[^}]*"classification"[^}]*"classification_reasoning"[^}]*\}', explanation, re.DOTALL)
                        
                        if not json_match:
                            # Try a simpler pattern
                            json_match = re.search(r'\{.*?"explanation".*?"classification".*?\}', explanation, re.DOTALL)
                        
                        if json_match:
                            json_str = json_match.group()
                            # Clean up common issues
                            json_str = json_str.replace('\n', ' ').strip()
                            parsed = json.loads(json_str)
                            
                            # Validate that we have the required fields
                            if "explanation" in parsed and "classification" in parsed:
                                if attempt > 0:
                                    print(f"Success on attempt {attempt + 1}/{max_retries} for {feature_id}")
                                return parsed
                            else:
                                raise ValueError("Missing required fields in JSON response")
                        else:
                            # Try to parse the entire response as JSON
                            try:
                                parsed = json.loads(explanation.strip())
                                if "explanation" in parsed and "classification" in parsed:
                                    if attempt > 0:
                                        print(f"Success on attempt {attempt + 1}/{max_retries} for {feature_id}")
                                    return parsed
                                else:
                                    raise ValueError("Missing required fields in JSON response")
                            except:
                                # If no valid JSON found, retry
                                print(f"Attempt {attempt + 1}/{max_retries} - No valid JSON found in response")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                return {
                                    "explanation": "Error: No valid JSON after retries",
                                    "classification": -1,
                                    "classification_reasoning": "Failed to parse response after retries",
                                    "raw_response": explanation[:500] if attempt == max_retries - 1 else None
                                }
                    except Exception as e:
                        print(f"Attempt {attempt + 1}/{max_retries} - Error parsing JSON: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {
                            "explanation": "Error parsing response after retries",
                            "classification": -1,
                            "classification_reasoning": f"Failed to parse after {max_retries} attempts",
                            "raw_response": explanation[:500] if attempt == max_retries - 1 else None
                        }
                        
        except asyncio.TimeoutError:
            print(f"Attempt {attempt + 1}/{max_retries} - Request timeout for {feature_id}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "explanation": "Error: Request timeout after retries",
                "classification": -1,
                "classification_reasoning": "Timeout after retries"
            }
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Error processing {feature_id}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "explanation": "Error generating explanation after retries",
                "classification": -1,
                "classification_reasoning": f"Exception after {max_retries} attempts: {str(e)[:100]}"
            }
    
    # Should never reach here, but just in case
    return {
        "explanation": "Error: Failed after all retry attempts",
        "classification": -1,
        "classification_reasoning": "Exhausted all retries"
    }


class BaseAutoInterp:
    """Base class for automated interpretation with shared functionality."""
    
    def __init__(self,
                 examples_file: str,
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-5-mini",
                 api_base: str = "https://api.openai.com/v1",
                 temperature: float = 0.0,
                 max_tokens_per_explanation: int = 16000,
                 max_examples_for_interp: int = 10,
                 activation_threshold: float = 0.5,
                 max_concurrent_requests: int = 20,
                 random_seed: Optional[int] = None):
        """
        Initialize interpretation system.
        
        Args:
            examples_file: Path to precomputed examples JSON
            api_key: OpenAI API key
            model_name: Model for generating explanations
            api_base: API base URL
            temperature: Temperature for generation (not used for gpt-5-mini)
            max_tokens_per_explanation: Max completion tokens
            max_examples_for_interp: Max examples to send to LLM
            activation_threshold: Minimum absolute activation to show
            max_concurrent_requests: Number of parallel API calls
            random_seed: Random seed for shuffling examples (None = different each time)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key required (set OPENAI_API_KEY or OPENROUTER_API_KEY)")
            
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens_per_explanation = max_tokens_per_explanation
        self.max_examples_for_interp = max_examples_for_interp
        self.activation_threshold = activation_threshold
        self.max_concurrent_requests = max_concurrent_requests
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.random_seed = random_seed
        
        # Load precomputed examples
        print(f"Loading precomputed examples from {examples_file}...")
        from tqdm import tqdm
        with open(examples_file, 'r') as f:
            data = json.load(f)
        
        self.features = data['features']
        self.metadata = data['metadata']
        print(f"Loaded {len(self.features)} features with {self.metadata['top_k']} examples each")
    
    async def process_feature(self,
                             feature_id: str,
                             feature_data: Dict,
                             session: aiohttp.ClientSession) -> Dict:
        """Process a single feature - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement process_feature")
    
    async def run(self,
                  feature_filter: Optional[List[str]] = None,
                  output_path: str = None):
        """
        Run interpretation on features.
        
        Args:
            feature_filter: List of feature IDs to process (None = all)
            output_path: Output file path
        """
        if output_path is None:
            raise ValueError("output_path must be specified")
            
        # Determine which features to process
        if feature_filter:
            features_to_process = {fid: self.features[fid] 
                                  for fid in feature_filter 
                                  if fid in self.features}
        else:
            features_to_process = self.features
        
        print(f"Processing {len(features_to_process)} features...")
        
        # Process features with parallelization
        results = []
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_feature(fid, fdata, session) 
                    for fid, fdata in features_to_process.items()]
            
            # Use a more informative progress bar
            for result in await async_tqdm.gather(*tasks, 
                                                  desc="Interpreting features",
                                                  ncols=100,
                                                  unit="feature"):
                results.append(result)
        
        elapsed = time.time() - start_time
        print(f"Processed {len(features_to_process)} features in {elapsed:.1f}s")
        
        # Save results
        output_data = {
            "metadata": {
                "num_features": len(results),
                "model": self.model_name,
                "processing_time": elapsed,
                "precomputed_metadata": self.metadata
            },
            "interpretations": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved interpretations to {output_path}")


# Classification label mapping
CLASSIFICATION_LABELS = {
    0: "monosemantic",
    1: "broad_but_consistent", 
    2: "polysemantic",
    -1: "error"
}