#!/usr/bin/env python3
"""
Parallelized context-aware automated interpretation for SAE features.

This implementation processes multiple features concurrently for much faster
interpretation, especially useful when dealing with thousands of features.
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import time


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


class AutoInterpContextParallel:
    """
    Parallelized context-aware automated interpretation for SAE features.
    
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
                 max_examples_per_rollout: int = 4,  # Max examples from same rollout
                 max_concurrent_requests: int = 10,  # Number of parallel API calls
                 rate_limit_per_minute: int = 60):  # API rate limit
        """
        Initialize the parallelized context-aware interpretation system.
        
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
            max_concurrent_requests: Number of concurrent API requests (default: 10)
            rate_limit_per_minute: Maximum requests per minute (default: 60)
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
        
        # Parallelization settings
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.rate_limiter = None  # Will be initialized in async context
        self.current_concurrency = max_concurrent_requests
        self.rate_limit_hits = 0
        self.consecutive_successes = 0
        
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
                                   feature_id: Optional[int] = None,
                                   session: aiohttp.ClientSession = None) -> str:
        """
        Generate explanation for a feature based on activation records.
        
        Args:
            activation_records: List of activation records for the feature
            max_activation: Maximum activation value
            feature_id: Feature ID for debugging
            session: Shared aiohttp session
            
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
        
        # Use the shared session if provided, otherwise create a new one
        if session is None:
            async with aiohttp.ClientSession() as new_session:
                response = await self._make_api_call(new_session, headers, payload)
        else:
            response = await self._make_api_call(session, headers, payload)
        
        # Extract explanation from response
        explanation = self._extract_explanation(response, feature_id)
        
        return explanation
    
    async def _make_api_call(self, session: aiohttp.ClientSession, headers: dict, payload: dict, 
                             max_retries: int = 3, base_delay: float = 1.0) -> dict:
        """Make the actual API call with rate limiting and retry logic."""
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        last_error = None
        for attempt in range(max_retries):
            try:
                async with self.request_semaphore:
                    async with session.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
                    ) as response:
                        result = await response.json()
                        
                        # Check for rate limit error
                        if response.status == 429 or (isinstance(result, dict) and 
                                                      result.get("error", {}).get("code") == "rate_limit_exceeded"):
                            # Rate limited - wait longer
                            retry_after = int(response.headers.get("Retry-After", 60))
                            self.rate_limit_hits += 1
                            
                            # Reduce concurrency if hitting rate limits
                            if self.rate_limit_hits >= 2 and self.current_concurrency > 1:
                                new_concurrency = max(1, self.current_concurrency // 2)
                                print(f"\nReducing concurrency from {self.current_concurrency} to {new_concurrency} due to rate limits")
                                self.current_concurrency = new_concurrency
                                # Update semaphore
                                self.request_semaphore = asyncio.Semaphore(new_concurrency)
                                self.rate_limit_hits = 0  # Reset counter
                            
                            print(f"\nRate limited. Waiting {retry_after}s before retry...")
                            await asyncio.sleep(retry_after)
                            continue
                        
                        # Check for server errors (5xx)
                        if response.status >= 500:
                            if attempt < max_retries - 1:
                                wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                                print(f"\nServer error {response.status}. Retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise Exception(f"Server error {response.status}: {result}")
                        
                        # Check for other errors in response
                        if "error" in result:
                            error_msg = result['error']
                            if isinstance(error_msg, dict):
                                error_msg = error_msg.get('message', str(error_msg))
                            
                            # Check if it's a transient error worth retrying
                            transient_errors = ["timeout", "connection", "temporary", "unavailable"]
                            if any(err in str(error_msg).lower() for err in transient_errors):
                                if attempt < max_retries - 1:
                                    wait_time = base_delay * (2 ** attempt)
                                    print(f"\nTransient error: {error_msg}. Retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                            
                            raise Exception(f"API error: {error_msg}")
                        
                        # Success!
                        self.consecutive_successes += 1
                        
                        # Gradually increase concurrency back up after many successes
                        if (self.consecutive_successes >= 50 and 
                            self.current_concurrency < self.max_concurrent_requests):
                            new_concurrency = min(self.max_concurrent_requests, 
                                                 self.current_concurrency + 2)
                            if new_concurrency > self.current_concurrency:
                                print(f"\nIncreasing concurrency from {self.current_concurrency} to {new_concurrency} after successful requests")
                                self.current_concurrency = new_concurrency
                                self.request_semaphore = asyncio.Semaphore(new_concurrency)
                                self.consecutive_successes = 0  # Reset counter
                        
                        return result
                        
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"\nRequest timeout. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Request timeout after {max_retries} attempts")
                    
            except aiohttp.ClientError as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"\nConnection error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Connection error after {max_retries} attempts: {e}")
            
            except Exception as e:
                # For unexpected errors, don't retry
                raise e
        
        # If we get here, all retries failed
        raise Exception(f"Failed after {max_retries} attempts. Last error: {last_error}")
    
    def _extract_explanation(self, result: dict, feature_id: Optional[int]) -> str:
        """Extract explanation from API response."""
        # Extract explanation - handle potential reasoning content
        if "choices" not in result or not result["choices"]:
            print(f"WARNING: No choices in response for feature {feature_id}")
            return "No explanation generated"
            
        choice = result["choices"][0]
        if "message" not in choice:
            print(f"WARNING: No message in choice for feature {feature_id}")
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
        
        return diverse_records
    
    async def process_single_feature(self, 
                                     feature_id: int,
                                     activation_records: List[ActivationRecord],
                                     session: aiohttp.ClientSession) -> Optional[ExplanationResult]:
        """
        Process a single feature asynchronously.
        
        Args:
            feature_id: Feature ID
            activation_records: Activation records for the feature
            session: Shared aiohttp session
            
        Returns:
            ExplanationResult or None if failed
        """
        if not activation_records:
            return None
            
        # Calculate max activation for normalization
        max_activation = max(
            max(record.activations) 
            for record in activation_records
        )
        
        if max_activation == 0:
            return None
        
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
                feature_id=feature_id,
                session=session
            )
            
            result = ExplanationResult(
                feature_id=feature_id,
                explanation=explanation,
                activation_records=activation_records
            )
            
            return result
            
        except Exception as e:
            print(f"Error processing feature {feature_id}: {e}")
            return None
    
    async def autointerp_features(self,
                                  activation_data_path: str,
                                  output_path: str,
                                  feature_ids: Optional[List[int]] = None):
        """
        Run automated interpretation on features with parallel processing.
        
        Args:
            activation_data_path: Path to activation data JSON
            output_path: Path to save results
            feature_ids: Specific features to interpret (None = all)
        """
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.rate_limit_per_minute)
        
        # Load activation data
        print("Loading activation data...")
        features_data = self.load_activation_data(activation_data_path)
        
        # Filter to requested features
        if feature_ids is not None:
            features_data = {fid: features_data[fid] 
                            for fid in feature_ids if fid in features_data}
        
        print(f"Interpreting {len(features_data)} features with {self.max_concurrent_requests} concurrent requests...")
        
        # Create shared aiohttp session
        async with aiohttp.ClientSession() as session:
            # Create tasks for all features
            tasks = []
            for feature_id, activation_records in features_data.items():
                task = self.process_single_feature(feature_id, activation_records, session)
                tasks.append(task)
            
            # Process with progress bar
            results = []
            start_time = time.time()
            
            # Use async_tqdm for progress tracking
            for result in await async_tqdm.gather(*tasks, desc="Processing features"):
                if result is not None:
                    results.append(result)
            
            elapsed_time = time.time() - start_time
            
        # Calculate processing rate
        features_per_second = len(features_data) / elapsed_time
        print(f"\nProcessed {len(features_data)} features in {elapsed_time:.1f}s ({features_per_second:.1f} features/s)")
        
        # Save results
        print(f"Saving results to {output_path}...")
        output_data = {
            "metadata": {
                "format": "context-aware",
                "model_name": self.model_name,
                "api_base": self.api_base,
                "num_features": len(results),
                "processing_time_seconds": elapsed_time,
                "features_per_second": features_per_second
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


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.interval:
                await asyncio.sleep(self.interval - time_since_last)
            self.last_call = time.time()


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallelized context-aware automated interpretation")
    parser.add_argument("--input", required=True, help="Path to activation data JSON")
    parser.add_argument("--output", required=True, help="Path to save explanations")
    parser.add_argument("--features", nargs="+", type=int, help="Specific features to interpret")
    parser.add_argument("--api-key", help="API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="openai/o3-mini", help="Model for explanations (OpenRouter format)")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1", help="API base URL")
    parser.add_argument("--threshold", type=float, default=0.1, help="Minimum activation to show")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning mode for CoT models")
    parser.add_argument("--max-per-rollout", type=int, default=4, help="Max examples from same rollout")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent API requests")
    parser.add_argument("--rate-limit", type=int, default=60, help="Max requests per minute")
    
    args = parser.parse_args()
    
    # Initialize autointerp
    autointerp = AutoInterpContextParallel(
        api_key=args.api_key,
        model_name=args.model,
        api_base=args.api_base,
        activation_threshold=args.threshold,
        use_reasoning_mode=not args.no_reasoning,
        max_examples_per_rollout=args.max_per_rollout,
        max_concurrent_requests=args.max_concurrent,
        rate_limit_per_minute=args.rate_limit
    )
    
    # Run interpretation
    asyncio.run(autointerp.autointerp_features(
        activation_data_path=args.input,
        output_path=args.output,
        feature_ids=args.features
    ))


if __name__ == "__main__":
    main()