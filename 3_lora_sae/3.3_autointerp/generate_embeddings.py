#!/usr/bin/env python3
"""
Generate embeddings for SAE feature interpretations using OpenAI's embedding API.

This script can work with:
1. OpenAI API directly (set OPENAI_API_KEY)
2. OpenRouter as a proxy (set OPENROUTER_API_KEY and use --use-openrouter flag)
"""

import json
import numpy as np
import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time
from openai import OpenAI


def load_interpretations(file_path: str) -> List[Dict]:
    """Load interpretation data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    explanations = data.get('explanations', [])
    print(f"Loaded {len(explanations)} interpretations from {file_path}")
    return explanations


def batch_texts(texts: List[str], batch_size: int = 100) -> List[List[str]]:
    """Batch texts for efficient API calls."""
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i + batch_size])
    return batches


def preprocess_texts(texts: List[str]) -> Tuple[List[str], List[int]]:
    """
    Preprocess texts to handle empty strings and other issues.
    
    Returns:
        Tuple of (processed_texts, empty_indices)
    """
    processed = []
    empty_indices = []
    
    for i, text in enumerate(texts):
        if text is None or text.strip() == '':
            # Replace empty strings with a placeholder
            processed.append("[Empty explanation]")
            empty_indices.append(i)
        else:
            # Keep the original text (OpenAI API handles Unicode fine)
            processed.append(text)
    
    if empty_indices:
        print(f"Found {len(empty_indices)} empty explanations, replaced with placeholder")
        print(f"  Indices: {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}")
    
    return processed, empty_indices


def generate_embeddings(
    texts: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    delay_between_batches: float = 0.1,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI's API with retry logic.
    
    Args:
        texts: List of text strings to embed
        client: OpenAI client instance
        model: Embedding model to use
        batch_size: Number of texts to process in each API call
        delay_between_batches: Delay in seconds between API calls
        max_retries: Maximum number of retries for failed batches
        initial_retry_delay: Initial delay for exponential backoff
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    # Preprocess texts to handle empty strings
    processed_texts, empty_indices = preprocess_texts(texts)
    
    all_embeddings = []
    batches = batch_texts(processed_texts, batch_size)
    failed_batches = []
    
    print(f"Generating embeddings for {len(texts)} texts in {len(batches)} batches...")
    
    # Process all batches with initial attempt
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        retry_count = 0
        retry_delay = initial_retry_delay
        success = False
        
        while retry_count <= max_retries and not success:
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                success = True
                
                # Small delay to avoid rate limits
                if delay_between_batches > 0:
                    time.sleep(delay_between_batches)
                    
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"\nError in batch {batch_idx}: {e}")
                    print(f"Retrying in {retry_delay:.1f} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"\nFailed batch {batch_idx} after {max_retries} retries: {e}")
                    failed_batches.append((batch_idx, batch))
                    # Add zero embeddings for failed batch to maintain alignment
                    embedding_dim = len(all_embeddings[0]) if all_embeddings else 1536
                    all_embeddings.extend([[0.0] * embedding_dim for _ in batch])
    
    # Report on any failures
    if failed_batches:
        print(f"\n⚠️  WARNING: {len(failed_batches)} batches failed after all retries")
        print("Failed batch indices:", [idx for idx, _ in failed_batches])
        
        # Try to process failed items individually as last resort
        print("\nAttempting to process failed items individually...")
        individual_successes = 0
        
        for batch_idx, batch in tqdm(failed_batches, desc="Retrying failed items"):
            batch_start = batch_idx * batch_size
            
            for i, text in enumerate(batch):
                item_idx = batch_start + i
                retry_count = 0
                retry_delay = initial_retry_delay
                success = False
                
                while retry_count <= 3 and not success:  # Fewer retries for individual items
                    try:
                        response = client.embeddings.create(
                            input=[text],  # Single item
                            model=model
                        )
                        # Replace the zero embedding with the actual one
                        all_embeddings[item_idx] = response.data[0].embedding
                        individual_successes += 1
                        success = True
                        time.sleep(0.5)  # Longer delay for individual requests
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count <= 3:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            print(f"  Failed item {item_idx}: {text[:50]}...")
        
        if individual_successes > 0:
            print(f"✓ Recovered {individual_successes} embeddings through individual processing")
    
    # Final check for any remaining zeros
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    zero_count = np.sum(np.all(embeddings_array == 0, axis=1))
    
    if empty_indices:
        print(f"\n📝 Note: {len(empty_indices)} empty explanations were replaced with '[Empty explanation]' placeholder")
    
    if zero_count > 0:
        print(f"\n⚠️  WARNING: {zero_count} embeddings are still zero after all retries")
        print("Consider increasing delay or batch size, or check API quota")
    else:
        print(f"\n✓ Successfully generated all {len(texts)} embeddings!")
    
    return embeddings_array


def setup_openai_client(use_openrouter: bool = False) -> OpenAI:
    """
    Setup OpenAI client, optionally using OpenRouter as proxy.
    
    Args:
        use_openrouter: If True, configure client to use OpenRouter
        
    Returns:
        Configured OpenAI client
    """
    if use_openrouter:
        # OpenRouter doesn't support embeddings, but we'll set it up anyway
        # in case they add support in the future
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Note: OpenRouter doesn't currently support embeddings
        # This is here for future compatibility
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print("WARNING: OpenRouter doesn't currently support embedding models.")
        print("You'll need to use OpenAI API directly for embeddings.")
        
    else:
        # Use OpenAI API directly
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = OpenAI(api_key=api_key)
        print("Using OpenAI API directly for embeddings")
    
    return client


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for SAE feature interpretations")
    parser.add_argument(
        "--input",
        type=str,
        default="all_interpretations_o3.json",
        help="Path to input JSON file with interpretations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="feature_embeddings.npy",
        help="Path to output numpy file for embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of texts to embed in each API call"
    )
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter as proxy (currently not supported for embeddings)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between API calls"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries for failed batches"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds (doubles with each retry)"
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save metadata JSON file alongside embeddings"
    )
    
    args = parser.parse_args()
    
    # Load interpretations
    interpretations = load_interpretations(args.input)
    
    # Extract texts (explanations) and feature IDs
    texts = [item['explanation'] for item in interpretations]
    feature_ids = [item['feature_id'] for item in interpretations]
    
    if not texts:
        print("No interpretations found!")
        return
    
    print(f"Found {len(texts)} interpretations to embed")
    print(f"Sample interpretations:")
    for i in range(min(5, len(texts))):
        print(f"  Feature {feature_ids[i]}: {texts[i]}")
    
    # Setup OpenAI client
    try:
        client = setup_openai_client(args.use_openrouter)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set either OPENAI_API_KEY or OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    # Generate embeddings
    print(f"\nGenerating embeddings using model: {args.model}")
    embeddings = generate_embeddings(
        texts,
        client,
        model=args.model,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        max_retries=args.max_retries,
        initial_retry_delay=args.retry_delay
    )
    
    # Save embeddings
    print(f"\nSaving embeddings to {args.output}")
    np.save(args.output, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Optionally save metadata
    if args.save_metadata:
        metadata_path = args.output.replace('.npy', '_metadata.json')
        metadata = {
            "num_features": len(feature_ids),
            "embedding_dim": embeddings.shape[1],
            "model": args.model,
            "feature_ids": feature_ids,
            "explanations": texts,
            "input_file": args.input
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    # Print summary statistics
    print("\nEmbedding statistics:")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std:  {np.std(embeddings):.4f}")
    print(f"  Min:  {np.min(embeddings):.4f}")
    print(f"  Max:  {np.max(embeddings):.4f}")
    print(f"  Shape: {embeddings.shape}")
    
    print("\nDone! Next step: Run visualize_umap.py to create UMAP visualizations")


if __name__ == "__main__":
    main()