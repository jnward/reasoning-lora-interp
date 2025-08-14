#!/usr/bin/env python3
"""
Generate embeddings for SAE feature interpretations using OpenAI's embedding API.
Fixed version that handles empty strings and provides better error diagnostics.
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


def batch_texts(texts: List[str], batch_size: int = 100) -> List[List[str]]:
    """Batch texts for efficient API calls."""
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i + batch_size])
    return batches


def diagnose_batch_error(batch: List[str], batch_idx: int) -> None:
    """Diagnose why a batch might be failing."""
    print(f"\nDiagnosing batch {batch_idx} with {len(batch)} items:")
    
    # Check for various issues
    for i, text in enumerate(batch):
        issues = []
        
        if text is None:
            issues.append("None value")
        elif text == '':
            issues.append("Empty string")
        elif len(text) > 8000:  # OpenAI has token limits
            issues.append(f"Very long ({len(text)} chars)")
        
        if issues:
            print(f"  Item {i}: {', '.join(issues)}")
            if text and len(text) < 100:
                print(f"    Content: {repr(text)}")
    
    # Check total batch size
    total_chars = sum(len(t) if t else 0 for t in batch)
    print(f"  Total batch size: {total_chars} characters")
    
    # Try to identify the specific problem item
    print("\n  Trying individual items to identify the problem...")


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
    Generate embeddings with improved error handling.
    """
    # Preprocess texts
    processed_texts, empty_indices = preprocess_texts(texts)
    
    all_embeddings = []
    batches = batch_texts(processed_texts, batch_size)
    failed_items = []  # Track individual failed items
    
    print(f"Generating embeddings for {len(processed_texts)} texts in {len(batches)} batches...")
    
    # Process all batches
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_start = batch_idx * batch_size
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
                    if "invalid" in str(e).lower():
                        # This is likely a content issue, not a rate limit
                        print(f"\nBatch {batch_idx} has invalid content: {e}")
                        diagnose_batch_error(batch, batch_idx)
                        
                        # Try processing items individually
                        print(f"Processing batch {batch_idx} items individually...")
                        batch_embeddings = []
                        
                        for i, text in enumerate(batch):
                            item_idx = batch_start + i
                            try:
                                response = client.embeddings.create(
                                    input=[text],
                                    model=model
                                )
                                batch_embeddings.append(response.data[0].embedding)
                                time.sleep(0.1)  # Small delay between items
                            except Exception as item_e:
                                print(f"  Failed item {item_idx}: {str(item_e)[:100]}")
                                print(f"    Text: {repr(text[:100])}...")
                                # Use zero embedding for this item
                                embedding_dim = len(batch_embeddings[0]) if batch_embeddings else 1536
                                batch_embeddings.append([0.0] * embedding_dim)
                                failed_items.append((item_idx, text[:50]))
                        
                        all_embeddings.extend(batch_embeddings)
                        success = True  # Mark as success since we handled it
                    else:
                        # Regular retry for rate limits or network issues
                        print(f"\nError in batch {batch_idx}: {e}")
                        print(f"Retrying in {retry_delay:.1f} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                else:
                    print(f"\nFailed batch {batch_idx} after {max_retries} retries")
                    # Add zero embeddings for failed batch
                    embedding_dim = len(all_embeddings[0]) if all_embeddings else 1536
                    for i in range(len(batch)):
                        all_embeddings.append([0.0] * embedding_dim)
                        failed_items.append((batch_start + i, batch[i][:50] if batch[i] else ""))
    
    # Report results
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    zero_count = np.sum(np.all(embeddings_array == 0, axis=1))
    
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*60)
    
    if failed_items:
        print(f"\n⚠️  {len(failed_items)} items failed to generate embeddings:")
        for idx, text_preview in failed_items[:10]:
            print(f"  Index {idx}: {text_preview}...")
        if len(failed_items) > 10:
            print(f"  ... and {len(failed_items) - 10} more")
    
    if empty_indices:
        print(f"\n📝 {len(empty_indices)} empty explanations were given placeholder embeddings")
    
    if zero_count > 0:
        print(f"\n⚠️  WARNING: {zero_count} embeddings are zero vectors")
    else:
        print(f"\n✅ Successfully generated all {len(texts)} embeddings!")
    
    return embeddings_array


def setup_openai_client() -> OpenAI:
    """Setup OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    print("Using OpenAI API for embeddings")
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
        default=50,
        help="Number of texts to embed in each API call"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls"
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save metadata JSON file alongside embeddings"
    )
    
    args = parser.parse_args()
    
    # Load interpretations
    interpretations = load_interpretations(args.input)
    
    # Extract texts and feature IDs
    texts = [item['explanation'] for item in interpretations]
    feature_ids = [item['feature_id'] for item in interpretations]
    
    if not texts:
        print("No interpretations found!")
        return
    
    print(f"Found {len(texts)} interpretations to embed")
    
    # Setup OpenAI client
    try:
        client = setup_openai_client()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Generate embeddings
    print(f"\nGenerating embeddings using model: {args.model}")
    embeddings = generate_embeddings(
        texts,
        client,
        model=args.model,
        batch_size=args.batch_size,
        delay_between_batches=args.delay
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
    
    print("\nDone! Next step: Run visualize_umap.py to create UMAP visualizations")


if __name__ == "__main__":
    main()