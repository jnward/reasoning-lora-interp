#!/usr/bin/env python3
"""
Test the speed improvement of parallel autointerp.
"""

import time
import asyncio
import json
import sys
from autointerp_context import AutoInterpContext
from autointerp_context_parallel import AutoInterpContextParallel


async def test_parallel_performance():
    """Compare sequential vs parallel performance."""
    
    # Test configuration
    test_features = list(range(5))  # Test with 5 features
    input_file = "../sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json"
    
    # Check if file exists
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            available_features = [int(k) for k in data['features'].keys()]
            test_features = available_features[:5]  # Use first 5 available features
            print(f"Testing with features: {test_features}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        print("Please generate feature data first using collect_sae_features_fast.py")
        sys.exit(1)
    
    # Test sequential version
    print("\n=== Testing Sequential Version ===")
    sequential_interp = AutoInterpContext(
        model_name="openai/gpt-4o-mini",  # Use faster model for testing
        max_examples_per_rollout=4
    )
    
    start_time = time.time()
    await sequential_interp.autointerp_features(
        activation_data_path=input_file,
        output_path="test_sequential_output.json",
        feature_ids=test_features
    )
    sequential_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.1f}s")
    print(f"Rate: {len(test_features)/sequential_time:.2f} features/s")
    
    # Test parallel version
    print("\n=== Testing Parallel Version ===")
    parallel_interp = AutoInterpContextParallel(
        model_name="openai/gpt-4o-mini",  # Use faster model for testing
        max_examples_per_rollout=4,
        max_concurrent_requests=5,  # Process all 5 features concurrently
        rate_limit_per_minute=60
    )
    
    start_time = time.time()
    await parallel_interp.autointerp_features(
        activation_data_path=input_file,
        output_path="test_parallel_output.json",
        feature_ids=test_features
    )
    parallel_time = time.time() - start_time
    
    print(f"Parallel time: {parallel_time:.1f}s")
    print(f"Rate: {len(test_features)/parallel_time:.2f} features/s")
    
    # Compare results
    print(f"\n=== Results ===")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x faster")
    print(f"Time saved: {sequential_time - parallel_time:.1f}s")
    
    # Estimate time for 2000 features
    sequential_2000 = (sequential_time / len(test_features)) * 2000
    parallel_2000 = (parallel_time / len(test_features)) * 2000
    
    print(f"\n=== Estimated Time for 2000 Features ===")
    print(f"Sequential: {sequential_2000/60:.1f} minutes ({sequential_2000/3600:.1f} hours)")
    print(f"Parallel: {parallel_2000/60:.1f} minutes ({parallel_2000/3600:.1f} hours)")
    print(f"Time saved: {(sequential_2000 - parallel_2000)/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(test_parallel_performance())