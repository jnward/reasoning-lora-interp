import time
import numpy as np

# Parameters for 10k token rollouts
n_tokens = 10_000
n_layers = 64
n_projections = 3
n_rollouts = 100
context_window = 10
top_k = 16

print(f"Recalculating for {n_tokens:,} tokens per rollout:")
print(f"Total ActivationExample objects per rollout: {n_tokens * n_layers * n_projections:,}")
print(f"Total objects for 100 rollouts: {n_tokens * n_layers * n_projections * n_rollouts:,}")
print("-" * 80)

# Based on our measurements:
# - Object creation: 4.0μs per object
# - Context slicing: 2.4μs per operation
# - Heap operations: 1.8μs per operation
# - Token decoding: 0.007ms per token (individual)

# Time estimates
object_time_per_rollout = (n_tokens * n_layers * n_projections * 4.0e-6)
slicing_time_per_rollout = (n_tokens * n_layers * n_projections * 2.4e-6)
heap_time_per_rollout = (n_tokens * n_layers * n_projections * 1.8e-6)
decode_time_per_rollout = (n_tokens * 0.007e-3)

print("\nPER ROLLOUT:")
print(f"1. Object creation: {object_time_per_rollout:.1f}s")
print(f"2. Context slicing: {slicing_time_per_rollout:.1f}s")
print(f"3. Heap operations: {heap_time_per_rollout:.1f}s")
print(f"4. Token decoding: {decode_time_per_rollout:.1f}s")
print(f"Total CPU per rollout: {object_time_per_rollout + slicing_time_per_rollout + heap_time_per_rollout + decode_time_per_rollout:.1f}s")

print("\nFOR 100 ROLLOUTS:")
total_object = object_time_per_rollout * n_rollouts
total_slicing = slicing_time_per_rollout * n_rollouts
total_heap = heap_time_per_rollout * n_rollouts
total_decode = decode_time_per_rollout * n_rollouts
total_cpu = total_object + total_slicing + total_heap + total_decode

print(f"1. Object creation: {total_object:.0f}s ({total_object/60:.1f} minutes) - {total_object/total_cpu*100:.1f}%")
print(f"2. Context slicing: {total_slicing:.0f}s ({total_slicing/60:.1f} minutes) - {total_slicing/total_cpu*100:.1f}%")
print(f"3. Heap operations: {total_heap:.0f}s ({total_heap/60:.1f} minutes) - {total_heap/total_cpu*100:.1f}%")
print(f"4. Token decoding: {total_decode:.0f}s ({total_decode/60:.1f} minutes) - {total_decode/total_cpu*100:.1f}%")
print(f"\nTotal CPU time: {total_cpu:.0f}s ({total_cpu/60:.1f} minutes)")
print(f"Total time you observed: 20 minutes")
print(f"CPU overhead accounts for: {total_cpu/60/20*100:.0f}% of total time")

print("\n" + "="*80)
print("MEMORY IMPACT:")
# Rough estimates for memory per ActivationExample
# - Base object overhead: ~100 bytes
# - Strings (token + context): ~200 bytes average
# - context_activations dict: ~200 bytes
# - Total per object: ~500 bytes

memory_per_object = 500  # bytes
total_objects_in_memory = n_layers * n_projections * top_k * 2  # Only top-k kept in memory
rolling_window_objects = n_tokens  # Objects created but discarded
total_memory_active = total_objects_in_memory * memory_per_object
memory_churn_per_rollout = n_tokens * n_layers * n_projections * memory_per_object

print(f"Active memory for top-k storage: {total_memory_active/1024/1024:.1f} MB")
print(f"Memory churn per rollout: {memory_churn_per_rollout/1024/1024:.0f} MB")
print(f"Total memory allocated (100 rollouts): {memory_churn_per_rollout * n_rollouts / 1024/1024/1024:.1f} GB")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print(f"- Creating {n_tokens * n_layers * n_projections:,} objects per rollout is the main bottleneck")
print(f"- Total of {n_tokens * n_layers * n_projections * n_rollouts/1e9:.1f} BILLION objects created")
print(f"- Only {total_objects_in_memory:,} objects actually needed (top-k × layers × projections)")
print(f"- Efficiency ratio: {total_objects_in_memory / (n_tokens * n_layers * n_projections * n_rollouts) * 100:.3f}%")
print(f"- You're creating ~{int(n_tokens * n_layers * n_projections / total_objects_in_memory):,}x more objects than needed!")