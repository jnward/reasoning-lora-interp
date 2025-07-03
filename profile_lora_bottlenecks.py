import time
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict
import heapq
import html as html_lib
import numpy as np

# Load tokenizer for testing
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Create sample data
sample_text = "This is a sample text " * 200  # ~1000 tokens
inputs = tokenizer(sample_text, return_tensors="pt")
input_ids = inputs.input_ids[0]
n_tokens = len(input_ids)
n_layers = 64
n_projections = 3
context_window = 10
top_k = 16

print(f"Testing with {n_tokens} tokens, {n_layers} layers, {n_projections} projections")
print(f"Total iterations: {n_tokens * n_layers * n_projections:,}")
print("-" * 80)

# Test 1: Individual token decoding vs batch
print("\n1. TOKEN DECODING:")
start = time.time()
tokens_individual = []
for token_id in input_ids:
    decoded = tokenizer.decode([token_id])
    tokens_individual.append(decoded)
individual_time = time.time() - start
print(f"   Individual decode: {individual_time:.3f}s ({individual_time/n_tokens*1000:.1f}ms per token)")

start = time.time()
tokens_batch = tokenizer.convert_ids_to_tokens(input_ids)
batch_time = time.time() - start
print(f"   Batch decode: {batch_time:.3f}s ({batch_time/n_tokens*1000:.1f}ms per token)")
print(f"   Speedup: {individual_time/batch_time:.1f}x")

# Test 2: Object creation overhead
@dataclass
class ActivationExample:
    rollout_idx: int
    token_idx: int
    token: str
    activation: float
    context_before: List[str]
    context_after: List[str]
    layer: int
    proj_type: str
    context_activations: Dict[int, float] = None

print("\n2. OBJECT CREATION OVERHEAD:")
# Simulate the inner loop
activations = np.random.randn(n_tokens).astype(np.float32)
tokens = tokens_batch

# Test with object creation
start = time.time()
created_objects = 0
for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
    for layer_idx in range(n_layers):
        for token_idx in range(n_tokens):
            context_start = max(0, token_idx - context_window)
            context_end = min(len(tokens), token_idx + context_window + 1)
            
            context_before = tokens[context_start:token_idx]
            context_after = tokens[token_idx+1:context_end]
            
            context_activations = {}
            for ctx_idx in range(context_start, context_end):
                if ctx_idx != token_idx:
                    context_activations[ctx_idx] = activations[ctx_idx]
            
            example = ActivationExample(
                rollout_idx=0,
                token_idx=token_idx,
                token=tokens[token_idx],
                activation=activations[token_idx],
                context_before=context_before,
                context_after=context_after,
                layer=layer_idx,
                proj_type=proj_type,
                context_activations=context_activations
            )
            created_objects += 1
object_creation_time = time.time() - start
print(f"   Created {created_objects:,} objects in {object_creation_time:.3f}s")
print(f"   Time per object: {object_creation_time/created_objects*1000000:.1f}μs")
print(f"   For 100 rollouts: ~{object_creation_time * 100:.1f}s")

# Test 3: Context slicing overhead
print("\n3. CONTEXT SLICING OVERHEAD:")
start = time.time()
for _ in range(n_tokens * n_layers * n_projections):
    token_idx = np.random.randint(0, n_tokens)
    context_start = max(0, token_idx - context_window)
    context_end = min(len(tokens), token_idx + context_window + 1)
    context_before = tokens[context_start:token_idx]
    context_after = tokens[token_idx+1:context_end]
slicing_time = time.time() - start
print(f"   List slicing: {slicing_time:.3f}s")
print(f"   Time per slice: {slicing_time/(n_tokens * n_layers * n_projections)*1000000:.1f}μs")

# Test 4: Heap operations
print("\n4. HEAP OPERATIONS:")
class TopKTracker:
    def __init__(self, k):
        self.k = k
        self.top_positive = []
        self.top_negative = []
        self.counter = 0
        
    def add(self, example):
        act = example.activation
        self.counter += 1
        
        if act >= 0:
            if len(self.top_positive) < self.k:
                heapq.heappush(self.top_positive, (act, self.counter, example))
            elif act > self.top_positive[0][0]:
                heapq.heapreplace(self.top_positive, (act, self.counter, example))
        else:
            if len(self.top_negative) < self.k:
                heapq.heappush(self.top_negative, (-act, self.counter, example))
            elif -act > self.top_negative[0][0]:
                heapq.heapreplace(self.top_negative, (-act, self.counter, example))

# Create simple examples for heap testing
simple_examples = []
for i in range(n_tokens):
    example = ActivationExample(
        rollout_idx=0,
        token_idx=i,
        token=tokens[i],
        activation=activations[i],
        context_before=[],
        context_after=[],
        layer=0,
        proj_type='gate_proj',
        context_activations={}
    )
    simple_examples.append(example)

start = time.time()
for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
    for layer_idx in range(n_layers):
        tracker = TopKTracker(top_k)
        for example in simple_examples:
            tracker.add(example)
heap_time = time.time() - start
print(f"   Heap operations: {heap_time:.3f}s")
print(f"   Time per add: {heap_time/(n_tokens * n_layers * n_projections)*1000000:.1f}μs")

# Test 5: HTML generation
print("\n5. HTML GENERATION:")
# Get some example data
tracker = TopKTracker(top_k)
for example in simple_examples[:100]:
    tracker.add(example)
top_examples = tracker.top_positive[:top_k//2]

start = time.time()
html_parts = []
for ex in top_examples:
    context_html = []
    for j in range(20):  # Simulate 20 context tokens
        token = tokens[min(j, len(tokens)-1)]
        token_display = html_lib.escape(token).replace('\n', '\\n')
        bg_color = f"rgba(255, 0, 0, 0.3)"
        context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
    html_parts.append(''.join(context_html))
html_time = time.time() - start
total_html_ops = top_k * 2 * n_layers * n_projections  # positive and negative
print(f"   HTML for {len(top_examples)} examples: {html_time:.3f}s")
print(f"   Estimated for all layers: ~{html_time * total_html_ops / len(top_examples):.1f}s")

# Summary
print("\n" + "="*80)
print("ESTIMATED TIME BREAKDOWN FOR 100 ROLLOUTS:")
print(f"1. Token decoding: ~{individual_time * 100:.1f}s")
print(f"2. Object creation: ~{object_creation_time * 100:.1f}s")
print(f"3. Context slicing: ~{slicing_time * 100:.1f}s")
print(f"4. Heap operations: ~{heap_time * 100:.1f}s")
print(f"5. HTML generation: ~{html_time * total_html_ops / len(top_examples):.1f}s")
print(f"\nTotal estimated: ~{(individual_time + object_creation_time + slicing_time + heap_time) * 100 + html_time * total_html_ops / len(top_examples):.1f}s")
print("\nNote: GPU forward pass time not included in above estimates")