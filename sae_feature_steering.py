# %%
import torch
import torch.nn as nn
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import sys
import os

# Add sae-interp to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'sae-interp'))
from sae_interp.batch_topk_sae import BatchTopKSAE

# %%
# Configuration - modify these values as needed
SAE_FEATURE_IDX = 84  # SAE feature index to use for steering
STEERING_STRENGTHS = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]  # List of steering strengths to test
PROBLEM_IDX = 10  # MATH500 problem index
PREFIX_TOKENS = 100  # Number of tokens to generate for prefix
ADDITIONAL_TOKENS = 100  # Additional tokens to generate with steering

print(f"Configuration:")
print(f"  SAE Feature: {SAE_FEATURE_IDX}")
print(f"  Steering strengths: {STEERING_STRENGTHS}")
print(f"  Problem index: {PROBLEM_IDX}")

# %%
# Model configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
sae_path = "/workspace/reasoning_interp/sae_interp/trained_sae.pt"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)

# Get the number of layers
n_layers = model.config.num_hidden_layers
print(f"Model has {n_layers} layers")

# %%
# Load SAE model
print("\nLoading SAE model...")
sae_checkpoint = torch.load(sae_path, map_location='cpu')
sae_config = sae_checkpoint['config']
print(f"SAE configuration:")
print(f"  d_model: {sae_config['d_model']}")
print(f"  dict_size: {sae_config['dict_size']}")
print(f"  k: {sae_config['k']}")

# Initialize SAE model
sae = BatchTopKSAE(
    d_model=sae_config['d_model'],
    dict_size=sae_config['dict_size'],
    k=sae_config['k']
)
sae.load_state_dict(sae_checkpoint['model_state_dict'])
sae.eval()
sae = sae.to(model.device)

# %%
# Extract SAE decoder weights for the selected feature
decoder_weights = sae.W_dec[SAE_FEATURE_IDX].detach()  # Shape: [192]
print(f"\nSAE Feature {SAE_FEATURE_IDX} decoder weights shape: {decoder_weights.shape}")

# Map the 192 dimensions to (layer, adapter) pairs
# Based on the training data structure: 64 layers × 3 adapters
# We need to check how the activations were ordered in training
# From the training script, activations are shape (n_tokens, 64, 3) and flattened to (n_tokens, 192)
# So the ordering is: [layer0_adapter0, layer0_adapter1, layer0_adapter2, layer1_adapter0, ...]

adapter_names = ['q_proj', 'k_proj', 'v_proj']  # The 3 adapters used in training

# Create mapping from flat index to (layer, adapter)
def flat_idx_to_layer_adapter(idx):
    layer = idx // 3
    adapter_idx = idx % 3
    return layer, adapter_names[adapter_idx]

# Find which LoRA features have significant weights
threshold = 0.01  # Only consider weights above this threshold
significant_weights = []

for idx in range(192):
    weight = decoder_weights[idx].item()
    if abs(weight) > threshold:
        layer, adapter = flat_idx_to_layer_adapter(idx)
        significant_weights.append({
            'layer': layer,
            'adapter': adapter,
            'weight': weight,
            'idx': idx
        })

print(f"\nSAE Feature {SAE_FEATURE_IDX} writes to {len(significant_weights)} LoRA features:")
for item in sorted(significant_weights, key=lambda x: abs(x['weight']), reverse=True)[:10]:
    print(f"  Layer {item['layer']:2d} {item['adapter']:6s}: {item['weight']:8.4f}")

# %%
# Extract all LoRA B matrices
print("\nExtracting LoRA B matrices...")
lora_b_matrices = {}

for layer_idx in range(n_layers):
    lora_b_matrices[layer_idx] = {}
    
    # Extract B matrices for each adapter type
    for adapter_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        try:
            if adapter_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                module = getattr(model.model.model.layers[layer_idx].self_attn, adapter_name)
            else:
                module = getattr(model.model.model.layers[layer_idx].mlp, adapter_name)
            
            if hasattr(module, 'lora_B'):
                lora_B_weight = module.lora_B['default'].weight.data
                b_direction = lora_B_weight.squeeze()
                lora_b_matrices[layer_idx][adapter_name] = b_direction
        except:
            # Some adapters might not exist
            pass

print(f"Extracted B matrices for {n_layers} layers")

# %%
# Load MATH500 dataset
print(f"\nLoading MATH500 dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
print(f"Dataset has {len(dataset)} examples")

# Select problem
if PROBLEM_IDX >= len(dataset):
    raise ValueError(f"Problem index {PROBLEM_IDX} is out of range. Dataset has {len(dataset)} examples")

problem = dataset[PROBLEM_IDX]
question = problem['problem']
print(f"\nSelected problem {PROBLEM_IDX}:")
print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")

# %%
# Format the prompt
system_prompt = "You are a helpful mathematics assistant."
prompt = (
    f"<|im_start|>system\n{system_prompt}\n"
    f"<|im_start|>user\n{question}\n"
    f"<|im_start|>assistant\n"
    f"<|im_start|>think\n"
)

# Tokenize the initial prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs.input_ids

print(f"\nInitial prompt has {input_ids.shape[1]} tokens")

# %%
# Generate prefix
print(f"\nGenerating {PREFIX_TOKENS} tokens for prefix...")
with torch.no_grad():
    prefix_output = model.generate(
        input_ids,
        max_new_tokens=PREFIX_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode the prefix
prefix_text = tokenizer.decode(prefix_output[0], skip_special_tokens=False)
print(f"\nPrefix generated:")
print("=" * 80)
print(prefix_text)
print("=" * 80)

prefix_ids = prefix_output

# %%
# Generate baseline (no steering)
print(f"\n\n{'='*80}")
print("Generating baseline (no steering)...")
print(f"{'='*80}")

with torch.no_grad():
    baseline_output = model.generate(
        prefix_ids,
        max_new_tokens=ADDITIONAL_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

baseline_new_tokens = baseline_output[0, prefix_ids.shape[1]:]
baseline_text = tokenizer.decode(baseline_new_tokens, skip_special_tokens=False)

print("\nBaseline output (no steering) - full:")
print(baseline_text)

# %%
# Create steering hooks
class SAESteeringHook:
    """Hook that applies SAE-based steering to multiple adapters in a layer"""
    def __init__(self, layer_idx, adapter_weights, lora_b_matrices, steering_strength, prefix_len):
        self.layer_idx = layer_idx
        self.adapter_weights = adapter_weights  # Dict of adapter_name -> sae_weight
        self.lora_b_matrices = lora_b_matrices  # Dict of adapter_name -> b_matrix
        self.steering_strength = steering_strength
        self.prefix_len = prefix_len
        self.applied_count = 0
        
        # Precompute steering vectors for each adapter
        self.steering_vectors = {}
        for adapter_name, sae_weight in adapter_weights.items():
            if adapter_name in lora_b_matrices:
                # Steering = SAE decoder weight * LoRA B matrix * steering strength
                self.steering_vectors[adapter_name] = (
                    sae_weight * lora_b_matrices[adapter_name] * steering_strength
                )
    
    def __call__(self, module, input, output):
        # This hook is applied to the layer output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Get current sequence length
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate total steering from all adapters
        total_steering = torch.zeros_like(hidden_states[0, 0])  # [hidden_size]
        
        for adapter_name, steering_vector in self.steering_vectors.items():
            # All steering vectors should be in residual stream dimension
            total_steering += steering_vector.to(hidden_states.dtype)
        
        # Apply steering
        if seq_len == 1:  # Single token generation
            steering = total_steering.unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states + steering
            self.applied_count += 1
        elif seq_len > self.prefix_len:
            # Full sequence processing - apply to positions after prefix
            hidden_states[:, self.prefix_len:, :] = (
                hidden_states[:, self.prefix_len:, :] + total_steering
            )
            self.applied_count += seq_len - self.prefix_len
        
        # Return in the same format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

# %%
# Iterate through different steering strengths
all_results = []

for steering_strength in STEERING_STRENGTHS:
    print(f"\n\n{'='*80}")
    print(f"Testing steering strength: {steering_strength}")
    print(f"{'='*80}")
    
    # Create hooks for each layer that has non-zero SAE weights
    hooks = []
    hook_handles = []
    
    # Group significant weights by layer
    weights_by_layer = {}
    for item in significant_weights:
        layer = item['layer']
        if layer not in weights_by_layer:
            weights_by_layer[layer] = {}
        weights_by_layer[layer][item['adapter']] = item['weight']
    
    print(f"Creating hooks for {len(weights_by_layer)} layers")
    
    # Create and register hooks
    for layer_idx, adapter_weights in weights_by_layer.items():
        # Get B matrices for this layer (only for adapters in training set)
        layer_b_matrices = {}
        for adapter_name in adapter_weights:
            if adapter_name in ['q_proj', 'k_proj', 'v_proj'] and adapter_name in lora_b_matrices[layer_idx]:
                layer_b_matrices[adapter_name] = lora_b_matrices[layer_idx][adapter_name]
        
        if layer_b_matrices:  # Only create hook if we have B matrices
            hook = SAESteeringHook(
                layer_idx, 
                adapter_weights, 
                layer_b_matrices, 
                steering_strength, 
                prefix_ids.shape[1]
            )
            hooks.append(hook)
            
            # Register hook on layer output
            handle = model.model.model.layers[layer_idx].register_forward_hook(hook)
            hook_handles.append(handle)
    
    print(f"Registered {len(hooks)} hooks")
    
    # Generate with steering
    with torch.no_grad():
        steered_output = model.generate(
            prefix_ids,
            max_new_tokens=ADDITIONAL_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Remove all hooks
    for handle in hook_handles:
        handle.remove()
    
    # Count total applications
    total_applications = sum(hook.applied_count for hook in hooks)
    print(f"Steering applied {total_applications} times across all layers")
    
    # Decode only the new tokens
    new_tokens = steered_output[0, prefix_ids.shape[1]:]
    steered_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    print(f"\nSteered output (full):")
    print(steered_text)
    
    # Show difference from baseline
    is_different = steered_text != baseline_text
    
    if is_different:
        print("\n[DIFFERENT FROM BASELINE]")
        
        # Calculate metrics
        baseline_tokens = tokenizer.encode(baseline_text, add_special_tokens=False)
        steered_tokens = tokenizer.encode(steered_text, add_special_tokens=False)
        
        # Find first difference
        first_diff_pos = None
        first_diff_baseline_token = None
        first_diff_steered_token = None
        
        for i, (b, s) in enumerate(zip(baseline_tokens, steered_tokens)):
            if b != s:
                first_diff_pos = i
                first_diff_baseline_token = tokenizer.decode([b])
                first_diff_steered_token = tokenizer.decode([s])
                break
        
        if first_diff_pos is not None:
            print(f"First token difference at position {first_diff_pos}")
            print(f"  Baseline token: '{first_diff_baseline_token}'")
            print(f"  Steered token: '{first_diff_steered_token}'")
            
            # Show context
            if first_diff_pos > 0:
                prefix_tokens = baseline_tokens[:first_diff_pos]
                prefix_text = tokenizer.decode(prefix_tokens)
                print(f"\nContext before first difference:")
                print(f"...{prefix_text[-100:]}[DIFF HERE]")
        
        # Calculate similarity
        min_len = min(len(baseline_tokens), len(steered_tokens))
        if min_len > 0:
            matching_tokens = sum(1 for b, s in zip(baseline_tokens[:min_len], steered_tokens[:min_len]) if b == s)
            similarity = matching_tokens / min_len
            print(f"\nToken-level similarity: {similarity:.2%}")
    else:
        print("\n[SAME AS BASELINE]")
    
    # Store results
    result = {
        "sae_feature": SAE_FEATURE_IDX,
        "steering_strength": steering_strength,
        "problem_idx": PROBLEM_IDX,
        "question": question,
        "prefix": prefix_text,
        "baseline": baseline_text,
        "steered": steered_text,
        "different": is_different,
        "num_layers_steered": len(hooks),
        "total_applications": total_applications,
        "significant_weights": significant_weights
    }
    
    all_results.append(result)

# %%
# Save all results
output_file = f"sae_feature_{SAE_FEATURE_IDX}_steering_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n\nAll results saved to {output_file}")

# %%
# Print summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"SAE Feature: {SAE_FEATURE_IDX}")
print(f"Problem: {question[:100]}...")
print(f"\nPrefix ended with: ...{tokenizer.decode(prefix_ids[0, -20:], skip_special_tokens=False)}")
print(f"\nBaseline continuation: {baseline_text[:100]}...")

print(f"\nSAE feature writes to {len(significant_weights)} LoRA features across {len(weights_by_layer)} layers")
print(f"\nTop 5 strongest connections:")
for item in sorted(significant_weights, key=lambda x: abs(x['weight']), reverse=True)[:5]:
    print(f"  Layer {item['layer']:2d} {item['adapter']:6s}: {item['weight']:8.4f}")

print(f"\nResults across steering strengths:")
for result in all_results:
    strength = result['steering_strength']
    different = result['different']
    status = 'DIFFERENT' if different else 'SAME'
    print(f"  Strength {strength:6.1f}: {status}")

# %%