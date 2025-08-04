# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json

# %%
# Configuration - modify these values as needed
LAYER = 30  # Layer index to steer
ADAPTER = 'up_proj'  # Adapter type: 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
STEERING_STRENGTHS = [0.0, -10.0, -25.0, -50.0, -100.0, -200.0, -500.0, -1000.0, -2000.0]  # List of steering strengths to test
PROBLEM_IDX = 10  # MATH500 problem index
PREFIX_TOKENS = 100  # Number of tokens to generate for prefix
ADDITIONAL_TOKENS = 100  # Additional tokens to generate with steering

print(f"Configuration:")
print(f"  Layer: {LAYER}")
print(f"  Adapter: {ADAPTER}")
print(f"  Steering strengths: {STEERING_STRENGTHS}")
print(f"  Problem index: {PROBLEM_IDX}")

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*544")
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

# Validate layer
if LAYER < 0 or LAYER >= n_layers:
    raise ValueError(f"Layer {LAYER} is out of range. Model has {n_layers} layers (0-{n_layers-1})")

# %%
# Extract B matrix from specified adapter at specified layer
print(f"\nExtracting LoRA B matrix from layer {LAYER} {ADAPTER}...")

# Navigate to the correct module based on adapter type
if ADAPTER in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    # Attention adapters
    module = getattr(model.model.model.layers[LAYER].self_attn, ADAPTER)
elif ADAPTER in ['gate_proj', 'up_proj', 'down_proj']:
    # MLP adapters
    module = getattr(model.model.model.layers[LAYER].mlp, ADAPTER)
else:
    raise ValueError(f"Unknown adapter type: {ADAPTER}")

# Extract the LoRA B matrix
if hasattr(module, 'lora_B'):
    # Get the B matrix from the LoRA adapter
    lora_B_weight = module.lora_B['default'].weight.data
    # For rank-1, this should be shape [output_dim, 1]
    # We want a 1D vector of shape [output_dim]
    b_direction = lora_B_weight.squeeze()
    print(f"B matrix shape: {b_direction.shape}")
    print(f"B matrix L2 norm: {torch.norm(b_direction).item():.3f}")
else:
    raise ValueError(f"No LoRA adapter found at layer {LAYER} {ADAPTER}")

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
# Format the prompt using the same format as the training data
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

# Decode the prefix to show what was generated
prefix_text = tokenizer.decode(prefix_output[0], skip_special_tokens=False)
print(f"\nPrefix generated:")
print("=" * 80)
print(prefix_text)
print("=" * 80)

# Now we'll use this prefix as the starting point for steering experiments
prefix_ids = prefix_output

# No longer calculating average L2 norms - steering directly with strength

# %%
# Generate baseline (no steering) for comparison
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
# Create steering hooks for different adapter types
class AttentionSteeringHook:
    """Hook for attention adapters (q_proj, k_proj, v_proj, o_proj) - steers residual stream"""
    def __init__(self, steering_vector, prefix_len):
        self.steering_vector = steering_vector
        self.prefix_len = prefix_len
        self.applied_count = 0
    
    def __call__(self, module, input, output):
        # The layer output is a tuple: (hidden_states, present_key_value)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Get current sequence length
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # During generation, we process one token at a time
        # Check if we're past the prefix
        if seq_len == 1:  # Single token generation
            # Ensure steering vector has same dtype as hidden states
            steering = self.steering_vector.to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states + steering
            self.applied_count += 1
        elif seq_len > self.prefix_len:
            # Full sequence processing - apply to positions after prefix
            steering = self.steering_vector.to(hidden_states.dtype)
            hidden_states[:, self.prefix_len:, :] = hidden_states[:, self.prefix_len:, :] + steering
            self.applied_count += seq_len - self.prefix_len
        
        # Return in the same format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

class MLPSteeringHook:
    """Hook for MLP adapters - steers at the appropriate point in MLP computation"""
    def __init__(self, steering_vector, prefix_len, adapter_type):
        self.steering_vector = steering_vector
        self.prefix_len = prefix_len
        self.adapter_type = adapter_type
        self.applied_count = 0
    
    def __call__(self, module, input, output):
        # For up_proj and gate_proj: output is MLP hidden dimension
        # For down_proj: output is residual stream dimension
        
        # Get current sequence length
        if len(output.shape) == 3:
            batch_size, seq_len, hidden_size = output.shape
        else:
            # Handle case where batch dimension might be squeezed
            seq_len, hidden_size = output.shape
            output = output.unsqueeze(0)
        
        # During generation, we process one token at a time
        if seq_len == 1:  # Single token generation
            steering = self.steering_vector.to(output.dtype).unsqueeze(0).unsqueeze(0)
            output = output + steering
            self.applied_count += 1
        elif seq_len > self.prefix_len:
            # Full sequence processing - apply to positions after prefix
            steering = self.steering_vector.to(output.dtype)
            output[:, self.prefix_len:, :] = output[:, self.prefix_len:, :] + steering
            self.applied_count += seq_len - self.prefix_len
        
        return output

# %%
# Iterate through different steering strengths
all_results = []

for steering_strength in STEERING_STRENGTHS:
    print(f"\n\n{'='*80}")
    print(f"Testing steering strength: {steering_strength}")
    print(f"{'='*80}")
    
    # Calculate steering vector - direct multiplication
    steering_vector = b_direction * steering_strength
    
    print(f"B matrix L2 norm: {torch.norm(b_direction).item():.3f}")
    print(f"Steering strength multiplier: {steering_strength}")
    print(f"Final steering vector L2 norm: {torch.norm(steering_vector).item():.3f}")
    
    # Create appropriate steering hook based on adapter type
    if ADAPTER in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        # For attention adapters, steer the residual stream after the layer
        steering_hook = AttentionSteeringHook(steering_vector, prefix_ids.shape[1])
        # Register hook on the layer output
        hook_handle = model.model.model.layers[LAYER].register_forward_hook(steering_hook)
    elif ADAPTER in ['gate_proj', 'up_proj']:
        # For gate_proj and up_proj, steer the MLP hidden states
        steering_hook = MLPSteeringHook(steering_vector, prefix_ids.shape[1], ADAPTER)
        # Register hook on the specific projection module
        module = getattr(model.model.model.layers[LAYER].mlp, ADAPTER)
        hook_handle = module.register_forward_hook(steering_hook)
    elif ADAPTER == 'down_proj':
        # For down_proj, we can steer either at the output or the residual stream
        # Let's steer at the residual stream for consistency
        steering_hook = AttentionSteeringHook(steering_vector, prefix_ids.shape[1])
        hook_handle = model.model.model.layers[LAYER].register_forward_hook(steering_hook)
    else:
        raise ValueError(f"Unknown adapter type: {ADAPTER}")
    
    # Generate with steering
    with torch.no_grad():
        steered_output = model.generate(
            prefix_ids,
            max_new_tokens=ADDITIONAL_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Remove hook
    hook_handle.remove()
    
    print(f"Steering applied {steering_hook.applied_count} times")
    
    # Decode only the new tokens (after prefix)
    new_tokens = steered_output[0, prefix_ids.shape[1]:]
    steered_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    print(f"\nSteered output (full):")
    print(steered_text)
    
    # Show difference from baseline
    is_different = steered_text != baseline_text
    
    if is_different:
        print("\n[DIFFERENT FROM BASELINE]")
        
        # Calculate simple metrics
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
            
            # Show context around first difference
            if first_diff_pos > 0:
                # Decode tokens up to the difference
                prefix_tokens = baseline_tokens[:first_diff_pos]
                prefix_text = tokenizer.decode(prefix_tokens)
                print(f"\nContext before first difference:")
                print(f"...{prefix_text[-100:]}[DIFF HERE]")
        
        # Calculate token-level similarity
        min_len = min(len(baseline_tokens), len(steered_tokens))
        if min_len > 0:
            matching_tokens = sum(1 for b, s in zip(baseline_tokens[:min_len], steered_tokens[:min_len]) if b == s)
            similarity = matching_tokens / min_len
            print(f"\nToken-level similarity: {similarity:.2%}")
    else:
        print("\n[SAME AS BASELINE]")
    
    # Store results for this strength
    result = {
        "layer": LAYER,
        "adapter": ADAPTER,
        "steering_strength": steering_strength,
        "problem_idx": PROBLEM_IDX,
        "question": question,
        "prefix": prefix_text,
        "baseline": baseline_text,
        "steered": steered_text,
        "different": is_different,
        "b_matrix_norm": torch.norm(b_direction).item(),
        "steering_vector_norm": torch.norm(steering_vector).item(),
        "steering_applications": steering_hook.applied_count
    }
    
    all_results.append(result)

# %%
# Save all results to file
output_file = f"lora_steering_layer{LAYER}_{ADAPTER}_multistrength.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n\nAll results saved to {output_file}")

# %%
# Print summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Layer: {LAYER}")
print(f"Adapter: {ADAPTER}")
print(f"Problem: {question[:100]}...")
print(f"\nPrefix ended with: ...{tokenizer.decode(prefix_ids[0, -20:], skip_special_tokens=False)}")
print(f"\nBaseline continuation: {baseline_text[:100]}...")

print(f"\nResults across steering strengths:")
for result in all_results:
    strength = result['steering_strength']
    different = result['different']
    status = 'DIFFERENT' if different else 'SAME'
    print(f"  Strength {strength:6.1f}: {status}")

# %%