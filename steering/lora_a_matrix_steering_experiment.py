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
import random

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1
prefix_tokens = 100  # Number of tokens to generate for prefix
additional_tokens = 100  # Additional tokens to generate with steering
steering_strength = 10.0  # Default multiplier for steering

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

# %%
# Extract A matrices from up_proj and gate_proj for all layers
print("Extracting LoRA A matrices from up_proj and gate_proj...")
a_matrices_up = {}
a_matrices_gate = {}

for layer_idx in range(n_layers):
    # Access the up_proj module
    up_proj = model.model.model.layers[layer_idx].mlp.up_proj
    
    # Extract the LoRA A matrix from up_proj
    if hasattr(up_proj, 'lora_A'):
        # Get the A matrix from the LoRA adapter
        lora_A_weight = up_proj.lora_A['default'].weight.data
        # For rank-1, this should be shape [1, input_dim]
        # We want a 1D vector of shape [input_dim]
        a_direction = lora_A_weight.squeeze()
        a_matrices_up[layer_idx] = a_direction
        print(f"Layer {layer_idx}: up_proj A matrix shape = {a_direction.shape}")
    
    # Access the gate_proj module
    gate_proj = model.model.model.layers[layer_idx].mlp.gate_proj
    
    # Extract the LoRA A matrix from gate_proj
    if hasattr(gate_proj, 'lora_A'):
        # Get the A matrix from the LoRA adapter
        lora_A_weight = gate_proj.lora_A['default'].weight.data
        # For rank-1, this should be shape [1, input_dim]
        # We want a 1D vector of shape [input_dim]
        a_direction = lora_A_weight.squeeze()
        a_matrices_gate[layer_idx] = a_direction
        print(f"Layer {layer_idx}: gate_proj A matrix shape = {a_direction.shape}")

print(f"Extracted A matrices for {len(a_matrices_up)} layers (up_proj)")
print(f"Extracted A matrices for {len(a_matrices_gate)} layers (gate_proj)")

# %%
# Load MATH500 dataset
print("\nLoading MATH500 dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
print(f"Dataset has {len(dataset)} examples")

# Sample a random prompt from MATH500
random_idx = 10  # Fixed for reproducibility
problem = dataset[random_idx]
question = problem['problem']
print(f"\nSelected problem {random_idx}:")
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
# Generate prefix (100 tokens)
print(f"\nGenerating {prefix_tokens} tokens for prefix...")
with torch.no_grad():
    prefix_output = model.generate(
        input_ids,
        max_new_tokens=prefix_tokens,
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

# %%
# Calculate average L2 norm of residual stream activations at each layer
print(f"\nCalculating average L2 norms of residual stream activations...")
residual_norms = {}

# Calculate where generated tokens start
original_prompt_len = input_ids.shape[1]
first_generated_pos = original_prompt_len
# Start calculating from 5 tokens after the first generated token
norm_start_pos = first_generated_pos + 5

print(f"Original prompt length: {original_prompt_len}")
print(f"First generated token position: {first_generated_pos}")
print(f"Starting norm calculation from position: {norm_start_pos}")

# Hook to capture residual stream activations (post-attn, pre-MLP)
def make_residual_hook(layer_idx):
    def hook(module, input, output):
        # For transformer layers, the output is (hidden_states, present_key_value)
        # We want the hidden_states which is the residual stream
        if isinstance(output, tuple):
            residual = output[0]
        else:
            residual = output
        
        # residual is [batch, seq_len, hidden_size]
        residual = residual[0]  # [seq_len, hidden_size]
        
        # Calculate L2 norm for each position
        l2_norms = torch.norm(residual, p=2, dim=-1)  # [seq_len]
        
        # Only average over generated tokens, starting 5 tokens after first generated
        if len(l2_norms) > norm_start_pos:
            relevant_norms = l2_norms[norm_start_pos:]
            residual_norms[layer_idx] = relevant_norms.mean().item()
        else:
            # If we haven't generated enough tokens yet, use what we have
            residual_norms[layer_idx] = l2_norms[first_generated_pos:].mean().item()
    return hook

# Register hooks to capture residual norms after attention, before MLP
hooks = []
for layer_idx in range(n_layers):
    # Hook the self_attn module to capture its output (which becomes MLP input)
    attn = model.model.model.layers[layer_idx].self_attn
    hook = attn.register_forward_hook(make_residual_hook(layer_idx))
    hooks.append(hook)

# Run forward pass to collect norms
with torch.no_grad():
    outputs = model(prefix_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

print("\nAverage L2 norms of residual stream per layer (calculated from generated tokens only):")
for layer_idx in range(n_layers):
    print(f"Layer {layer_idx}: {residual_norms[layer_idx]:.3f}")

# %%
# Plot average L2 norms across layers
import plotly.express as px
import pandas as pd

# Create dataframe for plotting
norm_data = pd.DataFrame({
    'Layer': list(range(n_layers)),
    'Average L2 Norm': [residual_norms[i] for i in range(n_layers)]
})

# Create bar chart
fig = px.bar(norm_data, x='Layer', y='Average L2 Norm',
             title='Average Residual Stream L2 Norms by Layer (Generated Tokens Only)',
             labels={'Layer': 'Layer Index', 'Average L2 Norm': 'Average L2 Norm'},
             color='Average L2 Norm',
             color_continuous_scale='Viridis')

# Update layout
fig.update_layout(
    xaxis_title="Layer",
    yaxis_title="Average L2 Norm",
    showlegend=False,
    height=600,
    width=1000
)

# Show the plot
fig.show()

# Save the plot
fig.write_html("residual_stream_norms_by_layer.html")
print("Saved norm plot to residual_stream_norms_by_layer.html")

# %%
# Generate baseline (no steering) for comparison
print(f"\n\n{'='*80}")
print("Generating baseline (no steering)...")
print(f"{'='*80}")

with torch.no_grad():
    baseline_output = model.generate(
        prefix_ids,
        max_new_tokens=additional_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

baseline_new_tokens = baseline_output[0, prefix_ids.shape[1]:]
baseline_text = tokenizer.decode(baseline_new_tokens, skip_special_tokens=False)

print("\nBaseline output (no steering):")
print(baseline_text)

# %%
# Steering hook that adds to residual stream after attention, before MLP
class ResidualSteeringHook:
    def __init__(self, target_layer, steering_vector, prefix_len):
        self.target_layer = target_layer
        self.steering_vector = steering_vector
        self.prefix_len = prefix_len
        self.applied_count = 0
    
    def __call__(self, module, input, output):
        # The attention output is a tuple: (hidden_states, present_key_value, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Get current sequence length
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # During generation, we process one token at a time
        # Check if we're past the prefix
        if seq_len == 1:  # Single token generation
            # Apply steering to all single-token forwards after prefix
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

# %%
# Test steering with up_proj A matrices
print(f"\n{'='*80}")
print("Testing steering with up_proj A matrices...")
print(f"{'='*80}")

up_proj_results = {}

for target_layer in tqdm(range(n_layers), desc="Testing up_proj layers"):
    print(f"\n\nSteering at Layer {target_layer} (up_proj)")
    print("-" * 40)
    
    # Get A matrix and scaling for this layer
    a_direction = a_matrices_up[target_layer].to(model.device)
    avg_norm = residual_norms[target_layer]
    steering_vector = a_direction * avg_norm * steering_strength
    
    print(f"A matrix L2 norm: {torch.norm(a_direction).item():.3f}")
    print(f"Average residual stream norm: {avg_norm:.3f}")
    print(f"Steering strength multiplier: {steering_strength}")
    print(f"Final steering vector L2 norm: {torch.norm(steering_vector).item():.3f}")
    
    # Create steering hook
    steering_hook = ResidualSteeringHook(target_layer, steering_vector, prefix_ids.shape[1])
    
    # Register hook on the attention output (which feeds into MLP)
    hook_handle = model.model.model.layers[target_layer].self_attn.register_forward_hook(steering_hook)
    
    # Generate with steering
    with torch.no_grad():
        steered_output = model.generate(
            prefix_ids,
            max_new_tokens=additional_tokens,
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
    
    # Store result
    up_proj_results[target_layer] = steered_text
    
    print(f"\nSteered output:")
    print(steered_text)
    
    # Show difference from baseline
    if steered_text != baseline_text:
        print("\n[DIFFERENT FROM BASELINE]")
    else:
        print("\n[SAME AS BASELINE]")

# %%
# Test steering with gate_proj A matrices

print(f"\n{'='*80}")
print("Testing steering with gate_proj A matrices...")
print(f"{'='*80}")

gate_proj_results = {}

for target_layer in tqdm(range(n_layers), desc="Testing gate_proj layers"):
    print(f"\n\nSteering at Layer {target_layer} (gate_proj)")
    print("-" * 40)
    
    # Get A matrix and scaling for this layer
    a_direction = a_matrices_gate[target_layer].to(model.device)
    avg_norm = residual_norms[target_layer]
    steering_vector = a_direction * np.sqrt(avg_norm) * steering_strength
    
    print(f"A matrix L2 norm: {torch.norm(a_direction).item():.3f}")
    print(f"Average residual stream norm: {avg_norm:.3f}")
    print(f"Steering strength multiplier: {steering_strength}")
    print(f"Final steering vector L2 norm: {torch.norm(steering_vector).item():.3f}")
    
    # Create steering hook
    steering_hook = ResidualSteeringHook(target_layer, steering_vector, prefix_ids.shape[1])
    
    # Register hook on the attention output
    hook_handle = model.model.model.layers[target_layer].self_attn.register_forward_hook(steering_hook)
    
    # Generate with steering
    with torch.no_grad():
        steered_output = model.generate(
            prefix_ids,
            max_new_tokens=additional_tokens,
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
    
    # Store result
    gate_proj_results[target_layer] = steered_text
    
    print(f"\nSteered output:")
    print(steered_text)
    
    # Show difference from baseline
    if steered_text != baseline_text:
        print("\n[DIFFERENT FROM BASELINE]")
    else:
        print("\n[SAME AS BASELINE]")

# %%
# Save results to file
results = {
    "problem_idx": random_idx,
    "question": question,
    "prefix": prefix_text,
    "baseline": baseline_text,
    "residual_stream_norms": residual_norms,
    "steering_strength": steering_strength,
    "up_proj_steered_outputs": up_proj_results,
    "gate_proj_steered_outputs": gate_proj_results
}

output_file = "lora_a_matrix_steering_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\nResults saved to {output_file}")

# %%
# Print summary
print(f"\n{'='*80}")
print("SUMMARY OF A-MATRIX STEERING EFFECTS")
print(f"{'='*80}")
print(f"\nProblem: {question[:100]}...")
print(f"\nPrefix ended with: ...{tokenizer.decode(prefix_ids[0, -20:], skip_special_tokens=False)}")
print(f"\nBaseline continuation: {baseline_text[:100]}...")

# Count how many layers had different outputs for up_proj
up_different_count = sum(1 for layer, text in up_proj_results.items() if text != baseline_text)
print(f"\nup_proj: {up_different_count} out of {n_layers} layers produced different outputs")

# Count how many layers had different outputs for gate_proj
gate_different_count = sum(1 for layer, text in gate_proj_results.items() if text != baseline_text)
print(f"gate_proj: {gate_different_count} out of {n_layers} layers produced different outputs")

print(f"\nup_proj steering effects by layer (first 5):")
for layer in range(min(5, n_layers)):
    is_different = up_proj_results[layer] != baseline_text
    print(f"\nLayer {layer} {'[DIFFERENT]' if is_different else '[SAME]'}: {up_proj_results[layer][:100]}...")

print(f"\ngate_proj steering effects by layer (first 5):")
for layer in range(min(5, n_layers)):
    is_different = gate_proj_results[layer] != baseline_text
    print(f"\nLayer {layer} {'[DIFFERENT]' if is_different else '[SAME]'}: {gate_proj_results[layer][:100]}...")

# %%