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
steering_strength = 50.0  # Default multiplier for steering

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
# Extract B matrices from down_proj for all layers
print("Extracting LoRA B matrices from down_proj...")
b_matrices = {}

for layer_idx in range(n_layers):
    # Access the down_proj module
    down_proj = model.model.model.layers[layer_idx].mlp.down_proj
    
    # Extract the LoRA B matrix
    if hasattr(down_proj, 'lora_B'):
        # Get the B matrix from the LoRA adapter
        lora_B_weight = down_proj.lora_B['default'].weight.data
        # For rank-1, this should be shape [output_dim, 1]
        # We want a 1D vector of shape [output_dim]
        b_direction = lora_B_weight.squeeze()
        b_matrices[layer_idx] = b_direction
        print(f"Layer {layer_idx}: B matrix shape = {b_direction.shape}")

print(f"Extracted B matrices for {len(b_matrices)} layers")

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
# First, calculate average L2 norm of mlp_out activations at each layer
print(f"\nCalculating average L2 norms of mlp_out activations...")
mlp_out_norms = {}

# Calculate where generated tokens start
original_prompt_len = input_ids.shape[1]
first_generated_pos = original_prompt_len
# Start calculating from 5 tokens after the first generated token
norm_start_pos = first_generated_pos + 5

print(f"Original prompt length: {original_prompt_len}")
print(f"First generated token position: {first_generated_pos}")
print(f"Starting norm calculation from position: {norm_start_pos}")

# Hook to capture mlp_out activations
def make_mlp_out_hook(layer_idx):
    def hook(module, input, output):
        # output is the mlp_out tensor [batch, seq_len, hidden_size]
        mlp_out = output[0]  # [seq_len, hidden_size]
        # Calculate L2 norm for each position
        l2_norms = torch.norm(mlp_out, p=2, dim=-1)  # [seq_len]
        # Only average over generated tokens, starting 5 tokens after first generated
        if len(l2_norms) > norm_start_pos:
            relevant_norms = l2_norms[norm_start_pos:]
            mlp_out_norms[layer_idx] = relevant_norms.mean().item()
        else:
            # If we haven't generated enough tokens yet, use what we have
            mlp_out_norms[layer_idx] = l2_norms[first_generated_pos:].mean().item()
    return hook

# Register hooks to capture mlp_out norms
hooks = []
for layer_idx in range(n_layers):
    mlp = model.model.model.layers[layer_idx].mlp
    hook = mlp.register_forward_hook(make_mlp_out_hook(layer_idx))
    hooks.append(hook)

# Run forward pass to collect norms
with torch.no_grad():
    outputs = model(prefix_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

print("\nAverage L2 norms per layer (calculated from generated tokens only):")
for layer_idx in range(n_layers):
    print(f"Layer {layer_idx}: {mlp_out_norms[layer_idx]:.3f}")

# %%
# Plot average L2 norms across layers
import plotly.express as px
import pandas as pd

# Create dataframe for plotting
norm_data = pd.DataFrame({
    'Layer': list(range(n_layers)),
    'Average L2 Norm': [mlp_out_norms[i] for i in range(n_layers)]
})

# Create bar chart
fig = px.bar(norm_data, x='Layer', y='Average L2 Norm',
             title='Average MLP Output L2 Norms by Layer (Generated Tokens Only)',
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
fig.write_html("mlp_out_norms_by_layer.html")
print("Saved norm plot to mlp_out_norms_by_layer.html")

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
# Now perform steering experiments for each layer
print(f"\n{'='*80}")
print("Starting steering experiments...")
print(f"{'='*80}")

# Store results for each layer
steering_results = {}

# Create a proper steering hook that adds to the residual stream after MLP
class SteeringHook:
    def __init__(self, target_layer, steering_vector, prefix_len):
        self.target_layer = target_layer
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
            # We need to track the total sequence position
            # This is a bit tricky - we'll apply steering to all single-token forwards
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

for target_layer in tqdm(range(n_layers), desc="Testing layers"):
    print(f"\n\nSteering at Layer {target_layer}")
    print("-" * 40)
    
    # Get B matrix and scaling for this layer
    b_direction = b_matrices[target_layer].to(model.device)
    avg_norm = mlp_out_norms[target_layer]
    steering_vector = b_direction * np.sqrt(avg_norm) * steering_strength
    
    print(f"B matrix L2 norm: {torch.norm(b_direction).item():.3f}")
    print(f"Average MLP out norm: {avg_norm:.3f}")
    print(f"Steering strength multiplier: {steering_strength}")
    print(f"Final steering vector L2 norm: {torch.norm(steering_vector).item():.3f}")
    
    # Create steering hook
    steering_hook = SteeringHook(target_layer, steering_vector, prefix_ids.shape[1])
    
    # Register hook on the target layer only
    hook_handle = model.model.model.layers[target_layer].register_forward_hook(steering_hook)
    
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
    steering_results[target_layer] = steered_text
    
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
    "mlp_out_norms": mlp_out_norms,
    "steering_strength": steering_strength,
    "steered_outputs": steering_results
}

output_file = "lora_steering_results_fixed.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\nResults saved to {output_file}")

# %%
# Print summary
print(f"\n{'='*80}")
print("SUMMARY OF STEERING EFFECTS")
print(f"{'='*80}")
print(f"\nProblem: {question[:100]}...")
print(f"\nPrefix ended with: ...{tokenizer.decode(prefix_ids[0, -20:], skip_special_tokens=False)}")
print(f"\nBaseline continuation: {baseline_text[:100]}...")

# Count how many layers had different outputs
different_count = sum(1 for layer, text in steering_results.items() if text != baseline_text)
print(f"\n{different_count} out of {n_layers} layers produced different outputs")

print(f"\nSteering effects by layer (first 5):")
for layer in range(min(5, n_layers)):
    is_different = steering_results[layer] != baseline_text
    print(f"\nLayer {layer} {'[DIFFERENT]' if is_different else '[SAME]'}: {steering_results[layer][:100]}...")

# %%