# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/root/reasoning_interp/ckpts_1.1"
rank = 1
context_window = 10  # Number of tokens before and after
top_k = 16  # Number of top activating examples

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

# %%
# Extract A matrices for gate_proj and up_proj from each layer
print("Extracting A matrices for gate_proj and up_proj...")

probe_directions = {
    'gate_proj': {},
    'up_proj': {}
}

# Get the number of layers
n_layers = model.config.num_hidden_layers

for layer_idx in range(n_layers):
    for proj_type in ['gate_proj', 'up_proj']:
        # Access the module directly
        module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
        
        # Extract the LoRA A matrix (which is a vector for rank-1)
        if hasattr(module, 'lora_A'):
            # Get the A matrix from the LoRA adapter
            lora_A_weight = module.lora_A['default'].weight.data
            # For rank-1, this could be shape [hidden_size, 1] or [1, hidden_size]
            # We want a 1D vector of shape [hidden_size]
            probe_direction = lora_A_weight.squeeze()
            probe_directions[proj_type][layer_idx] = probe_direction

print(f"Extracted directions for {len(probe_directions['gate_proj'])} layers")

# %%
# Load s1K-1.1 dataset
print("Loading s1K-1.1 dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"Dataset has {len(dataset)} examples")

# Check the structure of the dataset
print(f"Dataset features: {dataset.features}")
print(f"First example keys: {list(dataset[0].keys())}")
print(f"First example preview: {str(dataset[0])[:500]}...")

# %%
@dataclass
class ActivationExample:
    """Store activation with context"""
    rollout_idx: int
    token_idx: int
    token: str
    activation: float
    context_before: List[str]
    context_after: List[str]
    layer: int
    proj_type: str

# Storage for all activations
all_activations = {
    'gate_proj': {layer: [] for layer in range(n_layers)},
    'up_proj': {layer: [] for layer in range(n_layers)}
}

# %%
# Process rollouts
# Using only 10 examples for faster processing
num_examples = min(10, len(dataset))
print(f"Processing {num_examples} rollouts...")

# Process rollouts - using DeepSeek traces and attempts
for rollout_idx in tqdm(range(num_examples), desc="Processing rollouts"):
    # Get the rollout
    rollout = dataset[rollout_idx]
    
    # Extract question and DeepSeek thinking trajectory + attempt
    question = rollout['question']
    thinking_trajectory = rollout.get('deepseek_thinking_trajectory', '')
    attempt = rollout.get('deepseek_attempt', '')
    
    if not thinking_trajectory or not attempt:
        print(f"Skipping rollout {rollout_idx}: missing DeepSeek thinking trajectory or attempt")
        continue
    
    # Use the exact format for thinking traces
    # <|im_start|>system\n(system prompt)\n<|im_start|>user\n(question)\n<|im_start|>assistant\n<|im_start|>think\n(thinking trace)<|im_start|>answer\n(answer)<|im_end|>
    
    system_prompt = "You are a helpful mathematics assistant."
    
    full_text = (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{question}\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{thinking_trajectory}\n"
        f"<|im_start|>answer\n{attempt}<|im_end|>"
    )
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    
    # Get tokens for context
    tokens = []
    for token_id in input_ids:
        decoded = tokenizer.decode([token_id])
        tokens.append(decoded)
    
    # Storage for residual streams
    residual_streams = {}
    
    # Hook function to capture post-layernorm (pre-MLP) residual stream
    def make_hook(layer_idx):
        def hook(module, input, output):
            residual_streams[layer_idx] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in range(n_layers):
        layernorm = model.model.model.layers[layer_idx].post_attention_layernorm
        hook = layernorm.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute probe activations for each layer and token
    for proj_type in ['gate_proj', 'up_proj']:
        for layer_idx in range(n_layers):
            # Get probe direction and residual stream
            probe_dir = probe_directions[proj_type][layer_idx]
            residual = residual_streams[layer_idx][0]  # [seq_len, hidden_size]
            
            # Compute activations
            activations = torch.matmul(residual.float(), probe_dir)  # [seq_len]
            activations = activations.cpu().numpy()
            
            # Store each activation with context
            for token_idx in range(len(tokens)):
                # Get context window
                context_start = max(0, token_idx - context_window)
                context_end = min(len(tokens), token_idx + context_window + 1)
                
                context_before = tokens[context_start:token_idx]
                context_after = tokens[token_idx+1:context_end]
                
                example = ActivationExample(
                    rollout_idx=rollout_idx,
                    token_idx=token_idx,
                    token=tokens[token_idx],
                    activation=activations[token_idx],
                    context_before=context_before,
                    context_after=context_after,
                    layer=layer_idx,
                    proj_type=proj_type
                )
                
                all_activations[proj_type][layer_idx].append(example)
    
    # Clear GPU memory periodically
    if rollout_idx % 10 == 0:
        torch.cuda.empty_cache()

print("Finished processing all rollouts")

# %%
# Find top activating examples
print("\nFinding top activating examples...")

from IPython.display import HTML, display
import html as html_lib

def create_html_examples(examples: List[ActivationExample], title: str, all_examples: List[ActivationExample]) -> str:
    """Create HTML visualization of activation examples with context"""
    
    # Calculate activation range for normalization
    all_activations = [ex.activation for ex in all_examples]
    min_act = min(all_activations)
    max_act = max(all_activations)
    act_range = max_act - min_act if max_act > min_act else 1
    
    html_parts = [f"<h3>{title}</h3>"]
    
    for i, ex in enumerate(examples):
        # Build context with colored tokens
        context_html = []
        
        # Add context before
        for j, token in enumerate(ex.context_before[-10:]):  # Show last 10 tokens
            # Get activation for this context token if available
            ctx_activation = 0  # Default if we don't have the activation
            # Find activation for this token position
            # The position is: target position - (number of context tokens shown) + current index
            ctx_position = ex.token_idx - len(ex.context_before[-10:]) + j
            for other_ex in all_examples:
                if other_ex.rollout_idx == ex.rollout_idx and other_ex.token_idx == ctx_position:
                    ctx_activation = other_ex.activation
                    break
            
            # Normalize activation for coloring
            # Use absolute value for intensity, sign for color
            intensity = min(abs(ctx_activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
            if ctx_activation > 0:
                bg_color = f"rgba(255, 0, 0, {intensity})"  # Light red for positive
            else:
                bg_color = f"rgba(0, 0, 255, {intensity})"  # Light blue for negative
            
            token_display = html_lib.escape(token).replace('\n', '\\n')
            context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
        
        # Add the target token with red outline
        token_display = html_lib.escape(ex.token).replace('\n', '\\n')
        context_html.append(f'<span style="border: 2px solid red; background-color: yellow; padding: 2px; font-weight: bold;">{token_display}</span>')
        
        # Add context after
        for j, token in enumerate(ex.context_after[:10]):  # Show first 10 tokens
            # Get activation for this context token
            ctx_activation = 0
            for other_ex in all_examples:
                if other_ex.rollout_idx == ex.rollout_idx and other_ex.token_idx == ex.token_idx + j + 1:
                    ctx_activation = other_ex.activation
                    break
            
            # Normalize activation for coloring
            # Use absolute value for intensity, sign for color
            intensity = min(abs(ctx_activation) / max(abs(min_act), abs(max_act)), 1.0) * 0.3
            if ctx_activation > 0:
                bg_color = f"rgba(255, 0, 0, {intensity})"
            else:
                bg_color = f"rgba(0, 0, 255, {intensity})"
            
            token_display = html_lib.escape(token).replace('\n', '\\n')
            context_html.append(f'<span style="background-color: {bg_color}; padding: 2px;">{token_display}</span>')
        
        # Combine into a single line
        html_parts.append(f'<div class="example">')
        html_parts.append(f'<small style="color: #666;">Rollout {ex.rollout_idx}</small><br>')
        html_parts.append(''.join(context_html))
        html_parts.append('</div>')
    
    return '\n'.join(html_parts)

# %%
# Analyze each probe type and layer
html_outputs = {}

for proj_type in ['gate_proj', 'up_proj']:
    # Create HTML output for this probe type
    html_output = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{proj_type.upper()} Probe Activations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            h3 {{ color: #888; margin-top: 20px; }}
            .example {{ margin: 10px 0; font-family: monospace; line-height: 1.8; }}
            hr {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
    <h1>{proj_type.upper()} Probe Activations</h1>
    """
    
    # Process all layers
    for layer_idx in range(n_layers):
        examples = all_activations[proj_type][layer_idx]
        
        # Sort by activation value
        examples_sorted = sorted(examples, key=lambda x: x.activation)
        
        # Get top positive and negative examples
        top_negative = examples_sorted[:top_k]
        top_positive = examples_sorted[-top_k:][::-1]  # Reverse to get highest first
        
        # Add layer header
        html_output += f"<h2>Layer {layer_idx}</h2>"
        
        # Create HTML for positive examples
        html_output += create_html_examples(top_positive, f"Top {top_k} Positive Activations", examples)
        
        # Create HTML for negative examples  
        html_output += create_html_examples(top_negative, f"Top {top_k} Negative Activations", examples)
        
        html_output += "<hr style='margin: 30px 0;'>"
    
    html_output += """
    </body>
    </html>
    """
    
    # Save to file
    filename = f"lora_activations_{proj_type}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f"Saved {proj_type} activations to {filename}")
    
    # Also display in notebook
    display(HTML(html_output))
    
    # Store for combined output
    html_outputs[proj_type] = html_output

# %%
# Create a combined HTML file with both probe types
print("\nCreating combined HTML file...")

combined_html = """
<!DOCTYPE html>
<html>
<head>
    <title>LoRA Probe Activations - All Types</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        h3 { color: #888; margin-top: 20px; }
        .example { margin: 10px 0; font-family: monospace; line-height: 1.8; }
        hr { margin: 30px 0; }
        .toc { background: #f5f5f5; padding: 20px; margin-bottom: 30px; }
        .toc a { color: #333; text-decoration: none; margin: 0 10px; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>LoRA Probe Activations - Combined Results</h1>
    <div class="toc">
        <strong>Navigation:</strong>
        <a href="#gate_proj">Gate Projections</a> |
        <a href="#up_proj">Up Projections</a>
    </div>
"""

# Add gate_proj section
combined_html += '<div id="gate_proj">'
combined_html += html_outputs['gate_proj'].split('<body>')[1].split('</body>')[0]
combined_html += '</div>'

# Add separator
combined_html += '<hr style="border: 2px solid #333; margin: 50px 0;">'

# Add up_proj section
combined_html += '<div id="up_proj">'
combined_html += html_outputs['up_proj'].split('<body>')[1].split('</body>')[0]
combined_html += '</div>'

combined_html += """
</body>
</html>
"""

# Save combined file
with open('lora_activations_combined.html', 'w', encoding='utf-8') as f:
    f.write(combined_html)
print("Saved combined activations to lora_activations_combined.html")

# %%
# Create a summary visualization showing activation patterns
print("\n\nCreating summary statistics...")

summary_data = []
for proj_type in ['gate_proj', 'up_proj']:
    for layer_idx in range(n_layers):
        examples = all_activations[proj_type][layer_idx]
        activations = np.array([ex.activation for ex in examples])
        
        summary_data.append({
            'proj_type': proj_type,
            'layer': layer_idx,
            'mean': activations.mean(),
            'std': activations.std(),
            'min': activations.min(),
            'max': activations.max(),
            'q25': np.percentile(activations, 25),
            'q50': np.percentile(activations, 50),
            'q75': np.percentile(activations, 75),
            'q95': np.percentile(activations, 95),
            'q99': np.percentile(activations, 99)
        })

summary_df = pd.DataFrame(summary_data)
print("\nSummary statistics by layer:")
print(summary_df[summary_df['layer'].isin([0, n_layers//2, n_layers-1])].to_string())

# %%
# Save the top examples for further analysis
print("\nSaving top examples to JSON...")

top_examples_data = {}
for proj_type in ['gate_proj', 'up_proj']:
    top_examples_data[proj_type] = {}
    
    for layer_idx in range(n_layers):
        examples = all_activations[proj_type][layer_idx]
        examples_sorted = sorted(examples, key=lambda x: x.activation)
        
        # Get top examples
        top_negative = examples_sorted[:top_k]
        top_positive = examples_sorted[-top_k:][::-1]
        
        # Convert to serializable format
        layer_data = {
            'top_positive': [
                {
                    'rollout_idx': ex.rollout_idx,
                    'token_idx': ex.token_idx,
                    'token': ex.token,
                    'activation': float(ex.activation),
                    'context': ''.join(ex.context_before[-5:]) + f'[{ex.token}]' + ''.join(ex.context_after[:5])
                }
                for ex in top_positive
            ],
            'top_negative': [
                {
                    'rollout_idx': ex.rollout_idx,
                    'token_idx': ex.token_idx,
                    'token': ex.token,
                    'activation': float(ex.activation),
                    'context': ''.join(ex.context_before[-5:]) + f'[{ex.token}]' + ''.join(ex.context_after[:5])
                }
                for ex in top_negative
            ]
        }
        
        top_examples_data[proj_type][f'layer_{layer_idx}'] = layer_data

# Save to file
with open('lora_top_activating_examples.json', 'w') as f:
    json.dump(top_examples_data, f, indent=2)

print("Saved top examples to lora_top_activating_examples.json")

# %%