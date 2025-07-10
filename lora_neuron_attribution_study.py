# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc
from dataclasses import dataclass
from tabulate import tabulate
from tqdm import tqdm

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-2*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# %%
class StopGradientHooks:
    """Manages stop-gradient hooks for LayerNorm and Attention patterns"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_hooks = []
        self.layernorm_hooks = []
        self.original_forwards = {}
        
    def _stop_gradient_hook(self, module, input, output):
        """Hook that stops gradients through the output"""
        # Detach the output to stop gradient flow
        # But keep the computation graph intact for other paths
        return output.detach()
    
    def _create_attention_monkey_patch(self, original_forward):
        """Create a monkey-patched forward function that stops gradients through attention patterns"""
        def patched_forward(*args, **kwargs):
            # Get the original module (self in the forward context)
            module = args[0]
            
            # Extract inputs
            if len(args) > 1:
                hidden_states = args[1]
            else:
                hidden_states = kwargs.get('hidden_states')
            
            # Call original forward to get all computations
            # We need to handle output_attentions flag
            original_output_attentions = kwargs.get('output_attentions', False)
            kwargs['output_attentions'] = True  # Force attention weights output
            
            # Get full output with attention weights
            outputs = original_forward(*args, **kwargs)
            
            # Process outputs based on structure
            if isinstance(outputs, tuple):
                attention_output = outputs[0]
                attention_weights = outputs[1] if len(outputs) > 1 else None
                
                if attention_weights is not None:
                    # Detach attention weights to stop gradients
                    attention_weights = attention_weights.detach()
                
                # Reconstruct output based on original request
                if original_output_attentions:
                    return (attention_output, attention_weights) + outputs[2:]
                else:
                    # Don't include attention weights if not originally requested
                    return (attention_output,) + outputs[2:]
            
            return outputs
        
        return patched_forward
    
    def _monkey_patch_attention_modules(self):
        """Monkey-patch attention modules to stop gradients through attention patterns"""
        self.original_forwards = {}
        
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
                # Store original forward
                self.original_forwards[f'layer_{layer_idx}'] = attn_module.forward
                # Create patched forward
                patched_forward = self._create_attention_monkey_patch(attn_module.forward)
                # Apply monkey patch
                attn_module.forward = patched_forward.__get__(attn_module, attn_module.__class__)
        
        print(f\"Monkey-patched {len(self.original_forwards)} attention modules\")
    
    def register_layernorm_hooks(self):
        """Register stop-gradient hooks on all LayerNorm modules"""
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            # Hook input LayerNorm (pre-attention)
            if hasattr(layer, 'input_layernorm'):
                hook = layer.input_layernorm.register_forward_hook(self._stop_gradient_hook)
                self.layernorm_hooks.append(hook)
            
            # Hook post-attention LayerNorm (pre-MLP)
            if hasattr(layer, 'post_attention_layernorm'):
                hook = layer.post_attention_layernorm.register_forward_hook(self._stop_gradient_hook)
                self.layernorm_hooks.append(hook)
        
        # Hook final LayerNorm
        if hasattr(self.model.model.model, 'norm'):
            hook = self.model.model.model.norm.register_forward_hook(self._stop_gradient_hook)
            self.layernorm_hooks.append(hook)
        
        print(f"Registered {len(self.layernorm_hooks)} LayerNorm stop-gradient hooks")
    
    def register_attention_hooks(self):
        """Register hooks to stop gradients through attention patterns"""
        # Use monkey-patching instead of hooks for attention patterns
        self._monkey_patch_attention_modules()
    
    def register_all_hooks(self):
        """Register all stop-gradient hooks"""
        self.register_layernorm_hooks()
        self.register_attention_hooks()
        self.hooks = self.layernorm_hooks + self.attention_hooks
    
    def remove_all_hooks(self):
        """Remove all registered hooks and restore original functions"""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layernorm_hooks = []
        self.attention_hooks = []
        
        # Restore original attention forwards
        if hasattr(self, 'original_forwards'):
            for layer_idx in range(self.model.config.num_hidden_layers):
                layer = self.model.model.model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    key = f'layer_{layer_idx}'
                    if key in self.original_forwards:
                        layer.self_attn.forward = self.original_forwards[key]
            print(f"Restored {len(self.original_forwards)} original attention modules")


class LoRANeuronTracker:
    """Tracks LoRA neuron activations during forward pass while preserving gradient flow"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}  # {layer_name: tensor}
        self.hooks = []
        
    def _create_hook(self, layer_name: str, adapter_name: str = 'default'):
        """Create a forward hook that captures activations and maintains gradient flow"""
        
        def hook_fn(module, input, output):
            # output shape: [batch_size, seq_len, 1] for rank-1
            # CRITICAL: We need to ensure the tensor requires grad
            if not output.requires_grad:
                output.requires_grad_(True)
            
            # Now we can retain gradients
            output.retain_grad()
            
            # Store reference to the original tensor (not a clone!)
            key = f"{layer_name}.{adapter_name}"
            self.activations[key] = output
            
            # Return unchanged to preserve computation graph
            return output
            
        return hook_fn
    
    def register_hooks(self):
        """Register hooks on MLP LoRA A matrices only"""
        
        # Navigate through model structure
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            # Only check MLP projections (skip attention)
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                module = getattr(layer.mlp, proj_name, None)
                if module and hasattr(module, 'lora_A'):
                    for adapter_name, lora_A_module in module.lora_A.items():
                        hook = self._create_hook(f"layer{layer_idx}.mlp.{proj_name}", adapter_name)
                        handle = lora_A_module.register_forward_hook(hook)
                        self.hooks.append(handle)
        
        print(f"Registered {len(self.hooks)} hooks on MLP LoRA A matrices")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# %%
# Load tokenizer
print("Loading tokenizer...")
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
import json
import os

# Configuration for MATH-500
example_idx = 10  # 10th example as requested
max_new_tokens = 4096
generation_cache_file = f"math500_generation_example_{example_idx}.json"

# %%
# Load MATH-500 dataset
print("Loading MATH-500 dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# Get the 10th example
example = dataset[example_idx]
problem = example['problem']

print(f"\nUsing example {example_idx}:")
print(f"Problem: {problem[:200]}..." if len(problem) > 200 else f"Problem: {problem}")

# %%
# Check if generation already exists
if os.path.exists(generation_cache_file):
    print(f"\nLoading cached generation from {generation_cache_file}")
    with open(generation_cache_file, 'r') as f:
        cache_data = json.load(f)
    prompt = cache_data['full_text']
    generated_text = cache_data['generated_text']
    input_prompt = cache_data['input_prompt']
else:
    # Format prompt for generation
    system_prompt = "You are a helpful mathematics assistant. Please think step by step to solve the problem."
    input_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    print("\nGenerating response...")
    
    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
    
    # Generate with the model
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            # temperature=0.7,
            do_sample=False,
            # top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generation
    prompt = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    
    # Save to cache
    cache_data = {
        'example_idx': example_idx,
        'problem': problem,
        'input_prompt': input_prompt,
        'generated_text': generated_text,
        'full_text': prompt
    }
    
    with open(generation_cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Generation saved to {generation_cache_file}")

print(f"\nGenerated response preview: {generated_text[:200]}...")

# %%
# Tokenize and display tokens with positions
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs['input_ids'][:, :512]

# %%
print("\n" + "="*80)
print("TOKENIZED PROMPT - TOKENS WITH POSITION INDICES")
print("="*80)

# Decode tokens properly to handle special characters
tokens = []
for i in range(len(input_ids[0])):
    # Decode each token individually to get the exact string representation
    token_str = tokenizer.decode(input_ids[0][i:i+1])
    tokens.append(token_str)

for idx, (token_id, token) in enumerate(zip(input_ids[0], tokens)):
    # Escape special characters for display
    display_token = repr(token)[1:-1]  # Remove quotes from repr
    print(f"Position {idx:4d}: token={token:<20} (display: {display_token:<20}) id={token_id.item():<6}")

print("="*80)
print(f"Total tokens: {len(tokens)}")
print("="*80 + "\n")

# %%
# USER: HARDCODE YOUR TARGET POSITION HERE BASED ON THE PRINTED TOKENS ABOVE
target_position = 431  # Use -1 for last token, or specify a position

# %%
# Compute attribution
print("Computing LoRA neuron attributions...")

# Setup stop-gradient hooks
# This will freeze attention patterns and LayerNorm outputs during backward pass
# - LayerNorm outputs are detached to stop gradients
# - Attention patterns (softmax(QK^T/sqrt(d))) are detached in the attention computation
print("Setting up stop-gradient hooks for LayerNorm and attention patterns...")
stop_grad_hooks = StopGradientHooks(model)
stop_grad_hooks.register_all_hooks()

# Setup tracker
tracker = LoRANeuronTracker(model)
tracker.register_hooks()

# Enable gradient computation
model.eval()
torch.set_grad_enabled(True)

# Clear any existing gradients
model.zero_grad()

# Forward pass - hooks will capture activations with retain_grad()
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(input_ids=input_ids)

# Handle target position
if target_position == -1:
    target_position = input_ids.shape[1] - 1

print(f"\nTarget position: {target_position}")
print(f"Target token: '{tokens[target_position]}' (display: {repr(tokens[target_position])[1:-1]})")

# Get logits at target position
logits = outputs.logits[0, target_position]  # [vocab_size]

# Find top 10 tokens by logit value
top_k = 10
top_logits, top_indices = torch.topk(logits, top_k)

print(f"\nTop {top_k} predictions at position {target_position}:")
print("="*70)
print(f"{'Rank':<6} {'Token ID':<10} {'Token':<30} {'Logit':<10}")
print("="*70)

for i in range(top_k):
    token_id = top_indices[i].item()
    token = tokenizer.decode([token_id])
    logit_value = top_logits[i].item()
    display_token = repr(token)[1:-1][:30]
    print(f"{i:<6} {token_id:<10} {display_token:<30} {logit_value:<10.3f}")

print("="*70)

# %%
# USER: SPECIFY POSITIVE AND NEGATIVE LOGIT TOKEN IDS HERE
# Based on the table above, choose:
# - positive_token_id: The token you want the model to predict (target)
# - negative_token_id: The token you want to contrast against (counterfactual)

positive_token_id = 15277  # Set this to the token ID for positive logit
negative_token_id = 13824  # Set this to the token ID for negative logit

# If not specified, use top 2 tokens as default
if positive_token_id is None:
    positive_token_id = top_indices[0].item()
    print(f"\nUsing default positive token: {positive_token_id} (top prediction)")
if negative_token_id is None:
    negative_token_id = top_indices[1].item()
    print(f"Using default negative token: {negative_token_id} (second prediction)")

positive_token = tokenizer.decode([positive_token_id])
negative_token = tokenizer.decode([negative_token_id])
positive_logit = logits[positive_token_id]
negative_logit = logits[negative_token_id]

print(f"\nComputing attribution for logit difference:")
print(f"Positive token: '{positive_token}' (id: {positive_token_id}, logit: {positive_logit.item():.3f})")
print(f"Negative token: '{negative_token}' (id: {negative_token_id}, logit: {negative_logit.item():.3f})")
print(f"Logit difference: {positive_logit.item() - negative_logit.item():.3f}")

# Compute logit difference as target metric
target_metric = positive_logit - negative_logit

# %%
# Compute gradients and attributions for ALL positions
print("\nComputing gradients for all token positions...")
all_attributions = []

for name, activation in tqdm(tracker.activations.items(), desc="Computing attributions"):
    # Compute gradient of logit difference w.r.t. this activation
    grad = torch.autograd.grad(
        outputs=target_metric,
        inputs=activation,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Process each position up to and including target position
    # (positions after target cannot influence it due to causal mask)
    for pos in range(min(target_position + 1, activation.shape[1])):
        # Extract scalar values at this position
        activation_value = activation[0, pos, 0].item()
        gradient_value = grad[0, pos, 0].item()
        
        # Attribution = gradient × activation
        attribution = gradient_value * activation_value
        
        # Store with position and token information
        all_attributions.append({
            'layer': name,
            'position': pos,
            'token': tokens[pos],
            'token_id': input_ids[0][pos].item(),
            'attribution': attribution,
            'activation': activation_value,
            'gradient': gradient_value
        })

# Cleanup hooks
tracker.remove_hooks()
stop_grad_hooks.remove_all_hooks()

print(f"Computed {len(all_attributions)} total attributions")

# %%
# Sort by attribution (not absolute)
sorted_attributions = sorted(
    all_attributions,
    key=lambda x: x['attribution'],
    reverse=True
)

# Display top positive attributions
print(f"\nTop 30 LoRA neuron activations by attribution to logit difference at position {target_position}:")
print(f"Metric: logit('{positive_token}') - logit('{negative_token}') = {positive_logit.item() - negative_logit.item():.3f}")
print(f"(Attribution = Gradient × Activation)")

# Prepare table data for top attributions
table_data = []
for i, entry in enumerate(sorted_attributions[:30]):
    # Format token for display
    display_token = repr(entry['token'])[1:-1][:20]  # Truncate long tokens
    
    table_data.append([
        i + 1,
        entry['position'],
        display_token,
        entry['layer'],
        f"{entry['attribution']:.6f}",
        f"{entry['activation']:.6f}",
        f"{entry['gradient']:.6f}"
    ])

# Print table
headers = ["Rank", "Pos", "Token", "Layer/Module", "Attribution", "Activation", "Gradient"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Also show bottom attributions (most negative)
print(f"\nBottom 20 LoRA neuron activations (most negative attribution):")
table_data_bottom = []
for i, entry in enumerate(sorted_attributions[-20:]):
    # Format token for display
    display_token = repr(entry['token'])[1:-1][:20]
    
    table_data_bottom.append([
        len(sorted_attributions) - 19 + i,
        entry['position'],
        display_token,
        entry['layer'],
        f"{entry['attribution']:.6f}",
        f"{entry['activation']:.6f}",
        f"{entry['gradient']:.6f}"
    ])

print(tabulate(table_data_bottom, headers=headers, tablefmt="grid"))

# %%
# Summary statistics
attribution_values = [entry['attribution'] for entry in all_attributions]

print(f"\nAttribution Statistics:")
print(f"Total attribution entries: {len(attribution_values)}")
print(f"Sum of all attributions: {sum(attribution_values):.6f}")
print(f"Mean attribution: {np.mean(attribution_values):.6f}")
print(f"Max attribution: {max(attribution_values):.6f}")
print(f"Min attribution: {min(attribution_values):.6f}")
print(f"Std dev: {np.std(attribution_values):.6f}")

# Per-layer statistics
from collections import defaultdict
layer_stats = defaultdict(list)
for entry in all_attributions:
    layer_stats[entry['layer']].append(entry['attribution'])

print(f"\nTop 10 layers by mean attribution:")
layer_means = [(layer, np.mean(attrs)) for layer, attrs in layer_stats.items()]
layer_means.sort(key=lambda x: x[1], reverse=True)

for i, (layer, mean_attr) in enumerate(layer_means[:10]):
    print(f"{i+1:2d}. {layer}: {mean_attr:.6f}")

# %%
# Validation check: Are gradients flowing?
zero_grad_count = sum(1 for entry in all_attributions if entry['gradient'] == 0.0)
zero_activation_count = sum(1 for entry in all_attributions if entry['activation'] == 0.0)

print(f"\nGradient Flow Validation:")
print(f"Entries with zero gradient: {zero_grad_count}/{len(all_attributions)}")
print(f"Entries with zero activation: {zero_activation_count}/{len(all_attributions)}")

if zero_grad_count > len(all_attributions) * 0.5:
    print("WARNING: Many entries have zero gradients. This might indicate a gradient flow issue.")
else:
    print("✓ Gradient flow appears healthy")

# %%
# Save detailed results
import json
results_file = f"lora_attribution_results_math500_example_{example_idx}.json"

# Get top attributions for saving
top_positive = sorted_attributions[:100]
top_negative = sorted_attributions[-100:]

save_results = {
    'example_idx': example_idx,
    'problem': problem,
    'generated_text': generated_text,
    'target_position': target_position,
    'target_token': tokens[target_position],
    'positive_token': positive_token,
    'positive_token_id': positive_token_id,
    'positive_logit': positive_logit.item(),
    'negative_token': negative_token,
    'negative_token_id': negative_token_id,
    'negative_logit': negative_logit.item(),
    'logit_difference': positive_logit.item() - negative_logit.item(),
    'top_positive_attributions': [
        {
            'layer': entry['layer'],
            'position': entry['position'],
            'token': entry['token'],
            'attribution': entry['attribution'],
            'activation': entry['activation'],
            'gradient': entry['gradient']
        }
        for entry in top_positive
    ],
    'top_negative_attributions': [
        {
            'layer': entry['layer'],
            'position': entry['position'],
            'token': entry['token'],
            'attribution': entry['attribution'],
            'activation': entry['activation'],
            'gradient': entry['gradient']
        }
        for entry in top_negative
    ]
}

with open(results_file, 'w') as f:
    json.dump(save_results, f, indent=2)

print(f"\nDetailed results saved to {results_file}")

# %%
print("\nAttribution study complete!")

 # %%
