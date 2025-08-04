# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import json
import math
from collections import defaultdict
from tqdm import tqdm

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
class AttentionLinearizer:
    """Linearizes attention by treating attention patterns as constants"""
    
    def __init__(self, model):
        self.model = model
        self.original_attention_forwards = {}
        
    def linearize_all_attention_modules(self):
        """Monkey-patch all attention modules to linearize attention patterns"""
        count = 0
        
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            attn_module = layer.self_attn
            
            self.original_attention_forwards[layer_idx] = attn_module.forward
            
            linearized_forward = self._create_linearized_attention_forward(
                attn_module.forward, attn_module, layer_idx
            )
            
            attn_module.forward = linearized_forward
            count += 1
            
        print(f"Linearized {count} attention modules")
        
    def _create_linearized_attention_forward(self, original_forward, attn_module, layer_idx):
        """Create a linearized forward function for attention"""
        
        def linearized_forward(self, hidden_states, *args, **kwargs):
            original_sdpa = F.scaled_dot_product_attention
            
            def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                if is_causal:
                    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)
                    
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias += attn_mask
                        
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                
                # CRITICAL: Detach attention weights here
                attn_weight = attn_weight.detach()
                
                if dropout_p > 0.0:
                    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                    
                return attn_weight @ value
            
            F.scaled_dot_product_attention = patched_sdpa
            
            try:
                outputs = original_forward(hidden_states, *args, **kwargs)
            finally:
                F.scaled_dot_product_attention = original_sdpa
                
            return outputs
            
        return linearized_forward.__get__(attn_module, attn_module.__class__)


class LinearizedLayerNorm:
    """Manages linearization of LayerNorm modules via monkey-patching"""
    
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        
    def _create_linearized_forward(self, original_forward):
        """Create a linearized forward function for LayerNorm/RMSNorm"""
        def linearized_forward(self, input):
            is_rmsnorm = not hasattr(self, 'bias')
            
            if is_rmsnorm:
                variance = input.pow(2).mean(-1, keepdim=True)
                rms = torch.sqrt(variance + self.variance_epsilon).detach()
                normalized = input / rms
                return self.weight * normalized
            else:
                mean = input.mean(-1, keepdim=True).detach()
                var = input.var(-1, keepdim=True, unbiased=False).detach()
                normalized = (input - mean) / torch.sqrt(var + self.variance_epsilon)
                return self.weight * normalized + self.bias
            
        return linearized_forward
    
    def linearize_all_layernorms(self):
        """Monkey-patch all LayerNorm modules to use linearized forward"""
        count = 0
        
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
            if hasattr(layer, 'input_layernorm'):
                ln = layer.input_layernorm
                self.original_forwards[f'layer{layer_idx}_input'] = ln.forward
                ln.forward = self._create_linearized_forward(ln.forward).__get__(ln, ln.__class__)
                count += 1
            
            if hasattr(layer, 'post_attention_layernorm'):
                ln = layer.post_attention_layernorm
                self.original_forwards[f'layer{layer_idx}_post'] = ln.forward
                ln.forward = self._create_linearized_forward(ln.forward).__get__(ln, ln.__class__)
                count += 1
        
        if hasattr(self.model.model.model, 'norm'):
            ln = self.model.model.model.norm
            self.original_forwards['final_norm'] = ln.forward
            ln.forward = self._create_linearized_forward(ln.forward).__get__(ln, ln.__class__)
            count += 1
        
        print(f"Linearized {count} LayerNorm modules")


class LoRANeuronTracker:
    """Tracks LoRA neuron activations during forward pass while preserving gradient flow"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def _create_hook(self, layer_name: str, adapter_name: str = 'default'):
        """Create a forward hook that captures activations and maintains gradient flow"""
        
        def hook_fn(module, input, output):
            if not output.requires_grad:
                output.requires_grad_(True)
            
            output.retain_grad()
            
            key = f"{layer_name}.{adapter_name}"
            self.activations[key] = output
            
            return output
            
        return hook_fn
    
    def register_hooks(self):
        """Register hooks on MLP LoRA A matrices only"""
        
        for layer_idx in range(self.model.config.num_hidden_layers):
            layer = self.model.model.model.layers[layer_idx]
            
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
    device_map="cuda",
    trust_remote_code=True
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)

# %%
# Load MATH-500 dataset and select a problem
print("Loading MATH-500 dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# You can change this to any index from 0-499
example_idx = 25  # Change this to explore different problems
example = dataset[example_idx]
problem = example['problem']

print(f"\nUsing example {example_idx}:")
print(f"Problem: {problem[:200]}..." if len(problem) > 200 else f"Problem: {problem}")

# Format prompt
system_prompt = "You are a helpful mathematics assistant. Please think step by step to solve the problem."
prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{problem}
<|im_end|>
<|im_start|>assistant
<|im_start|>think
"""

# %%
# Generate completion
print("\nGenerating completion...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(generated_ids[0])
print(f"Generated {len(generated_ids[0]) - len(inputs.input_ids[0])} new tokens")

# %%
# Tokenize full text
inputs = tokenizer(generated_text, return_tensors="pt").to(model.device)
input_ids = inputs['input_ids'][:, :512]

# Decode tokens
tokens = []
for i in range(len(input_ids[0])):
    token_str = tokenizer.decode(input_ids[0][i:i+1])
    tokens.append(token_str)

print(f"Total tokens: {len(tokens)}")

# %%
# Compute attributions for ALL token positions
print("\nComputing attributions for all positions...")

# Setup linearization
layernorm_linearizer = LinearizedLayerNorm(model)
layernorm_linearizer.linearize_all_layernorms()

attention_linearizer = AttentionLinearizer(model)
attention_linearizer.linearize_all_attention_modules()

# Setup tracker
tracker = LoRANeuronTracker(model)
tracker.register_hooks()

# Enable gradient computation
model.eval()
torch.set_grad_enabled(True)

# Forward pass
print("Running forward pass...")
outputs = model(input_ids=input_ids)

# Store attribution data for each position
attribution_data = {}

# For each target position, compute attributions from ALL source positions
for target_pos in tqdm(range(len(tokens)), desc="Computing attributions"):
    # Get logits at this position
    logits = outputs.logits[0, target_pos]
    
    # Use top prediction as target
    top_token_id = torch.argmax(logits).item()
    target_logit = logits[top_token_id]
    
    # Clear gradients
    model.zero_grad()
    for name, act in tracker.activations.items():
        if act.grad is not None:
            act.grad = None
    
    # Backward pass
    target_logit.backward(retain_graph=True)
    
    # Collect attributions from each source position
    position_attributions = [0.0] * len(tokens)
    
    for name, activation in tracker.activations.items():
        if activation.grad is not None:
            grad = activation.grad
            
            # Sum attributions across all source positions
            for source_pos in range(min(target_pos + 1, activation.shape[1])):
                act_value = activation[0, source_pos, 0].item()
                grad_value = grad[0, source_pos, 0].item()
                attribution = grad_value * act_value
                
                # Add to position attribution
                position_attributions[source_pos] += attribution
    
    # Store data for this position
    attribution_data[target_pos] = {
        'token': tokens[target_pos],
        'token_id': input_ids[0][target_pos].item(),
        'top_prediction': tokenizer.decode([top_token_id]),
        'top_logit': target_logit.item(),
        'position_attributions': position_attributions[:target_pos + 1]  # Only up to target
    }

print("Attribution computation complete!")

# %%
# Generate interactive HTML visualization
print("\nGenerating interactive HTML...")

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LoRA Attribution Interactive Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        
        #token-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .token {
            display: inline-block;
            padding: 5px 10px;
            margin: 2px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            background-color: #f9f9f9;
            transition: all 0.2s;
            font-family: monospace;
            font-size: 14px;
        }
        
        .token:hover {
            background-color: #e9e9e9;
            border-color: #999;
        }
        
        .token.selected {
            background-color: #4CAF50;
            color: white;
            border-color: #45a049;
        }
        
        #info-panel {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        #chart-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 500px;
        }
        
        canvas {
            max-width: 100%;
            height: 100%;
        }
        
        .positive { color: #2196F3; }
        .negative { color: #f44336; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>LoRA Attribution Interactive Visualization - MATH-500 Example """ + str(example_idx) + """</h1>
    
    <div id="info-panel">
        <h3>Problem: """ + problem.replace('$', '').replace('\\', '\\\\') + """</h3>
        <p>Click on any token below to see how previous tokens contribute to predicting that token.</p>
        <p id="selection-info">No token selected</p>
    </div>
    
    <div id="token-container">
        <h3>Tokens (click to explore)</h3>
        <div id="tokens"></div>
    </div>
    
    <div id="chart-container">
        <canvas id="attribution-chart"></canvas>
    </div>
    
    <script>
        // Attribution data
        const attributionData = """ + json.dumps(attribution_data, indent=2) + """;
        
        // Chart instance
        let chart = null;
        
        // Create token elements
        const tokensDiv = document.getElementById('tokens');
        const numTokens = """ + str(len(tokens)) + """;
        
        for (let i = 0; i < numTokens; i++) {
            const tokenEl = document.createElement('span');
            tokenEl.className = 'token';
            tokenEl.textContent = `[${i}] ${attributionData[i].token}`;
            tokenEl.onclick = () => selectToken(i);
            tokensDiv.appendChild(tokenEl);
        }
        
        function selectToken(position) {
            // Update selected token styling
            document.querySelectorAll('.token').forEach((el, idx) => {
                el.classList.toggle('selected', idx === position);
            });
            
            // Update info panel
            const data = attributionData[position];
            document.getElementById('selection-info').innerHTML = `
                <strong>Target Position:</strong> ${position}<br>
                <strong>Target Token:</strong> "${data.token}"<br>
                <strong>Top Prediction:</strong> "${data.top_prediction}" (logit: ${data.top_logit.toFixed(2)})
            `;
            
            // Update chart
            updateChart(position);
        }
        
        function updateChart(position) {
            const data = attributionData[position];
            const attributions = data.position_attributions;
            
            // Create labels for source tokens
            const labels = [];
            for (let i = 0; i < attributions.length; i++) {
                labels.push(`[${i}] ${attributionData[i].token}`);
            }
            
            // Destroy existing chart
            if (chart) {
                chart.destroy();
            }
            
            // Create new chart
            const ctx = document.getElementById('attribution-chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Attribution',
                        data: attributions,
                        backgroundColor: attributions.map(v => v >= 0 ? 'rgba(33, 150, 243, 0.6)' : 'rgba(244, 67, 54, 0.6)'),
                        borderColor: attributions.map(v => v >= 0 ? 'rgba(33, 150, 243, 1)' : 'rgba(244, 67, 54, 1)'),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `Token Attributions to Position ${position} ("${data.token}")`
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    return `Attribution: ${context.parsed.y.toFixed(6)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Total Attribution (sum across all layers)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Source Token Position'
                            },
                            ticks: {
                                autoSkip: false,
                                maxRotation: 90,
                                minRotation: 45
                            }
                        }
                    }
                }
            });
        }
        
        // Select first token by default
        selectToken(0);
    </script>
</body>
</html>
"""

# Save HTML file
html_file = f"lora_attribution_interactive_math500_example_{example_idx}.html"
with open(html_file, 'w') as f:
    f.write(html_content)

print(f"Interactive visualization saved to {html_file}")
print("Open this file in a web browser to explore the attributions!")
print(f"\nProblem: {problem[:100]}..." if len(problem) > 100 else f"\nProblem: {problem}")

# %%
# Clean up
tracker.remove_hooks()
print("\nDone!")