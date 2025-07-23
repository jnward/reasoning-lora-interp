# %%
import torch
import torch.nn.functional as F
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import json
import gc
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

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
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# %%
# Hook storage for attention patterns
attention_patterns_storage = {}

def create_attention_hook(model_name: str, layer_idx: int):
    """Create a hook to capture attention patterns from a specific layer"""
    def hook(module, input, output):
        # For Qwen2 models, the attention output is a tuple
        # (hidden_states, attention_weights, past_key_values)
        if len(output) >= 2 and output[1] is not None:
            # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
            attention_weights = output[1].detach().cpu()
            key = f"{model_name}_layer_{layer_idx}"
            attention_patterns_storage[key] = attention_weights
    return hook

# %%
# Load base model
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Use eager attention to get attention weights
)

# Get number of layers and heads
n_layers = base_model.config.num_hidden_layers
n_heads = base_model.config.num_attention_heads
print(f"Model has {n_layers} layers and {n_heads} attention heads")

# %%
# Load LoRA model
print("\nLoading LoRA adapter...")
lora_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Use eager attention to get attention weights
)
lora_model = PeftModel.from_pretrained(lora_model, lora_dir, torch_dtype=torch.bfloat16)

# %%
# Generate or load a reasoning trace
print("\nPreparing reasoning trace...")

# Option 1: Use a cached generation if available
generation_cache_file = "math500_generation_example_10.json"
if os.path.exists(generation_cache_file):
    print(f"Loading cached generation from {generation_cache_file}")
    with open(generation_cache_file, 'r') as f:
        cache_data = json.load(f)
    full_text = cache_data['full_text']
    print(f"Loaded reasoning trace with {len(tokenizer.encode(full_text))} tokens")
else:
    # Option 2: Generate a new reasoning trace
    print("Generating new reasoning trace...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problem = dataset[10]['problem']
    
    system_prompt = "You are a helpful mathematics assistant. Please think step by step to solve the problem."
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
    
    with torch.no_grad():
        generated_ids = lora_model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

# Tokenize for analysis
inputs = tokenizer(full_text, return_tensors="pt", max_length=512, truncation=True)
input_ids = inputs.input_ids.to(base_model.device)
seq_len = input_ids.shape[1]

print(f"\nAnalyzing sequence of length: {seq_len}")

# %%
# Register hooks on both models
print("\nRegistering attention hooks...")

base_hooks = []
lora_hooks = []

for layer_idx in range(n_layers):
    # Hook for base model
    base_hook = base_model.model.layers[layer_idx].self_attn.register_forward_hook(
        create_attention_hook("base", layer_idx)
    )
    base_hooks.append(base_hook)
    
    # Hook for LoRA model
    lora_hook = lora_model.model.model.layers[layer_idx].self_attn.register_forward_hook(
        create_attention_hook("lora", layer_idx)
    )
    lora_hooks.append(lora_hook)

# %%
# Run forward passes and collect attention patterns
print("\nRunning forward passes...")
attention_patterns_storage.clear()

# Base model forward pass
with torch.no_grad():
    base_outputs = base_model(input_ids, output_attentions=True)

base_attention_patterns = {}
for layer_idx in range(n_layers):
    key = f"base_layer_{layer_idx}"
    if key in attention_patterns_storage:
        base_attention_patterns[layer_idx] = attention_patterns_storage[key]

attention_patterns_storage.clear()

# LoRA model forward pass
with torch.no_grad():
    lora_outputs = lora_model(input_ids, output_attentions=True)

lora_attention_patterns = {}
for layer_idx in range(n_layers):
    key = f"lora_layer_{layer_idx}"
    if key in attention_patterns_storage:
        lora_attention_patterns[layer_idx] = attention_patterns_storage[key]

# Remove hooks
for hook in base_hooks + lora_hooks:
    hook.remove()

# %%
# Compute KL divergences
print("\nComputing KL divergences...")

kl_divergences = np.zeros((n_layers, n_heads, seq_len))

for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
    if layer_idx not in base_attention_patterns or layer_idx not in lora_attention_patterns:
        continue
    
    base_attn = base_attention_patterns[layer_idx][0]  # [n_heads, seq_len, seq_len]
    lora_attn = lora_attention_patterns[layer_idx][0]  # [n_heads, seq_len, seq_len]
    
    for head_idx in range(n_heads):
        for pos_idx in range(seq_len):
            # Get attention distributions for this position
            base_dist = base_attn[head_idx, pos_idx, :pos_idx+1]  # Only look at previous positions
            lora_dist = lora_attn[head_idx, pos_idx, :pos_idx+1]
            
            if len(base_dist) == 0:  # Skip if no previous positions
                continue
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            base_dist = base_dist + epsilon
            lora_dist = lora_dist + epsilon
            
            # Renormalize
            base_dist = base_dist / base_dist.sum()
            lora_dist = lora_dist / lora_dist.sum()
            
            # Compute KL divergence: KL(base || lora)
            kl_div = (base_dist * (base_dist.log() - lora_dist.log())).sum().item()
            kl_divergences[layer_idx, head_idx, pos_idx] = kl_div

# %%
# Decode tokens
tokens = []
for token_id in input_ids[0]:
    token = tokenizer.decode([token_id])
    tokens.append(token)

# %%
# Compute aggregate statistics
print("\nComputing aggregate statistics...")

# Per-head statistics
head_stats = []
for layer_idx in range(n_layers):
    for head_idx in range(n_heads):
        head_kl = kl_divergences[layer_idx, head_idx, :]
        head_stats.append({
            'layer': layer_idx,
            'head': head_idx,
            'avg_kl': float(np.mean(head_kl)),
            'max_kl': float(np.max(head_kl)),
            'std_kl': float(np.std(head_kl))
        })

# Sort by average KL
head_stats.sort(key=lambda x: x['avg_kl'], reverse=True)

print(f"\nTop 10 heads by average KL divergence:")
for i, stat in enumerate(head_stats[:10]):
    print(f"{i+1}. Layer {stat['layer']}, Head {stat['head']}: avg_kl={stat['avg_kl']:.4f}, max_kl={stat['max_kl']:.4f}")

# %%
# Generate dashboard HTML
print("\nGenerating interactive dashboard...")

dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attention KL Divergence Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            display: flex;
            height: 100vh;
        }}
        .sidebar {{
            width: 350px;
            background-color: white;
            padding: 20px;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            overflow-y: auto;
        }}
        .main-content {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }}
        .controls {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .token-display {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .token {{
            display: inline-block;
            padding: 2px 1px;
            margin: 0;
            border-radius: 2px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 13px;
            cursor: pointer;
            position: relative;
            white-space: pre;
            line-height: 1.5;
        }}
        .token-tooltip {{
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
            z-index: 1000;
        }}
        .token:hover .token-tooltip {{
            opacity: 1;
        }}
        .token-line {{
            display: block;
            margin: 2px 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            line-height: 1.5;
        }}
        .head-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .head-item {{
            padding: 10px;
            margin: 5px 0;
            background-color: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        .head-item:hover {{
            background-color: #e9ecef;
        }}
        .head-item.selected {{
            background-color: #007bff;
            color: white;
        }}
        select, input {{
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        label {{
            display: block;
            margin-top: 10px;
            font-weight: 500;
        }}
        h1, h2, h3 {{
            margin-top: 0;
        }}
        .metric-display {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .colorbar {{
            height: 20px;
            background: linear-gradient(to right, #f0f0f0 0%, #ffffcc 25%, #ffeda0 50%, #feb24c 75%, #f03b20 100%);
            border-radius: 4px;
            margin: 10px 0;
        }}
        .special-token {{
            border: 2px solid #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Top Attention Heads</h2>
            <div style="margin-bottom: 10px;">
                <label>
                    <input type="radio" name="sort-metric" value="avg" checked> Sort by Average KL
                </label>
                <label>
                    <input type="radio" name="sort-metric" value="max"> Sort by Max KL
                </label>
            </div>
            <div class="head-list" id="head-list"></div>
        </div>
        
        <div class="main-content">
            <h1>Attention KL Divergence Analysis</h1>
            
            <div class="controls">
                <h3>Controls</h3>
                <label>View Mode:</label>
                <select id="view-mode">
                    <option value="head">Specific Head</option>
                    <option value="layer">Layer Average</option>
                    <option value="overall">Overall Average</option>
                </select>
                
                <div id="head-controls">
                    <label>Layer:</label>
                    <select id="layer-select"></select>
                    
                    <label>Head:</label>
                    <select id="head-select"></select>
                </div>
                
                <div id="layer-controls" style="display: none;">
                    <label>Layer:</label>
                    <select id="layer-select-avg"></select>
                </div>
                
                <div class="metric-display">
                    <div id="current-selection">Select a head or layer to view</div>
                </div>
            </div>
            
            <div class="token-display">
                <h3>Token-wise KL Divergence</h3>
                <div class="colorbar"></div>
                <div>
                    <small style="float: left;">Low KL</small>
                    <small style="float: right;">High KL</small>
                    <div style="clear: both;"></div>
                </div>
                <div id="tokens-container"></div>
            </div>
        </div>
    </div>

    <script>
        // Data embedded from Python
        const klDivergences = {json.dumps(kl_divergences.tolist())};
        const tokens = {json.dumps(tokens)};
        const headStats = {json.dumps(head_stats)};
        const nLayers = {n_layers};
        const nHeads = {n_heads};
        
        // State
        let currentLayer = 0;
        let currentHead = 0;
        let currentViewMode = 'head';
        let sortMetric = 'avg';
        
        // Initialize controls
        function initializeControls() {{
            // Layer select
            const layerSelect = document.getElementById('layer-select');
            const layerSelectAvg = document.getElementById('layer-select-avg');
            for (let i = 0; i < nLayers; i++) {{
                const option = new Option(`Layer ${{i}}`, i);
                layerSelect.add(option.cloneNode(true));
                layerSelectAvg.add(option);
            }}
            
            // Head select
            const headSelect = document.getElementById('head-select');
            for (let i = 0; i < nHeads; i++) {{
                headSelect.add(new Option(`Head ${{i}}`, i));
            }}
            
            // Event listeners
            document.getElementById('view-mode').addEventListener('change', updateViewMode);
            layerSelect.addEventListener('change', (e) => {{
                currentLayer = parseInt(e.target.value);
                updateDisplay();
            }});
            layerSelectAvg.addEventListener('change', (e) => {{
                currentLayer = parseInt(e.target.value);
                updateDisplay();
            }});
            headSelect.addEventListener('change', (e) => {{
                currentHead = parseInt(e.target.value);
                updateDisplay();
            }});
            
            document.querySelectorAll('input[name="sort-metric"]').forEach(radio => {{
                radio.addEventListener('change', (e) => {{
                    sortMetric = e.target.value;
                    updateHeadList();
                }});
            }});
        }}
        
        function updateViewMode() {{
            currentViewMode = document.getElementById('view-mode').value;
            document.getElementById('head-controls').style.display = 
                currentViewMode === 'head' ? 'block' : 'none';
            document.getElementById('layer-controls').style.display = 
                currentViewMode === 'layer' ? 'block' : 'none';
            updateDisplay();
        }}
        
        function getKLValues() {{
            if (currentViewMode === 'head') {{
                return klDivergences[currentLayer][currentHead];
            }} else if (currentViewMode === 'layer') {{
                // Average across heads for the layer
                const layerKL = [];
                for (let pos = 0; pos < tokens.length; pos++) {{
                    let sum = 0;
                    for (let h = 0; h < nHeads; h++) {{
                        sum += klDivergences[currentLayer][h][pos];
                    }}
                    layerKL.push(sum / nHeads);
                }}
                return layerKL;
            }} else {{
                // Average across all layers and heads
                const overallKL = [];
                for (let pos = 0; pos < tokens.length; pos++) {{
                    let sum = 0;
                    for (let l = 0; l < nLayers; l++) {{
                        for (let h = 0; h < nHeads; h++) {{
                            sum += klDivergences[l][h][pos];
                        }}
                    }}
                    overallKL.push(sum / (nLayers * nHeads));
                }}
                return overallKL;
            }}
        }}
        
        function updateDisplay() {{
            const klValues = getKLValues();
            const maxKL = Math.max(...klValues);
            const minKL = Math.min(...klValues);
            
            // Update current selection display
            let selectionText = '';
            if (currentViewMode === 'head') {{
                selectionText = `Layer ${{currentLayer}}, Head ${{currentHead}}`;
                const stats = headStats.find(h => h.layer === currentLayer && h.head === currentHead);
                if (stats) {{
                    selectionText += ` - Avg KL: ${{stats.avg_kl.toFixed(4)}}, Max KL: ${{stats.max_kl.toFixed(4)}}`;
                }}
            }} else if (currentViewMode === 'layer') {{
                selectionText = `Layer ${{currentLayer}} (averaged across heads)`;
            }} else {{
                selectionText = 'Overall (averaged across all layers and heads)';
            }}
            document.getElementById('current-selection').textContent = selectionText;
            
            // Update tokens display
            const container = document.getElementById('tokens-container');
            container.innerHTML = '';
            
            // Group tokens by lines (split on newlines)
            let currentLine = document.createElement('div');
            currentLine.className = 'token-line';
            container.appendChild(currentLine);
            
            tokens.forEach((token, idx) => {{
                // Check if token contains newline
                if (token === '\\n' || token.includes('\\n')) {{
                    // Add newline token to current line
                    const tokenSpan = createTokenElement(token, idx, klValues[idx], minKL, maxKL);
                    currentLine.appendChild(tokenSpan);
                    
                    // Start new line
                    currentLine = document.createElement('div');
                    currentLine.className = 'token-line';
                    container.appendChild(currentLine);
                }} else {{
                    // Add token to current line
                    const tokenSpan = createTokenElement(token, idx, klValues[idx], minKL, maxKL);
                    currentLine.appendChild(tokenSpan);
                }}
            }});
        }}
        
        function createTokenElement(token, idx, kl, minKL, maxKL) {{
            const tokenSpan = document.createElement('span');
            tokenSpan.className = 'token';
            
            // Special tokens
            if (token.includes('<|im_start|>') || token.includes('<|im_end|>')) {{
                tokenSpan.classList.add('special-token');
            }}
            
            // Color based on KL divergence
            const normalizedKL = (kl - minKL) / (maxKL - minKL + 1e-10);
            const color = getColorForValue(normalizedKL);
            tokenSpan.style.backgroundColor = color;
            
            // Set text color based on background
            const brightness = getBrightness(color);
            tokenSpan.style.color = brightness > 128 ? '#000' : '#fff';
            
            // Token text - escape HTML but preserve spaces
            let tokenDisplay = token
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;')
                .replace(/ /g, '&nbsp;');
            
            // Show newline as visible character
            if (token === '\\n') {{
                tokenDisplay = '↵';
                tokenSpan.style.color = '#888';
            }}
            
            tokenSpan.innerHTML = tokenDisplay;
            
            // Tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'token-tooltip';
            tooltip.textContent = `Pos: ${{idx}}, KL: ${{kl.toFixed(4)}}`;
            tokenSpan.appendChild(tooltip);
            
            return tokenSpan;
        }}
        
        function getColorForValue(value) {{
            // Color scale from light gray to red
            const colors = [
                [240, 240, 240],  // Very light gray
                [255, 255, 204],  // Light yellow
                [255, 237, 160],  // Yellow
                [254, 178, 76],   // Orange
                [240, 59, 32]     // Red
            ];
            
            const scaledValue = value * (colors.length - 1);
            const lowerIdx = Math.floor(scaledValue);
            const upperIdx = Math.ceil(scaledValue);
            const fraction = scaledValue - lowerIdx;
            
            const lowerColor = colors[Math.min(lowerIdx, colors.length - 1)];
            const upperColor = colors[Math.min(upperIdx, colors.length - 1)];
            
            const r = Math.round(lowerColor[0] + (upperColor[0] - lowerColor[0]) * fraction);
            const g = Math.round(lowerColor[1] + (upperColor[1] - lowerColor[1]) * fraction);
            const b = Math.round(lowerColor[2] + (upperColor[2] - lowerColor[2]) * fraction);
            
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}
        
        function getBrightness(color) {{
            const rgb = color.match(/\\d+/g);
            return (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
        }}
        
        function updateHeadList() {{
            const container = document.getElementById('head-list');
            container.innerHTML = '';
            
            // Sort heads
            const sortedHeads = [...headStats].sort((a, b) => {{
                return sortMetric === 'avg' ? b.avg_kl - a.avg_kl : b.max_kl - a.max_kl;
            }});
            
            // Display top heads
            sortedHeads.slice(0, 50).forEach((stat, idx) => {{
                const item = document.createElement('div');
                item.className = 'head-item';
                if (stat.layer === currentLayer && stat.head === currentHead && currentViewMode === 'head') {{
                    item.classList.add('selected');
                }}
                
                item.innerHTML = `
                    <strong>${{idx + 1}}. L${{stat.layer}}-H${{stat.head}}</strong><br>
                    Avg KL: ${{stat.avg_kl.toFixed(4)}}, Max KL: ${{stat.max_kl.toFixed(4)}}
                `;
                
                item.addEventListener('click', () => {{
                    currentLayer = stat.layer;
                    currentHead = stat.head;
                    document.getElementById('view-mode').value = 'head';
                    document.getElementById('layer-select').value = stat.layer;
                    document.getElementById('head-select').value = stat.head;
                    updateViewMode();
                    updateHeadList();
                }});
                
                container.appendChild(item);
            }});
        }}
        
        // Initialize
        initializeControls();
        updateHeadList();
        updateDisplay();
    </script>
</body>
</html>
"""

# Save dashboard
dashboard_file = "attention_kl_dashboard.html"
with open(dashboard_file, 'w') as f:
    f.write(dashboard_html)

print(f"\nDashboard saved to: {dashboard_file}")
print(f"Open this file in a web browser to explore the attention patterns interactively.")

# %%
# Clean up
del base_model
del lora_model
del base_attention_patterns
del lora_attention_patterns
gc.collect()
torch.cuda.empty_cache()

print("\nDashboard generation complete!")

# %%