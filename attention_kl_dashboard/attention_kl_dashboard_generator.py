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
import pickle
from functools import lru_cache
import hashlib

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_dir = "/workspace/reasoning_interp/lora_checkpoints/s1-lora-32B-r1-20250627_013544"
CACHE_DIR = "attention_kl_cache"

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Using LoRA from: {lora_dir}")

# %%
def get_cache_key(text: str, config: dict) -> str:
    """Generate a cache key based on input text and config"""
    config_str = json.dumps(config, sort_keys=True)
    combined = f"{text[:1000]}{config_str}"  # Use first 1000 chars of text
    return hashlib.md5(combined.encode()).hexdigest()

# %%
def extract_sparse_attention_patterns(base_patterns, lora_patterns, n_layers, n_heads, seq_len, 
                                    top_k=20, threshold=0.01):
    """Extract sparse attention patterns for efficient storage and visualization"""
    
    # Initialize storage for patterns
    sparse_patterns = {
        'base': {},
        'lora': {},
        'base_layer_avg': {},
        'lora_layer_avg': {},
        'base_overall_avg': {},
        'lora_overall_avg': {}
    }
    
    # Process individual head patterns
    print("Processing individual head patterns...")
    for layer_idx in tqdm(range(n_layers), desc="Extracting head patterns"):
        if layer_idx not in base_patterns or layer_idx not in lora_patterns:
            continue
            
        base_attn = base_patterns[layer_idx][0].float().numpy()  # [n_heads, seq_len, seq_len]
        lora_attn = lora_patterns[layer_idx][0].float().numpy()
        
        for head_idx in range(n_heads):
            key = f"{layer_idx}_{head_idx}"
            
            # Extract top-k attended positions for each query position
            sparse_patterns['base'][key] = {}
            sparse_patterns['lora'][key] = {}
            
            for pos in range(seq_len):
                # Get attention values for this position attending to all others
                base_att = base_attn[head_idx, pos, :pos+1]  # Only previous positions
                lora_att = lora_attn[head_idx, pos, :pos+1]
                
                if len(base_att) > 0:
                    # Get top-k positions
                    if len(base_att) > top_k:
                        top_k_idx = np.argpartition(base_att, -top_k)[-top_k:]
                        base_top_k = [(int(idx), float(base_att[idx])) for idx in top_k_idx 
                                     if base_att[idx] > threshold]
                        lora_top_k = [(int(idx), float(lora_att[idx])) for idx in top_k_idx 
                                     if lora_att[idx] > threshold]
                    else:
                        base_top_k = [(int(idx), float(base_att[idx])) for idx in range(len(base_att))
                                     if base_att[idx] > threshold]
                        lora_top_k = [(int(idx), float(lora_att[idx])) for idx in range(len(lora_att))
                                     if lora_att[idx] > threshold]
                    
                    if base_top_k:
                        sparse_patterns['base'][key][pos] = base_top_k
                    if lora_top_k:
                        sparse_patterns['lora'][key][pos] = lora_top_k
    
    # Compute layer-averaged patterns
    print("\nComputing layer-averaged patterns...")
    for layer_idx in tqdm(range(n_layers), desc="Layer averages"):
        if layer_idx not in base_patterns or layer_idx not in lora_patterns:
            continue
            
        base_attn = base_patterns[layer_idx][0].float().numpy()
        lora_attn = lora_patterns[layer_idx][0].float().numpy()
        
        # Average across heads
        base_layer_avg = base_attn.mean(axis=0)  # [seq_len, seq_len]
        lora_layer_avg = lora_attn.mean(axis=0)
        
        sparse_patterns['base_layer_avg'][layer_idx] = {}
        sparse_patterns['lora_layer_avg'][layer_idx] = {}
        
        for pos in range(seq_len):
            base_att = base_layer_avg[pos, :pos+1]
            lora_att = lora_layer_avg[pos, :pos+1]
            
            if len(base_att) > 0:
                # Store sparse representation
                base_sparse = [(int(idx), float(base_att[idx])) for idx in range(len(base_att))
                              if base_att[idx] > threshold]
                lora_sparse = [(int(idx), float(lora_att[idx])) for idx in range(len(lora_att))
                              if lora_att[idx] > threshold]
                
                if base_sparse:
                    sparse_patterns['base_layer_avg'][layer_idx][pos] = base_sparse[:top_k]
                if lora_sparse:
                    sparse_patterns['lora_layer_avg'][layer_idx][pos] = lora_sparse[:top_k]
    
    # Compute overall average patterns
    print("\nComputing overall average patterns...")
    base_overall = np.zeros((seq_len, seq_len))
    lora_overall = np.zeros((seq_len, seq_len))
    count = 0
    
    for layer_idx in range(n_layers):
        if layer_idx in base_patterns and layer_idx in lora_patterns:
            base_overall += base_patterns[layer_idx][0].float().numpy().mean(axis=0)
            lora_overall += lora_patterns[layer_idx][0].float().numpy().mean(axis=0)
            count += 1
    
    if count > 0:
        base_overall /= count
        lora_overall /= count
        
        for pos in range(seq_len):
            base_att = base_overall[pos, :pos+1]
            lora_att = lora_overall[pos, :pos+1]
            
            if len(base_att) > 0:
                base_sparse = [(int(idx), float(base_att[idx])) for idx in range(len(base_att))
                              if base_att[idx] > threshold]
                lora_sparse = [(int(idx), float(lora_att[idx])) for idx in range(len(lora_att))
                              if lora_att[idx] > threshold]
                
                if base_sparse:
                    sparse_patterns['base_overall_avg'][pos] = sorted(base_sparse, 
                                                                     key=lambda x: x[1], 
                                                                     reverse=True)[:top_k]
                if lora_sparse:
                    sparse_patterns['lora_overall_avg'][pos] = sorted(lora_sparse, 
                                                                     key=lambda x: x[1], 
                                                                     reverse=True)[:top_k]
    
    return sparse_patterns

# %%
def generate_attention_kl_data(text: str = None, max_length: int = 1024, include_attention_patterns: bool = True) -> dict:
    """Generate all attention KL divergence data. This is the expensive operation we want to cache."""
    
    cache_key = get_cache_key(text or "default", {"max_length": max_length, "lora_dir": lora_dir})
    cache_file = os.path.join(CACHE_DIR, f"attention_kl_{cache_key}.pkl")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Generating new attention KL data...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Hook storage for attention patterns
    attention_patterns_storage = {}

    def create_attention_hook(model_name: str, layer_idx: int):
        """Create a hook to capture attention patterns from a specific layer"""
        def hook(module, input, output):
            if len(output) >= 2 and output[1] is not None:
                attention_weights = output[1].detach().cpu()
                key = f"{model_name}_layer_{layer_idx}"
                attention_patterns_storage[key] = attention_weights
        return hook
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    n_layers = base_model.config.num_hidden_layers
    n_heads = base_model.config.num_attention_heads
    print(f"Model has {n_layers} layers and {n_heads} attention heads")
    
    # Load LoRA model
    print("\nLoading LoRA adapter...")
    lora_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    lora_model = PeftModel.from_pretrained(lora_model, lora_dir, torch_dtype=torch.bfloat16)
    
    # Prepare text
    if text is None:
        print("\nPreparing reasoning trace...")
        generation_cache_file = "math500_generation_example_25_1024.json"
        if os.path.exists(generation_cache_file):
            print(f"Loading cached generation from {generation_cache_file}")
            with open(generation_cache_file, 'r') as f:
                cache_data = json.load(f)
            text = cache_data['full_text']
        else:
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
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = inputs.input_ids.to(base_model.device)
    seq_len = input_ids.shape[1]
    
    print(f"\nAnalyzing sequence of length: {seq_len}")
    
    # Register hooks
    print("\nRegistering attention hooks...")
    base_hooks = []
    lora_hooks = []
    
    for layer_idx in range(n_layers):
        base_hook = base_model.model.layers[layer_idx].self_attn.register_forward_hook(
            create_attention_hook("base", layer_idx)
        )
        base_hooks.append(base_hook)
        
        lora_hook = lora_model.model.model.layers[layer_idx].self_attn.register_forward_hook(
            create_attention_hook("lora", layer_idx)
        )
        lora_hooks.append(lora_hook)
    
    # Run forward passes
    print("\nRunning forward passes...")
    attention_patterns_storage.clear()
    
    with torch.no_grad():
        base_outputs = base_model(input_ids, output_attentions=True)
    
    base_attention_patterns = {}
    for layer_idx in range(n_layers):
        key = f"base_layer_{layer_idx}"
        if key in attention_patterns_storage:
            base_attention_patterns[layer_idx] = attention_patterns_storage[key]
    
    attention_patterns_storage.clear()
    
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
    
    # Compute KL divergences
    print("\nComputing KL divergences...")
    kl_divergences = np.zeros((n_layers, n_heads, seq_len))
    
    for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
        if layer_idx not in base_attention_patterns or layer_idx not in lora_attention_patterns:
            continue
        
        base_attn = base_attention_patterns[layer_idx][0]
        lora_attn = lora_attention_patterns[layer_idx][0]
        
        for head_idx in range(n_heads):
            for pos_idx in range(seq_len):
                base_dist = base_attn[head_idx, pos_idx, :pos_idx+1]
                lora_dist = lora_attn[head_idx, pos_idx, :pos_idx+1]
                
                if len(base_dist) == 0:
                    continue
                
                epsilon = 1e-10
                base_dist = base_dist + epsilon
                lora_dist = lora_dist + epsilon
                
                base_dist = base_dist / base_dist.sum()
                lora_dist = lora_dist / lora_dist.sum()
                
                kl_div = (base_dist * (base_dist.log() - lora_dist.log())).sum().item()
                kl_divergences[layer_idx, head_idx, pos_idx] = kl_div
    
    # Decode tokens
    tokens = []
    for token_id in input_ids[0]:
        token = tokenizer.decode([token_id])
        tokens.append(token)
    
    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
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
    
    head_stats.sort(key=lambda x: x['avg_kl'], reverse=True)
    
    print(f"\nTop 10 heads by average KL divergence:")
    for i, stat in enumerate(head_stats[:10]):
        print(f"{i+1}. Layer {stat['layer']}, Head {stat['head']}: avg_kl={stat['avg_kl']:.4f}, max_kl={stat['max_kl']:.4f}")
    
    # Extract sparse attention patterns if requested
    attention_patterns = None
    if include_attention_patterns:
        print("\nExtracting sparse attention patterns...")
        attention_patterns = extract_sparse_attention_patterns(
            base_attention_patterns, lora_attention_patterns, n_layers, n_heads, seq_len
        )
    
    # Prepare data for caching
    data = {
        'kl_divergences': kl_divergences.tolist(),
        'tokens': tokens,
        'head_stats': head_stats,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'seq_len': seq_len,
        'text': text[:1000] + "..." if len(text) > 1000 else text,
        'attention_patterns': attention_patterns
    }
    
    # Save to cache
    print(f"\nSaving data to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Clean up models
    del base_model
    del lora_model
    del base_attention_patterns
    del lora_attention_patterns
    gc.collect()
    torch.cuda.empty_cache()
    
    return data

# %%
def generate_dashboard_from_data(data: dict, output_file: str = "attention_kl_dashboard.html"):
    """Generate the dashboard HTML from pre-computed data"""
    
    print("\nGenerating interactive dashboard...")
    
    kl_divergences = data['kl_divergences']
    tokens = data['tokens']
    head_stats = data['head_stats']
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    attention_patterns = data.get('attention_patterns', None)
    
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
            margin: 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            line-height: 1.5;
            min-height: 1.5em;
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
        .attention-mode {{
            background-color: #e7f0ff;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 14px;
        }}
        .hover-legend {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 13px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 15px;
            border-radius: 3px;
            border: 1px solid #ddd;
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
                <div class="attention-mode" id="attention-mode" style="display: none;">
                    <strong>Hover Mode:</strong> Showing attention patterns for position <span id="hover-position">-</span>
                    <div class="hover-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #4a90e2;"></div>
                            <span>Base Model Only</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #e74c3c;"></div>
                            <span>LoRA Model Only</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #9b59b6;"></div>
                            <span>Both Models</span>
                        </div>
                    </div>
                </div>
                <div id="tokens-container"></div>
            </div>
        </div>
    </div>

    <script>
        // Data embedded from Python
        const klDivergences = {json.dumps(kl_divergences)};
        const tokens = {json.dumps(tokens)};
        const headStats = {json.dumps(head_stats)};
        const nLayers = {n_layers};
        const nHeads = {n_heads};
        const attentionPatterns = {json.dumps(attention_patterns) if attention_patterns else 'null'};
        
        // State
        let currentLayer = 0;
        let currentHead = 0;
        let currentViewMode = 'head';
        let sortMetric = 'avg';
        let hoverPosition = null;
        let isHovering = false;
        
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
                const tokenSpan = createTokenElement(token, idx, klValues[idx], minKL, maxKL);
                currentLine.appendChild(tokenSpan);
                
                // Count newlines in the token and create that many new lines
                const newlineCount = (token.match(/\\n/g) || []).length;
                if (newlineCount > 0) {{
                    // For each newline, create a new line
                    for (let i = 0; i < newlineCount; i++) {{
                        currentLine = document.createElement('div');
                        currentLine.className = 'token-line';
                        if (i < newlineCount - 1) {{
                            // For empty lines between, add a non-breaking space
                            currentLine.innerHTML = '&nbsp;';
                        }}
                        container.appendChild(currentLine);
                    }}
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
            const textColor = brightness > 128 ? '#000' : '#fff';
            
            // Token text - escape HTML and show newlines as return symbol
            let tokenDisplay = token
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;')
                .replace(/ /g, '&nbsp;')
                .replace(/\\n/g, '↵');  // Temporarily replace newlines
            
            // Wrap entire content with color, then make return symbols gray
            tokenDisplay = '<span style="color: ' + textColor + ';">' + tokenDisplay + '</span>';
            tokenDisplay = tokenDisplay.replace(/↵/g, '</span><span style="color: #888;">↵</span><span style="color: ' + textColor + ';">');
            
            // Clean up any empty spans
            tokenDisplay = tokenDisplay.replace(/<span style="color: [^"]+;"><\/span>/g, '');
            
            tokenSpan.innerHTML = tokenDisplay;
            
            // Tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'token-tooltip';
            tooltip.textContent = `Pos: ${{idx}}, KL: ${{kl.toFixed(4)}}`;
            tokenSpan.appendChild(tooltip);
            
            // Hover events for attention patterns
            tokenSpan.addEventListener('mouseenter', () => {{
                showAttentionPatterns(idx);
            }});
            
            tokenSpan.addEventListener('mouseleave', () => {{
                hideAttentionPatterns();
            }});
            
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
        
        // Attention pattern visualization functions
        function getAttentionValue(position, targetPosition, modelType) {{
            if (!attentionPatterns) return 0;
            
            let patterns;
            if (currentViewMode === 'head') {{
                const key = `${{currentLayer}}_${{currentHead}}`;
                patterns = attentionPatterns[`${{modelType}}`][key];
            }} else if (currentViewMode === 'layer') {{
                patterns = attentionPatterns[`${{modelType}}_layer_avg`][currentLayer];
            }} else {{
                patterns = attentionPatterns[`${{modelType}}_overall_avg`];
            }}
            
            if (!patterns || !patterns[position]) return 0;
            
            // Find the target position in the sparse representation
            const sparseData = patterns[position];
            for (const [idx, value] of sparseData) {{
                if (idx === targetPosition) {{
                    return value;
                }}
            }}
            return 0;
        }}
        
        function getAttentionColor(baseValue, loraValue) {{
            // Normalize values for color intensity
            const maxValue = 0.5;  // Typical max attention value
            const baseIntensity = Math.min(baseValue / maxValue, 1);
            const loraIntensity = Math.min(loraValue / maxValue, 1);
            
            // Blue for base, red for LoRA, purple for both
            const blue = [74, 144, 226];   // #4a90e2
            const red = [231, 76, 60];      // #e74c3c
            const purple = [155, 89, 182];  // #9b59b6
            
            if (baseIntensity > 0.01 && loraIntensity > 0.01) {{
                // Both models attend - purple
                const intensity = Math.max(baseIntensity, loraIntensity);
                return blendWithWhite(purple, intensity);
            }} else if (baseIntensity > 0.01) {{
                // Only base model - blue
                return blendWithWhite(blue, baseIntensity);
            }} else if (loraIntensity > 0.01) {{
                // Only LoRA model - red
                return blendWithWhite(red, loraIntensity);
            }} else {{
                // Neither model attends significantly
                return 'rgb(250, 250, 250)';
            }}
        }}
        
        function blendWithWhite(color, intensity) {{
            const white = [255, 255, 255];
            const r = Math.round(white[0] + (color[0] - white[0]) * intensity);
            const g = Math.round(white[1] + (color[1] - white[1]) * intensity);
            const b = Math.round(white[2] + (color[2] - white[2]) * intensity);
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}
        
        function showAttentionPatterns(position) {{
            if (!attentionPatterns) return;
            
            hoverPosition = position;
            isHovering = true;
            
            // Update hover info display
            document.getElementById('hover-position').textContent = position;
            document.getElementById('attention-mode').style.display = 'block';
            
            // Update all token colors based on attention from the hovered position
            const tokenSpans = document.querySelectorAll('.token');
            tokenSpans.forEach((span, idx) => {{
                if (idx <= position) {{  // Only show attention to previous positions
                    const baseAttn = getAttentionValue(position, idx, 'base');
                    const loraAttn = getAttentionValue(position, idx, 'lora');
                    const color = getAttentionColor(baseAttn, loraAttn);
                    span.style.backgroundColor = color;
                    
                    // Update text color based on new background
                    const brightness = getBrightness(color);
                    const textColor = brightness > 128 ? '#000' : '#fff';
                    
                    // Re-apply text coloring with return symbols
                    const token = tokens[idx];
                    let tokenDisplay = token
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/"/g, '&quot;')
                        .replace(/'/g, '&#39;')
                        .replace(/ /g, '&nbsp;')
                        .replace(/\\n/g, '↵');
                    
                    tokenDisplay = '<span style="color: ' + textColor + ';">' + tokenDisplay + '</span>';
                    tokenDisplay = tokenDisplay.replace(/↵/g, '</span><span style="color: #888;">↵</span><span style="color: ' + textColor + ';">');
                    tokenDisplay = tokenDisplay.replace(/<span style="color: [^"]+;"><\\/span>/g, '');
                    
                    // Update the token content without recreating the tooltip
                    const tooltip = span.querySelector('.token-tooltip');
                    span.innerHTML = tokenDisplay;
                    if (tooltip) {{
                        span.appendChild(tooltip);
                    }}
                }}
            }});
        }}
        
        function hideAttentionPatterns() {{
            if (!isHovering) return;
            
            hoverPosition = null;
            isHovering = false;
            
            // Hide hover info
            document.getElementById('attention-mode').style.display = 'none';
            
            // Restore KL divergence colors
            updateDisplay();
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
    with open(output_file, 'w') as f:
        f.write(dashboard_html)
    
    print(f"\nDashboard saved to: {output_file}")
    print(f"Open this file in a web browser to explore the attention patterns interactively.")

# %%
if __name__ == "__main__":
    # Generate or load cached data
    data = generate_attention_kl_data()
    
    # Generate dashboard from data
    generate_dashboard_from_data(data)
    
    print("\nDashboard generation complete!")

# %%