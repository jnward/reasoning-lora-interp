# %% [markdown]
# # SAE-based Steering Experiments for LoRA Adapters
# 
# This notebook implements steering using SAE decoder directions applied incrementally 
# during the forward pass to LoRA adapter activations.

# %%
import torch
import torch.nn.functional as F
import glob
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import random
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass

# Add parent directory to path for SAE imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_interp.batch_topk_sae import BatchTopKSAE

# %%
# Configuration
@dataclass
class Config:
    base_model_id: str = "Qwen/Qwen2.5-32B-Instruct"
    lora_path: str = "/workspace/models/ckpts_1.1"
    sae_path: str = "/workspace/reasoning_interp/sae_interp/trained_sae_adapters_g-u-d-q-k-v-o.pt"
    rank: int = 1
    
    # Adapter configuration (must match SAE training)
    adapter_types: List[str] = None  # Will be set from SAE config
    n_layers: int = 64
    
    # Steering parameters
    default_steering_strength: float = 50.0  # Increased for better visibility
    decay_factor: float = 1.0  # 1.0 = no decay, <1.0 = exponential decay
    
    # Generation parameters
    prefix_tokens: int = 50  # Tokens to generate before steering
    steered_tokens: int = 100  # Tokens to generate with steering
    temperature: float = 0.7
    do_sample: bool = True
    
    def __post_init__(self):
        if self.adapter_types is None:
            self.adapter_types = ['gate_proj', 'up_proj', 'down_proj', 
                                 'q_proj', 'k_proj', 'v_proj', 'o_proj']

config = Config()

# %% [markdown]
# ## 1. Load Models and SAE

# %%
def load_sae_model(sae_path: str) -> Tuple[BatchTopKSAE, Dict]:
    """Load SAE model and configuration."""
    print(f"Loading SAE from {sae_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    
    # Extract configuration
    sae_config = checkpoint['config']
    # Convert numpy types to Python types for JSON serialization
    sae_config_clean = {}
    for k, v in sae_config.items():
        if isinstance(v, (np.integer, np.int64)):
            sae_config_clean[k] = int(v)
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            sae_config_clean[k] = float(v)
        else:
            sae_config_clean[k] = v
    print(f"SAE Config: {json.dumps(sae_config_clean, indent=2)}")
    
    # Create SAE model
    sae = BatchTopKSAE(
        d_model=sae_config['d_model'],
        dict_size=sae_config['dict_size'],
        k=sae_config['k']
    )
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    sae.W_enc.data = state_dict['W_enc']
    sae.b_enc.data = state_dict['b_enc']
    sae.W_dec.data = state_dict['W_dec']
    sae.b_dec.data = state_dict['b_dec']
    
    # Update config with SAE adapter types
    config.adapter_types = sae_config['adapter_types']
    
    return sae, sae_config

# Load SAE
sae_model, sae_config = load_sae_model(config.sae_path)
sae_model.eval()

print(f"\nSAE Details:")
print(f"  Input dimension: {sae_config['d_model']}")
print(f"  Dictionary size: {sae_config['dict_size']}")
print(f"  Sparsity (k): {sae_config['k']}")
print(f"  Adapter types: {sae_config['adapter_types']}")

# %%
# Find and load LoRA checkpoint
lora_dirs = glob.glob(f"{config.lora_path}/s1-lora-32B-r{config.rank}-*544")
if not lora_dirs:
    raise ValueError(f"No LoRA checkpoint found at {config.lora_path}")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with LoRA
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)
model.eval()

device = next(model.parameters()).device
print(f"Model loaded on device: {device}")

# Move SAE to same device
sae_model = sae_model.to(device)

# %% [markdown]
# ## 2. Extract LoRA Probe Directions

# %%
def extract_lora_probe_directions(model, n_layers: int, adapter_types: List[str]) -> Dict[str, Dict[int, torch.Tensor]]:
    """Extract LoRA A matrices (probe directions) from the model."""
    probe_directions = {adapter_type: {} for adapter_type in adapter_types}
    
    for layer_idx in range(n_layers):
        # MLP projections
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            if proj_type not in adapter_types:
                continue
            module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A'):
                lora_A_weight = module.lora_A['default'].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
        
        # Attention projections
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if proj_type not in adapter_types:
                continue
            module = model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A'):
                lora_A_weight = module.lora_A['default'].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
    
    return probe_directions

# Extract probe directions
probe_directions = extract_lora_probe_directions(model, config.n_layers, config.adapter_types)

# Verify extraction
for adapter_type in config.adapter_types:
    n_layers_with_adapter = len(probe_directions[adapter_type])
    print(f"{adapter_type}: {n_layers_with_adapter} layers with LoRA adapters")

# %% [markdown]
# ## 3. Create Steering Vector from SAE Features

# %%
def create_steering_vector_from_sae(
    sae_model: BatchTopKSAE,
    feature_indices: List[int],
    feature_strengths: Optional[List[float]] = None,
    base_strength: float = 1.0
) -> torch.Tensor:
    """
    Create a steering vector from SAE decoder directions.
    
    Args:
        sae_model: Trained SAE model
        feature_indices: List of SAE feature indices to use
        feature_strengths: Optional individual strengths for each feature
        base_strength: Overall scaling factor
    
    Returns:
        Steering vector of shape [448] (or appropriate dimension)
    """
    if feature_strengths is None:
        feature_strengths = [1.0] * len(feature_indices)
    
    # Initialize steering vector
    steering_vector = torch.zeros(sae_model.d_model, device=sae_model.W_dec.device)
    
    # Add weighted decoder directions
    for feat_idx, strength in zip(feature_indices, feature_strengths):
        decoder_direction = sae_model.W_dec[feat_idx]  # Shape: [d_model]
        steering_vector += strength * decoder_direction
    
    # Apply overall scaling
    steering_vector *= base_strength
    
    return steering_vector

# %%
# Example: Create steering vector from top SAE features
# For now, we'll select some arbitrary features - in practice, you'd choose based on 
# interpretability analysis from the SAE dashboard

example_features = [224, 288, 368, 402, 404, 421]  # Example feature indices
example_steering_vector = create_steering_vector_from_sae(
    sae_model, 
    example_features,
    base_strength=config.default_steering_strength
)

print(f"Created steering vector with shape: {example_steering_vector.shape}")
print(f"Steering vector L2 norm: {torch.norm(example_steering_vector).item():.4f}")

# Debug: Show some steering vector values
print(f"\n[DEBUG] Sample steering vector components:")
for i in range(0, min(20, len(example_steering_vector)), 5):
    print(f"  Component {i}: {example_steering_vector[i].item():.6f}")
print(f"  Min value: {example_steering_vector.min().item():.6f}")
print(f"  Max value: {example_steering_vector.max().item():.6f}")
print(f"  Mean absolute value: {example_steering_vector.abs().mean().item():.6f}")

# %% [markdown]
# ## 4. Implement Incremental Steering Hooks

# %%
class IncrementalSteeringManager:
    """Manages incremental application of steering vectors during forward pass."""
    
    def __init__(
        self,
        model,
        probe_directions: Dict[str, Dict[int, torch.Tensor]],
        steering_vector: torch.Tensor,
        adapter_types: List[str],
        n_layers: int,
        decay_factor: float = 1.0,
        start_position: Optional[int] = None,
        debug: bool = True
    ):
        self.model = model
        self.probe_directions = probe_directions
        self.steering_vector = steering_vector
        self.adapter_types = adapter_types
        self.n_layers = n_layers
        self.decay_factor = decay_factor
        self.start_position = start_position
        self.debug = debug
        
        self.hooks = []
        self.activation_cache = {}
        self.steering_applied = {}
        self.hook_call_count = 0
        self.cumulative_position = 0  # Track total tokens processed
        
        # Map adapter types to indices in steering vector
        self.adapter_to_idx = {adapter: i for i, adapter in enumerate(adapter_types)}
        
        # Extract LoRA components for steering
        self.lora_components = self._extract_lora_components()
        
        if self.debug:
            print(f"\n[DEBUG] IncrementalSteeringManager initialized:")
            print(f"  - Steering vector shape: {steering_vector.shape}")
            print(f"  - Steering vector L2 norm: {torch.norm(steering_vector).item():.4f}")
            print(f"  - Number of LoRA components found: {len(self.lora_components)}")
            print(f"  - Start position: {start_position}")
        
    def _extract_lora_components(self):
        """Extract LoRA B matrices and scaling factors for all adapters."""
        components = {}
        for layer_idx in range(self.n_layers):
            # MLP projections
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                if proj_type not in self.adapter_types:
                    continue
                module = self.model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
                if hasattr(module, 'lora_B'):
                    key = f"{proj_type}_{layer_idx}"
                    lora_B = module.lora_B['default'].weight.data
                    scaling = module.scaling['default']
                    components[key] = {
                        'lora_B': lora_B,  # Shape: [output_dim, rank]
                        'scaling': scaling,  # Scalar scaling factor
                        'layer_idx': layer_idx,
                        'adapter_type': proj_type
                    }
                    if self.debug and layer_idx == 0:  # Print info for first layer only
                        print(f"[DEBUG] {key}: LoRA B shape={lora_B.shape}, norm={torch.norm(lora_B).item():.4f}, scaling={scaling}")
            
            # Attention projections  
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if proj_type not in self.adapter_types:
                    continue
                module = self.model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
                if hasattr(module, 'lora_B'):
                    key = f"{proj_type}_{layer_idx}"
                    lora_B = module.lora_B['default'].weight.data
                    scaling = module.scaling['default']
                    components[key] = {
                        'lora_B': lora_B,  # Shape: [output_dim, rank]
                        'scaling': scaling,  # Scalar scaling factor
                        'layer_idx': layer_idx,
                        'adapter_type': proj_type
                    }
                    if self.debug and layer_idx == 0:
                        print(f"[DEBUG] {key}: LoRA B shape={lora_B.shape}, norm={torch.norm(lora_B).item():.4f}, scaling={scaling}")
        return components
        
    def _get_steering_component(self, layer_idx: int, adapter_type: str) -> float:
        """Get the specific component of steering vector for this adapter."""
        # Calculate index in flattened 448-d vector
        # Order: [layer0_gate, layer0_up, ..., layer63_o]
        adapter_idx = self.adapter_to_idx[adapter_type]
        component_idx = layer_idx * len(self.adapter_types) + adapter_idx
        
        # Apply decay based on layer depth
        decay = self.decay_factor ** layer_idx if self.decay_factor < 1.0 else 1.0
        
        return self.steering_vector[component_idx].item() * decay
    
    def _make_steering_hook(self, layer_idx: int, adapter_type: str):
        """Hook that adds steering contribution to adapter output."""
        def hook(module, input, output):
            self.hook_call_count += 1
            
            # Update cumulative position based on sequence length
            x = input[0] if isinstance(input, tuple) else input
            seq_len = x.shape[0] if len(x.shape) == 2 else x.shape[1]
            
            # On first call of each forward pass, update cumulative position
            # We detect this by checking if this is the first layer's first adapter
            if adapter_type == self.adapter_types[0] and layer_idx == 0:
                self.cumulative_position += seq_len
            
            # Debug: Print first few hook calls
            if self.debug and self.hook_call_count <= 5:
                print(f"\n[DEBUG] Hook called #{self.hook_call_count}: {adapter_type}_{layer_idx}")
                print(f"  Input shape: {x.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Cumulative position: {self.cumulative_position}")
                print(f"  Start position: {self.start_position}")
            
            # Check if we should apply steering based on cumulative position
            if self.start_position is not None:
                should_steer = self.cumulative_position > self.start_position
                
                if self.debug and self.hook_call_count <= 5:
                    print(f"  Current seq_len: {seq_len}")
                    print(f"  Applying steering: {should_steer}")
                
                if not should_steer:
                    return output
            
            # Get steering component for this adapter
            steering_scalar = self._get_steering_component(layer_idx, adapter_type)
            
            if self.debug and self.hook_call_count <= 5:
                print(f"  Steering scalar: {steering_scalar}")
            
            # Get LoRA components for this adapter
            key = f"{adapter_type}_{layer_idx}"
            if key not in self.lora_components:
                if self.debug:
                    print(f"  WARNING: No LoRA components for {key}")
                return output
            
            components = self.lora_components[key]
            lora_B = components['lora_B'].to(output.device)  # Shape: [output_dim, rank=1]
            scaling_factor = components['scaling']
            
            # For rank-1, lora_B has shape [output_dim, 1]
            # We want to add: steering_scalar * lora_B[:, 0] * scaling_factor
            # Convert to same dtype as output
            steering_contribution = steering_scalar * lora_B[:, 0] * scaling_factor
            steering_contribution = steering_contribution.to(output.dtype)
            
            if self.debug and self.hook_call_count <= 5:
                print(f"  LoRA B norm: {torch.norm(lora_B).item():.6f}")
                print(f"  Scaling factor: {scaling_factor}")
                print(f"  Contribution norm: {torch.norm(steering_contribution).item():.6f}")
                print(f"  Output dtype: {output.dtype}, Contribution dtype: {steering_contribution.dtype}")
            
            # Reshape to match output dimensions
            # output shape is typically [batch, seq_len, output_dim] or [seq_len, output_dim]
            if output.dim() == 3:
                steering_contribution = steering_contribution.unsqueeze(0).unsqueeze(0)
            elif output.dim() == 2:
                steering_contribution = steering_contribution.unsqueeze(0)
            
            # Add steering contribution to output
            steered_output = output + steering_contribution
            
            # Store for analysis
            self.activation_cache[key] = {
                'steering_scalar': steering_scalar,
                'contribution_norm': torch.norm(steering_contribution).item(),
                'output_norm_before': torch.norm(output).item(),
                'output_norm_after': torch.norm(steered_output).item()
            }
            
            return steered_output
        
        return hook
    
    
    def register_hooks(self):
        """Register steering hooks on all LoRA adapter modules."""
        hook_count = 0
        components_by_type = {}
        
        # Reset cumulative position when registering hooks
        self.cumulative_position = 0
        
        for layer_idx in range(self.n_layers):
            # MLP adapter hooks
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                if proj_type in self.adapter_types:
                    key = f"{proj_type}_{layer_idx}"
                    if key in self.lora_components:
                        module = self.model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
                        hook = module.register_forward_hook(
                            self._make_steering_hook(layer_idx, proj_type)
                        )
                        self.hooks.append(hook)
                        hook_count += 1
                        components_by_type[proj_type] = components_by_type.get(proj_type, 0) + 1
            
            # Attention adapter hooks
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if proj_type in self.adapter_types:
                    key = f"{proj_type}_{layer_idx}"
                    if key in self.lora_components:
                        module = self.model.model.model.layers[layer_idx].self_attn.__getattr__(proj_type)
                        hook = module.register_forward_hook(
                            self._make_steering_hook(layer_idx, proj_type)
                        )
                        self.hooks.append(hook)
                        hook_count += 1
                        components_by_type[proj_type] = components_by_type.get(proj_type, 0) + 1
        
        print(f"\n[INFO] Registered {hook_count} steering hooks across {self.n_layers} layers")
        if self.debug:
            print(f"[DEBUG] Hooks by type: {components_by_type}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activation_changes(self) -> pd.DataFrame:
        """Get a summary of steering contributions."""
        records = []
        for key, values in self.activation_cache.items():
            adapter_type, layer_idx = key.rsplit('_', 1)
            
            records.append({
                'adapter_type': adapter_type,
                'layer': int(layer_idx),
                'steering_scalar': values['steering_scalar'],
                'contribution_norm': values['contribution_norm'],
                'output_norm_before': values['output_norm_before'],
                'output_norm_after': values['output_norm_after'],
                'change': values['output_norm_after'] - values['output_norm_before'],
                'percent_change': ((values['output_norm_after'] - values['output_norm_before']) / 
                                 (values['output_norm_before'] + 1e-8)) * 100
            })
        
        return pd.DataFrame(records)

# %% [markdown]
# ## 5. Steering Experiments

# %%
def run_steering_experiment(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    probe_directions: Dict,
    config: Config,
    feature_description: str = "Custom features"
) -> Dict:
    """
    Run a steering experiment comparing baseline and steered generation.
    
    Returns:
        Dictionary containing baseline and steered outputs, plus analysis
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    print(f"\n{'='*80}")
    print(f"Steering Experiment: {feature_description}")
    print(f"{'='*80}")
    print(f"Prompt ({input_ids.shape[1]} tokens): {prompt[:200]}...")
    
    # Generate baseline (no steering)
    print("\n1. Generating baseline (no steering)...")
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_new_tokens=config.prefix_tokens + config.steered_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=False)
    
    # Generate with steering
    print("\n2. Generating with steering...")
    
    # First generate prefix without steering
    with torch.no_grad():
        prefix_output = model.generate(
            input_ids,
            max_new_tokens=config.prefix_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Now continue with steering
    steering_manager = IncrementalSteeringManager(
        model=model,
        probe_directions=probe_directions,
        steering_vector=steering_vector,
        adapter_types=config.adapter_types,
        n_layers=config.n_layers,
        decay_factor=config.decay_factor,
        start_position=prefix_output.shape[1],  # Start steering after prefix
        debug=True  # Enable debug logging
    )
    
    steering_manager.register_hooks()
    
    try:
        with torch.no_grad():
            steered_output = model.generate(
                prefix_output,
                max_new_tokens=config.steered_tokens,
                temperature=config.temperature,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    finally:
        steering_manager.remove_hooks()
        
        # Debug: Print activation cache info
        print(f"\n[DEBUG] Total hook calls: {steering_manager.hook_call_count}")
        print(f"[DEBUG] Final cumulative position: {steering_manager.cumulative_position}")
        print(f"[DEBUG] Activation cache size: {len(steering_manager.activation_cache)}")
        if len(steering_manager.activation_cache) > 0:
            # Show a few entries
            for i, (key, values) in enumerate(list(steering_manager.activation_cache.items())[:3]):
                print(f"[DEBUG] Cache entry {key}: steering_scalar={values['steering_scalar']:.6f}, contribution_norm={values['contribution_norm']:.6f}")
        else:
            print(f"[DEBUG] No steering was applied! Check cumulative_position vs start_position")
    
    steered_text = tokenizer.decode(steered_output[0], skip_special_tokens=False)
    
    # Analyze activation changes
    activation_changes = steering_manager.get_activation_changes()
    
    # Prepare results
    results = {
        'prompt': prompt,
        'baseline_text': baseline_text,
        'steered_text': steered_text,
        'baseline_new_tokens': baseline_text[len(prompt):],
        'steered_new_tokens': steered_text[len(prompt):],
        'activation_changes': activation_changes,
        'feature_description': feature_description,
        'steering_vector_norm': torch.norm(steering_vector).item()
    }
    
    return results

# %%
# First, let's test if hooks fire on a simple forward pass
print("\n" + "="*80)
print("TESTING HOOK FUNCTIONALITY")
print("="*80)

test_input = "Test prompt for debugging"
test_tokens = tokenizer(test_input, return_tensors="pt").to(model.device)

# Create a simple steering manager for testing
test_steering = create_steering_vector_from_sae(
    sae_model, [0, 1, 2], base_strength=10.0
)

test_manager = IncrementalSteeringManager(
    model=model,
    probe_directions=probe_directions,
    steering_vector=test_steering,
    adapter_types=config.adapter_types,
    n_layers=config.n_layers,
    decay_factor=1.0,
    start_position=None,  # No position restriction for test
    debug=True
)

test_manager.register_hooks()

print("\nRunning test forward pass...")
print(f"Input tokens: {test_tokens.input_ids.shape[1]}")
with torch.no_grad():
    test_output = model(test_tokens.input_ids)

test_manager.remove_hooks()

print(f"\n[TEST RESULTS]")
print(f"  Hook calls: {test_manager.hook_call_count}")
print(f"  Activation cache entries: {len(test_manager.activation_cache)}")
if test_manager.hook_call_count == 0:
    print("  WARNING: No hooks were called!")
elif len(test_manager.activation_cache) == 0:
    print("  WARNING: Hooks called but no activations cached!")
else:
    print("  SUCCESS: Hooks are working!")

print("\n" + "="*80)

# %%
# Load a test problem from MATH500
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
test_problem = dataset[42]  # Pick an arbitrary problem

# Format prompt
system_prompt = "You are a helpful mathematics assistant."
test_prompt = (
    f"<|im_start|>system\n{system_prompt}\n"
    f"<|im_start|>user\n{test_problem['problem']}\n"
    f"<|im_start|>assistant\n"
    f"<|im_start|>think\n"
)

# Run experiment with example steering vector
results = run_steering_experiment(
    model=model,
    tokenizer=tokenizer,
    prompt=test_prompt,
    steering_vector=example_steering_vector,
    probe_directions=probe_directions,
    config=config,
    feature_description="Example features [0, 10, 100, 500, 1000]"
)

# %% [markdown]
# ## 6. Visualize Results

# %%
def display_generation_comparison(results: Dict):
    """Display a comparison of baseline vs steered generation."""
    print("\n" + "="*80)
    print("GENERATION COMPARISON")
    print("="*80)
    
    print("\n--- BASELINE GENERATION ---")
    print(results['baseline_new_tokens'][:500])
    
    print("\n--- STEERED GENERATION ---")
    print(results['steered_new_tokens'][:500])
    
    print("\n--- STATISTICS ---")
    print(f"Steering vector L2 norm: {results['steering_vector_norm']:.4f}")
    print(f"Number of steering contributions: {len(results['activation_changes'])}")
    
    if len(results['activation_changes']) > 0:
        df = results['activation_changes']
        print(f"Mean contribution norm: {df['contribution_norm'].mean():.6f}")
        print(f"Max contribution norm: {df['contribution_norm'].max():.6f}")
        print(f"Mean output change: {df['change'].mean():.6f}")
        print(f"Max output change: {df['change'].abs().max():.6f}")

display_generation_comparison(results)

# %%
def plot_activation_changes(results: Dict):
    """Plot steering contributions across layers and adapter types."""
    df = results['activation_changes']
    
    if len(df) == 0:
        print("No activation changes to plot")
        return
    
    # Create heatmap of contribution norms
    pivot_df = df.pivot(index='layer', columns='adapter_type', values='contribution_norm')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Viridis',
        text=np.round(pivot_df.values, 6),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Contribution Norm")
    ))
    
    fig.update_layout(
        title=f"Steering Contribution Norms<br>{results['feature_description']}",
        xaxis_title="Adapter Type",
        yaxis_title="Layer",
        height=800,
        width=600
    )
    
    fig.show()
    
    # Bar chart of average contribution by adapter type
    avg_by_type = df.groupby('adapter_type')['contribution_norm'].mean().sort_values()
    
    fig2 = px.bar(
        x=avg_by_type.values,
        y=avg_by_type.index,
        orientation='h',
        title="Average Steering Contribution by Adapter Type",
        labels={'x': 'Average Contribution Norm', 'y': 'Adapter Type'},
        color=avg_by_type.values,
        color_continuous_scale='Viridis'
    )
    
    fig2.show()

plot_activation_changes(results)

# %% [markdown]
# ## 7. Interactive Feature Selection

# %%
def load_sae_feature_analysis(feature_data_path: Optional[str] = None):
    """Load SAE feature analysis data if available."""
    if feature_data_path is None:
        feature_data_path = "/workspace/reasoning_interp/sae_interp/sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json"
    
    if not os.path.exists(feature_data_path):
        print(f"Feature analysis data not found at {feature_data_path}")
        print("Run collect_sae_features_fast.py to generate this data")
        return None
    
    with open(feature_data_path, 'r') as f:
        feature_data = json.load(f)
    
    return feature_data

# Try to load feature analysis
feature_analysis = load_sae_feature_analysis()

if feature_analysis:
    # Handle features as dict (indexed by string keys)
    features_dict = feature_analysis['features']
    print(f"Loaded analysis for {len(features_dict)} SAE features")
    
    # Show top features by average activation
    features_with_examples = []
    for feat_idx_str, feat_data in features_dict.items():
        if isinstance(feat_data, dict) and 'top_examples' in feat_data:
            if feat_data['top_examples'] and len(feat_data['top_examples']) > 0:
                features_with_examples.append((int(feat_idx_str), feat_data))
    
    if features_with_examples:
        print(f"\nFeatures with examples: {len(features_with_examples)}")
        print("\nTop 10 features by average activation:")
        
        sorted_features = sorted(
            features_with_examples,
            key=lambda x: x[1].get('average_activation', 0),
            reverse=True
        )[:10]
        
        for idx, feat in sorted_features:
            avg_act = feat.get('average_activation', 0)
            print(f"  Feature {idx}: avg activation = {avg_act:.4f}")
            if feat['top_examples']:
                # Show first example
                ex = feat['top_examples'][0]
                tokens = ex.get('tokens', [])
                position = ex.get('position', 0)
                if tokens:
                    context = ''.join(tokens[max(0, position-5):position+5])
                    print(f"    Example: ...{context}...")

# %%
def experiment_with_top_features(
    model, 
    tokenizer,
    sae_model,
    probe_directions,
    config,
    feature_analysis,
    n_features: int = 5,
    test_prompt: Optional[str] = None
):
    """Run experiments with top-activating SAE features."""
    
    if feature_analysis is None:
        print("No feature analysis available")
        return
    
    # Get top features (handle dict structure)
    features_dict = feature_analysis['features']
    features_with_examples = []
    for feat_idx_str, feat_data in features_dict.items():
        if isinstance(feat_data, dict) and 'top_examples' in feat_data:
            if feat_data['top_examples'] and len(feat_data['top_examples']) > 0:
                features_with_examples.append((int(feat_idx_str), feat_data))
    
    sorted_features = sorted(
        features_with_examples,
        key=lambda x: x[1].get('average_activation', 0),
        reverse=True
    )[:n_features]
    
    top_feature_indices = [idx for idx, _ in sorted_features]
    
    print(f"\nTesting steering with top {n_features} features: {top_feature_indices}")
    
    # Create steering vector from top features
    steering_vector = create_steering_vector_from_sae(
        sae_model,
        top_feature_indices,
        base_strength=config.default_steering_strength
    )
    
    # Use provided prompt or default
    if test_prompt is None:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problem = dataset[10]
        test_prompt = (
            f"<|im_start|>system\nYou are a helpful mathematics assistant.\n"
            f"<|im_start|>user\n{problem['problem']}\n"
            f"<|im_start|>assistant\n<|im_start|>think\n"
        )
    
    # Run experiment
    results = run_steering_experiment(
        model=model,
        tokenizer=tokenizer,
        prompt=test_prompt,
        steering_vector=steering_vector,
        probe_directions=probe_directions,
        config=config,
        feature_description=f"Top {n_features} features by activation"
    )
    
    return results

# Run experiment with top features if analysis is available
if feature_analysis:
    top_features_results = experiment_with_top_features(
        model, tokenizer, sae_model, probe_directions, config, 
        feature_analysis, n_features=10
    )
    
    if top_features_results:
        display_generation_comparison(top_features_results)
        plot_activation_changes(top_features_results)

# %% [markdown]
# ## 8. Strength Ablation Study

# %%
def strength_ablation_study(
    model,
    tokenizer,
    sae_model,
    probe_directions,
    config,
    feature_indices: List[int],
    strengths: List[float],
    test_prompt: str
) -> pd.DataFrame:
    """Test different steering strengths and measure effects."""
    
    results_data = []
    
    for strength in tqdm(strengths, desc="Testing strengths"):
        # Create steering vector with current strength
        steering_vector = create_steering_vector_from_sae(
            sae_model,
            feature_indices,
            base_strength=strength
        )
        
        # Run experiment
        results = run_steering_experiment(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            steering_vector=steering_vector,
            probe_directions=probe_directions,
            config=config,
            feature_description=f"Strength={strength}"
        )
        
        # Calculate metrics
        activation_changes = results['activation_changes']
        
        if len(activation_changes) > 0:
            mean_contribution = activation_changes['contribution_norm'].mean()
            max_contribution = activation_changes['contribution_norm'].max()
            mean_change = activation_changes['change'].mean()
            max_change = activation_changes['change'].abs().max()
        else:
            mean_contribution = 0
            max_contribution = 0
            mean_change = 0
            max_change = 0
        
        results_data.append({
            'strength': strength,
            'vector_norm': results['steering_vector_norm'],
            'mean_contribution_norm': mean_contribution,
            'max_contribution_norm': max_contribution,
            'mean_output_change': mean_change,
            'max_output_change': max_change,
            'baseline_text': results['baseline_new_tokens'][:100],
            'steered_text': results['steered_new_tokens'][:100]
        })
    
    return pd.DataFrame(results_data)

# %%
# Run strength ablation with example features
strengths_to_test = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

print("Running strength ablation study...")
strength_results = strength_ablation_study(
    model, tokenizer, sae_model, probe_directions, config,
    feature_indices=[0, 10, 100],  # Use a few features
    strengths=strengths_to_test,
    test_prompt=test_prompt
)

# Plot results
fig = px.line(
    strength_results,
    x='strength',
    y=['mean_contribution_norm', 'max_contribution_norm'],
    title="Steering Strength vs Contribution Norms",
    labels={'value': 'Contribution Norm', 'strength': 'Steering Strength'},
    markers=True
)
fig.show()

print("\nStrength ablation results:")
print(strength_results[['strength', 'vector_norm', 'mean_contribution_norm', 'max_contribution_norm', 'mean_output_change', 'max_output_change']])

# %% [markdown]
# ## 9. Save and Load Experiments

# %%
def save_experiment_results(results: Dict, filename: str):
    """Save experiment results to file."""
    # Convert DataFrame to dict for JSON serialization
    results_copy = results.copy()
    if 'activation_changes' in results_copy:
        results_copy['activation_changes'] = results_copy['activation_changes'].to_dict('records')
    
    with open(filename, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"Saved results to {filename}")

def load_experiment_results(filename: str) -> Dict:
    """Load experiment results from file."""
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert activation_changes back to DataFrame
    if 'activation_changes' in results:
        results['activation_changes'] = pd.DataFrame(results['activation_changes'])
    
    return results

# Save example results
save_experiment_results(
    results,
    "/workspace/reasoning_interp/steering/example_steering_results.json"
)

print("\n" + "="*80)
print("Steering experiment notebook ready!")
print("="*80)
print("\nKey functions available:")
print("  - create_steering_vector_from_sae(): Create steering vectors from SAE features")
print("  - run_steering_experiment(): Run full steering experiment")
print("  - IncrementalSteeringManager: Class for managing incremental steering")
print("  - strength_ablation_study(): Test different steering strengths")
print("\nNext steps:")
print("  1. Run collect_sae_features_fast.py to generate feature analysis data")
print("  2. Identify interesting SAE features from the dashboard")
print("  3. Test steering with those specific features")
print("  4. Analyze which features have the strongest steering effects")
# %%
