# %%
import torch
import numpy as np
from safetensors import safe_open
import plotly.graph_objects as go
from pathlib import Path
import glob

# %%
# Configuration
lora_path = "/workspace/models/ckpts_1.1"
rank = 1  # Analyzing rank-1 LoRA

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*544")
if not lora_dirs:
    raise ValueError(f"No rank-{rank} LoRA found in {lora_path}")

lora_dir = sorted(lora_dirs)[-1]  # Get the most recent one
print(f"Loading LoRA from: {lora_dir}")

# %%
# Load LoRA weights from safetensors
adapter_path = Path(lora_dir) / "adapter_model.safetensors"
if not adapter_path.exists():
    raise FileNotFoundError(f"adapter_model.safetensors not found in {lora_dir}")

lora_weights = {}
with safe_open(adapter_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        lora_weights[key] = f.get_tensor(key)

print(f"Loaded {len(lora_weights)} weight tensors")

# %%
# Extract LoRA A and B matrices for MLP projections
# Reading matrices: up_proj and gate_proj use lora_A (read from residual stream)
# Writing matrices: down_proj uses lora_B (write to residual stream)

reading_matrices = {'up_proj': {}, 'gate_proj': {}}
writing_matrices = {'down_proj': {}}

# Parse the weights
for key, tensor in lora_weights.items():
    # Extract layer number
    parts = key.split('.')
    layer_num = None
    for part in parts:
        if part.isdigit():
            layer_num = int(part)
            break
    
    if layer_num is None:
        continue
    
    # Check projection type and matrix type
    if '.up_proj.' in key and 'lora_A' in key:
        reading_matrices['up_proj'][layer_num] = tensor.squeeze(0)  # Shape: [d_model]
    elif '.gate_proj.' in key and 'lora_A' in key:
        reading_matrices['gate_proj'][layer_num] = tensor.squeeze(0)  # Shape: [d_model]
    elif '.down_proj.' in key and 'lora_B' in key:
        writing_matrices['down_proj'][layer_num] = tensor.squeeze(-1)  # Shape: [d_model]

# %%
# Create ordered lists of matrices
n_layers = 64  # Qwen-2.5-32B has 64 layers

# Collect all reading vectors (up_proj and gate_proj from all layers)
reading_vectors = []
reading_labels = []
for layer in range(n_layers):
    if layer in reading_matrices['up_proj']:
        reading_vectors.append(reading_matrices['up_proj'][layer])
        reading_labels.append(f"L{layer}_up")
    if layer in reading_matrices['gate_proj']:
        reading_vectors.append(reading_matrices['gate_proj'][layer])
        reading_labels.append(f"L{layer}_gate")

# Collect all writing vectors (down_proj from all layers)
writing_vectors = []
writing_labels = []
for layer in range(n_layers):
    if layer in writing_matrices['down_proj']:
        writing_vectors.append(writing_matrices['down_proj'][layer])
        writing_labels.append(f"L{layer}_down")

# Stack into tensors
reading_tensor = torch.stack(reading_vectors)  # Shape: [n_reading, d_model]
writing_tensor = torch.stack(writing_vectors)  # Shape: [n_writing, d_model]

print(f"Reading vectors (up_proj + gate_proj): {reading_tensor.shape}")
print(f"Writing vectors (down_proj): {writing_tensor.shape}")

# %%
# Compute weight interference matrix (cosine similarity)
# Normalize vectors
reading_norms = torch.norm(reading_tensor, dim=1, keepdim=True)
writing_norms = torch.norm(writing_tensor, dim=1, keepdim=True)

reading_normalized = reading_tensor / (reading_norms + 1e-8)
writing_normalized = writing_tensor / (writing_norms + 1e-8)

# Compute cosine similarity matrix
# Rows: reading matrices (up_proj, gate_proj)
# Columns: writing matrices (down_proj)
interference_matrix = torch.mm(reading_normalized, writing_normalized.t()).numpy()

print(f"Interference matrix shape: {interference_matrix.shape}")

# %%
# Apply causality mask: early layers can't read from later layers
# Create mask where entry (i,j) is True if reading layer i can read from writing layer j
n_reading = len(reading_vectors)
n_writing = len(writing_vectors)
causal_mask = np.ones((n_reading, n_writing), dtype=bool)

for i, reading_label in enumerate(reading_labels):
    reading_layer = int(reading_label.split('_')[0][1:])  # Extract layer number
    
    for j, writing_label in enumerate(writing_labels):
        writing_layer = int(writing_label.split('_')[0][1:])  # Extract layer number
        
        # Reading layer can only read from earlier or same layer
        if writing_layer > reading_layer:
            causal_mask[i, j] = False

# Apply mask by setting non-causal entries to NaN
interference_matrix_causal = interference_matrix.copy()
interference_matrix_causal[~causal_mask] = np.nan

# %%
# Create interactive heatmap
fig = go.Figure(data=go.Heatmap(
    z=interference_matrix_causal,
    x=writing_labels,
    y=reading_labels,
    colorscale='RdBu',
    zmid=0,
    zmin=-0.2,
    zmax=0.2,
    connectgaps=False,  # Show NaN values as gaps (white with proper background)
    colorbar=dict(
        title=dict(text="Cosine Similarity"),
        tickmode="linear",
        tick0=-0.2,
        dtick=0.1
    ),
    hovertemplate='Reading: %{y}<br>Writing: %{x}<br>Cosine Similarity: %{z:.3f}<extra></extra>'
))

fig.update_layout(
    title=dict(
        text="Weight Interference: LoRA Writing → Reading Matrices",
        font=dict(size=14)
    ),
    xaxis_title="Writing (down_proj)",
    yaxis_title="Reading (up_proj, gate_proj)",
    width=600,
    height=500,
    xaxis=dict(
        tickangle=-45,
        tickmode='linear',
        dtick=4,
        tickfont=dict(size=8)
    ),
    yaxis=dict(
        tickmode='linear',
        dtick=4,
        tickfont=dict(size=8)
    ),
    margin=dict(l=60, r=20, t=40, b=60),
    plot_bgcolor='white'  # Set background to white
)

fig.show()

# %%
# Compute statistics
valid_similarities = interference_matrix_causal[~np.isnan(interference_matrix_causal)]

print("\nWeight Interference Statistics (Causal Only):")
print(f"  Number of valid connections: {len(valid_similarities)}")
print(f"  Mean similarity: {valid_similarities.mean():.3f}")
print(f"  Std deviation: {valid_similarities.std():.3f}")
print(f"  Min similarity: {valid_similarities.min():.3f}")
print(f"  Max similarity: {valid_similarities.max():.3f}")
print(f"  Median similarity: {np.median(valid_similarities):.3f}")

# Find strongest positive and negative interferences
threshold = 0.5
strong_positive = np.where((interference_matrix_causal > threshold) & ~np.isnan(interference_matrix_causal))
strong_negative = np.where((interference_matrix_causal < -threshold) & ~np.isnan(interference_matrix_causal))

print(f"\nStrong positive interferences (> {threshold}):")
for i, j in zip(strong_positive[0], strong_positive[1]):
    print(f"  {reading_labels[i]} ← {writing_labels[j]}: {interference_matrix_causal[i, j]:.3f}")

print(f"\nStrong negative interferences (< {-threshold}):")
for i, j in zip(strong_negative[0], strong_negative[1]):
    print(f"  {reading_labels[i]} ← {writing_labels[j]}: {interference_matrix_causal[i, j]:.3f}")

# %%
# Create a summary heatmap aggregated by layer
# Average the similarities for each layer pair
n_layers = 64
layer_interference = np.full((n_layers, n_layers), np.nan)

for reading_layer in range(n_layers):
    for writing_layer in range(n_layers):
        if writing_layer > reading_layer:
            continue  # Causal constraint
        
        # Find all relevant entries
        similarities = []
        for i, rlabel in enumerate(reading_labels):
            if int(rlabel.split('_')[0][1:]) == reading_layer:
                for j, wlabel in enumerate(writing_labels):
                    if int(wlabel.split('_')[0][1:]) == writing_layer:
                        sim = interference_matrix_causal[i, j]
                        if not np.isnan(sim):
                            similarities.append(sim)
        
        if similarities:
            layer_interference[reading_layer, writing_layer] = np.mean(similarities)

# Plot layer-aggregated interference
fig_layer = go.Figure(data=go.Heatmap(
    z=layer_interference,
    colorscale='RdBu',
    zmid=0,
    zmin=-0.2,
    zmax=0.2,
    connectgaps=False,
    colorbar=dict(
        title=dict(text="Mean Cosine Similarity")
    ),
    hovertemplate='Reading Layer: %{y}<br>Writing Layer: %{x}<br>Mean Similarity: %{z:.3f}<extra></extra>'
))

fig_layer.update_layout(
    title=dict(
        text="Layer-Aggregated Weight Interference",
        font=dict(size=14)
    ),
    xaxis_title="Writing Layer (down_proj)",
    yaxis_title="Reading Layer (up_proj + gate_proj)",
    width=500,
    height=500,
    xaxis=dict(tickmode='linear', dtick=8, tickfont=dict(size=10)),
    yaxis=dict(tickmode='linear', dtick=8, tickfont=dict(size=10)),
    margin=dict(l=60, r=20, t=40, b=60),
    plot_bgcolor='white'  # Set background to white
)

fig_layer.show()

# %%
# BASELINE EXPERIMENT: Random directions
print("\n" + "="*60)
print("BASELINE EXPERIMENT: Random Directions")
print("="*60)

# Get d_model dimension from the loaded vectors
d_model = reading_tensor.shape[1]
print(f"\nGenerating random directions with d_model={d_model}")

# Generate same number of random vectors as we have LoRA vectors
np.random.seed(42)  # For reproducibility

# Random reading vectors (same count as up_proj + gate_proj)
n_random_reading = len(reading_vectors)
random_reading_vectors = []
random_reading_labels = []
for i in range(n_random_reading):
    # Generate random unit vector
    vec = np.random.randn(d_model)
    vec = vec / np.linalg.norm(vec)
    random_reading_vectors.append(torch.tensor(vec, dtype=torch.float32))
    # Create label similar to original format
    layer = i // 2  # Assuming 2 reading types per layer
    proj_type = "up" if i % 2 == 0 else "gate"
    random_reading_labels.append(f"L{layer}_rand_{proj_type}")

# Random writing vectors (same count as down_proj)
n_random_writing = len(writing_vectors)
random_writing_vectors = []
random_writing_labels = []
for i in range(n_random_writing):
    # Generate random unit vector
    vec = np.random.randn(d_model)
    vec = vec / np.linalg.norm(vec)
    random_writing_vectors.append(torch.tensor(vec, dtype=torch.float32))
    random_writing_labels.append(f"L{i}_rand_down")

# Stack into tensors
random_reading_tensor = torch.stack(random_reading_vectors)
random_writing_tensor = torch.stack(random_writing_vectors)

print(f"Random reading vectors: {random_reading_tensor.shape}")
print(f"Random writing vectors: {random_writing_tensor.shape}")

# %%
# Compute interference matrix for random vectors
# Normalize vectors (already normalized, but doing it again for consistency)
random_reading_norms = torch.norm(random_reading_tensor, dim=1, keepdim=True)
random_writing_norms = torch.norm(random_writing_tensor, dim=1, keepdim=True)

random_reading_normalized = random_reading_tensor / (random_reading_norms + 1e-8)
random_writing_normalized = random_writing_tensor / (random_writing_norms + 1e-8)

# Compute cosine similarity matrix
random_interference_matrix = torch.mm(random_reading_normalized, random_writing_normalized.t()).numpy()

# Apply same causal mask
n_random_reading = len(random_reading_vectors)
n_random_writing = len(random_writing_vectors)
random_causal_mask = np.ones((n_random_reading, n_random_writing), dtype=bool)

for i, reading_label in enumerate(random_reading_labels):
    reading_layer = int(reading_label.split('_')[0][1:])
    
    for j, writing_label in enumerate(random_writing_labels):
        writing_layer = int(writing_label.split('_')[0][1:])
        
        if writing_layer > reading_layer:
            random_causal_mask[i, j] = False

# Apply mask
random_interference_matrix_causal = random_interference_matrix.copy()
random_interference_matrix_causal[~random_causal_mask] = np.nan

# %%
# Create heatmap for random baseline
fig_random = go.Figure(data=go.Heatmap(
    z=random_interference_matrix_causal,
    x=random_writing_labels,
    y=random_reading_labels,
    colorscale='RdBu',
    zmid=0,
    zmin=-1,
    zmax=1,
    connectgaps=False,
    colorbar=dict(
        title=dict(text="Cosine Similarity"),
        tickmode="linear",
        tick0=-1,
        dtick=0.5
    ),
    hovertemplate='Reading: %{y}<br>Writing: %{x}<br>Cosine Similarity: %{z:.3f}<extra></extra>'
))

fig_random.update_layout(
    title=dict(
        text="Random Baseline: Writing → Reading Matrices",
        font=dict(size=14)
    ),
    xaxis_title="Writing (random)",
    yaxis_title="Reading (random)",
    width=600,
    height=500,
    xaxis=dict(
        tickangle=-45,
        tickmode='linear',
        dtick=4,
        tickfont=dict(size=8)
    ),
    yaxis=dict(
        tickmode='linear',
        dtick=4,
        tickfont=dict(size=8)
    ),
    margin=dict(l=60, r=20, t=40, b=60),
    plot_bgcolor='white'
)

fig_random.show()

# %%
# Compute statistics for random baseline
random_valid_similarities = random_interference_matrix_causal[~np.isnan(random_interference_matrix_causal)]

print("\nRandom Baseline Statistics (Causal Only):")
print(f"  Number of valid connections: {len(random_valid_similarities)}")
print(f"  Mean similarity: {random_valid_similarities.mean():.3f}")
print(f"  Std deviation: {random_valid_similarities.std():.3f}")
print(f"  Min similarity: {random_valid_similarities.min():.3f}")
print(f"  Max similarity: {random_valid_similarities.max():.3f}")
print(f"  Median similarity: {np.median(random_valid_similarities):.3f}")

# Compare with LoRA statistics
print("\nComparison - LoRA vs Random:")
print(f"  Mean similarity - LoRA: {valid_similarities.mean():.3f}, Random: {random_valid_similarities.mean():.3f}")
print(f"  Std deviation - LoRA: {valid_similarities.std():.3f}, Random: {random_valid_similarities.std():.3f}")

# Find strong correlations in random baseline
random_strong_positive = np.where((random_interference_matrix_causal > threshold) & ~np.isnan(random_interference_matrix_causal))
random_strong_negative = np.where((random_interference_matrix_causal < -threshold) & ~np.isnan(random_interference_matrix_causal))

print(f"\nStrong correlations (|r| > {threshold}):")
print(f"  LoRA: {len(strong_positive[0])} positive, {len(strong_negative[0])} negative")
print(f"  Random: {len(random_strong_positive[0])} positive, {len(random_strong_negative[0])} negative")

# %%
# Create histogram of all cosine similarity values
print("\n" + "-"*60)
print("Distribution of All Cosine Similarity Values")
print("-"*60)

# Extract all valid (non-NaN) values from LoRA interference matrix
lora_all_values = interference_matrix_causal[~np.isnan(interference_matrix_causal)]

# Extract all valid values from random baseline
random_all_values = random_interference_matrix_causal[~np.isnan(random_interference_matrix_causal)]

print(f"\nTotal values analyzed:")
print(f"  LoRA: {len(lora_all_values):,}")
print(f"  Random: {len(random_all_values):,}")

# Compute statistics
print(f"\nStatistics for all cosine similarities:")
print(f"  LoRA - Mean: {lora_all_values.mean():.4f}, Std: {lora_all_values.std():.4f}")
print(f"  Random - Mean: {random_all_values.mean():.4f}, Std: {random_all_values.std():.4f}")

# Plot distribution of all values
fig_all_values = go.Figure()

# Define common bin edges for both histograms
bin_edges = np.linspace(-0.2, 0.2, 101)  # 100 bins from -0.2 to 0.2

fig_all_values.add_trace(go.Histogram(
    x=lora_all_values,
    name='LoRA',
    opacity=0.7,
    xbins=dict(
        start=-0.2,
        end=0.2,
        size=0.004  # (0.2 - (-0.2)) / 100 = 0.004
    ),
    histnorm='probability'
))

fig_all_values.add_trace(go.Histogram(
    x=random_all_values,
    name='Random',
    opacity=0.7,
    xbins=dict(
        start=-0.2,
        end=0.2,
        size=0.004  # Same bin size
    ),
    histnorm='probability'
))

fig_all_values.update_layout(
    title="Distribution of All Cosine Similarities (LoRA Writing ↔ Reading)",
    xaxis_title="Cosine Similarity",
    yaxis_title="Probability",
    barmode='overlay',
    width=700,
    height=500,
    showlegend=True,
    xaxis=dict(range=[-0.2, 0.2])  # Match the heatmap range
)

fig_all_values.show()

# %%
# Compute row-wise max statistics (keeping for later analysis)
print("\n" + "-"*60)
print("Row-wise Maximum Analysis")
print("-"*60)

# For LoRA
lora_row_maxes = []
for i in range(interference_matrix_causal.shape[0]):
    row = interference_matrix_causal[i, :]
    valid_values = row[~np.isnan(row)]
    if len(valid_values) > 0:
        lora_row_maxes.append(np.max(valid_values))

# For Random baseline
random_row_maxes = []
for i in range(random_interference_matrix_causal.shape[0]):
    row = random_interference_matrix_causal[i, :]
    valid_values = row[~np.isnan(row)]
    if len(valid_values) > 0:
        random_row_maxes.append(np.max(valid_values))

# Compute statistics
lora_median_row_max = np.median(lora_row_maxes)
random_median_row_max = np.median(random_row_maxes)

print(f"\nMedian of row-wise maximum cosine similarities:")
print(f"  LoRA: {lora_median_row_max:.3f}")
print(f"  Random: {random_median_row_max:.3f}")
print(f"  Difference: {lora_median_row_max - random_median_row_max:.3f}")

# %%
# Statistical significance analysis
print("\n" + "-"*60)
print("Statistical Analysis of the Difference")
print("-"*60)

# Perform Mann-Whitney U test (non-parametric test for difference in distributions)
from scipy import stats
statistic, p_value = stats.mannwhitneyu(lora_row_maxes, random_row_maxes, alternative='greater')

print(f"\nMann-Whitney U test (LoRA > Random):")
print(f"  Test statistic: {statistic}")
print(f"  p-value: {p_value:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Effect size (Cohen's d)
lora_mean = np.mean(lora_row_maxes)
random_mean = np.mean(random_row_maxes)
pooled_std = np.sqrt((np.std(lora_row_maxes)**2 + np.std(random_row_maxes)**2) / 2)
cohens_d = (lora_mean - random_mean) / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
print(f"  Interpretation: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'} effect")

# Analyze which layers show the strongest interference
print("\n" + "-"*60)
print("Layer-wise Analysis of Strong Connections")
print("-"*60)

# Find which reading layers have the strongest connections
strong_connection_threshold = 0.1  # Cosine similarity > 0.1
for i, (row_max, label) in enumerate(zip(lora_row_maxes, reading_labels)):
    if row_max > strong_connection_threshold:
        # Find which writing layer this reading layer connects to most strongly
        row = interference_matrix_causal[i, :]
        valid_mask = ~np.isnan(row)
        if np.any(valid_mask):
            max_idx = np.nanargmax(row)
            print(f"  {label} → {writing_labels[max_idx]}: {row[max_idx]:.3f}")

# %%
# MLP NEURON ANALYSIS
print("\n" + "="*60)
print("MLP NEURON ANALYSIS")
print("="*60)

# Load base model to access MLP weights
print("\nLoading base model for MLP weight access...")
from transformers import AutoModelForCausalLM

base_model_id = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load on CPU to save GPU memory
    trust_remote_code=True
)

# Extract MLP down_proj neuron directions
print("\nExtracting MLP neuron directions (vectorized)...")
mlp_neuron_directions_by_layer = []
neurons_per_layer = None

for layer_idx in range(n_layers):
    # Access down_proj weight matrix
    down_proj_weight = model.model.layers[layer_idx].mlp.down_proj.weight.data.float()  # Shape: [d_model, d_intermediate]
    
    # Normalize all columns at once
    weight_norms = torch.norm(down_proj_weight, dim=0, keepdim=True)
    normalized_weight = down_proj_weight / (weight_norms + 1e-8)
    
    mlp_neuron_directions_by_layer.append(normalized_weight)
    
    if layer_idx == 0:
        neurons_per_layer = down_proj_weight.shape[1]
        print(f"  Neurons per layer: {neurons_per_layer}, d_model={down_proj_weight.shape[0]}")
    
    if layer_idx % 10 == 0:
        print(f"  Processed layer {layer_idx}")

total_neurons = neurons_per_layer * n_layers
print(f"\nTotal MLP neurons: {total_neurons}")

# Free model memory
del model
torch.cuda.empty_cache()

# %%
# Compute row-wise max similarities between LoRA reading and MLP neurons
print("\nComputing similarities between LoRA reading directions and MLP neurons (vectorized)...")

from tqdm import tqdm

# For each LoRA reading vector, compute max similarity with MLP neurons
mlp_lora_row_maxes = []

for i, reading_vec in enumerate(tqdm(reading_vectors, desc="Processing LoRA reading vectors")):
    reading_layer = int(reading_labels[i].split('_')[0][1:])
    
    max_similarity = -1.0
    
    # Check neurons from causally valid layers (same or earlier)
    for layer_idx in range(reading_layer + 1):
        # Compute similarities with all neurons in this layer at once
        # mlp_neuron_directions_by_layer[layer_idx] has shape [d_model, n_neurons]
        similarities = torch.matmul(reading_vec.unsqueeze(0), mlp_neuron_directions_by_layer[layer_idx]).squeeze()
        layer_max = torch.max(similarities).item()
        max_similarity = max(max_similarity, layer_max)
    
    mlp_lora_row_maxes.append(max_similarity)

print(f"Computed {len(mlp_lora_row_maxes)} row-wise max similarities")

# %%
# Generate random baseline with same structure as MLP neurons
print(f"\nGenerating random baseline (same structure as MLP)...")
np.random.seed(43)  # Different seed from before

# Generate random directions with same structure as MLP
random_neuron_directions_by_layer = []
for layer_idx in tqdm(range(n_layers)):
    # Generate random matrix with same shape as MLP down_proj
    random_matrix = np.random.randn(d_model, neurons_per_layer).astype(np.float32)
    # Normalize columns
    norms = np.linalg.norm(random_matrix, axis=0, keepdims=True)
    random_matrix = random_matrix / (norms + 1e-8)
    random_neuron_directions_by_layer.append(torch.tensor(random_matrix))

# Compute row-wise max for random baseline
print("\nComputing similarities between LoRA reading directions and random neurons...")
random_neuron_row_maxes = []

for i, reading_vec in enumerate(tqdm(reading_vectors, desc="Processing LoRA reading vectors (random baseline)")):
    reading_layer = int(reading_labels[i].split('_')[0][1:])
    
    max_similarity = -1.0
    
    # Check neurons from causally valid layers
    for layer_idx in range(reading_layer + 1):
        # Compute similarities with all neurons in this layer at once
        similarities = torch.matmul(reading_vec.unsqueeze(0), random_neuron_directions_by_layer[layer_idx]).squeeze()
        layer_max = torch.max(similarities).item()
        max_similarity = max(max_similarity, layer_max)
    
    random_neuron_row_maxes.append(max_similarity)

# %%
# Compare MLP vs Random baseline statistics
print("\n" + "-"*60)
print("MLP Neuron Analysis Results")
print("-"*60)

mlp_median = np.median(mlp_lora_row_maxes)
random_median = np.median(random_neuron_row_maxes)

print(f"\nMedian of row-wise maximum cosine similarities:")
print(f"  MLP neurons: {mlp_median:.3f}")
print(f"  Random baseline: {random_median:.3f}")
print(f"  Difference: {mlp_median - random_median:.3f}")

# Compare with previous LoRA writing analysis
print(f"\nComparison of gaps:")
print(f"  LoRA writing vs random gap: {lora_median_row_max - random_median_row_max:.3f}")
print(f"  MLP neurons vs random gap: {mlp_median - random_median:.3f}")

# Statistical test
statistic, p_value = stats.mannwhitneyu(mlp_lora_row_maxes, random_neuron_row_maxes, alternative='greater')
print(f"\nMann-Whitney U test (MLP > Random):")
print(f"  p-value: {p_value:.6f}")

# Effect size
mlp_mean = np.mean(mlp_lora_row_maxes)
random_mean = np.mean(random_neuron_row_maxes)
pooled_std = np.sqrt((np.std(mlp_lora_row_maxes)**2 + np.std(random_neuron_row_maxes)**2) / 2)
cohens_d = (mlp_mean - random_mean) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f}")

# %%
# Create histogram comparison
fig_mlp = go.Figure()

fig_mlp.add_trace(go.Histogram(
    x=mlp_lora_row_maxes,
    name='MLP Neurons',
    opacity=0.7,
    nbinsx=50,
    histnorm='probability'
))

fig_mlp.add_trace(go.Histogram(
    x=random_neuron_row_maxes,
    name='Random Baseline',
    opacity=0.7,
    nbinsx=50,
    histnorm='probability'
))

# Add previous LoRA results for comparison
fig_mlp.add_trace(go.Histogram(
    x=lora_row_maxes,
    name='LoRA Writing (prev)',
    opacity=0.5,
    nbinsx=30,
    histnorm='probability',
    marker_color='green'
))

fig_mlp.update_layout(
    title="Row-wise Max Similarities: MLP Neurons vs Random Baseline",
    xaxis_title="Maximum Cosine Similarity per LoRA Reading Vector",
    yaxis_title="Probability",
    barmode='overlay',
    width=700,
    height=500,
    showlegend=True,
    legend=dict(x=0.7, y=0.95)
)

fig_mlp.show()

# %%
# Percentile analysis
print("\n" + "-"*60)
print("Percentile Analysis")
print("-"*60)

percentiles = [10, 25, 50, 75, 90, 95, 99]
print("\nPercentiles of row-wise max similarities:")
print("Percentile | MLP Neurons | Random | Difference")
print("-" * 50)
for p in percentiles:
    mlp_p = np.percentile(mlp_lora_row_maxes, p)
    random_p = np.percentile(random_neuron_row_maxes, p)
    print(f"{p:>10} | {mlp_p:>11.3f} | {random_p:>6.3f} | {mlp_p - random_p:>10.3f}")

# %%
