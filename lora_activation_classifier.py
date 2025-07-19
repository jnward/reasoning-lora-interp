# %%
import torch
import torch.nn.functional as F
import glob
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import json
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1
last_n_tokens = 16  # Extract activations from last 16 tokens

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-2*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# %%
# Load and explore dataset
print("Loading s1K-1.1 dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"Dataset has {len(dataset)} examples")

# Explore deepseek_grade column
print("\nExploring deepseek_grade column:")
grades = [ex['deepseek_grade'] for ex in dataset]
unique_grades = set(grades)
print(f"Unique values: {unique_grades}")
print(f"Distribution: Yes = {grades.count('Yes')}, No = {grades.count('No')}")

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("\nLoading base model...")
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
# Extract A matrices for all projections
print("\nExtracting LoRA A matrices...")

probe_directions = {
    'gate_proj': {},
    'up_proj': {},
    'down_proj': {}
}

# Get the number of layers
n_layers = model.config.num_hidden_layers
print(f"Model has {n_layers} layers")

for layer_idx in range(n_layers):
    # Extract A matrices for all projections
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        # Access the module directly
        module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
        
        # Extract the LoRA A matrix (which is a vector for rank-1)
        if hasattr(module, 'lora_A'):
            # Get the A matrix from the LoRA adapter
            lora_A_weight = module.lora_A['default'].weight.data
            # For rank-1, this should be shape [1, input_dim]
            # We want a 1D vector of shape [input_dim]
            probe_direction = lora_A_weight.squeeze()
            probe_directions[proj_type][layer_idx] = probe_direction

print(f"Extracted directions for {len(probe_directions['gate_proj'])} layers")
print(f"Total features: {3 * n_layers} (3 projections × {n_layers} layers)")

# %%
# Function to extract all activations for a single example
def extract_all_lora_activations(rollout_idx: int) -> Optional[Dict]:
    """Extract LoRA activations for all tokens in an example."""
    
    # Get the rollout
    rollout = dataset[rollout_idx]
    
    # Extract question and DeepSeek thinking trajectory + attempt
    question = rollout['question']
    thinking_trajectory = rollout.get('deepseek_thinking_trajectory', '')
    attempt = rollout.get('deepseek_attempt', '')
    
    if not thinking_trajectory or not attempt:
        return None
    
    # Use the exact format for thinking traces
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
    seq_len = len(input_ids)
    
    # Storage for projected activations
    projected_activations = {
        'gate_proj': {},
        'up_proj': {},
        'down_proj': {}
    }
    
    # Hook function to compute gate/up projections from pre-MLP residual
    def make_pre_mlp_hook(layer_idx):
        def hook(module, input, output):
            pre_mlp = output.detach()[0]  # [seq_len, hidden_size]
            # Compute projections for gate and up
            for proj_type in ['gate_proj', 'up_proj']:
                probe_dir = probe_directions[proj_type][layer_idx]
                activations = torch.matmul(pre_mlp.float(), probe_dir)  # [seq_len]
                projected_activations[proj_type][layer_idx] = activations.cpu().numpy()
        return hook
    
    # Hook function to compute down_proj projections from post-SwiGLU
    def make_down_proj_hook(layer_idx):
        def hook(module, input, output):
            # Get the post-SwiGLU activations (input to down_proj)
            post_swiglu = input[0].detach()[0]  # [seq_len, intermediate_size]
            # Project onto the A matrix
            probe_dir = probe_directions['down_proj'][layer_idx]
            activations = torch.matmul(post_swiglu.float(), probe_dir)  # [seq_len]
            projected_activations['down_proj'][layer_idx] = activations.cpu().numpy()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in range(n_layers):
        # Pre-MLP hook (computes gate/up projections)
        layernorm = model.model.model.layers[layer_idx].post_attention_layernorm
        hook = layernorm.register_forward_hook(make_pre_mlp_hook(layer_idx))
        hooks.append(hook)
        
        # Down-proj hook (computes down projections)
        down_proj = model.model.model.layers[layer_idx].mlp.down_proj
        hook = down_proj.register_forward_hook(make_down_proj_hook(layer_idx))
        hooks.append(hook)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Return all activations
    return {
        'activations': projected_activations,
        'seq_len': seq_len,
        'rollout_idx': rollout_idx
    }

# %%
# Function to compute feature vector from all activations
def compute_feature_vector(all_activations: Dict, last_n_tokens: int = 16) -> np.ndarray:
    """Compute feature vector by averaging over last N tokens."""
    seq_len = all_activations['seq_len']
    projected_activations = all_activations['activations']
    
    # Determine token range for last N tokens
    start_idx = max(0, seq_len - last_n_tokens)
    
    # Create activation vector: 192 dimensions (3 projections × 64 layers)
    activation_vector = []
    
    # For each projection type and layer, average activations over last N tokens
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        for layer_idx in range(n_layers):
            layer_activations = projected_activations[proj_type][layer_idx]
            # Take mean activation over last N tokens
            mean_activation = np.mean(layer_activations[start_idx:])
            activation_vector.append(mean_activation)
    
    return np.array(activation_vector)

# %%
# Test on a single example
print("\nTesting activation extraction on a single example...")
test_all_activations = extract_all_lora_activations(0)
if test_all_activations is not None:
    test_vector = compute_feature_vector(test_all_activations)
    print(f"Activation vector shape: {test_vector.shape}")
    print(f"Expected shape: ({3 * n_layers},)")
    assert test_vector.shape[0] == 3 * n_layers, "Incorrect activation vector size!"
    print(f"Full activations shape: {test_all_activations['activations']['gate_proj'][0].shape[0]} tokens")

# %%
# Collect activations for examples
NUM_EXAMPLES = 1000  # Change this to increase sample size
CACHE_DIR = "lora_activation_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Create cache filenames
full_cache_filename = f"{CACHE_DIR}/full_activations_n{NUM_EXAMPLES}_r{rank}.h5"
summary_cache_filename = f"{CACHE_DIR}/summary_activations_n{NUM_EXAMPLES}_last{last_n_tokens}_r{rank}.npz"

# Check if we need to compute full activations
need_full_compute = not os.path.exists(full_cache_filename)

if need_full_compute:
    print(f"\nCollecting full activations for {NUM_EXAMPLES} examples...")
    import h5py
    
    # Create HDF5 file for efficient storage of variable-length sequences
    with h5py.File(full_cache_filename, 'w') as hf:
        # Process specified number of examples
        num_examples = min(NUM_EXAMPLES, len(dataset))
        
        all_labels = []
        all_valid_indices = []
        all_seq_lens = []
        
        for rollout_idx in tqdm(range(num_examples), desc="Processing examples"):
            # Extract all activations
            all_activations = extract_all_lora_activations(rollout_idx)
            
            if all_activations is not None:
                # Get label
                grade = dataset[rollout_idx]['deepseek_grade']
                label = 1 if grade == "Yes" else 0
                all_labels.append(label)
                all_valid_indices.append(rollout_idx)
                all_seq_lens.append(all_activations['seq_len'])
                
                # Save activations for this example
                grp = hf.create_group(f'example_{rollout_idx}')
                grp.attrs['seq_len'] = all_activations['seq_len']
                grp.attrs['label'] = label
                
                # Save each projection type
                for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                    proj_grp = grp.create_group(proj_type)
                    for layer_idx in range(n_layers):
                        proj_grp.create_dataset(
                            f'layer_{layer_idx}', 
                            data=all_activations['activations'][proj_type][layer_idx],
                            compression='gzip',
                            compression_opts=4
                        )
                
                # Clear memory periodically
                if rollout_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Save metadata
        hf.attrs['num_examples'] = len(all_valid_indices)
        hf.attrs['num_layers'] = n_layers
        hf.create_dataset('labels', data=np.array(all_labels))
        hf.create_dataset('valid_indices', data=np.array(all_valid_indices))
        hf.create_dataset('seq_lens', data=np.array(all_seq_lens))
    
    print(f"Saved full activations to {full_cache_filename}")

# Now compute or load summary features
if os.path.exists(summary_cache_filename):
    print(f"\nLoading cached summary features from {summary_cache_filename}")
    cache_data = np.load(summary_cache_filename)
    X = cache_data['X']
    y = cache_data['y']
    valid_indices = cache_data['valid_indices'].tolist()
else:
    print(f"\nComputing summary features from full activations...")
    import h5py
    
    activation_vectors = []
    labels = []
    valid_indices = []
    
    with h5py.File(full_cache_filename, 'r') as hf:
        # Get metadata
        stored_labels = hf['labels'][:]
        stored_indices = hf['valid_indices'][:]
        
        # Pre-allocate array for faster processing
        activation_vectors = np.zeros((len(stored_indices), 3 * n_layers))
        
        # Process each stored example
        for i, rollout_idx in enumerate(tqdm(stored_indices, desc="Computing features")):
            grp = hf[f'example_{rollout_idx}']
            seq_len = grp.attrs['seq_len']
            
            # Determine token range for last N tokens
            start_idx = max(0, seq_len - last_n_tokens)
            
            # Batch read all layers for each projection type
            feat_idx = 0
            for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                proj_grp = grp[proj_type]
                
                # Read all layers at once for this projection
                for layer_idx in range(n_layers):
                    # Only load the last N tokens using HDF5 slicing
                    layer_activations = proj_grp[f'layer_{layer_idx}'][start_idx:]
                    # Compute mean and store directly
                    activation_vectors[i, feat_idx] = np.mean(layer_activations)
                    feat_idx += 1
            
            labels.append(stored_labels[i])
            valid_indices.append(rollout_idx)
        
    # X is already a numpy array from pre-allocation
    X = activation_vectors
    y = np.array(labels)
    
    # Save summary cache
    print(f"Saving summary features to {summary_cache_filename}")
    np.savez(summary_cache_filename, X=X, y=y, valid_indices=np.array(valid_indices))

print(f"\nLoaded {len(X)} examples")
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.sum(y)} Yes, {len(y) - np.sum(y)} No")

# %%
# Save the extracted features
print("\nSaving extracted features...")
np.save('lora_activation_features.npy', X)
np.save('lora_activation_labels.npy', y)
np.save('lora_activation_indices.npy', np.array(valid_indices))
print("Features saved!")

# %%
# Train linear classifier with regularization
print("\nTraining linear classifier...")

# Split data
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, valid_indices, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")
print(f"Features: {X.shape[1]}")
print(f"Feature/sample ratio: {X.shape[1]/len(X_train):.2f}")

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different regularization strengths with L1 and L2
print("\nTrying different regularization strengths...")
C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
results_l1 = []
results_l2 = []

# L1 regularization (sparse solution)
print("\nL1 Regularization (Sparse):")
for C in C_values:
    # Train with L1 regularization - use saga solver which supports L1
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42, penalty='l1', solver='saga')
    clf.fit(X_train_scaled, y_train)
    
    # Count non-zero coefficients
    non_zero = np.sum(np.abs(clf.coef_[0]) > 1e-6)
    
    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)
    
    results_l1.append({
        'C': C,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc,
        'non_zero': non_zero
    })
    
    print(f"C={C}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={train_acc-test_acc:.4f}, Non-zero features={non_zero}/192")

# L2 regularization (dense solution)
print("\nL2 Regularization (Dense):")
for C in C_values:
    # Train with L2 regularization
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42, penalty='l2')
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)
    
    results_l2.append({
        'C': C,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc
    })
    
    print(f"C={C}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={train_acc-test_acc:.4f}")

# Compare best L1 vs L2
best_l1 = max(results_l1, key=lambda x: x['test_acc'])
best_l2 = max(results_l2, key=lambda x: x['test_acc'])

if best_l1['test_acc'] > best_l2['test_acc']:
    print(f"\nBest: L1 with C={best_l1['C']} (test_acc={best_l1['test_acc']:.4f}, {best_l1['non_zero']} non-zero features)")
    best_penalty = 'l1'
    best_C = best_l1['C']
    best_solver = 'saga'
else:
    print(f"\nBest: L2 with C={best_l2['C']} (test_acc={best_l2['test_acc']:.4f})")
    best_penalty = 'l2'
    best_C = best_l2['C']
    best_solver = 'lbfgs'

# Train final model with best settings
clf = LogisticRegression(C=best_C, max_iter=2000, random_state=42, penalty=best_penalty, solver=best_solver)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)

# Calculate metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"\nFinal results with C={best_C}:")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# %%
# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No', 'Yes']))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %%
# Try dimensionality reduction
print("\nTrying dimensionality reduction...")

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Method 1: PCA
print("\n1. PCA approach:")
n_components_list = [10, 20, 30, 50]
for n_comp in n_components_list:
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    clf_pca = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_pca.fit(X_train_pca, y_train)
    
    train_acc = clf_pca.score(X_train_pca, y_train)
    test_acc = clf_pca.score(X_test_pca, y_test)
    
    print(f"PCA {n_comp} components: Train={train_acc:.4f}, Test={test_acc:.4f}")

# Method 2: Feature selection
print("\n2. Feature selection approach:")
k_values = [10, 20, 30, 50]
for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    clf_selected = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_selected.fit(X_train_selected, y_train)
    
    train_acc = clf_selected.score(X_train_selected, y_train)
    test_acc = clf_selected.score(X_test_selected, y_test)
    
    print(f"Top {k} features: Train={train_acc:.4f}, Test={test_acc:.4f}")

# %%
# Analyze feature importance
print("\nAnalyzing feature importance...")

# Get coefficients
coefficients = clf.coef_[0]

# If using L1, show sparsity pattern
if best_penalty == 'l1':
    non_zero_mask = np.abs(coefficients) > 1e-6
    non_zero_count = np.sum(non_zero_mask)
    print(f"\nL1 selected {non_zero_count} out of 192 features ({non_zero_count/192*100:.1f}%)")
    
    # Show which features were selected
    non_zero_indices = np.where(non_zero_mask)[0]
    print("\nSelected features by projection type:")
    for proj_idx, proj_type in enumerate(['gate_proj', 'up_proj', 'down_proj']):
        proj_start = proj_idx * n_layers
        proj_end = (proj_idx + 1) * n_layers
        selected_in_proj = [idx for idx in non_zero_indices if proj_start <= idx < proj_end]
        selected_layers = [idx - proj_start for idx in selected_in_proj]
        print(f"{proj_type}: {len(selected_in_proj)} features from layers {selected_layers[:10]}{'...' if len(selected_layers) > 10 else ''}")

# Reshape coefficients to (3, n_layers) for better visualization
coef_reshaped = coefficients.reshape(3, n_layers)

# Create projection type labels
proj_types = ['gate_proj', 'up_proj', 'down_proj']

# Plot heatmap of coefficients
plt.figure(figsize=(20, 6))
sns.heatmap(coef_reshaped, 
            xticklabels=range(n_layers), 
            yticklabels=proj_types,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Coefficient'})
plt.title('Linear Classifier Coefficients by Projection Type and Layer')
plt.xlabel('Layer')
plt.ylabel('Projection Type')
plt.tight_layout()
plt.show()

# %%
# Find most important features
top_k = 20
abs_coef = np.abs(coefficients)
top_indices = np.argsort(abs_coef)[-top_k:][::-1]

print(f"\nTop {top_k} most important features:")
for idx in top_indices:
    proj_idx = idx // n_layers
    layer_idx = idx % n_layers
    proj_type = proj_types[proj_idx]
    coef_value = coefficients[idx]
    print(f"{proj_type} Layer {layer_idx}: {coef_value:.4f}")

# %%
# Plot distribution of coefficients by projection type
plt.figure(figsize=(12, 6))

for i, proj_type in enumerate(proj_types):
    plt.subplot(1, 3, i+1)
    layer_coeffs = coef_reshaped[i, :]
    plt.bar(range(n_layers), layer_coeffs)
    plt.title(f'{proj_type} Coefficients')
    plt.xlabel('Layer')
    plt.ylabel('Coefficient')
    plt.ylim(-0.1, 0.1)  # Adjust based on actual range

plt.tight_layout()
plt.show()

# %%
# Analyze prediction probabilities
print("\nAnalyzing prediction probabilities...")

# Get probabilities for test set
y_test_proba = clf.predict_proba(X_test)[:, 1]

# Plot probability distribution
plt.figure(figsize=(10, 6))
plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.5, label='Actual No', color='red')
plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.5, label='Actual Yes', color='green')
plt.xlabel('Predicted Probability of "Yes"')
plt.ylabel('Count')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.show()

# %%
# Save the trained model
import joblib
joblib.dump(clf, 'lora_activation_classifier.joblib')
print("\nClassifier saved to 'lora_activation_classifier.joblib'")

# %%
# Analyze performance by projection type
print("\nAnalyzing performance by projection type...")

# Test each projection type separately
proj_types = ['gate_proj', 'up_proj', 'down_proj']
for proj_idx, proj_type in enumerate(proj_types):
    # Extract features for just this projection type
    start_idx = proj_idx * n_layers
    end_idx = (proj_idx + 1) * n_layers
    
    X_proj = X[:, start_idx:end_idx]
    X_train_proj, X_test_proj = X_proj[idx_train], X_proj[idx_test]
    
    # Standardize
    scaler_proj = StandardScaler()
    X_train_proj_scaled = scaler_proj.fit_transform(X_train_proj)
    X_test_proj_scaled = scaler_proj.transform(X_test_proj)
    
    # Train classifier
    clf_proj = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf_proj.fit(X_train_proj_scaled, y_train)
    
    train_acc = clf_proj.score(X_train_proj_scaled, y_train)
    test_acc = clf_proj.score(X_test_proj_scaled, y_test)
    
    print(f"{proj_type}: Train={train_acc:.4f}, Test={test_acc:.4f}")

# %%
print("\nExperiment complete!")
print(f"To improve results, consider:")
print(f"1. Increasing NUM_EXAMPLES (currently {NUM_EXAMPLES})")
print(f"2. Using more sophisticated feature extraction (e.g., max instead of mean)")
print(f"3. Trying different token ranges (not just last 16)")
print(f"4. Using ensemble methods instead of single classifier")
# %%
