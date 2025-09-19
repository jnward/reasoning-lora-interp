# Rank-1 Reasoning: Minimal Parameter Diffs Encode Interpretable Reasoning Signals

This repository contains the code for the paper "Rank-1 Reasoning: Minimal Parameter Diffs Encode Interpretable Reasoning Signals", which demonstrates that a rank-1 LoRA can recover 73-90% of full finetuning performance on reasoning benchmarks while maintaining interpretable adapter directions.

## Installation

### Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Setup
Ensure you have access to the Qwen-2.5-32B-Instruct model from Hugging Face and sufficient GPU memory (recommended: 8x H200 or equivalent for training).

## Repository Structure

```
.
├── 0_train_lora.sh                      # Main LoRA training script
├── 1_lora_adapter_ablations/            # Adapter ablation experiments (Figure 1)
├── 2_lora_activation_interp/            # LoRA activation interpretation (Figure 2)
├── 3_lora_sae/                         # SAE training and interpretation (Figure 3)
├── s1_peft/                            # Core training implementation
└── rank-1_reasoning.pdf                # Paper
```

## Usage

### 1. Training the Rank-1 LoRA

Train a rank-1 LoRA adapter on Qwen-2.5-32B-Instruct using the s1K-1.1 dataset (1000 DeepSeek R1 chain-of-thought trajectories):

```bash
bash 0_train_lora.sh
```

This script:
- Adapts all MLP matrices (gate_proj, up_proj, down_proj) and attention matrices (q_proj, k_proj, v_proj, o_proj)
- Trains for 5 epochs with cosine learning rate scheduling
- Uses rank=1 with alpha=16 (RSLoRA scaling)
- Saves checkpoint to `s1_peft/ckpts_lora/`

Optional arguments:
- `--model_size`: Model size variant (default: 32B)
- `--rank`: LoRA rank (default: 1)
- `--lr`: Learning rate (default: 1e-3)
- `--layer_start/--layer_end`: Train specific layer range
- `--layer_indices`: Train specific layers (comma-separated)
- `--layer_stride`: Train every nth layer

### 2. Adapter Ablation Experiments (Figure 1)

Perform layer-wise and component-wise ablation studies to understand which LoRA components contribute most to performance:

#### Generate ablation data:
```bash
cd 1_lora_adapter_ablations
python 1.0_generate_ablation_data.py
```
This script:
- Performs leave-one-out ablation for each layer
- Measures KL divergence between ablated and full model
- Ablates individual adapter types (MLP vs attention)
- Saves results to timestamped JSON files

#### Visualize ablation results:
```bash
python 1.1_visualize_ablation_results.py
```
Creates:
- Per-layer KL divergence bar plots
- Component-wise ablation heatmap
- Saved as `lora_component_ablation_final.pdf`

Additional analysis scripts:
- `lora_ablation_all_by_type.py`: Ablates all MLP or attention components simultaneously
- `iterative_mlp_ablation_by_matrix.py`: Iterative ablation of MLP matrices
- `lora_layer_ablation_kl_visualization.py`: Alternative KL visualization

### 3. LoRA Activation Interpretation (Figure 2)

Interpret individual LoRA adapter directions and compare with MLP neurons:

#### Extract LoRA activations:
```bash
cd 2_lora_activation_interp
python 2.0_extract_lora_activations.py --num-gpus 4
```
Extracts scalar activations from all LoRA adapters during forward passes.

#### Find max-activating examples:
```bash
python 2.1_precompute_lora_topk.py
```
Identifies top-k activating examples for each adapter direction.

#### Run autointerpretation:
```bash
python 2.2_run_lora_autointerp.py
```
Uses LLM-based interpretation to understand what each direction detects.

#### Analyze interpretability:
```bash
python 2.3_analyze_lora_interpretability.py
```
Classifies features as monosemantic, fuzzy, or polysemantic.

#### Compare with MLP neurons:
Similar pipeline for MLP neurons (scripts 2.4-2.7), then:
```bash
python 2.8_compare_lora_mlp.py
```
Generates comparative analysis showing LoRA adapters activate more for reasoning-specific features.

### 4. SAE Interpretation (Figure 3)

Train sparse autoencoders on the entire LoRA activation state to discover fine-grained features:

#### Train SAE:
```bash
cd 3_lora_sae
python 3.0_train_sae.py
```
Trains a batch top-k SAE with:
- Expansion factor: 8x (creates ~3584 features from 448-dimensional input)
- Sparsity: k=16 active features per input
- Saves model to `trained_sae_adapters_*.pt`

#### Collect SAE features:
```bash
python 3.1_collect_sae_features.py --model-path trained_sae_adapters_g-u-d-q-k-v-o.pt
```
Finds max-activating examples for each SAE feature.

#### Generate interactive dashboard:
```bash
python 3.2_generate_sae_dashboard.py --input sae_features_data_*.json
```
Creates `sae_dashboard_new.html` - an interactive visualization showing:
- Max-activating examples for each feature
- Token-level activation heatmaps
- Feature interpretations and categories

#### Autointerpretation and Analysis:
```bash
cd 3.3_autointerp

# Run autointerpretation on SAE features
python 3.3.0_autointerp.py

# Categorize features into semantic groups
python 3.3.1_categorize.py

# Compute activation densities across categories
python 3.3.2_compute_activation_densities.py

# Create donut chart visualization (Figure 3)
python 3.3.3_create_donut_fig.py
```
