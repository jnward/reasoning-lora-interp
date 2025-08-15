# SAE Interpretation Pipeline

This directory contains the Sparse Autoencoder (SAE) training and interpretation pipeline for analyzing LoRA adapter activations.

## Overview

The pipeline trains sparse autoencoders on LoRA activation vectors to discover interpretable features in the rank-1 adaptation space. This helps understand what the LoRA adapters are doing during mathematical reasoning.

## Key Components

### 1. Activation Generation
- **`generate_activations_multigpu.py`**: Multi-GPU script to generate H5 files containing LoRA activations
  - Processes mathematical reasoning examples from s1K-1.1 dataset
  - Extracts scalar activations between LoRA A and B matrices
  - Supports all 7 adapter types: gate_proj, up_proj, down_proj, q_proj, k_proj, v_proj, o_proj
  - Saves to `activations_all_adapters/` directory by default

### 2. SAE Training
- **`train_sae.py`**: Trains a Top-K sparse autoencoder on the activations
  - Auto-detects activation dimensions based on adapter types
  - Uses batch Top-K activation function for sparsity
  - Supports variable input dimensions (192 for MLP-only, 448 for all adapters)
  - Saves trained models with adapter configuration in filename

- **`batch_topk_sae.py`**: SAE model implementation with efficient Top-K sparsity

### 3. Feature Collection
- **`collect_sae_features_fast.py`**: Optimized script to find max-activating examples
  - 100x faster than original implementation
  - Uses float16 precision and batch processing
  - Finds top-k examples for each SAE feature
  - Extracts token contexts with activation values
  - Handles numpy type serialization properly

- **`collect_sae_features_multigpu.py`**: Multi-GPU version (use fast version instead)

### 4. Dashboard Generation
- **`generate_sae_dashboard.py`**: Creates interactive HTML dashboard
  - Visualizes max-activating examples for each feature
  - Shows activation heatmaps on token sequences
  - Supports filtering by features with full examples
  - Self-contained HTML with embedded data

## Typical Workflow

1. **Generate activations** (if not already done):
   ```bash
   python generate_activations_multigpu.py --num-examples 1000 --num-gpus 4
   ```

2. **Train SAE**:
   ```bash
   python train_sae.py --expansion-factor 16 --k 50 --batch-size 1024
   ```

3. **Collect max-activating examples**:
   ```bash
   python collect_sae_features_fast.py --model-path trained_sae_adapters_g-u-d-q-k-v-o.pt --top-k 10
   ```

4. **Generate dashboard**:
   ```bash
   python generate_sae_dashboard.py --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json
   ```

5. **View dashboard**: Open the generated HTML file in a browser

## Important Notes

- **Adapter Selection**: The pipeline auto-detects which adapters to use based on the training configuration
- **Directory Structure**: 
  - MLP-only: `activations/`
  - All adapters: `activations_all_adapters/`
  - Custom selection: `activations_{adapter_initials}/`
- **Memory Usage**: The fast collection script uses ~10GB GPU memory with default settings
- **Performance**: Context extraction can process ~1000 examples/second on a single GPU

## Configuration Tips

- **Expansion Factor**: Controls SAE dictionary size (typically 8-32x input dimension)
- **Sparsity (k)**: Number of active features per input (typically 10-100)
- **Context Window**: Tokens on each side of target (default 10)
- **Batch Sizes**: Larger = faster but more memory usage

## Troubleshooting

1. **JSON Serialization Errors**: Fixed in fast script with NumpyEncoder
2. **CUDA Graph Errors**: Disabled torch.compile to avoid issues
3. **Memory Issues**: Reduce batch sizes or use fewer examples
4. **Dashboard Shows "Loading"**: Ensure field names match (tokens/activations)

## Recent Improvements

- Added attention adapter support (q/k/v/o projections)
- 100x speedup in feature collection
- Fixed numpy serialization issues
- Optimized context extraction with batched encoding
- Added configurable batch sizes