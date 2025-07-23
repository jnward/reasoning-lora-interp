# Reasoning Interpretability with Rank-1 LoRA

## Project Overview

This project investigates the interpretability of rank-1 LoRA (Low-Rank Adaptation) adapters for mathematical reasoning. The key insight is that rank-1 LoRA recovers most of the reasoning performance of full finetuning while being highly interpretable, making it an ideal starting point for understanding which features and circuits are important for reasoning.

**Research Goal**: Understand what reasoning models do that base models don't - specifically, which features and circuits enable better performance on reasoning benchmarks.

## Model Details

- **Base Model**: Qwen/Qwen2.5-32B-Instruct
- **Adaptation**: Rank-1 LoRA (r=1)
- **Training Dataset**: s1k-1.1 (math reasoning dataset)
- **Model Location**: `/workspace/models/ckpts_1.1/`

## Key Findings

1. **LoRA Direction Interpretability**: LoRA directions tend to be interpretable through analysis of max-activating examples
2. **Steering Capability**: Steering with LoRA directions shows some promise but needs more investigation
3. **Feature Importance**: Rank-1 constraint creates interpretable feature directions that can be analyzed

## Experiment Types

### 1. LoRA Ablation Studies (`lora_ablation_*.py`)
- Systematically ablate LoRA weights by layer/type
- Measure impact on model outputs using KL divergence
- Identify most important layers for reasoning

### 2. Activation Analysis
- **Max Activating Examples** (`lora_max_activating_examples.py`): Find inputs that maximally activate LoRA neurons
- **Activation Caching**: Store activations for efficient analysis

### 3. Steering Experiments (`lora_steering_experiment.py`)
- Test if adding LoRA directions to activations changes model behavior
- Scale by average MLP output norms for appropriate magnitude
- Preliminary results show some effect but needs refinement

### 4. Attribution Studies (`lora_neuron_attribution_study.py`)
- Compute gradients through linearized attention/LayerNorm
- Attribute model predictions to specific LoRA neuron activations
- Currently in progress, not yet yielding clear insights

### 5. Counterfactual Generation (`cf/generate_aime_rollouts.py`)
- Generate multiple solution attempts for AIME problems
- Find problems where model sometimes succeeds and sometimes fails
- Use for analyzing what changes between correct/incorrect attempts

### 6. Classification Experiments (`lora_activation_classifier.py`)
- Train linear probes on LoRA activations
- Attempt to predict solution correctness from activation patterns
- Limited success so far

## Tokenization Format

The model uses specific chat templates with special tokens:

```
<|im_start|>system
{system_prompt}
<|im_start|>user
{question}
<|im_start|>assistant
<|im_start|>think
{thinking_trajectory}
<|im_start|>answer
{answer}<|im_end|>
```

**Important**: When analyzing model outputs, search for `<|im_start|>` to understand token boundaries and special sections.

## Main Workflows

### 1. Ablation Analysis
```bash
# Run layer-wise ablation
python lora_ablation_experiment.py

# Analyze results
python analyze_lora_weights.py
```

### 3. Steering Experiments
```bash
# Test steering with LoRA directions
python lora_steering_experiment.py
```

## Key Files

- `analyze_lora_weights.py`: Visualize LoRA weight matrices and compute similarities
- `lora_ablation_experiment.py`: Layer-wise ablation studies
- `lora_max_activating_examples.py`: Find max-activating dataset examples
- `lora_steering_experiment.py`: Test steering with LoRA directions
- `lora_neuron_attribution_study.py`: Attribution analysis (experimental)
- `lora_activation_classifier.py`: Train probes on activations
- `cf/generate_aime_rollouts.py`: Generate counterfactual examples

## Dependencies

- PyTorch with CUDA support
- Transformers, PEFT for LoRA
- vLLM for efficient generation (counterfactuals)
- Standard ML/visualization libraries (numpy, pandas, plotly, etc.)

## Future Work

1. **Deep Feature Understanding**: Thoroughly analyze interesting features discovered through max-activating examples
2. **Steering Refinement**: Better understand when/why steering works and improve methodology
3. **Attribution Methods**: Explore more sophisticated attribution techniques
4. **Circuit Discovery**: Map out full circuits involved in reasoning
5. **Comparative Analysis**: Compare with higher-rank LoRA and full finetuning
6. **Mechanistic Understanding**: Connect findings to broader mechanistic interpretability

## Notes

- Model paths are hardcoded to `/workspace/models/ckpts_1.1/` - adjust as needed
- Many scripts are exploratory one-offs
- Classification hasn't been very successful - activation patterns may be too high-dimensional
- Attribution is promising but needs more work on methodology