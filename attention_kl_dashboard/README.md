# Attention KL Dashboard Generator

Visualizes KL divergence between base and LoRA model attention patterns.

## Usage

```python
python attention_kl_dashboard_generator.py
```

Generates `attention_kl_dashboard.html` - open in browser to explore.

## Features

- **Interactive Dashboard**: Token-level KL divergence heatmap
- **Attention Patterns**: Hover tokens to see attention differences (blue=base, red=LoRA, purple=both)
- **Multi-view**: Analyze by head, layer average, or overall average
- **Caching**: Automatic caching of expensive computations
- **1024 Token Support**: Handles long sequences efficiently with sparse representations

## Output

- HTML dashboard with embedded data and visualization
- No external dependencies needed for viewing
- Sidebar shows top divergent attention heads
- Color-coded tokens show KL divergence intensity

## Requirements

- PyTorch, Transformers, PEFT
- Base model: Qwen/Qwen2.5-32B-Instruct
- LoRA checkpoint: `/workspace/reasoning_interp/lora_checkpoints/s1-lora-32B-r1-20250627_013544`