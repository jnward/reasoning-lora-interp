# Running Automated Interpretation Guide

## Quick Start

1. **Get an API Key**
   - OpenRouter (recommended): https://openrouter.ai/keys
   - Or use OpenAI: https://platform.openai.com/api-keys

2. **Set your API key**
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

3. **Run interpretation**
   ```bash
   # Test on a few features first
   python autointerp_context.py \
       --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
       --output interpretations.json \
       --features 7 8 9 10
   ```

## Cost Estimation

Before running on many features, estimate costs:

```bash
python estimate_costs.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --model openai/gpt-4o-mini \
    --num-features 100
```

### Approximate Costs (as of late 2024)
- **GPT-4o-mini**: ~$0.0001 per feature (recommended for initial runs)
- **GPT-4o**: ~$0.002 per feature
- **Claude 3.5 Sonnet**: ~$0.001 per feature
- **Claude 3 Haiku**: ~$0.0002 per feature

## Model Selection

### For Initial Exploration (Fast & Cheap)
```bash
--model "openai/gpt-4o-mini"      # GPT-4o mini (default)
--model "anthropic/claude-3-haiku" # Claude 3 Haiku
```

### For Better Quality
```bash
--model "openai/gpt-4o"                # GPT-4o
--model "anthropic/claude-3.5-sonnet"  # Claude 3.5 Sonnet
```

### Open Source Options
```bash
--model "meta-llama/llama-3.1-70b-instruct"  # Llama 3.1 70B
--model "google/gemini-pro-1.5"              # Gemini Pro
```

## Full Examples

### 1. Quick Test (10 features with GPT-4o-mini)
```bash
export OPENROUTER_API_KEY="your-key"
python autointerp_context.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output test_interpretations.json \
    --features 7 8 9 10 14 19 21 35 40 41
```

### 2. First 100 Features
```bash
python autointerp_context.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output interpretations_first_100.json \
    --features $(seq 0 99)
```

### 3. All Features (Expensive!)
```bash
# First check cost!
python estimate_costs.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --model openai/gpt-4o-mini

# If okay with cost, run:
python autointerp_context.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output interpretations_all.json
```

### 4. Using Different Models
```bash
# Claude 3.5 Sonnet for higher quality
python autointerp_context.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output interpretations_claude.json \
    --features 7 8 9 10 \
    --model "anthropic/claude-3.5-sonnet"
```

### 5. Using OpenAI Directly
```bash
export OPENAI_API_KEY="your-openai-key"
python autointerp_context.py \
    --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output interpretations_openai.json \
    --features 7 8 9 10 \
    --model "gpt-4o-mini" \
    --api-base "https://api.openai.com/v1"
```

## Output Format

The output JSON will look like:
```json
{
  "metadata": {
    "format": "context-aware",
    "model_name": "openai/gpt-4o-mini",
    "api_base": "https://openrouter.ai/api/v1",
    "num_features": 10
  },
  "explanations": [
    {
      "feature_id": 7,
      "explanation": "This neuron activates on mathematical units and notation, particularly 'm' for meters and 'J' for joules in physics equations."
    },
    ...
  ]
}
```

## Tips

1. **Start Small**: Test on 5-10 features first to ensure everything works
2. **Monitor Costs**: Use the cost estimator before large runs
3. **Save Progress**: The script saves after each feature, so you can interrupt and resume
4. **Model Choice**: Start with GPT-4o-mini for exploration, upgrade to better models for final analysis
5. **Rate Limits**: OpenRouter has generous rate limits, but for large runs you might need to add delays

## Troubleshooting

- **API Key Error**: Make sure `OPENROUTER_API_KEY` is set correctly
- **Model Not Found**: Check model name matches OpenRouter's format (e.g., "openai/gpt-4o-mini")
- **Empty Responses**: Some features might not generate explanations if they have no activations
- **Rate Limits**: Add a delay between requests if hitting limits
