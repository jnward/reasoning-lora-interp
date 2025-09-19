# Automated Interpretation for SAE Features

This directory contains tools for automatically interpreting SAE features using language models.

## Quick Start

```bash
# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Run interpretation on a few features (sequential - slow)
python autointerp_context.py \
    --input ../sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output interpretations.json \
    --features 7 8 9 10

# Run interpretation with parallel processing (10x faster!)
python autointerp_context_parallel.py \
    --input ../sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output interpretations.json \
    --features 7 8 9 10 \
    --max-concurrent 10 \
    --rate-limit 60
```

## Files

- `autointerp_context.py` - Main implementation with OpenRouter support (sequential)
- `autointerp_context_parallel.py` - Parallelized version for faster processing (recommended)
- `autointerp_context_demo.py` - Demo script to test formatting without API calls
- `estimate_costs.py` - Estimate costs before running
- `test_parallel_speed.py` - Test script to compare sequential vs parallel performance
- `run_autointerp_openrouter.sh` - Example script
- `AUTOINTERP_GUIDE.md` - Detailed usage guide
- `AUTOINTERP_CONTEXT_SUMMARY.md` - Technical details about the implementation

## Format

The system uses a context-aware format that shows the full text first, then lists activating tokens:

```
Example 1:
... the Boh r radius. Evaluate the magnitude of the integral $ |\math...
math 9.65
integral 7.23

Example 2:  
... \ times 1 0 ^{- 2 6} \, \ text {JT}\), \( a _ 0 = 5. 2 9...
text 8.90
times 6.12
```

## Rollout Diversity

To ensure diverse examples, the system limits examples from the same rollout:
- Default: Maximum 4 examples per rollout
- This helps capture different contexts where features activate
- Use `--max-per-rollout N` to adjust this limit
- Examples are selected by highest activation value within diversity constraints

## Models

- Default: `openai/o3-mini` (reasoning model with CoT)
- High-quality alternative: `openai/gpt-4o` 
- Budget option: `openai/gpt-4o-mini` 
- Also supports: Claude, Gemini, Llama, etc.

## Reasoning Models

o3-mini is a powerful reasoning model that uses chain-of-thought reasoning. Important notes:
- **OpenRouter requirements**: 
  - Must include `include_reasoning: true` (automatically added)
  - Requires at least 1000 `max_tokens` for complex prompts (set to 4000)
- The system automatically configures these parameters for o3 models
- Uses ~1200-2000 tokens per response (mostly for internal reasoning)
- Despite extensive reasoning, the final answer remains concise as requested
- Cost: ~$0.002-0.009 per feature interpretation (depending on actual usage)

See `AUTOINTERP_GUIDE.md` for full model options and pricing.

## Performance

The parallel version (`autointerp_context_parallel.py`) processes features concurrently:
- **Sequential**: ~20 seconds/feature
- **Parallel (10 concurrent)**: ~2 seconds/feature (10x faster!)
- **Time for 2000 features**: ~11 hours → ~1.1 hours

Configure parallelization with:
- `--max-concurrent N`: Number of concurrent API requests (default: 10)
- `--rate-limit N`: Max requests per minute (default: 60)

### Automatic Rate Limit Handling

The parallel version includes robust error handling:
- **Automatic retries**: Up to 3 retries with exponential backoff
- **Rate limit detection**: Automatically waits when rate limited
- **Adaptive concurrency**: Reduces concurrency when hitting rate limits, gradually increases when stable
- **Timeout handling**: 60-second timeout per request with retry
- **Connection error recovery**: Retries on transient network errors

You can safely set `--max-concurrent` higher than your rate limit - the system will automatically adapt!
