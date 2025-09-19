#!/bin/bash
# Example script to run autointerp with OpenRouter

# Set your OpenRouter API key
# Get one at https://openrouter.ai/keys
export OPENROUTER_API_KEY="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"

# Run interpretation on specific features with GPT-4o (default)
python autointerp_context.py \
    --input ../sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
    --output all_interpretations_gpt-5.json \
    --model "openai/gpt-5"

# Other model options on OpenRouter:
# --model "openai/gpt-4o-mini"               # Faster and cheaper alternative
# --model "anthropic/claude-3.5-sonnet"      # Claude 3.5 Sonnet
# --model "anthropic/claude-3-haiku"         # Claude 3 Haiku (fast/cheap)
# --model "google/gemini-pro-1.5"            # Gemini Pro
# --model "meta-llama/llama-3.1-70b-instruct" # Llama 3.1 70B

# Run on all features (be careful - this will cost money!)
# python autointerp_context.py \
#     --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
#     --output interpretations_all.json \
#     --model "openai/gpt-4o-mini"

# You can also use regular OpenAI API by changing the base URL:
# export OPENAI_API_KEY="your-openai-key"
# python autointerp_context.py \
#     --input sae_features_data_trained_sae_adapters_g-u-d-q-k-v-o.json \
#     --output interpretations.json \
#     --features 7 8 9 10 \
#     --model "gpt-4o-mini" \
#     --api-base "https://api.openai.com/v1"
