#!/usr/bin/env python3
"""
Test different models to debug empty responses.
"""

import asyncio
import aiohttp
import os
import json

async def test_models():
    """Test different models with simple and complex prompts."""
    
    api_key = "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"
    
    # Test prompts
    simple_prompt = "What is 2+2? Answer with just the number."
    
    neuron_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document where the neuron activates and describe what it's looking for.

Your explanation should be just a few words or a short phrase. Don't write complete sentences.

The activation format shows the full text first, then lists tokens where the neuron fired along with their activation strengths (0-10 scale).

<neuron_activations>
Example 1:
The cat sat on the mat looking at the bird.
cat 8.5
mat 7.2
bird 6.1

Example 2:
A small cat jumped over the fence quickly.
cat 9.1
small 3.2
</neuron_activations>

What is this neuron looking for? Answer in just a few words:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Model Test"
    }
    
    # Test configurations
    tests = [
        ("openai/gpt-4o-mini", simple_prompt, "GPT-4o-mini simple"),
        ("openai/gpt-4o", simple_prompt, "GPT-4o simple"),
        ("openai/o3-mini", simple_prompt, "o3-mini simple"),
        ("openai/gpt-4o-mini", neuron_prompt, "GPT-4o-mini neuron"),
        ("openai/gpt-4o", neuron_prompt, "GPT-4o neuron"),
        ("openai/o3-mini", neuron_prompt, "o3-mini neuron"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for model, prompt, test_name in tests:
            print(f"\n{'='*60}")
            print(f"Testing: {test_name} (model: {model})")
            print(f"{'='*60}")
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100
            }
            
            # Add include_reasoning for o3 models
            if "o3" in model:
                payload["include_reasoning"] = True
                payload["max_tokens"] = 500
            
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    
                    print(f"Status: {response.status}")
                    
                    if "error" in result:
                        print(f"ERROR: {result['error']}")
                        continue
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        message = choice.get("message", {})
                        
                        content = message.get("content", "")
                        reasoning = message.get("reasoning", "")
                        refusal = message.get("refusal", "")
                        
                        print(f"Content: '{content}'")
                        if refusal:
                            print(f"REFUSAL: '{refusal}'")
                        if reasoning:
                            print(f"Reasoning length: {len(reasoning)} chars")
                            if len(reasoning) < 200:
                                print(f"Reasoning: {reasoning}")
                            else:
                                print(f"Reasoning preview: {reasoning[:200]}...")
                        else:
                            print("No reasoning field or empty")
                        
                        # Check usage
                        if "usage" in result:
                            usage = result["usage"]
                            print(f"Tokens used - prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}")
                    
            except Exception as e:
                print(f"Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_models())