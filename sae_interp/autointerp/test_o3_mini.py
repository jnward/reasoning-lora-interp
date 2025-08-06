#!/usr/bin/env python3
"""
Test script to debug o3-mini issues with explanations.
"""

import asyncio
import aiohttp
import os
import json

async def test_o3_mini():
    """Test o3-mini directly with a simple prompt."""
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY")
        return
    
    # Simple test prompt
    test_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document where the neuron activates and describe what it's looking for.

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

Explanation (just a few words):"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "SAE Feature Interpretation Test"
    }
    
    # Test different configurations
    configs = [
        {
            "name": "Basic o3-mini",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.0,
                "max_tokens": 60
            }
        },
        {
            "name": "o3-mini with more tokens",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.0,
                "max_tokens": 500
            }
        },
        {
            "name": "o3-mini with include_reasoning",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
                "include_reasoning": True
            }
        },
        {
            "name": "o3-mini with temperature",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Testing: {config['name']}")
            print(f"{'='*60}")
            
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=config['payload']
                ) as response:
                    result = await response.json()
                    
                    print(f"Status: {response.status}")
                    print(f"Response keys: {result.keys()}")
                    
                    if "error" in result:
                        print(f"ERROR: {result['error']}")
                        continue
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        print(f"Choice keys: {choice.keys()}")
                        
                        if "message" in choice:
                            message = choice["message"]
                            print(f"Message keys: {message.keys()}")
                            
                            content = message.get("content", "")
                            print(f"\nContent length: {len(content)}")
                            print(f"Content: {content}")
                            
                            if "reasoning" in message:
                                reasoning = message.get('reasoning')
                                if reasoning:
                                    print(f"\nReasoning present: {len(reasoning)} chars")
                                    print(f"Reasoning preview: {reasoning[:500]}...")
                                    # Check if the answer is in reasoning
                                    if "Answer in just a few words:" in reasoning:
                                        parts = reasoning.split("Answer in just a few words:")
                                        if len(parts) > 1:
                                            print(f"Found answer in reasoning: {parts[-1].strip()}")
                                else:
                                    print("\nReasoning field exists but is None/empty")
                    
            except Exception as e:
                print(f"Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_o3_mini())