#!/usr/bin/env python3
"""
Test o3-mini with very high token limits.
"""

import asyncio
import aiohttp

async def test_token_limits():
    api_key = "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"
    
    # Full prompt
    prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document where the neuron activates and describe what it's looking for.

If a feature always activates for the same token, you should note this in your explanation. You may need to look at words surrounding activating tokens in order to understand why a feature is firing.
If there isn't a clear explanation, note this.

Your explanation should not exceed ten words. Don't write complete sentences. The neuron might be responding to:
- Individual tokens or specific words
- Common phrases or expressions  
- Abstract concepts or behaviors
- Broader context or topics

The activation format shows the full text first, then lists tokens where the neuron fired along with their activation strengths (0-10 scale). Higher values mean stronger activation.

For example:
cat and mouse ran around the tree. They quickly
tree 7.81
ran 2.30
around 1.01

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
        "Content-Type": "application/json"
    }
    
    # Test different token limits
    token_limits = [500, 1000, 2000, 4000, 8000, 16000]
    
    for limit in token_limits:
        print(f"\n{'='*60}")
        print(f"Testing with max_tokens = {limit}")
        print(f"{'='*60}")
        
        payload = {
            "model": "openai/o3-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": limit,
            "include_reasoning": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                
                if "error" in result:
                    print(f"ERROR: {result['error']}")
                    continue
                
                if "choices" in result and result["choices"]:
                    message = result["choices"][0]["message"]
                    content = message.get("content", "")
                    
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    
                    if content:
                        print(f"✓ SUCCESS!")
                        print(f"Response: {content}")
                        print(f"Tokens used: {prompt_tokens} prompt, {completion_tokens} completion")
                    else:
                        print(f"✗ FAILED - Empty response")
                        print(f"Tokens: {prompt_tokens} prompt, {completion_tokens} completion")

if __name__ == "__main__":
    asyncio.run(test_token_limits())