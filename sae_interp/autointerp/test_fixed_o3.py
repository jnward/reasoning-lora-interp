#!/usr/bin/env python3
"""
Test o3-mini with our fixed implementation.
"""

import asyncio
import aiohttp

async def test_fixed():
    api_key = "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"
    
    # Our full neuron prompt
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
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "SAE Feature Interpretation"
    }
    
    # Test with our fixed parameters
    payload = {
        "model": "openai/o3-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,  # Higher limit as per our fix
        "include_reasoning": True  # Required for o3-mini on OpenRouter
    }
    
    print("Testing o3-mini with fixed parameters...")
    print(f"Prompt length: {len(prompt)} chars")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            result = await response.json()
            
            print(f"\nStatus: {response.status}")
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
                return
            
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                message = choice.get("message", {})
                
                content = message.get("content", "")
                reasoning = message.get("reasoning", "")
                
                print(f"\nContent: '{content}'")
                print(f"Success: {'YES' if content else 'NO'}")
                
                if reasoning:
                    print(f"\nReasoning present: {len(reasoning)} chars")
                
                usage = result.get("usage", {})
                print(f"\nTokens - prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}")

if __name__ == "__main__":
    asyncio.run(test_fixed())