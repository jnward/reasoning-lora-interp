#!/usr/bin/env python3
"""
Debug why o3-mini fails on neuron interpretation.
"""

import asyncio
import aiohttp
import json

async def test_o3_mini_variations():
    """Test o3-mini with different prompt variations to isolate the issue."""
    
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "o3-mini Debug"
    }
    
    # Test different prompts to isolate the issue
    prompts = [
        # 1. Very simple
        ("What is a cat?", "simple_question"),
        
        # 2. With instructions
        ("Look at this word: cat. What does it mean? Answer briefly.", "with_instructions"),
        
        # 3. With XML tags
        ("<data>cat</data>\nWhat is this?", "with_xml"),
        
        # 4. With neuron language but no examples
        ("A neuron activates on the word 'cat'. What might it be looking for?", "neuron_simple"),
        
        # 5. With one example
        ("""Look at where a neuron activates:
Example: The cat sat on the mat.
Activations: cat 8.5

What is it looking for?""", "one_example"),
        
        # 6. With formatted example
        ("""<neuron_activations>
The cat sat on the mat.
cat 8.5
</neuron_activations>

What pattern?""", "formatted_example"),
        
        # 7. Our full prompt but shorter
        ("""A neuron activates on certain tokens. Look at the examples:

Example 1:
The cat sat on the mat.
cat 8.5

What is this neuron looking for?""", "short_neuron"),
        
        # 8. Full prompt without special instructions
        ("""<neuron_activations>
Example 1:
The cat sat on the mat looking at the bird.
cat 8.5
mat 7.2

Example 2:
A small cat jumped over the fence.
cat 9.1
</neuron_activations>

What is this neuron looking for?""", "full_simple"),
        
        # 9. With system message
        ("""You are analyzing neural network activations.

The cat sat on the mat.
cat 8.5

What pattern do you see?""", "with_context"),
        
        # 10. Original full prompt
        ("""We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document where the neuron activates and describe what it's looking for.

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

What is this neuron looking for? Answer in just a few words:""", "original_full"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for prompt, name in prompts:
            print(f"\n{'='*60}")
            print(f"Test: {name}")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"{'='*60}")
            
            payload = {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100,
                "include_reasoning": True
            }
            
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        print(f"HTTP Status: {response.status}")
                    
                    if "error" in result:
                        print(f"ERROR: {result['error']}")
                        continue
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        message = choice.get("message", {})
                        
                        content = message.get("content", "")
                        refusal = message.get("refusal")
                        finish_reason = choice.get("finish_reason")
                        
                        print(f"Content: '{content}'")
                        if refusal:
                            print(f"REFUSAL: '{refusal}'")
                        print(f"Finish reason: {finish_reason}")
                        
                        # Check token usage
                        usage = result.get("usage", {})
                        print(f"Tokens - prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}")
                        
                        # Success/failure
                        if content:
                            print("✓ SUCCESS - Got response")
                        else:
                            print("✗ FAILED - Empty response")
                    
            except Exception as e:
                print(f"Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_o3_mini_variations())