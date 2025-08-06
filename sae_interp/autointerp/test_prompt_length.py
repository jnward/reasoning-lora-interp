#!/usr/bin/env python3
"""
Test to find what prompt length/complexity works with o3-mini.
"""

import asyncio
import aiohttp

async def test_prompt_lengths():
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Different prompts of increasing complexity
    prompts = [
        # Simple prompt that we know works
        ("What is a cat?", "simple"),
        
        # Add neuron context
        ("A neuron fires on the word 'cat'. What is it detecting?", "neuron_basic"),
        
        # Add example
        ("""A neuron fires on certain words.
Example: The cat sat.
Word: cat
What pattern?""", "with_example"),
        
        # Multiple examples
        ("""Examples where neuron fires:
1. The cat sat (cat: 8.5)
2. A cat jumped (cat: 9.0)
What pattern?""", "multiple_examples"),
        
        # With XML tags
        ("""<examples>
The cat sat.
cat 8.5
</examples>
What pattern?""", "with_xml"),
        
        # Shorter version of our prompt
        ("""Look at where a neuron activates:

Example 1:
The cat sat on the mat.
cat 8.5
mat 7.2

What is it looking for? Brief answer:""", "short_structured"),
        
        # Medium version
        ("""We're studying neurons. Look at where this one activates.

Examples:
The cat sat on the mat.
cat 8.5

A small cat jumped.
cat 9.1

What pattern? Answer briefly:""", "medium"),
        
        # With instructions but no long preamble
        ("""Analyze these neuron activations. Answer in few words.

<neuron_activations>
Example 1:
The cat sat on the mat looking at the bird.
cat 8.5
mat 7.2
</neuron_activations>

What is this neuron looking for?""", "instructions_short"),
    ]
    
    for prompt, name in prompts:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"Length: {len(prompt)} chars")
        print(f"{'='*60}")
        
        payload = {
            "model": "openai/o3-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
            "include_reasoning": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    
                    if content:
                        print(f"✓ SUCCESS")
                        print(f"Response: {content[:100]}...")
                        print(f"Tokens: {usage.get('prompt_tokens', 0)} prompt, {usage.get('completion_tokens', 0)} completion")
                    else:
                        print(f"✗ FAILED - Empty response")
                        print(f"Tokens: {usage.get('prompt_tokens', 0)} prompt, {usage.get('completion_tokens', 0)} completion")

if __name__ == "__main__":
    asyncio.run(test_prompt_lengths())