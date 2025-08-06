#!/usr/bin/env python3
"""
Test o3-mini with exact same prompt that worked before.
"""

import asyncio
import aiohttp
import json

async def test_exact():
    """Test with exact prompts that worked/failed before."""
    
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Model Test"
    }
    
    # Test with different payloads
    tests = [
        # Exact payload that worked
        {
            "name": "2+2 (worked before)",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                "temperature": 0.0,
                "max_tokens": 100
            }
        },
        # Same but with include_reasoning
        {
            "name": "2+2 with include_reasoning",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                "temperature": 0.0,
                "max_tokens": 100,
                "include_reasoning": True
            }
        },
        # Without include_reasoning for cat
        {
            "name": "cat without include_reasoning",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": "What is a cat?"}],
                "temperature": 0.0,
                "max_tokens": 100
            }
        },
        # Try with higher max_tokens
        {
            "name": "cat with 500 tokens",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": "What is a cat?"}],
                "temperature": 0.0,
                "max_tokens": 500
            }
        },
        # Try the exact payload from test_models.py that worked
        {
            "name": "Exact working payload",
            "payload": {
                "model": "openai/o3-mini",
                "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                "temperature": 0.0,
                "max_tokens": 500,
                "include_reasoning": True
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for test in tests:
            print(f"\n{'='*60}")
            print(f"Test: {test['name']}")
            print(f"Payload: {json.dumps(test['payload'], indent=2)}")
            print(f"{'='*60}")
            
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=test['payload']
                ) as response:
                    result = await response.json()
                    
                    print(f"Status: {response.status}")
                    
                    if "error" in result:
                        print(f"ERROR: {result['error']}")
                        if "error" in result and isinstance(result["error"], dict):
                            print(f"Error details: {json.dumps(result['error'], indent=2)}")
                        continue
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        message = choice.get("message", {})
                        
                        content = message.get("content", "")
                        print(f"Content: '{content}'")
                        print(f"Finish reason: {choice.get('finish_reason')}")
                        
                        usage = result.get("usage", {})
                        print(f"Tokens - prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}")
                        
                        # Print full response for debugging
                        print(f"\nFull message object: {json.dumps(message, indent=2)}")
                    
            except Exception as e:
                print(f"Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_exact())