#!/usr/bin/env python3
"""Generate a longer reasoning trace (1024 tokens) for attention analysis."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
import glob

# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/workspace/models/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*544")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA model
print("Loading LoRA model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)

# Load dataset and pick a different problem
print("Loading dataset...")
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# Let's use problem 25 for variety
problem_idx = 25
problem = dataset[problem_idx]['problem']
print(f"\nProblem {problem_idx}: {problem[:200]}...")

# Create prompt
system_prompt = "You are a helpful mathematics assistant. Please think step by step to solve the problem."
prompt = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{problem}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_length = inputs.input_ids.shape[1]
print(f"Prompt length: {prompt_length} tokens")

# Generate with more tokens
print("\nGenerating 1024-token response...")
with torch.no_grad():
    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=1024 - prompt_length,  # Total 1024 including prompt
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.0
    )

# Decode
full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
total_tokens = generated_ids.shape[1]
print(f"Generated total of {total_tokens} tokens")

# Save the generation
output_file = f"math500_generation_example_{problem_idx}_1024.json"
with open(output_file, 'w') as f:
    json.dump({
        'problem_idx': problem_idx,
        'problem': problem,
        'full_text': full_text,
        'total_tokens': total_tokens,
        'prompt_tokens': prompt_length,
        'generated_tokens': total_tokens - prompt_length
    }, f, indent=2)

print(f"\nSaved generation to {output_file}")
print(f"\nFirst 500 chars of response:")
print(full_text[len(prompt):len(prompt)+500])