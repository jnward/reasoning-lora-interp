#!/usr/bin/env python3
"""
Generate multiple rollouts for AIME problems using vLLM with temperature sampling
"""

import os
import json
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import time
from datetime import datetime
import re

# Configuration
ADAPTER_PATH = "/workspace/models/ckpts_1.1/s1-lora-32B-r1-20250627_013544"  # Change this to your adapter
N_ROLLOUTS = 4  # Number of rollouts to generate
TEMPERATURE = 0.7  # Temperature for sampling
MAX_TOKENS = 32768  # Maximum tokens to generate per rollout
OUTPUT_DIR = "/workspace/reasoning_interp/cf/rollouts"  # Directory to save rollouts
MIN_INCORRECT = 1  # Minimum number of incorrect rollouts needed
MIN_CORRECT = 1  # Minimum number of correct rollouts needed
START_IDX = 13  # Dataset index to start from (0-29)

# Base model is auto-detected from adapter path
def get_base_model_from_adapter_path(adapter_path):
    """Extract base model from adapter path"""
    import re
    match = re.search(r's1-lora-(\d+B)-r\d+', adapter_path)
    if not match:
        raise ValueError(f"Could not extract model size from adapter path: {adapter_path}")
    model_size = match.group(1)
    return f"Qwen/Qwen2.5-{model_size}-Instruct"

def merge_lora_model(adapter_path, base_model_name=None):
    """Merge LoRA adapter with base model"""
    merged_path = f"{adapter_path}-merged"
    
    # Check if merged model already exists
    if os.path.exists(merged_path):
        print(f"Merged model already exists at: {merged_path}")
        return merged_path
    
    # Auto-detect base model if not provided
    if base_model_name is None:
        base_model_name = get_base_model_from_adapter_path(adapter_path)
        print(f"Auto-detected base model: {base_model_name}")
    
    print(f"Loading base model: {base_model_name}")
    # Load base model in the same precision as training
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter: {adapter_path}")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Merging adapter with base model...")
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {merged_path}")
    # Save merged model
    merged_model.save_pretrained(merged_path)
    
    # Also save tokenizer for convenience
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(merged_path)
    
    # Free memory
    del model
    del merged_model
    torch.cuda.empty_cache()
    
    print(f"Merge complete! Model saved to: {merged_path}")
    return merged_path

def load_aime_problems():
    """Load AIME24 problems from the JSON file"""
    problems_file = "/workspace/s1_peft/aime24_nofigures_problems.json"
    
    # If file doesn't exist, download it
    if not os.path.exists(problems_file):
        from datasets import load_dataset
        dataset = load_dataset("simplescaling/aime24_nofigures", split="train")
        problems = []
        for idx, item in enumerate(dataset):
            problems.append({
                "index": idx,
                "problem": item["problem"],
                "answer": item["answer"],
                "solution": item["solution"],
                "url": item["url"],
                "id": item["id"]
            })
        with open(problems_file, "w") as f:
            json.dump(problems, f, indent=2)
        print(f"Downloaded and saved AIME problems to: {problems_file}")
    
    with open(problems_file, "r") as f:
        problems = json.load(f)
    
    return problems

def check_answer_in_text(text, correct_answer):
    """Check if the correct answer appears in boxed format in the last 100 characters
    Returns: (is_correct, has_boxed_answer)
    """
    # Look at last 100 characters
    last_chars = text[-100:] if len(text) > 100 else text
    
    # Look for boxed{answer} pattern
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, last_chars)
    
    if matches:
        # Check if any boxed answer matches the correct answer
        for match in matches:
            if match == correct_answer:
                return True, True
        return False, True  # Has boxed answer but it's wrong
    
    return False, False  # No boxed answer at all (incomplete)

def generate_rollouts(llm, tokenizer, problem, n_rollouts, temperature, max_tokens):
    """Generate n rollouts for a given problem using vLLM"""
    
    # Format the problem as a chat message
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": problem["problem"]}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    # Add assistant tag for generation
    prompt += "<|im_start|>assistant\n"
    
    # Create prompts list (same prompt repeated n times)
    prompts = [prompt] * n_rollouts
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    print(f"Generating {n_rollouts} rollouts with temperature={temperature}...")
    
    # Generate rollouts
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Extract generated texts and check answers
    rollouts = []
    correct_count = 0
    incorrect_count = 0
    incomplete_count = 0
    
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        is_correct, has_boxed = check_answer_in_text(generated_text, problem["answer"])
        
        if is_correct:
            correct_count += 1
        elif has_boxed:
            incorrect_count += 1
        else:
            incomplete_count += 1
        
        rollout = {
            "rollout_id": idx,
            "problem_idx": problem["index"],
            "problem": problem["problem"],
            "answer": problem["answer"],
            "generated_text": generated_text,
            "is_correct": is_correct,
            "has_boxed_answer": has_boxed,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat()
        }
        rollouts.append(rollout)
    
    print(f"Results: {correct_count} correct, {incorrect_count} incorrect, {incomplete_count} incomplete")
    
    return rollouts, correct_count, incorrect_count, incomplete_count

def save_rollouts(rollouts, output_dir, problem_idx):
    """Save rollouts to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aime_problem_{problem_idx}_rollouts_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(rollouts, f, indent=2)
    
    print(f"Saved {len(rollouts)} rollouts to: {filepath}")
    return filepath

def main():
    print("=" * 80)
    print("AIME Rollout Generation with vLLM")
    print("=" * 80)
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"Number of rollouts per problem: {N_ROLLOUTS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Looking for problems with at least {MIN_CORRECT} correct & complete AND {MIN_INCORRECT} incorrect & complete")
    print(f"Starting from problem index: {START_IDX}")
    print("=" * 80)
    
    # Step 1: Merge LoRA model (or load if already merged)
    print("\nStep 1: Merging LoRA model...")
    merged_model_path = merge_lora_model(ADAPTER_PATH)
    
    # Step 2: Load AIME problems
    print("\nStep 2: Loading AIME problems...")
    problems = load_aime_problems()
    
    # Initialize vLLM once
    print("\nInitializing vLLM...")
    tensor_parallel_size = 1
    llm = LLM(
        model=merged_model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16"
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    
    # Step 3: Iterate through problems until we find one with enough errors
    print("\nStep 3: Iterating through problems...")
    
    for problem_idx in range(START_IDX, len(problems)):
        problem = problems[problem_idx]
        print(f"\n{'='*60}")
        print(f"Testing problem {problem_idx}:")
        print(f"Problem text: {problem['problem'][:200]}...")
        print(f"Correct answer: {problem['answer']}")
        
        # Generate rollouts
        rollouts, correct_count, incorrect_count, incomplete_count = generate_rollouts(
            llm,
            tokenizer,
            problem,
            N_ROLLOUTS,
            TEMPERATURE,
            MAX_TOKENS
        )
        
        # Save rollouts
        print("\nSaving rollouts...")
        output_file = save_rollouts(rollouts, OUTPUT_DIR, problem_idx)
        
        # Print detailed results
        print("\n" + "=" * 80)
        print("Generation Complete!")
        print("=" * 80)
        print(f"Problem index: {problem_idx}")
        print(f"Generated {len(rollouts)} rollouts")
        print(f"Correct: {correct_count}/{N_ROLLOUTS}")
        print(f"Incorrect: {incorrect_count}/{N_ROLLOUTS}")
        print(f"Incomplete: {incomplete_count}/{N_ROLLOUTS}")
        print(f"Output file: {output_file}")
        
        # Show which rollouts were correct/incorrect/incomplete
        print("\nDetailed results:")
        for i, rollout in enumerate(rollouts):
            if rollout["is_correct"]:
                status = "✓ correct"
            elif rollout["has_boxed_answer"]:
                status = "✗ incorrect"
            else:
                status = "⚠ incomplete"
            last_100 = rollout["generated_text"][-100:]
            print(f"  Rollout {i}: {status} - Last 100 chars: ...{last_100}")
        
        # Check if we have enough incorrect AND correct complete answers
        # Only complete answers (with boxed{}) count for criteria
        meets_criteria = incorrect_count >= MIN_INCORRECT and correct_count >= MIN_CORRECT
        if meets_criteria:
            print(f"\n✅ Found problem meeting criteria: {correct_count} correct & complete, {incorrect_count} incorrect & complete!")
            break
    else:
        print(f"\nNo problem found with at least {MIN_CORRECT} correct & complete AND {MIN_INCORRECT} incorrect & complete rollouts after checking all {len(problems)} problems.")

if __name__ == "__main__":
    main()