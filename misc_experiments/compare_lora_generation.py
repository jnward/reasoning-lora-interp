# %%
import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import textwrap

# %%
# Configuration
base_model_id = "Qwen/Qwen2.5-32B-Instruct"
lora_path = "/root/s1_peft/ckpts_1.1"
rank = 1

# Find the rank-1 LoRA checkpoint
lora_dirs = glob.glob(f"{lora_path}/s1-lora-32B-r{rank}-*")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# %%
# Load MATH500 dataset from Hugging Face
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
problem = dataset[0]['problem']
print(f"Problem: {problem}")

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# %%
# Load base model and add LoRA adapters
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model.load_adapter(lora_dir)

# %%
# Format prompt
messages = [
    {"role": "system", "content": "You are a helpful mathematics assistant."},
    {"role": "user", "content": problem}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# %%
# Generate with base model (LoRA disabled)
model.disable_adapters()
with torch.no_grad():
    base_outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

base_text = tokenizer.decode(base_outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print("BASE MODEL OUTPUT (LoRA disabled):")
for line in base_text.split('\n'):
    print('\n'.join(textwrap.wrap(line, width=80)) if line else '')

# %%
# Generate with LoRA model (LoRA enabled)
model.enable_adapters()
with torch.no_grad():
    lora_outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

lora_text = tokenizer.decode(lora_outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print("\nLoRA MODEL OUTPUT (LoRA enabled):")
for line in lora_text.split('\n'):
    print('\n'.join(textwrap.wrap(line, width=80)) if line else '')

# %%