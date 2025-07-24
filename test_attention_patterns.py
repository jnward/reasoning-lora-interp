import json

# Test a few files to see the attention pattern structure
test_files = [
    'attention_kl_data/heads/0_0.json',
    'attention_kl_data/layer_avg/0.json',
    'attention_kl_data/overall.json'
]

for file_path in test_files:
    print(f"\n=== {file_path} ===")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'attention_patterns' in data:
        print("Has attention patterns: Yes")
        
        # Check base patterns
        if 'base' in data['attention_patterns']:
            base_keys = list(data['attention_patterns']['base'].keys())
            print(f"Base pattern positions: {base_keys[:5]}... (total: {len(base_keys)})")
            
            # Check a specific position
            if '10' in data['attention_patterns']['base']:
                pos10_data = data['attention_patterns']['base']['10']
                print(f"Position 10 base data: {pos10_data[:3]}...")
                
                # Check if we can find attention to position 0
                attn_to_0 = None
                for pair in pos10_data:
                    if pair[0] == 0:
                        attn_to_0 = pair[1]
                        break
                print(f"Position 10 attention to position 0: {attn_to_0}")
        
        # Check lora patterns
        if 'lora' in data['attention_patterns']:
            lora_keys = list(data['attention_patterns']['lora'].keys())
            print(f"LoRA pattern positions: {lora_keys[:5]}... (total: {len(lora_keys)})")
    else:
        print("Has attention patterns: No")