#!/usr/bin/env python3
"""
Extract validation samples from dataset for use with train_controlnet.py
"""
import json
import sys

def extract_validation_samples(jsonl_path, num_samples=2):
    """Extract first N samples from JSONL for validation"""
    samples = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            sample = json.loads(line.strip())
            samples.append(sample)
    
    return samples

if __name__ == "__main__":
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "/home/user/StreamDiffusion/temporalnet2_celebs_exp.jsonl"
    
    samples = extract_validation_samples(jsonl_path, num_samples=2)
    
    print("Extracted validation samples:")
    print()
    
    # Generate command-line arguments
    prompts = [s["prompt"] for s in samples]
    prev_images = [s["prev_img_path"] for s in samples]
    curr_images = [s["curr_img_path"] for s in samples]
    
    print("Add these to your training command:")
    print()
    print(f'  --validation_prompt "{prompts[0]}" "{prompts[1]}" \\')
    print(f'  --validation_prev_image "{prev_images[0]}" "{prev_images[1]}" \\')
    print(f'  --validation_curr_image "{curr_images[0]}" "{curr_images[1]}" \\')
    print(f'  --validation_steps 100 \\')
    print(f'  --num_validation_images 4')
    print()
    
    print("Full paths:")
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Prompt: {sample['prompt'][:80]}...")
        print(f"  Prev: {sample['prev_img_path']}")
        print(f"  Curr: {sample['curr_img_path']}")

