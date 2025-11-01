#!/usr/bin/env python3
"""
Extract Validation Samples for TemporalNet2 Training

This script helps you select random or specific samples from your dataset
to use as validation samples during training.

Usage:
    python extract_validation_samples.py /path/to/dataset.jsonl --num-samples 3
"""

import argparse
import json
import random
import os


def extract_validation_samples(jsonl_path, num_samples=3, random_seed=42, output_file=None):
    """Extract validation samples from JSONL dataset."""
    
    print(f"Extracting {num_samples} validation samples from: {jsonl_path}")
    print("=" * 80)
    
    # Load all samples
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"Total samples in dataset: {len(samples)}")
    
    # Randomly select samples
    random.seed(random_seed)
    selected = random.sample(samples, min(num_samples, len(samples)))
    
    print(f"Selected {len(selected)} samples for validation")
    print("\n" + "=" * 80)
    print("VALIDATION SAMPLES")
    print("=" * 80)
    
    # Display selected samples
    for i, sample in enumerate(selected, 1):
        print(f"\nSample {i}:")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  Prompt: {sample['prompt'][:80]}...")
        print(f"  Prev Image: {sample['prev_img_path']}")
        print(f"  Optical Flow: {sample['optical_flow_path']}")
        
        # Verify files exist
        if not os.path.exists(sample['prev_img_path']):
            print(f"  ⚠️  WARNING: prev_img_path does not exist!")
        if not os.path.exists(sample['optical_flow_path']):
            print(f"  ⚠️  WARNING: optical_flow_path does not exist!")
    
    # Generate bash script snippet
    print("\n" + "=" * 80)
    print("BASH SCRIPT SNIPPET")
    print("=" * 80)
    print("\n# Add these to your train_temporalnet2.sh:\n")
    
    for i, sample in enumerate(selected, 1):
        print(f'export VALIDATION_PREV_IMG_{i}="{sample["prev_img_path"]}"')
        print(f'export VALIDATION_FLOW_{i}="{sample["optical_flow_path"]}"')
        print(f'export VALIDATION_PROMPT_{i}="{sample["prompt"]}"')
        if i < len(selected):
            print()
    
    print("\n# In your accelerate launch command, add:")
    
    prev_imgs = " ".join([f'"$VALIDATION_PREV_IMG_{i+1}"' for i in range(len(selected))])
    flows = " ".join([f'"$VALIDATION_FLOW_{i+1}"' for i in range(len(selected))])
    prompts = " ".join([f'"$VALIDATION_PROMPT_{i+1}"' for i in range(len(selected))])
    
    print(f'    --validation_prev_image {prev_imgs} \\')
    print(f'    --validation_optical_flow {flows} \\')
    print(f'    --validation_prompt {prompts} \\')
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("# Validation Samples\n\n")
            for i, sample in enumerate(selected, 1):
                f.write(f'export VALIDATION_PREV_IMG_{i}="{sample["prev_img_path"]}"\n')
                f.write(f'export VALIDATION_FLOW_{i}="{sample["optical_flow_path"]}"\n')
                f.write(f'export VALIDATION_PROMPT_{i}="{sample["prompt"]}"\n')
                f.write('\n')
            
            f.write("# Command line arguments:\n")
            f.write(f'# --validation_prev_image {prev_imgs}\n')
            f.write(f'# --validation_optical_flow {flows}\n')
            f.write(f'# --validation_prompt {prompts}\n')
        
        print(f"\n✓ Saved to: {output_file}")
    
    print("\n" + "=" * 80)
    
    return selected


def extract_specific_samples(jsonl_path, video_ids):
    """Extract specific samples by video ID."""
    
    print(f"Extracting specific samples from: {jsonl_path}")
    print(f"Looking for video IDs: {', '.join(video_ids)}")
    print("=" * 80)
    
    samples = []
    found_ids = set()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            if sample['video_id'] in video_ids:
                samples.append(sample)
                found_ids.add(sample['video_id'])
    
    print(f"Found {len(samples)} samples")
    
    missing_ids = set(video_ids) - found_ids
    if missing_ids:
        print(f"⚠️  WARNING: Could not find video IDs: {', '.join(missing_ids)}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract validation samples from JSONL dataset")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL dataset file")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of random samples to extract (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save validation config to this file")
    parser.add_argument("--video-ids", type=str, nargs="+",
                        help="Extract specific video IDs instead of random samples")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        print(f"Error: Dataset file not found: {args.jsonl_path}")
        return 1
    
    if args.video_ids:
        samples = extract_specific_samples(args.jsonl_path, args.video_ids)
        if samples:
            extract_validation_samples(args.jsonl_path, num_samples=len(samples), 
                                      random_seed=args.seed, output_file=args.output)
    else:
        extract_validation_samples(args.jsonl_path, num_samples=args.num_samples,
                                  random_seed=args.seed, output_file=args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())

