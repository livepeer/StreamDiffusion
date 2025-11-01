#!/usr/bin/env python3
"""
Dataset Verification Script for TemporalNet2 Training

This script verifies that your JSONL dataset is in the correct format
and that all referenced image files exist.

Usage:
    python verify_dataset.py /path/to/dataset.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def verify_dataset(jsonl_path, max_samples=None):
    """Verify JSONL dataset format and file existence."""
    
    print(f"Verifying dataset: {jsonl_path}")
    print("=" * 80)
    
    required_fields = ["video_id", "prompt", "prev_img_path", "curr_img_path", "optical_flow_path"]
    optional_fields = ["negative_prompt"]
    
    total_samples = 0
    valid_samples = 0
    errors = []
    
    # Count total lines first
    with open(jsonl_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total samples in dataset: {total_lines}")
    
    if max_samples:
        total_lines = min(total_lines, max_samples)
        print(f"Verifying first {total_lines} samples...")
    
    print("\nStarting verification...\n")
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Verifying"), 1):
            if max_samples and line_num > max_samples:
                break
            
            total_samples += 1
            sample_errors = []
            
            try:
                # Parse JSON
                data = json.loads(line.strip())
                
                # Check required fields
                for field in required_fields:
                    if field not in data:
                        sample_errors.append(f"Missing required field: {field}")
                
                # Check image paths exist
                if "prev_img_path" in data:
                    if not os.path.exists(data["prev_img_path"]):
                        sample_errors.append(f"prev_img_path does not exist: {data['prev_img_path']}")
                    else:
                        # Try to open the image
                        try:
                            img = Image.open(data["prev_img_path"])
                            img.verify()
                        except Exception as e:
                            sample_errors.append(f"Cannot open prev_img_path: {e}")
                
                if "curr_img_path" in data:
                    if not os.path.exists(data["curr_img_path"]):
                        sample_errors.append(f"curr_img_path does not exist: {data['curr_img_path']}")
                    else:
                        # Try to open the image
                        try:
                            img = Image.open(data["curr_img_path"])
                            img.verify()
                        except Exception as e:
                            sample_errors.append(f"Cannot open curr_img_path: {e}")
                
                if "optical_flow_path" in data:
                    if not os.path.exists(data["optical_flow_path"]):
                        sample_errors.append(f"optical_flow_path does not exist: {data['optical_flow_path']}")
                    else:
                        # Try to open the image
                        try:
                            img = Image.open(data["optical_flow_path"])
                            img.verify()
                        except Exception as e:
                            sample_errors.append(f"Cannot open optical_flow_path: {e}")
                
                # Check prompt is not empty
                if "prompt" in data and not data["prompt"].strip():
                    sample_errors.append("prompt field is empty")
                
                if not sample_errors:
                    valid_samples += 1
                else:
                    errors.append({
                        "line": line_num,
                        "video_id": data.get("video_id", "unknown"),
                        "errors": sample_errors
                    })
            
            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_num,
                    "video_id": "unknown",
                    "errors": [f"JSON decode error: {e}"]
                })
            except Exception as e:
                errors.append({
                    "line": line_num,
                    "video_id": "unknown",
                    "errors": [f"Unexpected error: {e}"]
                })
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Total samples checked: {total_samples}")
    print(f"Valid samples: {valid_samples}")
    print(f"Invalid samples: {len(errors)}")
    print(f"Success rate: {valid_samples/total_samples*100:.2f}%")
    
    if errors:
        print("\n" + "=" * 80)
        print("ERRORS FOUND")
        print("=" * 80)
        
        # Show first 10 errors
        max_errors_to_show = 10
        for i, error in enumerate(errors[:max_errors_to_show]):
            print(f"\nError {i+1} (Line {error['line']}, Video ID: {error['video_id']}):")
            for err in error["errors"]:
                print(f"  - {err}")
        
        if len(errors) > max_errors_to_show:
            print(f"\n... and {len(errors) - max_errors_to_show} more errors")
        
        print("\n" + "=" * 80)
        return False
    else:
        print("\n✓ All samples are valid!")
        print("=" * 80)
        return True


def show_sample(jsonl_path, sample_index=0):
    """Display a sample from the dataset."""
    print(f"\nShowing sample {sample_index} from dataset:")
    print("=" * 80)
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == sample_index:
                data = json.loads(line.strip())
                print(json.dumps(data, indent=2))
                
                # Show image info if they exist
                if os.path.exists(data.get("prev_img_path", "")):
                    img = Image.open(data["prev_img_path"])
                    print(f"\nPrevious frame: {img.size} {img.mode}")
                
                if os.path.exists(data.get("curr_img_path", "")):
                    img = Image.open(data["curr_img_path"])
                    print(f"Current frame: {img.size} {img.mode}")
                
                if os.path.exists(data.get("optical_flow_path", "")):
                    img = Image.open(data["optical_flow_path"])
                    print(f"Optical flow: {img.size} {img.mode}")
                
                break
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Verify TemporalNet2 JSONL dataset")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL dataset file")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum number of samples to verify (default: all)")
    parser.add_argument("--show-sample", type=int, default=None,
                        help="Show a specific sample by index (0-based)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        print(f"Error: Dataset file not found: {args.jsonl_path}")
        return 1
    
    if args.show_sample is not None:
        show_sample(args.jsonl_path, args.show_sample)
    else:
        success = verify_dataset(args.jsonl_path, args.max_samples)
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())

