#!/usr/bin/env python3
"""
Utility script to prepare a temporal dataset in JSONL format from a directory of videos.

This script helps you create the JSONL dataset needed for Temporal Prior ControlNet training.

Usage:
    python prepare_temporal_dataset.py \
        --video_dir /path/to/stylized/videos \
        --output_jsonl dataset.jsonl \
        --prompt "your style description"

Input structure:
    video_dir/
        video1/
            frame_0000.jpg
            frame_0001.jpg
            frame_0002.jpg
            ...
        video2/
            frame_0000.jpg
            ...

Output: JSONL file where each line is:
{
    "video_id": "video1_0001",
    "prompt": "your style description",
    "negative_prompt": "blurry, low resolution, watermark, text",
    "prev_img_path": "/abs/path/to/video1/frame_0000.jpg",
    "curr_img_path": "/abs/path/to/video1/frame_0001.jpg"
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def find_video_directories(video_dir: str) -> List[Path]:
    """Find all subdirectories in video_dir that contain images."""
    video_dir = Path(video_dir)
    video_dirs = []
    
    for subdir in video_dir.iterdir():
        if subdir.is_dir():
            # Check if directory contains images
            image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            if len(image_files) > 1:  # Need at least 2 frames for a pair
                video_dirs.append(subdir)
    
    return sorted(video_dirs)


def get_frame_pairs(video_dir: Path) -> List[tuple]:
    """Get consecutive frame pairs from a video directory."""
    # Find all image files
    image_files = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
    
    # Create pairs of consecutive frames
    pairs = []
    for i in range(len(image_files) - 1):
        pairs.append((image_files[i], image_files[i + 1]))
    
    return pairs


def create_temporal_dataset(
    video_dir: str,
    output_jsonl: str,
    prompt: str,
    negative_prompt: str = "blurry, low resolution, watermark, text",
    max_pairs_per_video: int = None,
) -> None:
    """
    Create a temporal dataset in JSONL format.
    
    Args:
        video_dir: Directory containing video subdirectories with frame images
        output_jsonl: Output JSONL file path
        prompt: Default prompt describing the style
        negative_prompt: Default negative prompt
        max_pairs_per_video: Maximum number of pairs to extract per video (None = all)
    """
    video_dirs = find_video_directories(video_dir)
    
    if not video_dirs:
        print(f"No video directories found in {video_dir}")
        return
    
    print(f"Found {len(video_dirs)} video directories")
    
    total_pairs = 0
    records = []
    
    for video_path in video_dirs:
        video_id = video_path.name
        pairs = get_frame_pairs(video_path)
        
        if max_pairs_per_video:
            pairs = pairs[:max_pairs_per_video]
        
        print(f"  {video_id}: {len(pairs)} frame pairs")
        
        for idx, (prev_frame, curr_frame) in enumerate(pairs):
            record = {
                "video_id": f"{video_id}_{idx:04d}",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "prev_img_path": str(prev_frame.absolute()),
                "curr_img_path": str(curr_frame.absolute()),
            }
            records.append(record)
        
        total_pairs += len(pairs)
    
    # Write to JSONL
    with open(output_jsonl, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nCreated dataset with {total_pairs} temporal frame pairs")
    print(f"Saved to: {output_jsonl}")
    print(f"\nFirst record:")
    print(json.dumps(records[0], indent=2))


def create_from_csv(
    csv_path: str,
    output_jsonl: str,
    default_negative_prompt: str = "blurry, low resolution, watermark, text",
) -> None:
    """
    Create temporal dataset from a CSV file with custom prompts.
    
    CSV format:
        video_id,prompt,prev_img_path,curr_img_path
        video1_0001,description of style,/path/to/frame0.jpg,/path/to/frame1.jpg
        ...
    
    Optional column: negative_prompt
    """
    import csv
    
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                "video_id": row["video_id"],
                "prompt": row["prompt"],
                "negative_prompt": row.get("negative_prompt", default_negative_prompt),
                "prev_img_path": os.path.abspath(row["prev_img_path"]),
                "curr_img_path": os.path.abspath(row["curr_img_path"]),
            }
            records.append(record)
    
    # Write to JSONL
    with open(output_jsonl, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Created dataset with {len(records)} temporal frame pairs from CSV")
    print(f"Saved to: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare temporal dataset in JSONL format for ControlNet training"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        help="Directory containing video subdirectories with frame images",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Alternative: CSV file with custom prompts per frame pair",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Default prompt describing the style (required if using --video_dir)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low resolution, watermark, text",
        help="Default negative prompt",
    )
    parser.add_argument(
        "--max_pairs_per_video",
        type=int,
        default=None,
        help="Maximum number of pairs to extract per video",
    )
    
    args = parser.parse_args()
    
    if args.csv_path:
        print("Creating dataset from CSV file...")
        create_from_csv(args.csv_path, args.output_jsonl, args.negative_prompt)
    elif args.video_dir:
        if not args.prompt:
            print("Error: --prompt is required when using --video_dir")
            return
        print("Creating dataset from video directory...")
        create_temporal_dataset(
            args.video_dir,
            args.output_jsonl,
            args.prompt,
            args.negative_prompt,
            args.max_pairs_per_video,
        )
    else:
        print("Error: Specify either --video_dir or --csv_path")
        parser.print_help()


if __name__ == "__main__":
    main()


