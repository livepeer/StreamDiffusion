#!/usr/bin/env python3
"""
ControlNet Video Test Demo for StreamDiffusion

This script processes a video file through ControlNet and saves the results
for testing purposes. It saves output video, performance metrics, and copies
of the config and input video to an output directory.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys
import time
import os
import shutil
import json
from collections import deque


def process_video(config_path, input_video, output_dir, engine_only=False):
    """Process video through ControlNet pipeline"""
    print(f"process_video: Loading config from {config_path}")
    
    # Import here to avoid loading at module level
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from streamdiffusion import load_config, create_wrapper_from_config
    
    # Load configuration
    config = load_config(config_path)
    
    # Get width and height from config (with defaults)
    width = config.get('width', 512)
    height = config.get('height', 512)
    
    print(f"process_video: Using dimensions: {width}x{height}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config and input video to output directory
    config_copy_path = output_dir / f"config_{Path(config_path).name}"
    shutil.copy2(config_path, config_copy_path)
    print(f"process_video: Copied config to {config_copy_path}")
    
    input_copy_path = output_dir / f"input_{Path(input_video).name}"
    shutil.copy2(input_video, input_copy_path)
    print(f"process_video: Copied input video to {input_copy_path}")
    
    # Create wrapper using the built-in function (width/height from config)
    wrapper = create_wrapper_from_config(config)
    
    if engine_only:
        print("Engine-only mode: TensorRT engines have been built (if needed). Exiting.")
        return None
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"process_video: Input video - {frame_count} frames at {fps} FPS")
    
    # Setup output video writer
    output_video_path = output_dir / "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width + width, height))
    
    # Performance tracking
    frame_times = []
    total_start_time = time.time()
    
    print("process_video: Starting video processing...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        
        # Resize frame
        frame_resized = cv2.resize(frame, (width, height))
        
        # Convert frame to PIL
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Update control image for all configured ControlNets
        if hasattr(wrapper.stream, '_controlnet_module') and wrapper.stream._controlnet_module:
            controlnet_count = len(wrapper.stream._controlnet_module.controlnets)
            print(f"process_video: Updating control image for {controlnet_count} ControlNet(s) on frame {frame_idx}")
            for i in range(controlnet_count):
                wrapper.update_control_image(i, frame_pil)
        else:
            print(f"process_video: No ControlNet module found for frame {frame_idx}")
        output_image = wrapper(frame_pil)
        
        # Convert output to display format
        output_array = np.array(output_image)
        output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        
        # Create side-by-side display
        combined = np.hstack([frame_resized, output_bgr])
        
        # Add labels
        cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Generated", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(combined)
        
        # Track performance
        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
            print(f"process_video: Processed {frame_idx}/{frame_count} frames (Avg FPS: {avg_fps:.2f})")
    
    total_time = time.time() - total_start_time
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate performance metrics
    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1.0 / avg_frame_time
        min_frame_time = min(frame_times)
        max_frame_time = max(frame_times)
        max_fps = 1.0 / min_frame_time
        min_fps = 1.0 / max_frame_time
    else:
        avg_frame_time = avg_fps = min_frame_time = max_frame_time = max_fps = min_fps = 0
    
    # Performance metrics
    metrics = {
        "input_video": str(input_video),
        "config_file": str(config_path),
        "width": width,
        "height": height,
        "total_frames": frame_idx,
        "total_time_seconds": total_time,
        "avg_fps": avg_fps,
        "min_fps": min_fps,
        "max_fps": max_fps,
        "avg_frame_time_seconds": avg_frame_time,
        "min_frame_time_seconds": min_frame_time,
        "max_frame_time_seconds": max_frame_time,
        "model_id": config['model_id'],
        "acceleration": config.get('acceleration', 'none'),
        "frame_buffer_size": config.get('frame_buffer_size', 1),
        "num_inference_steps": config.get('num_inference_steps', 50),
        "guidance_scale": config.get('guidance_scale', 1.1),
        "controlnets": [cn['model_id'] for cn in config.get('controlnets', [])]
    }
    
    # Save metrics
    metrics_path = output_dir / "performance_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"process_video: Processing completed!")
    print(f"process_video: Output video saved to: {output_video_path}")
    print(f"process_video: Performance metrics saved to: {metrics_path}")
    print(f"process_video: Average FPS: {avg_fps:.2f}")
    print(f"process_video: Total time: {total_time:.2f} seconds")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="ControlNet Video Test Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent.parent / "configs" / "controlnet_examples" / "multi_controlnet_example.yaml"
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to ControlNet configuration file")
    parser.add_argument("--input-video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: creates timestamped directory)")
    parser.add_argument("--engine-only", action="store_true", help="Only build TensorRT engines and exit (no video processing)")
    
    args = parser.parse_args()
    
    # Create default output directory if not specified
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_name = Path(args.input_video).stem
        config_name = Path(args.config).stem
        args.output_dir = f"controlnet_test_{config_name}_{input_name}_{timestamp}"
        print(f"main: No output directory specified, using: {args.output_dir}")
    
    # Validate input files
    if not Path(args.config).exists():
        print(f"main: Error - Config file not found: {args.config}")
        return 1
    
    if not Path(args.input_video).exists():
        print(f"main: Error - Input video not found: {args.input_video}")
        return 1
    
    print("ControlNet Video Test Demo")
    print(f"main: Config: {args.config}")
    print(f"main: Input video: {args.input_video}")
    print(f"main: Output directory: {args.output_dir}")
    
    try:
        metrics = process_video(args.config, args.input_video, args.output_dir, engine_only=args.engine_only)
        if args.engine_only:
            print("main: Engine-only mode completed successfully!")
            return 0
        print("main: Video processing completed successfully!")
        return 0
    except Exception as e:
        import traceback
        print(f"main: Error during processing: {e}")
        print(f"main: Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        return 1


if __name__ == "__main__":
    exit(main()) 