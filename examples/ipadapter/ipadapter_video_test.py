#!/usr/bin/env python3
"""
IPAdapter Video Test Demo for StreamDiffusion

This script processes a video file through IPAdapter and saves the results
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


def process_video(config_path, input_video, output_dir, resolution=None, engine_only=False):
    """Process video through IPAdapter pipeline"""
    print(f"process_video: Loading config from {config_path}")
    
    # Import here to avoid loading at module level
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from streamdiffusion.config import load_config, create_wrapper_from_config
    
    # Load configuration
    config = load_config(config_path)
    
    # Set default resolution based on pipeline type if not specified
    if resolution is None:
        pipeline_type = config.get('pipeline_type', 'sd1.5')
        if pipeline_type == 'sdxlturbo':
            resolution = 1024
        else:
            resolution = 512
    
    print(f"process_video: Using resolution: {resolution}x{resolution}")
    
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
    
    # Create wrapper using the built-in function
    wrapper = create_wrapper_from_config(config, width=resolution, height=resolution)
    
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
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (resolution * 2, resolution))
    
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
        frame_resized = cv2.resize(frame, (resolution, resolution))
        
        # Convert frame to PIL
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Generate with IPAdapter style conditioning
        # IPAdapter uses the style image from config for conditioning
        # and processes the input frame as the base image
        if hasattr(wrapper, 'preprocess_image'):
            # For img2img mode, preprocess the input frame
            preprocessed_frame = wrapper.preprocess_image(frame_pil)
            output_image = wrapper(image=preprocessed_frame)
        else:
            # For txt2img mode or direct processing
            output_image = wrapper(frame_pil)
        
        # Convert output to display format
        output_array = np.array(output_image)
        output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        
        # Create side-by-side display
        combined = np.hstack([frame_resized, output_bgr])
        
        # Add labels
        cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "IPAdapter Output", (resolution + 10, 30), 
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
    
    # Get IPAdapter information from config
    ipadapters_info = []
    if 'ipadapters' in config:
        for ip_config in config['ipadapters']:
            ipadapters_info.append({
                'model_path': ip_config.get('ipadapter_model_path', ''),
                'image_encoder_path': ip_config.get('image_encoder_path', ''),
                'style_image': ip_config.get('style_image', ''),
                'scale': ip_config.get('scale', 1.0),
                'enabled': ip_config.get('enabled', True)
            })
    
    # Performance metrics
    metrics = {
        "input_video": str(input_video),
        "config_file": str(config_path),
        "resolution": f"{resolution}x{resolution}",
        "total_frames": frame_idx,
        "total_time_seconds": total_time,
        "avg_fps": avg_fps,
        "min_fps": min_fps,
        "max_fps": max_fps,
        "avg_frame_time_seconds": avg_frame_time,
        "min_frame_time_seconds": min_frame_time,
        "max_frame_time_seconds": max_frame_time,
        "pipeline_type": config.get('pipeline_type', 'sd1.5'),
        "model_id": config['model_id'],
        "acceleration": config.get('acceleration', 'none'),
        "frame_buffer_size": config.get('frame_buffer_size', 1),
        "num_inference_steps": config.get('num_inference_steps', 50),
        "guidance_scale": config.get('guidance_scale', 1.1),
        "mode": config.get('mode', 'img2img'),
        "ipadapters": ipadapters_info
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
    print(f"process_video: IPAdapters used: {len(ipadapters_info)}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="IPAdapter Video Test Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir / "ipadapter_img2img_config_example.yaml"
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to IPAdapter configuration file")
    parser.add_argument("--input-video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: creates timestamped directory)")
    parser.add_argument("--resolution", type=int, default=None,
                       help="Video resolution (auto-detects from pipeline type if not specified)")
    parser.add_argument("--engine-only", action="store_true", 
                       help="Only build TensorRT engines and exit (no video processing)")
    
    args = parser.parse_args()
    
    # Create default output directory if not specified
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_name = Path(args.input_video).stem
        config_name = Path(args.config).stem
        args.output_dir = f"ipadapter_test_{config_name}_{input_name}_{timestamp}"
        print(f"main: No output directory specified, using: {args.output_dir}")
    
    # Validate input files
    if not Path(args.config).exists():
        print(f"main: Error - Config file not found: {args.config}")
        return 1
    
    if not Path(args.input_video).exists():
        print(f"main: Error - Input video not found: {args.input_video}")
        return 1
    
    print("IPAdapter Video Test Demo")
    print(f"main: Config: {args.config}")
    print(f"main: Input video: {args.input_video}")
    print(f"main: Output directory: {args.output_dir}")
    
    try:
        metrics = process_video(args.config, args.input_video, args.output_dir, args.resolution, engine_only=args.engine_only)
        if args.engine_only:
            print("main: Engine-only mode completed successfully!")
            return 0
        print("main: Video processing completed successfully!")
        return 0
    except Exception as e:
        print(f"main: Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 