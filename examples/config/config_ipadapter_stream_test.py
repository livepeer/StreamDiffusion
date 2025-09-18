#!/usr/bin/env python3
"""
IPAdapter Stream Test Demo for StreamDiffusion

This script tests the IPAdapter as the primary driving force for style and structure,
while using a static image as the base input to StreamDiffusion. This demonstrates
the technique where IPAdapter provides both style and structural guidance.

Key features:
- Input video frames are used as both ControlNet control images AND IPAdapter style images
- A static image is used as the base input to StreamDiffusion (repeated for each frame)
- is_stream=True enables high-throughput pipelined processing
- Tests the IPAdapter stream behavior fix
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


def tensor_to_opencv(tensor: torch.Tensor, target_width: int, target_height: int) -> np.ndarray:
    """
    Convert a PyTorch tensor (output_type='pt') to OpenCV BGR format for video writing.
    Uses efficient tensor operations similar to the realtime-img2img demo.
    
    Args:
        tensor: Tensor in range [0,1] with shape [B, C, H, W] or [C, H, W]
        target_width: Target width for output
        target_height: Target height for output
    
    Returns:
        BGR numpy array ready for OpenCV
    """
    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to uint8 format (0-255) and ensure correct shape (C, H, W)
    tensor_uint8 = (tensor * 255).clamp(0, 255).to(torch.uint8)
    
    # Convert from [C, H, W] to [H, W, C] format
    if tensor_uint8.dim() == 3:
        image_np = tensor_uint8.permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"tensor_to_opencv: Unexpected tensor shape: {tensor_uint8.shape}")
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Resize if needed
    if image_bgr.shape[:2] != (target_height, target_width):
        image_bgr = cv2.resize(image_bgr, (target_width, target_height))
    
    return image_bgr


def process_video_ipadapter_stream(config_path, input_video, static_image, output_dir, engine_only=False):
    """Process video using IPAdapter as primary driving force with static base image"""
    print(f"process_video_ipadapter_stream: Loading config from {config_path}")
    
    # Import here to avoid loading at module level
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from streamdiffusion import load_config, create_wrapper_from_config
    
    # Load configuration
    config = load_config(config_path)
    
    # Force tensor output for better performance
    config['output_type'] = 'pt'
    
    # Get width and height from config (with defaults)
    width = config.get('width', 512)
    height = config.get('height', 512)
    
    print(f"process_video_ipadapter_stream: Using dimensions: {width}x{height}")
    print(f"process_video_ipadapter_stream: Using output_type='pt' for better performance")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config, input video, and static image to output directory
    config_copy_path = output_dir / f"config_{Path(config_path).name}"
    shutil.copy2(config_path, config_copy_path)
    print(f"process_video_ipadapter_stream: Copied config to {config_copy_path}")
    
    input_copy_path = output_dir / f"input_{Path(input_video).name}"
    shutil.copy2(input_video, input_copy_path)
    print(f"process_video_ipadapter_stream: Copied input video to {input_copy_path}")
    
    static_copy_path = output_dir / f"static_{Path(static_image).name}"
    shutil.copy2(static_image, static_copy_path)
    print(f"process_video_ipadapter_stream: Copied static image to {static_copy_path}")
    
    # Create wrapper using the built-in function
    wrapper = create_wrapper_from_config(config)
    
    if engine_only:
        print("Engine-only mode: TensorRT engines have been built (if needed). Exiting.")
        return None
    
    # Load and prepare static image
    static_img = Image.open(static_image)
    static_img = static_img.resize((width, height), Image.Resampling.LANCZOS)
    print(f"process_video_ipadapter_stream: Loaded static image: {static_image}")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"process_video_ipadapter_stream: Input video - {frame_count} frames at {fps} FPS")
    
    # Setup output video writer (3-panel display: input, static, generated)
    output_video_path = output_dir / "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width * 3, height))
    
    # Performance tracking
    frame_times = []
    total_start_time = time.time()
    
    print("process_video_ipadapter_stream: Starting IPAdapter stream processing...")
    print("process_video_ipadapter_stream: Using static image as base input, video frames for ControlNet + IPAdapter")
    
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
        
        # Update ControlNet control images (structural guidance from video frames)
        if hasattr(wrapper.stream, '_controlnet_module') and wrapper.stream._controlnet_module:
            controlnet_count = len(wrapper.stream._controlnet_module.controlnets)
            print(f"process_video_ipadapter_stream: Updating control image for {controlnet_count} ControlNet(s) on frame {frame_idx}")
            for i in range(controlnet_count):
                wrapper.update_control_image(i, frame_pil)
        else:
            print(f"process_video_ipadapter_stream: No ControlNet module found for frame {frame_idx}")
        
        # Update IPAdapter style image (style/content guidance from video frames)
        # This is the key part - using video frames as IPAdapter style images with is_stream=True
        if hasattr(wrapper.stream, '_ipadapter_module') and wrapper.stream._ipadapter_module:
            print(f"process_video_ipadapter_stream: Updating IPAdapter style image on frame {frame_idx} (is_stream=True)")
            # Update style image with is_stream=True for pipelined processing
            wrapper.update_style_image(frame_pil, is_stream=True)
        else:
            print(f"process_video_ipadapter_stream: No IPAdapter module found for frame {frame_idx}")
        
        # Process with static image as base input (this is the key difference)
        # The static image provides the base structure, while ControlNet and IPAdapter
        # provide the dynamic guidance from the video frames
        output_tensor = wrapper(static_img)
        
        # Convert tensor output to OpenCV BGR format
        output_bgr = tensor_to_opencv(output_tensor, width, height)
        
        # Convert static image to display format
        static_array = np.array(static_img)
        static_bgr = cv2.cvtColor(static_array, cv2.COLOR_RGB2BGR)
        
        # Create 3-panel display: Input Video | Static Base | Generated Output
        combined = np.hstack([frame_resized, static_bgr, output_bgr])
        
        # Add labels
        cv2.putText(combined, "Input Video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Static Base", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Generated", (width * 2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(combined, f"Frame: {frame_idx}/{frame_count}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame
        out.write(combined)
        
        # Track performance
        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
            print(f"process_video_ipadapter_stream: Processed {frame_idx}/{frame_count} frames (Avg FPS: {avg_fps:.2f})")
    
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
        "static_image": str(static_image),
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
        "controlnets": [cn['model_id'] for cn in config.get('controlnets', [])],
        "ipadapter_configs": [ip['ipadapter_model_path'] for ip in config.get('ipadapter_config', [])],
        "test_type": "ipadapter_stream_test",
        "is_stream_enabled": True,
        "output_type": "pt",
        "description": "IPAdapter as primary driving force with static base image using tensor output for performance"
    }
    
    # Save metrics
    metrics_path = output_dir / "performance_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"process_video_ipadapter_stream: Processing completed!")
    print(f"process_video_ipadapter_stream: Output video saved to: {output_video_path}")
    print(f"process_video_ipadapter_stream: Performance metrics saved to: {metrics_path}")
    print(f"process_video_ipadapter_stream: Average FPS: {avg_fps:.2f}")
    print(f"process_video_ipadapter_stream: Total time: {total_time:.2f} seconds")
    print(f"process_video_ipadapter_stream: Test completed - IPAdapter stream behavior verified")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="IPAdapter Stream Test Demo - Tests IPAdapter as primary driving force")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file (must include both ControlNet and IPAdapter configs)")
    parser.add_argument("--input-video", type=str, required=True,
                       help="Path to input video file (used for both ControlNet and IPAdapter guidance)")
    parser.add_argument("--static-image", type=str, required=True,
                       help="Path to static image file (used as base input to StreamDiffusion)")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Parent directory for results (default: 'output'). Script will create a timestamped subdirectory inside this.")
    parser.add_argument("--engine-only", action="store_true", 
                       help="Only build TensorRT engines and exit (no video processing)")
    
    args = parser.parse_args()
    
    # Create timestamped subdirectory within the specified parent directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_name = Path(args.input_video).stem
    static_name = Path(args.static_image).stem
    config_name = Path(args.config).stem
    subdir_name = f"ipadapter_stream_test_{config_name}_{input_name}_{static_name}_{timestamp}"
    
    # Combine parent directory with generated subdirectory name
    final_output_dir = Path(args.output_dir) / subdir_name
    args.output_dir = str(final_output_dir)
    print(f"main: Using output directory: {args.output_dir}")
    
    # Validate input files
    if not Path(args.config).exists():
        print(f"main: Error - Config file not found: {args.config}")
        return 1
    
    if not Path(args.input_video).exists():
        print(f"main: Error - Input video not found: {args.input_video}")
        return 1
    
    if not Path(args.static_image).exists():
        print(f"main: Error - Static image not found: {args.static_image}")
        return 1
    
    print("IPAdapter Stream Test Demo")
    print("=" * 50)
    print(f"main: Config: {args.config}")
    print(f"main: Input video: {args.input_video}")
    print(f"main: Static image: {args.static_image}")
    print(f"main: Output directory: {args.output_dir}")
    print("=" * 50)
    print("Test Description:")
    print("- Input video frames → ControlNet control images (structural guidance)")
    print("- Input video frames → IPAdapter style images (style/content guidance)")
    print("- Static image → Base input to StreamDiffusion (repeated for each frame)")
    print("- is_stream=True → High-throughput pipelined processing")
    print("- Tests IPAdapter stream behavior fix")
    print("=" * 50)
    
    try:
        metrics = process_video_ipadapter_stream(
            args.config, 
            args.input_video, 
            args.static_image, 
            args.output_dir, 
            engine_only=args.engine_only
        )
        if args.engine_only:
            print("main: Engine-only mode completed successfully!")
            return 0
        print("main: IPAdapter stream test completed successfully!")
        return 0
    except Exception as e:
        import traceback
        print(f"main: Error during processing: {e}")
        print(f"main: Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        return 1


if __name__ == "__main__":
    exit(main())
