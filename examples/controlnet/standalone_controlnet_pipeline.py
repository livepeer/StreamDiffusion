#!/usr/bin/env python3
"""
Standalone Multi-ControlNet StreamDiffusion Pipeline

Self-contained script demonstrating multiple ControlNets + StreamDiffusion integration.
Shows depth + canny edge conditioning.

Designed for reference for porting into other production systems.
No GUI, no webcam complexity - just core pipeline logic with hardcoded configs.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Optional
import argparse

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from PIL import Image
from streamdiffusion.controlnet.config import load_config, create_wrapper_from_config
# ============================================================================
# PIPELINE IMPLEMENTATION
# ============================================================================


class MultiControlNetStreamDiffusionPipeline:
    """
    Multi-ControlNet StreamDiffusion pipeline.
    """
    
    def __init__(self, config_file: str, resolution: int = 512):
        self.config_file = config_file
        self.resolution = resolution
        self.wrapper = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize the StreamDiffusion pipeline from config file"""
        print("Initializing Multi-ControlNet StreamDiffusion pipeline...")
        print(f"Using config file: {self.config_file}")
        # Load configuration and create wrapper
        config_data = load_config(self.config_file)
        self.wrapper = create_wrapper_from_config(config_data, width=self.resolution, height=self.resolution)
        self.warmup_steps = config_data.get('warmup', 10)
        self._warmed_up = False
        
        print("Pipeline created - warmup will occur with first input image")
        
        # Check TensorRT status
        if hasattr(self.wrapper.stream, 'unet') and hasattr(self.wrapper.stream.unet, 'engine'):
            print("TensorRT acceleration active")
        else:
            print("Running in PyTorch mode")
    

    
    def process_image(self, image: Image.Image) -> Image.Image:
        """
        Process a single image through the multi-ControlNet pipeline.
        
        This is the core controlnet inference method that would be used in production.
        The conditioning for all of the controlnets that are defined in the config will be applied automatically.
        """
        # Run warmup with actual input image on first call
        if not self._warmed_up and self.warmup_steps > 0:
            print(f"Running {self.warmup_steps} warmup iterations with input image...")
            for i in range(self.warmup_steps):
                if i % 3 == 0:  # Print progress every 3 steps
                    print(f"  Warmup step {i+1}/{self.warmup_steps}")
                self.wrapper.update_control_image_efficient(image)
                _ = self.wrapper(image)
            self._warmed_up = True
            print("Warmup completed!")
        
        # Update control image for all ControlNets
        self.wrapper.update_control_image_efficient(image)
        
        # Generate output with multi-ControlNet conditioning
        return self.wrapper(image)
    
    def update_controlnet_strength(self, index: int, strength: float):
        """Dynamically update ControlNet strength. This will be required for Product."""
        if hasattr(self.wrapper, 'update_controlnet_scale'):
            self.wrapper.update_controlnet_scale(index, strength)
            print(f"update_controlnet_strength: Updated ControlNet {index+1} strength to {strength}")
        else:
            print("update_controlnet_strength: Not supported for this pipeline")
    
    def update_stream_params(self, guidance_scale: float = None, delta: float = None, num_inference_steps: int = None):
        """Dynamically update StreamDiffusion parameters during inference"""
        if hasattr(self.wrapper.stream, 'update_stream_params'):
            self.wrapper.stream.update_stream_params(
                guidance_scale=guidance_scale,
                delta=delta,
                num_inference_steps=num_inference_steps
            )
        else:
            print("update_stream_params: Not supported for this pipeline")


def load_input_image(image_path: str, target_resolution: int) -> Image.Image:
    """Load and prepare input image"""
    print(f"Loading input image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Resize to target resolution while maintaining aspect ratio
    original_size = image.size
    print(f"Original size: {original_size}")
    
    # Resize to target resolution
    image = image.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
    print(f"Resized to: {image.size}")
    
    return image


def setup_output_directory():
    """Create output directory next to the script"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "standalone_controlnet_pipeline_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def run_demo(config_file: str, input_image_path: str, resolution: int = 512, engine_only: bool = False):
    """
    Demonstration of the multi-ControlNet pipeline.
    Shows how depth + canny ControlNets work together.
    """
    # Setup output directory
    output_dir = setup_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Validate paths
    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found at {config_file}")
        return False
    
    if not os.path.exists(input_image_path):
        print(f"ERROR: Input image not found at {input_image_path}")
        return False
    
    try:
        # Initialize pipeline (this will trigger engine building if needed)
        pipeline = MultiControlNetStreamDiffusionPipeline(config_file, resolution)
        
        if engine_only:
            print("Engine-only mode: TensorRT engines have been built (if needed). Exiting.")
            return True
        
        # Load input image
        input_image = load_input_image(input_image_path, resolution)
        
        
        print("Running multi-ControlNet inference...")
        start_time = time.time()
        
        output_image = pipeline.process_image(input_image)
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f}s")
        
        # Save results to output directory
        timestamp = int(time.time())
        input_path = output_dir / f"input_{timestamp}.png"
        output_path = output_dir / f"output_{timestamp}.png"
        
        input_image.save(input_path)
        output_image.save(output_path)
        
        print(f"Results saved:")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        
        # Demonstrate combined parameter updates
        pipeline.update_controlnet_strength(0, 0.2)  # Reduce depth influence
        pipeline.update_stream_params(guidance_scale=1.2, delta=1.0)  # Adjust guidance and noise
        
        adjusted_output = pipeline.process_image(input_image)
        adjusted_path = output_dir / f"output_adjusted_{timestamp}.png"
        adjusted_output.save(adjusted_path)
        print(f"  Adjusted output: {adjusted_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Standalone Multi-ControlNet StreamDiffusion Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to ControlNet configuration file")
    parser.add_argument("--input-image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--engine-only", action="store_true", help="Only build TensorRT engines and exit (no inference)")
    args = parser.parse_args()

    print("=" * 70)
    print("Standalone Multi-ControlNet StreamDiffusion Pipeline")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Input Image: {args.input_image}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("=" * 70)
    
    success = run_demo(
        config_file=args.config,
        input_image_path=args.input_image,
        resolution=args.resolution,
        engine_only=args.engine_only
    )
    
    if success:
        print("\n✓ Multi-ControlNet demo completed successfully!")
    else:
        print("\n✗ Demo failed - check configuration and dependencies")


if __name__ == "__main__":
    main() 