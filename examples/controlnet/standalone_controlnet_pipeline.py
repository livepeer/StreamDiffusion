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
from utils.wrapper import StreamDiffusionWrapper

# ============================================================================
# HARDCODED CONFIGURATION - EDIT THESE FOR YOUR SETUP
# ============================================================================

class Config:
    """All configuration"""
    
    # Model paths
    MODEL_PATH = r"C:\_dev\comfy\ComfyUI\models\checkpoints\sd_turbo.safetensors"
    TENSORRT_ENGINE_PATH = r"C:\_dev\comfy\ComfyUI\models\tensorrt\depth-anything\depth_anything_vits14-fp16.engine"
    
    # Input image path
    INPUT_IMAGE_PATH = r"C:\Users\ryanf\Downloads\test_512x512.png"
    
    # Generation settings
    PROMPT = "an anime render of a girl with purple hair, masterpiece"
    NEGATIVE_PROMPT = "blurry, low quality, flat, 2d"
    GUIDANCE_SCALE = 1.1
    SEED = 789
    
    # Pipeline settings
    RESOLUTION = 512
    FRAME_BUFFER_SIZE = 1
    DELTA = 0.7
    T_INDEX_LIST = [0, 16]  
    
    # Performance settings
    ACCELERATION = "tensorrt"
    USE_LCM_LORA = False
    USE_TINY_VAE = True
    CFG_TYPE = "self"
    
    # Multi-ControlNet settings
    CONTROLNETS = [
        {
            "name": "Depth ControlNet",
            "model_id": "thibaud/controlnet-sd21-depth-diffusers",
            "preprocessor": "depth_tensorrt",
            "conditioning_scale": 0.5,
            "enabled": True,
            "preprocessor_params": {
                "engine_path": TENSORRT_ENGINE_PATH,
                "detect_resolution": 518,
                "image_resolution": 512
            }
        },
        {
            "name": "Canny ControlNet", 
            "model_id": "thibaud/controlnet-sd21-canny-diffusers",
            "preprocessor": "canny",
            "conditioning_scale": 0.5,
            "enabled": True,
            "preprocessor_params": {
                "low_threshold": 50,
                "high_threshold": 100
            }
        }
    ]


class MultiControlNetStreamDiffusionPipeline:
    """
    Multi-ControlNet StreamDiffusion pipeline.
    """
    
    def __init__(self, config: Config = None, config_file: Optional[str] = None):
        self.config = config or Config()
        self.config_file = config_file
        self.wrapper = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize the StreamDiffusion pipeline with multiple ControlNets"""
        print("Initializing Multi-ControlNet StreamDiffusion pipeline...")
        
        # If config file is provided, use the built-in config system
        if self.config_file:
            print(f"Using config file: {self.config_file}")
            from streamdiffusion.controlnet.config import load_config, create_wrapper_from_config
            
            config_data = load_config(self.config_file)
            self.wrapper = create_wrapper_from_config(config_data)
            print("Pipeline ready for inference!")
            return
        
        # Otherwise use hardcoded config (original behavior)
        print(f"Model: {self.config.MODEL_PATH}")
        print(f"Acceleration: {self.config.ACCELERATION}")
        
        # Create ControlNet configurations
        controlnet_configs = []
        for i, cn_config in enumerate(self.config.CONTROLNETS):
            print(f"  ControlNet {i+1}: {cn_config['model_id']} ({cn_config['preprocessor']})")
            
            config_dict = {
                'model_id': cn_config['model_id'],
                'preprocessor': cn_config['preprocessor'],
                'conditioning_scale': cn_config['conditioning_scale'],
                'enabled': cn_config['enabled'],
                'preprocessor_params': cn_config.get('preprocessor_params', {}),
                'pipeline_type': 'sdturbo',
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0,
            }
            controlnet_configs.append(config_dict)
        
        # Initialize wrapper with multiple ControlNets
        self.wrapper = StreamDiffusionWrapper(
            model_id_or_path=self.config.MODEL_PATH,
            t_index_list=self.config.T_INDEX_LIST,
            mode="img2img",
            output_type="pil",
            dtype=torch.float16,
            frame_buffer_size=self.config.FRAME_BUFFER_SIZE,
            width=self.config.RESOLUTION,
            height=self.config.RESOLUTION,
            warmup=10,
            acceleration=self.config.ACCELERATION,
            do_add_noise=True,
            use_lcm_lora=self.config.USE_LCM_LORA,
            use_tiny_vae=self.config.USE_TINY_VAE,
            use_denoising_batch=True,
            cfg_type=self.config.CFG_TYPE,
            seed=self.config.SEED,
            use_safety_checker=False,
            use_controlnet=True,
            controlnet_config=controlnet_configs,  # Pass list of ControlNet configs
        )
        
        # Prepare pipeline
        self.wrapper.prepare(
            prompt=self.config.PROMPT,
            negative_prompt=self.config.NEGATIVE_PROMPT,
            num_inference_steps=50,
            guidance_scale=self.config.GUIDANCE_SCALE,
            delta=self.config.DELTA,
        )
        
        print("Pipeline ready for inference!")
        
        # Check TensorRT status
        if hasattr(self.wrapper.stream, 'unet') and hasattr(self.wrapper.stream.unet, 'engine'):
            print("✓ TensorRT acceleration active")
        else:
            print("⚠ Running in PyTorch mode")
    
    def warmup(self, test_image: Image.Image):
        """Warm up the pipeline for stable performance"""
        print("Warming up pipeline...")
        
        # Run several warmup iterations to stabilize performance
        warmup_iterations = 3
        for i in range(warmup_iterations):
            print(f"  Warmup iteration {i+1}/{warmup_iterations}")
            self.wrapper.update_control_image_efficient(test_image)
            _ = self.wrapper(test_image)
        
        print("✓ Pipeline warmed up")
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """
        Process a single image through the multi-ControlNet pipeline.
        
        This is the core controlnet inference method that would be used in production.
        Both depth and canny conditioning will be applied automatically.
        """
        # Update control image for all ControlNets
        self.wrapper.update_control_image_efficient(image)
        
        # Generate output with multi-ControlNet conditioning
        return self.wrapper(image)
    
    def update_controlnet_strength(self, index: int, strength: float):
        """Dynamically update ControlNet strength. This will be required for Product."""
        if hasattr(self.wrapper, 'update_controlnet_scale'):
            self.wrapper.update_controlnet_scale(index, strength)
            print(f"Updated ControlNet {index+1} strength to {strength}")
        else:
            print("⚠ update_controlnet_strength not supported for this pipeline")


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


def run_demo(engine_only=False, config_file=None):
    """
    Demonstration of the multi-ControlNet pipeline.
    Shows how depth + canny ControlNets work together.
    """
    # Setup output directory
    output_dir = setup_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Skip validation if using config file
    if not config_file:
        # Validate paths
        config = Config()
        if not os.path.exists(config.MODEL_PATH):
            print(f"ERROR: Model not found at {config.MODEL_PATH}")
            print("Please update MODEL_PATH in the Config class")
            return False
        
        if not os.path.exists(config.INPUT_IMAGE_PATH):
            print(f"ERROR: Input image not found at {config.INPUT_IMAGE_PATH}")
            print("Please update INPUT_IMAGE_PATH in the Config class")
            return False
        
        if not os.path.exists(config.TENSORRT_ENGINE_PATH):
            print(f"WARNING: TensorRT engine not found at {config.TENSORRT_ENGINE_PATH}")
            print("Will build engine on first run (may take 5-10 minutes)")
    else:
        config = Config()  # Still need config for some paths when using config file
    
    try:
        # Initialize pipeline (this will trigger engine building if needed)
        pipeline = MultiControlNetStreamDiffusionPipeline(config, config_file=config_file)
        
        if engine_only:
            print("Engine-only mode: TensorRT engines have been built (if needed). Exiting.")
            return True
        
        # Load input image
        input_image = load_input_image(config.INPUT_IMAGE_PATH, config.RESOLUTION)
        
        # Warm up the pipeline
        pipeline.warmup(input_image)
        
        # Run inference
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
        

        pipeline.update_controlnet_strength(0, 0.2)  # Reduce depth influence
        
        weak_depth_output = pipeline.process_image(input_image)
        weak_depth_path = output_dir / f"output_weak_depth_{timestamp}.png"
        weak_depth_output.save(weak_depth_path)
        print(f"  Weak depth output: {weak_depth_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Standalone Multi-ControlNet StreamDiffusion Pipeline")
    parser.add_argument("--engine-only", action="store_true", help="Only build TensorRT engines and exit (no inference)")
    parser.add_argument("--config", type=str, help="Optional config file path (uses hardcoded config if not provided)")
    args = parser.parse_args()

    print("=" * 70)
    print("Standalone Multi-ControlNet StreamDiffusion Pipeline")
    print("=" * 70)
    print(f"Model: {Config.MODEL_PATH}")
    print(f"Input Image: {Config.INPUT_IMAGE_PATH}")
    print(f"Prompt: {Config.PROMPT}")
    print(f"Resolution: {Config.RESOLUTION}x{Config.RESOLUTION}")
    
    print("\nControlNets:")
    for i, cn in enumerate(Config.CONTROLNETS):
        print(f"  {i+1}. {cn['name']}: {cn['model_id']}")
        print(f"     Preprocessor: {cn['preprocessor']}, Strength: {cn['conditioning_scale']}")
    
    print("=" * 70)
    
    success = run_demo(engine_only=args.engine_only, config_file=args.config)
    
    if success:
        print("\n✓ Multi-ControlNet demo completed successfully!")
       
    else:
        print("\n✗ Demo failed - check configuration and dependencies")


if __name__ == "__main__":
    main() 