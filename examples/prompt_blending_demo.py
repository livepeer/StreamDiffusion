#!/usr/bin/env python3
"""
StreamDiffusion Prompt & Seed Blending Demo

Demonstrates real-time prompt and seed blending capabilities in StreamDiffusion.
Shows how to blend multiple prompts and/or seeds with adjustable weights at runtime.

Based on the standalone pipeline structure but focuses on blending without ControlNets.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Optional, Dict
import argparse

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from PIL import Image
from streamdiffusion import load_config, create_wrapper_from_config

# ============================================================================
# PROMPT BLENDING PIPELINE IMPLEMENTATION
# ============================================================================


class PromptBlendingPipeline:
    """
    StreamDiffusion pipeline with advanced prompt blending capabilities.
    """
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.wrapper = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize the StreamDiffusion pipeline from config file"""
        print("Initializing StreamDiffusion Prompt Blending pipeline...")
        print(f"Using config file: {self.config_file}")
        
        # Load configuration and create wrapper
        config_data = load_config(self.config_file)
        
        # Show if config has prompt blending setup
        if 'prompt_blending' in config_data:
            print("Config contains prompt blending configuration:")
            blend_config = config_data['prompt_blending']
            for i, (prompt, weight) in enumerate(blend_config['prompt_list']):
                print(f"  {i+1}. [{weight:.1f}] {prompt[:50]}...")
            print(f"  Method: {blend_config['interpolation_method']}")
        else:
            print("Config uses single prompt mode")
        
        # Create wrapper - prompt blending applied automatically if in config
        self.wrapper = create_wrapper_from_config(config_data)
        self.warmup_steps = config_data.get('warmup', 10)
        self._warmed_up = False
        
        print("Pipeline created with prompt blending capabilities")
        
        # Check TensorRT status
        if hasattr(self.wrapper.stream, 'unet') and hasattr(self.wrapper.stream.unet, 'engine'):
            print("TensorRT acceleration active")
        else:
            print("Running in PyTorch mode")
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """Process a single image through the pipeline."""
        # Run warmup with actual input image on first call
        if not self._warmed_up and self.warmup_steps > 0:
            print(f"Running {self.warmup_steps} warmup iterations with input image...")
            for i in range(self.warmup_steps):
                if i % 3 == 0:  # Print progress every 3 steps
                    print(f"  Warmup step {i+1}/{self.warmup_steps}")
                _ = self.wrapper(image)
            self._warmed_up = True
            print("Warmup completed!")
        
        # Generate output
        return self.wrapper(image)
    

    
    def demonstrate_smooth_blending(self, image: Image.Image) -> List[Image.Image]:
        """Demonstrate smooth blending between prompts and seeds over exactly 20 images."""
        print("\n=== Smooth Prompt & Seed Blending Demo ===")
        
        # Check if config has blending configurations
        config_data = load_config(self.config_file)
        
        has_prompt_blending = 'prompt_blending' in config_data
        has_seed_blending = 'seed_blending' in config_data
        
        if has_prompt_blending or has_seed_blending:
            results = []
            num_images = 20
            
            # Setup prompt blending if available
            prompt_a, prompt_b = None, None
            if has_prompt_blending:
                blend_config = config_data['prompt_blending']
                if len(blend_config['prompt_list']) >= 2:
                    prompt_a = blend_config['prompt_list'][0][0]
                    prompt_b = blend_config['prompt_list'][1][0]
                    print("Prompt blending enabled:")
                    print(f"  Prompt A: {prompt_a}")
                    print(f"  Prompt B: {prompt_b}")
            
            # Setup seed blending if available
            seed_a, seed_b = None, None
            if has_seed_blending:
                seed_config = config_data['seed_blending']
                if len(seed_config['seed_list']) >= 2:
                    seed_a = seed_config['seed_list'][0][0]
                    seed_b = seed_config['seed_list'][1][0]
                    print("Seed blending enabled:")
                    print(f"  Seed A: {seed_a}")
                    print(f"  Seed B: {seed_b}")
            
            print(f"  Generating {num_images} images with smooth transitions")
            print(f"\nGenerating {num_images} images:")
            
            for i in range(num_images):
                # Calculate weights: from 100% A to 100% B
                t = i / (num_images - 1)  # 0.0 to 1.0
                weight_a = 1.0 - t
                weight_b = t
                
                status_parts = []
                if prompt_a and prompt_b:
                    status_parts.append(f"Prompt A={weight_a:.2f}, B={weight_b:.2f}")
                if seed_a and seed_b:
                    status_parts.append(f"Seed A={weight_a:.2f}, B={weight_b:.2f}")
                
                print(f"  Image {i+1:02d}/{num_images}: {' | '.join(status_parts)}")
                
                # Prepare update parameters
                update_params = {}
                
                # Add prompt blending if available
                if prompt_a and prompt_b:
                    update_params['prompt_list'] = [
                        [prompt_a, weight_a],
                        [prompt_b, weight_b]
                    ]
                    update_params['interpolation_method'] = "slerp"
                
                # Add seed blending if available
                if seed_a and seed_b:
                    update_params['seed_list'] = [
                        [seed_a, weight_a],
                        [seed_b, weight_b]
                    ]
                    update_params['seed_interpolation_method'] = "linear"
                
                # Apply blending with new weights
                self.wrapper.update_stream_params(**update_params)
                
                # Generate image
                result = self.process_image(image)
                results.append(result)
            
            # Show cache performance
            cache_info = self.wrapper.get_cache_info()
            print(f"\nCache performance: {cache_info}")
            
        else:
            # Fall back to single prompt mode
            print("No prompt or seed blending in config - using single prompt mode")
            results = self._generate_single_prompt_sequence(image, config_data)
        
        return results
    
    def _generate_single_prompt_sequence(self, image: Image.Image, config_data: Dict) -> List[Image.Image]:
        """Generate 20 identical images using single prompt from config."""
        single_prompt = config_data.get('prompt', 'a beautiful landscape')
        
        print(f"Using single prompt: {single_prompt}")
        print(f"Generating 20 images (all identical):")
        
        results = []
        num_images = 20
        
        for i in range(num_images):
            print(f"  Image {i+1:02d}/{num_images}: Single prompt")
            result = self.process_image(image)
            results.append(result)
        
        print("Note: All images are identical since no prompt blending was configured")
        return results


def load_input_image(image_path: str, target_width: int, target_height: int) -> Image.Image:
    """Load and prepare input image"""
    print(f"Loading input image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Resize to target resolution
    original_size = image.size
    print(f"Original size: {original_size}")
    
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"Resized to: {image.size}")
    
    return image


def setup_output_directory():
    """Create output directory next to the script"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "prompt_blending_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def run_prompt_blending_demo(config_file: str, input_image_path: str, engine_only: bool = False):
    """
    Run a focused prompt blending demonstration with exactly 20 images.
    Each image shows a different weight ratio between two prompts.
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
        # Load configuration to get dimensions
        config_data = load_config(config_file)
        target_width = config_data.get('width', 512)
        target_height = config_data.get('height', 512)
        
        # Initialize pipeline
        pipeline = PromptBlendingPipeline(config_file)
        
        if engine_only:
            print("Engine-only mode: TensorRT engines have been built (if needed). Exiting.")
            return True
        
        # Load input image
        input_image = load_input_image(input_image_path, target_width, target_height)
        
        timestamp = int(time.time())
        
        # Save input image for reference
        input_path = output_dir / f"input_{timestamp}.png"
        input_image.save(input_path)
        print(f"Input saved: {input_path}")
        
        print("\n" + "="*70)
        print("PROMPT & SEED BLENDING DEMONSTRATION - 20 IMAGE SEQUENCE")
        print("="*70)
        
        # Generate the 20-image smooth blend sequence
        start_time = time.time()
        blend_results = pipeline.demonstrate_smooth_blending(input_image)
        blend_time = time.time() - start_time
        
        # Save all 20 images
        for i, result in enumerate(blend_results):
            result_path = output_dir / f"blend_{i+1:02d}_{timestamp}.png"
            result.save(result_path)
        
        print(f"\nAll {len(blend_results)} images saved to: {output_dir}")
        print(f"Generation time: {blend_time:.2f}s")
        
        # Final summary
        print("\n" + "="*70)
        print("DEMO COMPLETE - SMOOTH PROMPT & SEED BLENDING")
        print("="*70)
        print(f"Total images generated: {len(blend_results)} (plus 1 input)")
        print(f"Each image represents different blending ratios:")
        print("  • Image 01: 100% Source A,   0% Source B")
        print("  • Image 10:  53% Source A,  47% Source B") 
        print("  • Image 20:   0% Source A, 100% Source B")
        print("  (Source A/B can be prompts and/or seeds depending on config)")
        print(f"\nFiles saved with pattern: blend_XX_{timestamp}.png")
        print("\nTo create an animation:")
        print(f"ffmpeg -r 10 -i {output_dir}/blend_%02d_{timestamp}.png -pix_fmt yuv420p blend_animation.mp4")
        
        # Clean up caches
        pipeline.wrapper.clear_caches()
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StreamDiffusion Prompt & Seed Blending Demo")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sd15_img2img.yaml"),
                        help="Path to configuration file")
    parser.add_argument("--input-image", type=str, required=True, 
                        help="Path to input image file")
    parser.add_argument("--engine-only", action="store_true", 
                        help="Only build TensorRT engines and exit (no inference)")
    args = parser.parse_args()

    print("=" * 70)
    print("StreamDiffusion Prompt & Seed Blending Demo")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Input Image: {args.input_image}")
    print("=" * 70)
    print("This demo generates exactly 20 images showing:")
    print("• Smooth blending between prompts and/or seeds")
    print("• Each image has different weight ratios (100%A → 100%B)")
    print("• SLERP interpolation for prompts, linear/SLERP for seeds")
    print("• Efficient caching for both prompts and noise tensors")
    print("• Perfect for creating smooth animations with varied content")
    print("")
    print("📊 Output: 20 images + 1 input = 21 total files")
    print("🎬 Animation: Use ffmpeg command shown at the end")
    print("=" * 70)
    
    success = run_prompt_blending_demo(
        config_file=args.config,
        input_image_path=args.input_image,
        engine_only=args.engine_only
    )
    
    if success:
        print("\n✓ Prompt & seed blending demo completed successfully!")
        print("Check the output images to see the blending effects in action.")
    else:
        print("\n✗ Demo failed - check configuration and dependencies")


if __name__ == "__main__":
    main() 