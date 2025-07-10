import os
import sys
import torch
from pathlib import Path
from PIL import Image

# Add paths to import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.streamdiffusion.config import create_wrapper_from_config, load_config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "ipadapter_img2img_config_example.yaml")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "..", "output")

def main():
    """IPAdapter img2img example using configuration system with multiple strength values."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"main: Loading img2img configuration from {CONFIG_PATH}")
    
    # Load configuration
    config = load_config(CONFIG_PATH)
    print(f"main: Device: {device}")
    print(f"main: Mode: {config.get('mode', 'img2img')}")
    
    # Get the original scale from config
    original_scale = config.get('ipadapters', [{}])[0].get('scale', 1.0)
    
    # Define strength values to test
    strength_values = [original_scale]
    
    # Load input image for img2img transformation
    input_image_path = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "hand_up512.png")
    
    # Check if input image exists, if not use alternative paths
    if not os.path.exists(input_image_path):
        print(f"main: Input image not found at {input_image_path}")
        print("main: Please place an input image at the specified path or update the path")
        # For demonstration, try alternative paths
        alt_paths = [
            os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "input.png"),
            os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "style.webp"),
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                input_image_path = alt_path
                print(f"main: Using alternative input image: {input_image_path}")
                break
        else:
            print("main: No suitable input image found. Please provide an input image.")
            return
    
    try:
        print(f"main: Testing IPAdapter img2img with strength values: {strength_values}")
        
        for i, strength in enumerate(strength_values):
            print(f"\nmain: Creating pipeline {i+1}/4 with strength {strength}")
            
            # Create a copy of config for this strength
            current_config = config.copy()
            if 'ipadapters' in current_config and len(current_config['ipadapters']) > 0:
                current_config['ipadapters'][0]['scale'] = strength
            
            # Create fresh wrapper for this strength (clean slate)
            wrapper = create_wrapper_from_config(current_config, device=device)
            
            # Preprocess the input image
            print(f"main: Loading and preprocessing input image from {input_image_path}")
            input_image = wrapper.preprocess_image(input_image_path)
            
            print(f"main: Generating img2img with IPAdapter strength {strength}")
            
            # Warm up the pipeline
            for _ in range(wrapper.batch_size - 1):
                wrapper(image=input_image)
            
            # Generate final image
            output_image = wrapper(image=input_image)
            
            # Save result with strength in filename
            if strength == original_scale:
                output_path = os.path.join(OUTPUT_DIR, f"ipadapter_img2img_strength_{strength:.2f}_config.png")
            else:
                output_path = os.path.join(OUTPUT_DIR, f"ipadapter_img2img_strength_{strength:.2f}.png")
            
            output_image.save(output_path)
            print(f"main: IPAdapter img2img image (strength {strength}) saved to: {output_path}")
            
            # Clean up wrapper to ensure no state interference
            del wrapper
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\nmain: IPAdapter img2img demonstration completed successfully!")
        print(f"main: Generated 4 images with different IPAdapter strengths:")
        print(f"main:   - 0.00: No IPAdapter influence (pure img2img)")
        print(f"main:   - 0.50: Balanced IPAdapter and input image")
        print(f"main:   - 1.00: Strong IPAdapter influence")
        print(f"main:   - {original_scale:.2f}: Original config value")
        
    except Exception as e:
        print(f"main: Error - {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 