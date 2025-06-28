import os
import sys
import torch
from pathlib import Path
from PIL import Image

# Add paths to import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.streamdiffusion.controlnet.config import create_wrapper_from_config, load_config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "ipadapter_img2img_config_example.yaml")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "..", "output")

def main():
    """IPAdapter img2img example using configuration system."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"main: Loading img2img configuration from {CONFIG_PATH}")
    
    # Load configuration
    config = load_config(CONFIG_PATH)
    print(f"main: Device: {device}")
    print(f"main: Mode: {config.get('mode', 'img2img')}")
    
    try:
        # Create wrapper from config (this automatically sets up IPAdapter for img2img)
        print("main: Creating StreamDiffusion wrapper with IPAdapter for img2img...")
        wrapper = create_wrapper_from_config(config, device=device)
        
        # Load input image for img2img transformation (hand_up512.png)
        # Style conditioning comes from input.png via config (different images for clear demo)
        input_image_path = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "hand_up512.png")
        
        # Check if input image exists, if not use a default path
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
        
        # Preprocess the input image
        print(f"main: Loading and preprocessing input image from {input_image_path}")
        input_image = wrapper.preprocess_image(input_image_path)
        
        print("main: Generating img2img with IPAdapter style conditioning...")
        
        # Generate images with img2img + IPAdapter
        # The IPAdapter embeddings will be automatically applied
        for _ in range(wrapper.batch_size - 1):
            wrapper(image=input_image)
        
        output_image = wrapper(image=input_image)
        
        # Save result
        output_path = os.path.join(OUTPUT_DIR, "ipadapter_img2img_config.png")
        output_image.save(output_path)
        print(f"main: IPAdapter img2img image saved to: {output_path}")
        
        # Demonstrate runtime style updates
        print("main: Demonstrating runtime style image update...")
        
        # Update style image at runtime
        style_image_path = config['ipadapters'][0]['style_image']
        if os.path.exists(style_image_path):
            wrapper.update_style_image(style_image_path, index=0)
            
            # Generate another image with updated style
            output_image2 = wrapper(image=input_image)
            output_path2 = os.path.join(OUTPUT_DIR, "ipadapter_img2img_config_updated.png")
            output_image2.save(output_path2)
            print(f"main: Updated IPAdapter img2img image saved to: {output_path2}")
        
        # Demonstrate scale adjustment for img2img
        print("main: Demonstrating scale adjustment for img2img...")
        wrapper.update_scale(0, 0.5)  # Reduce IPAdapter influence to balance with input image
        
        output_image3 = wrapper(image=input_image)
        output_path3 = os.path.join(OUTPUT_DIR, "ipadapter_img2img_config_scaled.png")
        output_image3.save(output_path3)
        print(f"main: Scaled IPAdapter img2img image saved to: {output_path3}")
        
        # Demonstrate stronger style conditioning
        print("main: Demonstrating stronger style conditioning...")
        wrapper.update_scale(0, 1.0)  # Increase IPAdapter influence
        
        output_image4 = wrapper(image=input_image)
        output_path4 = os.path.join(OUTPUT_DIR, "ipadapter_img2img_config_strong.png")
        output_image4.save(output_path4)
        print(f"main: Strong IPAdapter img2img image saved to: {output_path4}")
        
        print("main: IPAdapter img2img demonstration completed successfully!")
        print("main: Generated images demonstrate:")
        print("main:   1. Basic IPAdapter img2img with style conditioning")
        print("main:   2. Runtime style image updates")
        print("main:   3. Scale adjustments for balancing input image vs style")
        print("main:   4. Strong style conditioning effects")
        
    except Exception as e:
        print(f"main: Error - {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 