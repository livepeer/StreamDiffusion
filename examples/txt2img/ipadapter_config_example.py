import os
import sys
import torch
from pathlib import Path

# Add paths to import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.streamdiffusion.controlnet.config import create_wrapper_from_config, load_config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "ipadapter_config_example.yaml")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "..", "output")

def main():
    """IPAdapter example using configuration system."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"main: Loading configuration from {CONFIG_PATH}")
    
    # Load configuration
    config = load_config(CONFIG_PATH)
    print(f"main: Device: {device}")
    
    # Note: Models will be automatically downloaded from HuggingFace if using model IDs
    # No manual download needed when using "h94/IP-Adapter" format in config
    
    try:
        # Create wrapper from config (this automatically sets up IPAdapter)
        print("main: Creating StreamDiffusion wrapper with IPAdapter from config...")
        wrapper = create_wrapper_from_config(config, device=device)
        
        # The wrapper is now an IPAdapterPipeline with everything configured
        print("main: Generating image with IPAdapter...")
        
        # Generate images
        for _ in range(wrapper.batch_size - 1):
            wrapper()
        
        output_image = wrapper()
        
        # Save result
        output_path = os.path.join(OUTPUT_DIR, "streamdiffusion_ipadapter_config.png")
        output_image.save(output_path)
        print(f"main: IPAdapter image saved to: {output_path}")
        
        # Demonstrate runtime updates
        print("main: Demonstrating runtime style image update...")
        
        # You can still update the style image at runtime
        style_image_path = config['ipadapters'][0]['style_image']
        if os.path.exists(style_image_path):
            wrapper.update_style_image(style_image_path, index=0)
            
            # Generate another image with updated style
            output_image2 = wrapper()
            output_path2 = os.path.join(OUTPUT_DIR, "streamdiffusion_ipadapter_config_updated.png")
            output_image2.save(output_path2)
            print(f"main: Updated IPAdapter image saved to: {output_path2}")
        
        # Demonstrate scale update
        print("main: Demonstrating scale update...")
        # wrapper.update_scale(0, 0.5)  # Reduce IPAdapter influence
        
        output_image3 = wrapper()
        output_path3 = os.path.join(OUTPUT_DIR, "streamdiffusion_ipadapter_config_scaled.png")
        output_image3.save(output_path3)
        print(f"main: Scaled IPAdapter image saved to: {output_path3}")
        
    except Exception as e:
        print(f"main: Error - {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 