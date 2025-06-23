import os
import sys
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

# Add paths to import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Diffusers_IPAdapter"))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "input")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs")

def main():
    """Generate one image using IPAdapter."""
    
    # Hard-coded settings
    prompt = "a beautiful woman with long brown hair, smiling, portrait photography"
    style_image_path = os.path.join(INPUT_DIR, "hand_up512.png")
    output_dir = OUTPUT_DIR
    model_id = "KBlueLeaf/kohaku-v2.1"
    ipadapter_model_id = "h94/IP-Adapter"  # Will load ip-adapter_sd15.bin from this repo
    image_encoder_id = "h94/IP-Adapter"    # Will load image encoder from this repo
    width, height = 512, 512
    seed = 42
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating image with IPAdapter...")
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Style image: {style_image_path}")
    
    # Load style image
    if not os.path.exists(style_image_path):
        print(f"Error: Style image not found at {style_image_path}")
        return
    
    style_img = Image.open(style_image_path).convert("RGB")
    print(f"Style image loaded: {style_img.size}")
    
    try:
        # First, set up StreamDiffusion pipeline
        print("Setting up StreamDiffusion...")
        stream = StreamDiffusionWrapper(
            model_id_or_path=model_id,
            t_index_list=[0, 16, 32, 45],
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            acceleration="xformers",
            mode="txt2img",
            use_denoising_batch=False,
            cfg_type="none",
            seed=seed,
        )
        
        # Access the underlying pipeline for IPAdapter integration
        pipe = stream.stream.pipe
        
        # Download IPAdapter models
        print("Downloading IPAdapter models...")
        ipadapter_model_path = hf_hub_download(
            repo_id="h94/IP-Adapter", 
            filename="models/ip-adapter_sd15.bin"
        )
        
        # Download image encoder (it's a directory)
        repo_path = snapshot_download(
            repo_id="h94/IP-Adapter",
            allow_patterns=["models/image_encoder/*"]
        )
        image_encoder_path = os.path.join(repo_path, "models", "image_encoder")
        
        # Import and initialize IPAdapter on the StreamDiffusion pipeline
        from ip_adapter.ip_adapter import IPAdapter
        ip_adapter = IPAdapter(
            pipe, 
            ipadapter_model_path,
            image_encoder_path,
            device=device
        )
        
        # Generate IPAdapter-conditioned prompt embeddings
        print("Generating IPAdapter embeddings...")
        prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
            style_img,
            prompt=prompt,
            negative_prompt="blurry, horror, worst quality, low quality",
        )
        
        # Prepare StreamDiffusion with IPAdapter embeddings
        # We'll modify the stream to use our custom embeddings
        stream.stream.prompt_embeds = prompt_embeds
        stream.stream.negative_prompt_embeds = negative_prompt_embeds
        
        # Generate using StreamDiffusion with IPAdapter conditioning
        print("Generating with StreamDiffusion + IPAdapter...")
        for _ in range(stream.batch_size - 1):
            stream()
        
        output_image = stream()
        
        # Save image
        output_path = os.path.join(output_dir, "streamdiffusion_ipadapter_output.png")
        output_image.save(output_path)
        print(f"StreamDiffusion + IPAdapter image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 