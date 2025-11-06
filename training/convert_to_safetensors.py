#!/usr/bin/env python3
"""
Convert TemporalNet2 ControlNet checkpoint to safetensors format for HuggingFace upload
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import torch
from safetensors.torch import save_file


def convert_checkpoint_to_safetensors(checkpoint_dir, output_dir):
    """
    Convert a DeepSpeed checkpoint to safetensors format with proper config for HuggingFace
    
    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., checkpoint-25000)
        output_dir: Path where to save the converted model
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting checkpoint from {checkpoint_dir} to {output_dir}")
    
    # Load the PyTorch model
    pytorch_model_dir = checkpoint_dir / "pytorch_model.bin"
    
    if not pytorch_model_dir.exists():
        raise ValueError(
            f"pytorch_model.bin not found in {checkpoint_dir}. "
            "Please run zero_to_fp32.py first to convert DeepSpeed checkpoint."
        )
    
    # Load the model index to understand the sharding
    index_file = pytorch_model_dir / "pytorch_model.bin.index.json"
    
    if index_file.exists():
        print(f"Loading sharded model from {pytorch_model_dir}")
        with open(index_file, "r") as f:
            index = json.load(f)
        
        # Load all shards
        state_dict = {}
        weight_map = index["weight_map"]
        shard_files = set(weight_map.values())
        
        for shard_file in sorted(shard_files):
            shard_path = pytorch_model_dir / shard_file
            print(f"Loading shard: {shard_file}")
            shard_state = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_state)
        
        metadata = index.get("metadata", {})
    else:
        # Single file model
        model_file = checkpoint_dir / "pytorch_model.bin"
        print(f"Loading single model file from {model_file}")
        state_dict = torch.load(model_file, map_location="cpu")
        metadata = {}
    
    print(f"Loaded model with {len(state_dict)} parameters")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,}")
    
    # Create config.json for ControlNet SDXL with 6 conditioning channels
    config = {
        "_class_name": "ControlNetModel",
        "_diffusers_version": "0.35.2",
        "act_fn": "silu",
        "addition_embed_type": "text_time",
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": 256,
        "attention_head_dim": [5, 10, 20],
        "block_out_channels": [320, 640, 1280],
        "class_embed_type": None,
        "conditioning_channels": 6,
        "conditioning_embedding_out_channels": [16, 32, 96, 256],
        "controlnet_conditioning_channel_order": "rgb",
        "cross_attention_dim": 2048,
        "down_block_types": [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
        ],
        "downsample_padding": 1,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "global_pool_conditions": False,
        "in_channels": 4,
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        "num_attention_heads": None,
        "num_class_embeds": None,
        "only_cross_attention": False,
        "projection_class_embeddings_input_dim": 2816,
        "resnet_time_scale_shift": "default",
        "transformer_layers_per_block": [1, 2, 10],
        "upcast_attention": None,
        "use_linear_projection": True
    }
    
    # Save config
    config_path = output_dir / "config.json"
    print(f"Saving config to {config_path}")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Check model size to determine if we need to shard
    # HuggingFace recommends sharding models larger than 5GB
    total_size = sum(p.element_size() * p.numel() for p in state_dict.values())
    size_gb = total_size / (1024**3)
    print(f"Model size: {size_gb:.2f} GB")
    
    # For models over 4GB, we'll create shards
    max_shard_size = 4 * 1024**3  # 4GB
    
    if total_size > max_shard_size:
        print(f"Model is large ({size_gb:.2f} GB), creating sharded safetensors...")
        
        # Split into shards
        shards = []
        current_shard = {}
        current_size = 0
        weight_map = {}
        
        for key, tensor in sorted(state_dict.items()):
            tensor_size = tensor.element_size() * tensor.numel()
            
            # If adding this tensor would exceed shard size, start new shard
            if current_size + tensor_size > max_shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[key] = tensor
            current_size += tensor_size
            shard_idx = len(shards) + 1
            weight_map[key] = f"diffusion_pytorch_model-{shard_idx:05d}-of-{len(shards)+1:05d}.safetensors"
        
        # Add the last shard
        if current_shard:
            shards.append(current_shard)
        
        # Update weight_map with correct total count
        total_shards = len(shards)
        weight_map = {
            key: f"diffusion_pytorch_model-{idx+1:05d}-of-{total_shards:05d}.safetensors"
            for idx, shard in enumerate(shards)
            for key in shard.keys()
        }
        
        # Save each shard
        for idx, shard in enumerate(shards):
            shard_name = f"diffusion_pytorch_model-{idx+1:05d}-of-{total_shards:05d}.safetensors"
            shard_path = output_dir / shard_name
            print(f"Saving shard {idx+1}/{total_shards}: {shard_name}")
            save_file(shard, shard_path, metadata={"format": "pt"})
        
        # Create index file
        index = {
            "metadata": {
                "total_size": total_size,
                **metadata
            },
            "weight_map": weight_map
        }
        
        index_path = output_dir / "diffusion_pytorch_model.safetensors.index.json"
        print(f"Saving index to {index_path}")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    else:
        # Save as single file
        safetensors_path = output_dir / "diffusion_pytorch_model.safetensors"
        print(f"Saving to {safetensors_path}")
        save_file(state_dict, safetensors_path, metadata={"format": "pt"})
    
    # Create a README.md for the model card
    readme_content = f"""---
license: openrail++
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
  - stable-diffusion-xl
  - controlnet
  - temporal
  - video
  - diffusers
inference: true
---

# TemporalNet2 ControlNet for SDXL

This is a TemporalNet2 ControlNet model trained on SDXL (Stable Diffusion XL base 1.0).

## Model Description

TemporalNet2 is a ControlNet variant designed for temporal coherence in video generation. It takes two conditioning inputs:
- **Previous Frame**: The previous frame in the video sequence (3 channels)
- **Optical Flow**: The optical flow between the previous and current frame (3 channels)

Total conditioning channels: **6 channels**

This model was trained to generate temporally coherent frames by learning from both the visual content of the previous frame and the motion information encoded in optical flow.

## Usage

```python
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
from PIL import Image
import torch

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "YOUR_USERNAME/temporalnet2-sdxl-controlnet",
    torch_dtype=torch.float16
)

# Create the pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Load your conditioning images
prev_frame = Image.open("previous_frame.jpg")
optical_flow = Image.open("optical_flow.jpg")

# Concatenate conditioning images (they will be concatenated in the pipeline)
# Note: You'll need to prepare the 6-channel input by concatenating prev_frame and optical_flow
prompt = "your prompt describing the scene"

# Generate
image = pipe(
    prompt=prompt,
    image=[prev_frame, optical_flow],  # The pipeline will handle concatenation
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

image.save("output.jpg")
```

## Training Details

- **Base Model**: stabilityai/stable-diffusion-xl-base-1.0
- **Training Resolution**: Multi-resolution (512, 640, 768, 896, 1024px)
- **Conditioning Channels**: 6 (3 for previous frame + 3 for optical flow)
- **Training Steps**: 25,000
- **Mixed Precision**: bfloat16

## Limitations

This model requires specific conditioning inputs:
1. The previous frame from your video sequence
2. The optical flow computed between frames

For best results, ensure your optical flow visualization uses a consistent color scheme and magnitude representation.

## License

This model is released under the same license as SDXL (OpenRAIL++).
"""
    
    readme_path = output_dir / "README.md"
    print(f"Saving README to {readme_path}")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print("\n" + "="*80)
    print("✓ Conversion complete!")
    print("="*80)
    print(f"\nModel saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - config.json")
    print(f"  - diffusion_pytorch_model.safetensors (or sharded versions)")
    print(f"  - README.md")
    print(f"\nTo upload to HuggingFace:")
    print(f"  1. Install: pip install huggingface_hub")
    print(f"  2. Login: huggingface-cli login")
    print(f"  3. Upload:")
    print(f"     from huggingface_hub import HfApi")
    print(f"     api = HfApi()")
    print(f"     api.create_repo('YOUR_USERNAME/temporalnet2-sdxl-controlnet', repo_type='model')")
    print(f"     api.upload_folder(")
    print(f"         folder_path='{output_dir}',")
    print(f"         repo_id='YOUR_USERNAME/temporalnet2-sdxl-controlnet',")
    print(f"         repo_type='model'")
    print(f"     )")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Convert TemporalNet2 checkpoint to safetensors")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory (e.g., checkpoint-25000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir + '_hf')"
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_hf"
    
    convert_checkpoint_to_safetensors(checkpoint_dir, output_dir)


if __name__ == "__main__":
    main()


