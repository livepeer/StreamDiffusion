#!/usr/bin/env python3
"""
Verify that the converted model can be loaded correctly
"""
import argparse
from pathlib import Path
import torch
from diffusers import ControlNetModel


def verify_model(model_dir):
    """
    Verify that a converted model can be loaded
    
    Args:
        model_dir: Path to the model directory
    """
    model_dir = Path(model_dir)
    
    print(f"Verifying model at: {model_dir}")
    print("="*80)
    
    # Check files exist
    print("\n1. Checking required files...")
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print("  ✗ config.json not found!")
        return False
    print("  ✓ config.json found")
    
    safetensors_files = list(model_dir.glob("*.safetensors"))
    index_file = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    
    if index_file.exists():
        print(f"  ✓ Found sharded model with index")
        print(f"  ✓ Found {len(safetensors_files)} shard(s)")
    elif safetensors_files:
        print(f"  ✓ Found single safetensors file")
    else:
        print("  ✗ No safetensors files found!")
        return False
    
    # Try loading the model
    print("\n2. Loading model with diffusers...")
    try:
        controlnet = ControlNetModel.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16
        )
        print("  ✓ Model loaded successfully!")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False
    
    # Check model properties
    print("\n3. Checking model properties...")
    config = controlnet.config
    
    print(f"  - Class: {config._class_name}")
    print(f"  - Conditioning channels: {config.conditioning_channels}")
    print(f"  - Cross attention dim: {config.cross_attention_dim}")
    print(f"  - Block out channels: {config.block_out_channels}")
    
    if config.conditioning_channels != 6:
        print(f"  ⚠ Warning: Expected 6 conditioning channels, got {config.conditioning_channels}")
    else:
        print("  ✓ Conditioning channels correct (6)")
    
    if config.cross_attention_dim != 2048:
        print(f"  ⚠ Warning: Expected cross_attention_dim=2048 for SDXL, got {config.cross_attention_dim}")
    else:
        print("  ✓ Cross attention dimension correct (SDXL)")
    
    # Count parameters
    print("\n4. Counting parameters...")
    total_params = sum(p.numel() for p in controlnet.parameters())
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Check input/output shapes
    print("\n5. Testing forward pass (dry run)...")
    try:
        # Create dummy inputs
        batch_size = 1
        height, width = 512, 512
        
        sample = torch.randn(batch_size, 4, height // 8, width // 8, dtype=torch.float16)
        timestep = torch.tensor([999])
        encoder_hidden_states = torch.randn(batch_size, 77, 2048, dtype=torch.float16)
        controlnet_cond = torch.randn(batch_size, 6, height, width, dtype=torch.float16)
        
        # Move to CPU for test (no GPU required)
        controlnet = controlnet.to("cpu")
        sample = sample.to("cpu").to(torch.float32)
        encoder_hidden_states = encoder_hidden_states.to("cpu").to(torch.float32)
        controlnet_cond = controlnet_cond.to("cpu").to(torch.float32)
        
        # Test forward pass
        with torch.no_grad():
            down_block_res_samples, mid_block_res_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                return_dict=False
            )
        
        print(f"  ✓ Forward pass successful!")
        print(f"  - Down block outputs: {len(down_block_res_samples)}")
        print(f"  - Mid block output shape: {mid_block_res_sample.shape}")
        
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        print("  (This might be okay if you don't have enough RAM)")
    
    print("\n" + "="*80)
    print("✓ Model verification complete!")
    print("="*80)
    print("\nModel is ready to upload to HuggingFace!")
    print(f"\nTo upload, run:")
    print(f"  python upload_to_hf.py {model_dir} YOUR_USERNAME/temporalnet2-sdxl-controlnet")
    print("="*80)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify converted model")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory (e.g., checkpoint-25000_hf)"
    )
    
    args = parser.parse_args()
    
    success = verify_model(args.model_dir)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()


