#!/usr/bin/env python3
"""
Debug script to inspect UNet architecture and IPAdapter processor installation

This script will help us understand:
1. What attention processor names exist in the UNet
2. Which ones are self-attention vs cross-attention
3. Why IPAdapter processors aren't being installed
4. What the actual UNet structure looks like
"""

import torch
import sys
from pathlib import Path
from diffusers import UNet2DConditionModel

# Add StreamDiffusion to path
current_dir = Path(__file__).parent
streamdiffusion_path = current_dir / "src"
sys.path.insert(0, str(streamdiffusion_path))

def inspect_unet_attention_processors():
    """Load UNet and inspect its attention processors"""
    print("Loading UNet2DConditionModel...")
    
    # Load the actual UNet model
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="unet",
        torch_dtype=torch.float16
    )
    
    print(f"UNet config:")
    print(f"  - cross_attention_dim: {unet.config.cross_attention_dim}")
    print(f"  - block_out_channels: {unet.config.block_out_channels}")
    print(f"  - in_channels: {unet.config.in_channels}")
    print(f"  - out_channels: {unet.config.out_channels}")
    
    print(f"\nInspecting attention processors...")
    attn_processors = unet.attn_processors
    print(f"Total attention processors: {len(attn_processors)}")
    
    # Categorize processors
    self_attn_names = []
    cross_attn_names = []
    
    for name in attn_processors.keys():
        if name.endswith("attn1.processor"):
            self_attn_names.append(name)
        elif name.endswith("attn2.processor"):
            cross_attn_names.append(name)
        else:
            print(f"Unknown attention processor pattern: {name}")
    
    print(f"\nSelf-attention processors (attn1): {len(self_attn_names)}")
    for i, name in enumerate(self_attn_names[:5]):  # Show first 5
        print(f"  {i+1}. {name}")
    if len(self_attn_names) > 5:
        print(f"  ... and {len(self_attn_names)-5} more")
    
    print(f"\nCross-attention processors (attn2): {len(cross_attn_names)}")
    for i, name in enumerate(cross_attn_names[:5]):  # Show first 5
        print(f"  {i+1}. {name}")
    if len(cross_attn_names) > 5:
        print(f"  ... and {len(cross_attn_names)-5} more")
    
    return unet, self_attn_names, cross_attn_names

def test_ipadapter_processor_installation(unet, cross_attn_names):
    """Test IPAdapter processor installation logic"""
    print(f"\n{'='*60}")
    print("Testing IPAdapter processor installation...")
    
    # Add the IPAdapter path
    current_dir = Path(__file__).parent
    ipadapter_path = current_dir / "src" / "streamdiffusion" / "ipadapter" / "Diffusers_IPAdapter"
    sys.path.insert(0, str(ipadapter_path))
    
    try:
        print(f"Attempting to import IPAdapter attention processors...")
        
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
            print("✓ Using IPAttnProcessor2_0 (scaled_dot_product_attention available)")
        else:
            from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
            print("✓ Using IPAttnProcessor (standard version)")
            
        print("✓ Successfully imported IPAdapter attention processors")
        
        # Test creating processors for cross-attention layers
        print(f"\nTesting processor creation for {len(cross_attn_names)} cross-attention layers...")
        
        successful_processors = 0
        failed_processors = 0
        
        for name in cross_attn_names[:3]:  # Test first 3
            try:
                # Extract hidden size using the logic from the wrapper
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                else:
                    print(f"  ❌ Unknown block pattern for {name}")
                    failed_processors += 1
                    continue
                
                # Create IPAdapter processor
                processor = IPAttnProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=unet.config.cross_attention_dim,
                    num_tokens=4
                )
                
                print(f"  ✓ {name}: hidden_size={hidden_size}, cross_attn_dim={unet.config.cross_attention_dim}")
                successful_processors += 1
                
            except Exception as e:
                print(f"  ❌ Failed to create processor for {name}: {e}")
                failed_processors += 1
        
        print(f"\nProcessor creation results:")
        print(f"  ✓ Successful: {successful_processors}")
        print(f"  ❌ Failed: {failed_processors}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import IPAdapter processors: {e}")
        print(f"   IPAdapter path: {ipadapter_path}")
        print(f"   Path exists: {ipadapter_path.exists()}")
        return False

def test_wrapper_creation():
    """Test the actual IPAdapter wrapper creation"""
    print(f"\n{'='*60}")
    print("Testing IPAdapter wrapper creation...")
    
    try:
        from streamdiffusion.acceleration.tensorrt.ipadapter_wrapper import create_ipadapter_wrapper
        
        print("Loading UNet for wrapper test...")
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet",
            torch_dtype=torch.float32  # Use float32 like the wrapper does
        )
        
        print("Creating IPAdapter wrapper...")
        wrapped_unet = create_ipadapter_wrapper(unet, num_tokens=4)
        
        print(f"✓ Wrapper created successfully")
        print(f"  - Wrapper class: {wrapped_unet.__class__.__name__}")
        print(f"  - Number of tokens: {wrapped_unet.num_image_tokens}")
        print(f"  - Cross attention dim: {wrapped_unet.cross_attention_dim}")
        
        # Check what processors were actually installed
        installed_processors = wrapped_unet.unet.attn_processors
        cross_attention_count = len([n for n in installed_processors.keys() if 'attn2' in n])
        print(f"  - Cross-attention processors installed: {cross_attention_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debugging tests"""
    print("UNet Architecture Debugging Script")
    print("=" * 60)
    
    try:
        # Step 1: Inspect basic architecture
        unet, self_attn_names, cross_attn_names = inspect_unet_attention_processors()
        
        # Step 2: Test IPAdapter processor installation
        processor_test_passed = test_ipadapter_processor_installation(unet, cross_attn_names)
        
        # Step 3: Test wrapper creation
        wrapper_test_passed = test_wrapper_creation()
        
        # Summary
        print(f"\n{'='*60}")
        print("DEBUGGING SUMMARY:")
        print(f"✓ UNet loaded successfully")
        print(f"✓ Found {len(self_attn_names)} self-attention processors")
        print(f"✓ Found {len(cross_attn_names)} cross-attention processors")
        print(f"{'✓' if processor_test_passed else '❌'} IPAdapter processor creation test")
        print(f"{'✓' if wrapper_test_passed else '❌'} Wrapper creation test")
        
        if not processor_test_passed or not wrapper_test_passed:
            print(f"\n🔍 Issues detected that explain why 0 processors are installed!")
        else:
            print(f"\n🤔 All tests passed - need to investigate further...")
            
    except Exception as e:
        print(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 