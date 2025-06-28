#!/usr/bin/env python3
"""
Test script for IPAdapter TensorRT Phase 1 implementation
Tests basic components: wrapper creation, model detection, UNet extensions
"""

import torch
import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_model_detection():
    """Test model architecture detection for IPAdapter"""
    print("=== Testing Model Detection ===")
    
    try:
        from model_detection import detect_model_from_diffusers_unet
        from diffusers import UNet2DConditionModel
        
        # Test with a dummy UNet config (SD1.5-like)
        config = {
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": (320, 640, 1280, 1280),
            "cross_attention_dim": 768,
            "attention_head_dim": 8,
        }
        
        # Create a minimal UNet for testing
        print("Creating test UNet with SD1.5-like config...")
        unet = UNet2DConditionModel(**config)
        
        # Test detection
        model_type = detect_model_from_diffusers_unet(unet)
        cross_attention_dim = unet.config.cross_attention_dim
        
        print(f"Detected model type: {model_type}")
        print(f"Cross attention dim: {cross_attention_dim}")
        
        assert model_type == "SD15", f"Expected SD15, got {model_type}"
        assert cross_attention_dim == 768, f"Expected 768, got {cross_attention_dim}"
        
        print("Model detection test PASSED")
        return True
        
    except Exception as e:
        print(f"Model detection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ipadapter_wrapper():
    """Test IPAdapter wrapper creation"""
    print("\n=== Testing IPAdapter Wrapper ===")
    
    try:
        from ipadapter_wrapper import IPAdapterUNetWrapper, create_ipadapter_wrapper
        from diffusers import UNet2DConditionModel
        
        # Create test UNet
        config = {
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": (320, 640, 1280, 1280),
            "cross_attention_dim": 768,
            "attention_head_dim": 8,
        }
        
        print("Creating test UNet...")
        unet = UNet2DConditionModel(**config)
        
        # Test wrapper creation
        print("Creating IPAdapter wrapper...")
        wrapper = create_ipadapter_wrapper(unet)
        
        assert isinstance(wrapper, IPAdapterUNetWrapper), "Failed to create wrapper"
        assert wrapper.cross_attention_dim == 768, "Wrong cross attention dim"
        assert wrapper.num_image_tokens == 4, "Wrong token count"
        
        # Test forward pass shapes
        print("Testing wrapper forward pass...")
        batch_size = 2
        latent_height, latent_width = 64, 64
        text_maxlen = 77
        
        with torch.no_grad():
            sample = torch.randn(batch_size, 4, latent_height, latent_width)
            timestep = torch.ones(batch_size)
            encoder_hidden_states = torch.randn(batch_size, text_maxlen, 768)
            image_embeddings = torch.randn(batch_size, 4, 768)
            
            # Test forward pass
            output = wrapper(sample, timestep, encoder_hidden_states, image_embeddings)
            
            assert output[0].shape == sample.shape, f"Wrong output shape: {output[0].shape}"
            
        print("IPAdapter wrapper test PASSED")
        return True
        
    except Exception as e:
        print(f"IPAdapter wrapper test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_model_extensions():
    """Test UNet model class extensions for IPAdapter"""
    print("\n=== Testing UNet Model Extensions ===")
    
    try:
        from models import UNet
        
        # Test UNet creation with IPAdapter support
        print("Creating UNet model with IPAdapter support...")
        unet_model = UNet(
            fp16=True,
            device="cpu",  # Use CPU for testing
            max_batch=2,
            min_batch_size=1,
            embedding_dim=768,
            use_ipadapter=True,
        )
        
        assert unet_model.use_ipadapter == True, "IPAdapter not enabled"
        assert unet_model.num_image_tokens == 4, "Wrong token count"
        
        # Test input names
        input_names = unet_model.get_input_names()
        expected_names = ["sample", "timestep", "encoder_hidden_states", "image_embeddings"]
        assert input_names == expected_names, f"Wrong input names: {input_names}"
        
        # Test input profile
        profile = unet_model.get_input_profile(1, 512, 512, static_batch=True, static_shape=True)
        assert "image_embeddings" in profile, "Missing image_embeddings in profile"
        
        expected_shape = (1, 4, 768)
        actual_shape = profile["image_embeddings"][1]  # opt shape
        assert actual_shape == expected_shape, f"Wrong image embeddings shape: {actual_shape}"
        
        # Test sample input
        sample_input = unet_model.get_sample_input(1, 512, 512)
        assert len(sample_input) == 4, f"Wrong number of inputs: {len(sample_input)}"
        
        image_embeds = sample_input[3]  # image_embeddings should be last
        assert image_embeds.shape == (2, 4, 768), f"Wrong image embedding shape: {image_embeds.shape}"
        
        print("UNet model extensions test PASSED")
        return True
        
    except Exception as e:
        print(f"UNet model extensions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 1 tests"""
    print("Starting IPAdapter TensorRT Phase 1 validation tests...\n")
    
    tests = [
        test_model_detection,
        test_ipadapter_wrapper,
        test_unet_model_extensions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All Phase 1 tests PASSED! ✓")
        print("\nPhase 1 foundation is ready for Phase 2 implementation.")
        return True
    else:
        print("Some tests FAILED! ✗")
        print("\nPlease fix issues before proceeding to Phase 2.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 