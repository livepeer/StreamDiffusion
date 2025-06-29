#!/usr/bin/env python3
"""
Simple test to verify IPAdapter model resolution works correctly.

This test only checks if:
1. HuggingFace model resolution works
2. Model paths are correctly downloaded

This is much faster than testing the full pipeline.
"""

import os
import sys
import torch
from pathlib import Path

# Add paths to import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_model_resolution():
    """Test that IPAdapter model resolution works correctly."""
    
    print("=" * 60)
    print("IPAdapter Model Resolution Test")
    print("=" * 60)
    
    try:
        from utils.wrapper import StreamDiffusionWrapper
        
        # Create a minimal wrapper instance just for testing model resolution
        wrapper = StreamDiffusionWrapper.__new__(StreamDiffusionWrapper)
        wrapper.device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper.dtype = torch.float16
        
        print("Test 1: Testing IPAdapter model resolution...")
        
        # Test resolving IPAdapter model
        ipadapter_path = wrapper._resolve_ipadapter_model_path("h94/IP-Adapter", "ipadapter")
        print(f"✓ IPAdapter model resolved to: {ipadapter_path}")
        
        # Verify the file exists
        if os.path.exists(ipadapter_path):
            print(f"✓ IPAdapter model file exists at: {ipadapter_path}")
        else:
            print(f"❌ IPAdapter model file not found at: {ipadapter_path}")
            return False
        
        print("\nTest 2: Testing image encoder resolution...")
        
        # Test resolving image encoder
        encoder_path = wrapper._resolve_ipadapter_model_path("h94/IP-Adapter", "image_encoder")
        print(f"✓ Image encoder resolved to: {encoder_path}")
        
        # Verify the directory exists
        if os.path.exists(encoder_path) and os.path.isdir(encoder_path):
            print(f"✓ Image encoder directory exists at: {encoder_path}")
        else:
            print(f"❌ Image encoder directory not found at: {encoder_path}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ Model resolution test passed!")
        print("IPAdapter models can be downloaded and resolved correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Model resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run model resolution test."""
    
    success = test_model_resolution()
    
    if success:
        print("\n🎉 Model resolution works correctly!")
        print("Now you can test the full pipeline with confidence.")
    else:
        print("\n❌ Model resolution failed.")
        print("Fix the model resolution issues before testing the full pipeline.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 