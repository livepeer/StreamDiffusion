#!/usr/bin/env python3
"""
IP Adapter Architecture Inspector

This script loads and inspects the IP Adapter model to understand:
1. Model structure and layer dimensions
2. CLIP vision model dimensions
3. Expected input/output shapes
4. Model type detection (Plus vs Standard)
"""

import os
import torch
from safetensors.torch import load_file
from transformers import CLIPVisionModel, CLIPImageProcessor
from pathlib import Path

def inspect_safetensors_structure(file_path):
    """Inspect the structure of a safetensors file"""
    print(f"Loading safetensors from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return None
    
    try:
        state_dict = load_file(file_path)
        print(f"✓ Successfully loaded {len(state_dict)} tensors")
        return state_dict
    except Exception as e:
        print(f"ERROR loading safetensors: {e}")
        return None

def analyze_ip_adapter_structure(state_dict):
    """Analyze IP Adapter model structure"""
    print("\n" + "="*60)
    print("IP ADAPTER MODEL ANALYSIS")
    print("="*60)
    
    # Separate image_proj and ip_adapter keys
    image_proj_keys = [k for k in state_dict.keys() if k.startswith("image_proj")]
    ip_adapter_keys = [k for k in state_dict.keys() if k.startswith("ip_adapter")]
    
    print(f"Image Projection Keys: {len(image_proj_keys)}")
    print(f"IP Adapter Keys: {len(ip_adapter_keys)}")
    
    # Detect model type
    is_plus = any("latents" in k for k in image_proj_keys)
    is_full = "image_proj.proj.3.weight" in state_dict
    has_perceiver = "image_proj.perceiver_resampler.proj_in.weight" in state_dict
    
    print(f"\nModel Type Detection:")
    print(f"  Is Plus Model: {is_plus}")
    print(f"  Is Full Model: {is_full}")
    print(f"  Has Perceiver Resampler: {has_perceiver}")
    
    # Analyze image projection structure
    print(f"\nImage Projection Structure:")
    for key in sorted(image_proj_keys):
        tensor = state_dict[key]
        print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    
    # Analyze IP adapter structure (show first few)
    print(f"\nIP Adapter Structure (first 10 keys):")
    for key in sorted(ip_adapter_keys)[:10]:
        tensor = state_dict[key]
        print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    
    if len(ip_adapter_keys) > 10:
        print(f"  ... and {len(ip_adapter_keys) - 10} more keys")
    
    # Key dimension analysis
    print(f"\nKey Dimension Analysis:")
    
    # Check proj_in dimensions (if it exists)
    proj_in_key = "image_proj.proj_in.weight"
    if proj_in_key in state_dict:
        proj_in_shape = state_dict[proj_in_key].shape
        print(f"  proj_in: {proj_in_shape} -> embedding_dim={proj_in_shape[1]}, dim={proj_in_shape[0]}")
    
    # Check latents dimensions (if it exists)
    latents_key = "image_proj.latents"
    if latents_key in state_dict:
        latents_shape = state_dict[latents_key].shape
        print(f"  latents: {latents_shape} -> num_queries={latents_shape[1]}, dim={latents_shape[2]}")
    
    # Check proj_out dimensions (if it exists)
    proj_out_key = "image_proj.proj_out.weight"
    if proj_out_key in state_dict:
        proj_out_shape = state_dict[proj_out_key].shape
        print(f"  proj_out: {proj_out_shape} -> output_dim={proj_out_shape[0]}, dim={proj_out_shape[1]}")
    
    # Check a K/V layer
    k_keys = [k for k in ip_adapter_keys if "to_k_ip" in k]
    v_keys = [k for k in ip_adapter_keys if "to_v_ip" in k]
    
    if k_keys:
        first_k_key = sorted(k_keys)[0]
        k_shape = state_dict[first_k_key].shape
        print(f"  First K layer: {first_k_key}: {k_shape}")
    
    if v_keys:
        first_v_key = sorted(v_keys)[0]
        v_shape = state_dict[first_v_key].shape
        print(f"  First V layer: {first_v_key}: {v_shape}")
    
    return {
        'is_plus': is_plus,
        'is_full': is_full,
        'has_perceiver': has_perceiver,
        'image_proj_keys': image_proj_keys,
        'ip_adapter_keys': ip_adapter_keys,
        'state_dict': state_dict
    }

def analyze_clip_vision_model():
    """Analyze CLIP vision model dimensions"""
    print("\n" + "="*60)
    print("CLIP VISION MODEL ANALYSIS")
    print("="*60)
    
    try:
        print("Loading CLIP-ViT-Large-14...")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        print(f"✓ Model loaded successfully")
        
        # Check model config
        config = model.config
        print(f"\nModel Configuration:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  intermediate_size: {config.intermediate_size}")
        print(f"  num_hidden_layers: {config.num_hidden_layers}")
        print(f"  num_attention_heads: {config.num_attention_heads}")
        print(f"  image_size: {config.image_size}")
        print(f"  patch_size: {config.patch_size}")
        print(f"  projection_dim: {config.projection_dim}")
        
        # Test with dummy input
        dummy_image = torch.randn(1, 3, 224, 224)
        
        model.config.output_hidden_states = True
        with torch.no_grad():
            outputs = model(pixel_values=dummy_image)
        
        print(f"\nOutput Shapes:")
        print(f"  pooler_output: {outputs.pooler_output.shape}")
        if hasattr(outputs, 'last_hidden_state'):
            print(f"  last_hidden_state: {outputs.last_hidden_state.shape}")
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            print(f"  num_hidden_states: {len(outputs.hidden_states)}")
            print(f"  final_hidden_state: {outputs.hidden_states[-1].shape}")
            print(f"  penultimate_hidden_state: {outputs.hidden_states[-2].shape}")
        
        return {
            'hidden_size': config.hidden_size,
            'projection_dim': config.projection_dim,
            'pooler_output_shape': outputs.pooler_output.shape,
            'penultimate_shape': outputs.hidden_states[-2].shape if outputs.hidden_states else None
        }
        
    except Exception as e:
        print(f"ERROR loading CLIP model: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_config_recommendations(ip_analysis, clip_analysis):
    """Generate configuration recommendations based on analysis"""
    print("\n" + "="*60)
    print("CONFIGURATION RECOMMENDATIONS")
    print("="*60)
    
    if not ip_analysis or not clip_analysis:
        print("Cannot generate recommendations due to missing analysis data")
        return
    
    state_dict = ip_analysis['state_dict']
    
    print("Based on the analysis, here are the recommended configurations:")
    print()
    
    # Determine embedding dimension
    if "image_proj.proj_in.weight" in state_dict:
        embedding_dim = state_dict["image_proj.proj_in.weight"].shape[1]
        print(f"✓ embedding_dim = {embedding_dim}")
    else:
        embedding_dim = clip_analysis['hidden_size']
        print(f"? embedding_dim = {embedding_dim} (from CLIP config)")
    
    # Determine cross attention dimension
    if "image_proj.proj_out.weight" in state_dict:
        output_dim = state_dict["image_proj.proj_out.weight"].shape[0]
        print(f"✓ output_dim = {output_dim}")
    else:
        output_dim = 768  # Default for SD 1.5
        print(f"? output_dim = {output_dim} (default for SD 1.5)")
    
    # Determine internal dimension
    if "image_proj.proj_in.weight" in state_dict:
        dim = state_dict["image_proj.proj_in.weight"].shape[0]
        print(f"✓ dim = {dim}")
    else:
        dim = output_dim
        print(f"? dim = {dim} (same as output_dim)")
    
    # Determine number of queries
    if "image_proj.latents" in state_dict:
        num_queries = state_dict["image_proj.latents"].shape[1]
        print(f"✓ num_queries = {num_queries}")
    else:
        num_queries = 4  # Default for standard models
        print(f"? num_queries = {num_queries} (default)")
    
    # Determine model architecture
    if ip_analysis['is_plus']:
        print(f"\n✓ Model Type: IP Adapter Plus")
        print(f"  -> Use Resampler architecture")
        print(f"  -> Use penultimate hidden states from CLIP")
    else:
        print(f"\n✓ Model Type: IP Adapter Standard")
        print(f"  -> Use ImageProjModel architecture")
        print(f"  -> Use pooler output from CLIP")
    
    print(f"\nRecommended Resampler Configuration:")
    print(f"```python")
    print(f"self.image_proj_model = Resampler(")
    print(f"    dim={dim},")
    print(f"    depth=4,")
    print(f"    dim_head=64,")
    print(f"    heads={dim//64},")
    print(f"    num_queries={num_queries},")
    print(f"    embedding_dim={embedding_dim},")
    print(f"    output_dim={output_dim},")
    print(f"    ff_mult=4")
    print(f")")
    print(f"```")
    
    print(f"\nCLIP Feature Extraction:")
    if ip_analysis['is_plus']:
        print(f"```python")
        print(f"# For Plus models, use penultimate hidden states")
        print(f"clip_features = image_outputs.hidden_states[-2]  # Shape: {clip_analysis['penultimate_shape']}")
        print(f"```")
    else:
        print(f"```python")
        print(f"# For standard models, use pooler output")
        print(f"clip_features = image_outputs.pooler_output.unsqueeze(1)  # Shape: {clip_analysis['pooler_output_shape']}")
        print(f"```")

def main():
    """Main inspection function"""
    print("IP Adapter Architecture Inspector")
    print("="*60)
    
    # Configuration
    IP_ADAPTER_PATH = r"C:\_dev\comfy\ComfyUI\models\ipadapter\ip-adapter-plus_sd15.safetensors"
    
    print(f"Target IP Adapter: {IP_ADAPTER_PATH}")
    
    # Inspect IP Adapter
    state_dict = inspect_safetensors_structure(IP_ADAPTER_PATH)
    if not state_dict:
        return
    
    ip_analysis = analyze_ip_adapter_structure(state_dict)
    
    # Inspect CLIP Vision Model
    clip_analysis = analyze_clip_vision_model()
    
    # Generate recommendations
    generate_config_recommendations(ip_analysis, clip_analysis)
    
    print(f"\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    print("Use the recommendations above to configure your IP Adapter integration.")

if __name__ == "__main__":
    main() 