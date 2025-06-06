#!/usr/bin/env python3
"""
check_engines.py
================
Check if your TensorRT engine filenames match what the inference script expects.

This helps debug the "rebuilding from scratch" issue by showing exactly what
engine filenames are expected based on your configuration.

Example
-------
python scripts/check_engines.py \
       --config configs/controlnet_examples/sdturbo_canny_example.yaml \
       --model-path /workspace/ComfyUI/models/checkpoints/ \
       --engine-dir ./engines
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to Python path to find utils module
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from streamdiffusion.controlnet import load_controlnet_config


def check_engine_paths(config_path, model_path=None, engine_dir="engines"):
    """
    Check what engine paths are expected vs what exists
    """
    # Load config and override paths
    cfg = load_controlnet_config(config_path)
    
    # Override model path if provided
    if model_path is not None:
        original_model = cfg.model_id
        model_filename = os.path.basename(original_model.replace('\\', '/'))
        cfg.model_id = str(Path(model_path) / model_filename)
        print(f"🔄 Using model: {cfg.model_id}")
    
    # Simulate the engine path construction logic from StreamDiffusionWrapper
    model_path_obj = Path(cfg.model_id)
    model_stem = model_path_obj.stem if model_path_obj.exists() else cfg.model_id
    
    # Default parameters that match StreamDiffusionWrapper defaults
    use_lcm_lora = getattr(cfg, 'use_lcm_lora', True)
    use_tiny_vae = getattr(cfg, 'use_tiny_vae', True)
    mode = "img2img"  # Default mode used by controlnet_infer.py
    
    # Calculate batch sizes (these come from StreamDiffusion internals)
    frame_buffer_size = 1  # Default from controlnet_infer.py
    t_index_list = cfg.t_index_list
    use_denoising_batch = True
    denoising_steps_num = len(t_index_list)
    
    # This matches the logic in StreamDiffusion.__init__
    if use_denoising_batch:
        batch_size = denoising_steps_num * frame_buffer_size
        trt_unet_batch_size = denoising_steps_num * frame_buffer_size
    else:
        trt_unet_batch_size = frame_buffer_size
        batch_size = frame_buffer_size
    
    # Account for CFG - this logic is from StreamDiffusion
    if cfg.guidance_scale > 1.0:
        if cfg.cfg_type == "initialize":
            trt_unet_batch_size = (denoising_steps_num + 1) * frame_buffer_size
        elif cfg.cfg_type == "full":
            trt_unet_batch_size = 2 * denoising_steps_num * frame_buffer_size
    
    print(f"🔍 Calculated batch sizes:")
    print(f"   UNet batch: {trt_unet_batch_size}")
    print(f"   VAE batch: {batch_size}")
    print(f"   Frame buffer: {frame_buffer_size}")
    print(f"   Denoising steps: {denoising_steps_num}")
    print(f"   CFG type: {cfg.cfg_type}")
    print(f"   Guidance scale: {cfg.guidance_scale}")
    
    # Construct expected engine prefixes (matching create_prefix logic)
    unet_prefix = f"{model_stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{trt_unet_batch_size}--min_batch-{trt_unet_batch_size}--mode-{mode}"
    vae_prefix = f"{model_stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{batch_size}--min_batch-{batch_size}--mode-{mode}"
    
    # Construct full paths
    engine_dir_path = Path(engine_dir)
    unet_path = engine_dir_path / unet_prefix / "unet.engine"
    vae_encoder_path = engine_dir_path / vae_prefix / "vae_encoder.engine"
    vae_decoder_path = engine_dir_path / vae_prefix / "vae_decoder.engine"
    
    print(f"\n🎯 Expected engine paths:")
    print(f"   UNet: {unet_path}")
    print(f"   VAE Encoder: {vae_encoder_path}")
    print(f"   VAE Decoder: {vae_decoder_path}")
    
    # Check if they exist
    print(f"\n✅ Engine status:")
    unet_exists = unet_path.exists()
    vae_enc_exists = vae_encoder_path.exists()
    vae_dec_exists = vae_decoder_path.exists()
    
    print(f"   UNet: {'✅ EXISTS' if unet_exists else '❌ MISSING'}")
    print(f"   VAE Encoder: {'✅ EXISTS' if vae_enc_exists else '❌ MISSING'}")
    print(f"   VAE Decoder: {'✅ EXISTS' if vae_dec_exists else '❌ MISSING'}")
    
    if all([unet_exists, vae_enc_exists, vae_dec_exists]):
        print(f"\n🎉 All engines found! Inference should use existing engines.")
    else:
        print(f"\n⚠️  Missing engines will be rebuilt from scratch during inference.")
        print(f"\n💡 To build these engines, run:")
        print(f"   python scripts/build_trt_engine.py \\")
        print(f"          --model {model_path_obj.name} \\")
        if model_path:
            print(f"          --model-path {model_path} \\")
        if cfg.controlnets:
            print(f"          --controlnet-id {cfg.controlnets[0].model_id} \\")
        print(f"          --engine-dir {engine_dir}")
    
    # List what actually exists in the engine directory
    print(f"\n📁 What's actually in {engine_dir}:")
    if engine_dir_path.exists():
        for item in engine_dir_path.iterdir():
            if item.is_dir():
                engines_in_dir = list(item.glob("*.engine"))
                print(f"   {item.name}/ ({len(engines_in_dir)} engines)")
                for engine in engines_in_dir:
                    print(f"      {engine.name}")
    else:
        print(f"   Directory doesn't exist yet")


def main():
    parser = argparse.ArgumentParser(description="Check if TensorRT engine filenames match expectations")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to ControlNet configuration file")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override base model directory")
    parser.add_argument("--engine-dir", type=str, default="engines",
                       help="TensorRT engines directory")
    
    args = parser.parse_args()
    
    print(f"🔍 Checking engine paths for config: {args.config}")
    check_engine_paths(args.config, args.model_path, args.engine_dir)


if __name__ == "__main__":
    main() 