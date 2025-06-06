#!/usr/bin/env python3
"""
controlnet_infer.py
===================
Run a ControlNet-enabled StreamDiffusion pipeline **accelerated by TensorRT**
on a single static input image and save the result.

This script can override the hardcoded model paths in Ryan's configs with your
own ComfyUI model paths while preserving the model filenames.

Requirements
------------
The TensorRT engines for the specified base model must already exist.  They can
be generated with ``build_trt_engine.py``.

Example
-------
# Use your ComfyUI paths, overriding the hardcoded Windows paths in the config
python scripts/controlnet_infer.py \
       --config configs/controlnet_examples/canny_example.yaml \
       --input  path/to/photo.jpg \
       --output result.png \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --controlnet-path /path/to/your/ComfyUI/models/controlnet \
       --engine-dir ./trt_engines
"""

import argparse
from pathlib import Path
from dataclasses import asdict
import os
import sys

# Add the parent directory to Python path to find utils module
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import torch
from PIL import Image

from streamdiffusion.controlnet import load_controlnet_config
from utils.wrapper import StreamDiffusionWrapper  # Wrapper provides convenient TRTEngine loading


def override_model_paths(cfg, model_path=None, controlnet_path=None):
    """Override model paths in config while preserving filenames"""
    
    # Override base model path
    if model_path is not None:
        original_model = cfg.model_id
        # Extract just the filename from the original path - handle both Windows and Unix separators
        model_filename = os.path.basename(original_model.replace('\\', '/'))
        # Combine with new base path
        cfg.model_id = str(Path(model_path) / model_filename)
        print(f"🔄 Overriding base model path:")
        print(f"   Original: {original_model}")
        print(f"   New: {cfg.model_id}")
    
    # Override ControlNet paths
    if controlnet_path is not None:
        for i, cn in enumerate(cfg.controlnets):
            original_model = cn.model_id
            
            # Check if this looks like a local path (has file separators)
            if ('/' in original_model and not original_model.startswith('http')) or ('\\' in original_model):
                # Extract filename and combine with new path - handle both Windows and Unix separators
                model_filename = os.path.basename(original_model.replace('\\', '/'))
                cn.model_id = str(Path(controlnet_path) / model_filename)
                print(f"🔄 Overriding ControlNet {i} path:")
                print(f"   Original: {original_model}")
                print(f"   New: {cn.model_id}")
            else:
                # Looks like a HuggingFace repo name, leave as-is
                print(f"ℹ️  ControlNet {i} appears to be HuggingFace repo, keeping as-is: {original_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ControlNet StreamDiffusion inference with TensorRT acceleration.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON ControlNet configuration file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image.")
    parser.add_argument("--output", type=str, default="output.png", help="Where to save the generated image.")
    parser.add_argument("--engine-dir", type=str, default="engines", help="Directory that contains the compiled TensorRT engines.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Base directory containing model files (e.g., /path/to/ComfyUI/models/checkpoints). "
                             "Overrides the base model path in config while keeping the filename.")
    parser.add_argument("--controlnet-path", type=str, default=None,
                        help="Base directory containing ControlNet model files (e.g., /path/to/ComfyUI/models/controlnet). "
                             "Overrides ControlNet paths in config while keeping filenames.")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 1.  Load high-level configuration
    # ---------------------------------------------------------------------
    cfg = load_controlnet_config(args.config)
    
    # ---------------------------------------------------------------------
    # 2.  Override model paths if provided
    # ---------------------------------------------------------------------
    override_model_paths(cfg, args.model_path, args.controlnet_path)

    # ---------------------------------------------------------------------
    # 3.  Convert ControlNetConfig dataclasses into plain dictionaries –
    #     this is the format expected by *StreamDiffusionWrapper*.
    # ---------------------------------------------------------------------
    controlnet_cfg_dicts = []
    for cn in cfg.controlnets:
        cn_dict = asdict(cn)
        cn_dict["pipeline_type"] = cfg.pipeline_type  # keep track of which pipeline to use
        controlnet_cfg_dicts.append(cn_dict)

    # ---------------------------------------------------------------------
    # 4.  Build / load the pipeline (TensorRT engines are re-used if present)
    #     *** USE ALL CONFIG PARAMETERS TO MATCH ENGINE BUILDING ***
    # ---------------------------------------------------------------------
    print(f"🚀 Creating StreamDiffusion pipeline...")
    print(f"📝 Base model: {cfg.model_id}")
    print(f"📝 Pipeline type: {cfg.pipeline_type}")
    print(f"📝 ControlNets: {len(cfg.controlnets)}")
    print(f"📝 Key parameters from config:")
    print(f"   use_lcm_lora: {getattr(cfg, 'use_lcm_lora', True)}")
    print(f"   use_tiny_vae: {getattr(cfg, 'use_tiny_vae', True)}")
    print(f"   acceleration: {getattr(cfg, 'acceleration', 'tensorrt')}")
    print(f"   cfg_type: {getattr(cfg, 'cfg_type', 'self')}")
    print(f"   t_index_list: {cfg.t_index_list}")
    
    # Parse dtype from config string to torch dtype
    dtype_str = getattr(cfg, 'dtype', 'float16')
    if dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'float32':
        dtype = torch.float32
    else:
        dtype = torch.float16  # default fallback
    
    try:
        wrapper = StreamDiffusionWrapper(
            model_id_or_path=cfg.model_id,
            t_index_list=cfg.t_index_list,
            mode="img2img",  # ControlNet inference is always img2img
            width=cfg.width,
            height=cfg.height,
            device=getattr(cfg, 'device', 'cuda'),
            dtype=dtype,
            # *** Use config parameters instead of hardcoded defaults ***
            acceleration=getattr(cfg, 'acceleration', 'tensorrt'),
            use_lcm_lora=getattr(cfg, 'use_lcm_lora', True),
            use_tiny_vae=getattr(cfg, 'use_tiny_vae', True),
            cfg_type=getattr(cfg, 'cfg_type', 'self'),
            seed=getattr(cfg, 'seed', 2),
            use_denoising_batch=True,  # Required for img2img mode
            frame_buffer_size=1,  # Single image inference
            warmup=10,
            do_add_noise=True,
            engine_dir=args.engine_dir,
            use_controlnet=True,
            controlnet_config=controlnet_cfg_dicts,
        )
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        print("💡 Make sure:")
        print("   - Model files exist at the specified paths")
        print("   - TensorRT engines are built (use build_trt_engine.py)")
        print("   - ControlNet models are accessible")
        print("   - Config parameters match what was used during engine building")
        return

    # Prepare with the supplied prompt / generation settings
    wrapper.prepare(
        prompt=cfg.prompt or "",
        negative_prompt=cfg.negative_prompt or "",
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
    )

    # ---------------------------------------------------------------------
    # 5.  Load the input image and feed it into the pipeline
    # ---------------------------------------------------------------------
    print(f"📷 Loading input image: {args.input}")
    input_image = Image.open(args.input).convert("RGB").resize((cfg.width, cfg.height))

    # Update ControlNet(s) with the image (efficient batch version)
    wrapper.update_control_image_efficient(input_image)

    # Run inference
    print("🚀 Generating …")
    output = wrapper(input_image)

    # *wrapper* may return either a single PIL image or a list depending on
    # internal settings – normalise that here.
    output_image = output[0] if isinstance(output, list) else output

    # ---------------------------------------------------------------------
    # 6.  Save the result
    # ---------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(out_path)
    print("✅ Done.  Saved to", out_path)


if __name__ == "__main__":
    main() 