#!/usr/bin/env python3
"""
build_trt_engine.py
===================
Simple helper script that compiles (or re-uses) TensorRT engines for a
StreamDiffusion model.  The resulting ``.engine`` files are placed inside
``--engine-dir`` ( default: ``./engines`` ).  The script can **optionally**
prepare the UNet for ControlNet by adding a dummy ``controlnets`` attribute so
that the builder includes the extra ControlNet inputs.

Example
-------
# Build engines for SD-1.5 with ControlNet support, using your ComfyUI model paths
python scripts/build_trt_engine.py \
       --model kohaku-v2.1.safetensors \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --controlnet-id control_v11p_sd15_canny.safetensors \
       --controlnet-path /path/to/your/ComfyUI/models/controlnet \
       --engine-dir ./trt_engines

# Or use HuggingFace repo for ControlNet (no --controlnet-path needed)
python scripts/build_trt_engine.py \
       --model kohaku-v2.1.safetensors \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --controlnet-id lllyasviel/control_v11p_sd15_canny \
       --engine-dir ./trt_engines

# Build engines without ControlNet support (slightly faster)
python scripts/build_trt_engine.py --model stabilityai/sd-turbo
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def build_engines(model: str,
                  engine_dir: Path,
                  width: int = 512,
                  height: int = 512,
                  batch_size: int = 1,
                  controlnet_id: str | None = None,
                  model_path: str | None = None,
                  controlnet_path: str | None = None) -> None:
    """Compile TensorRT engines for *model* and save them to *engine_dir*.

    If *controlnet_id* is given a **dummy** controlnets attribute will be
    appended to the StreamDiffusion instance so that the TensorRT builder
    generates UNet engines capable of accepting ControlNet inputs.
    
    Args:
        model: Model name/filename or HuggingFace model ID
        engine_dir: Directory to save engines
        width: Image width for optimization
        height: Image height for optimization  
        batch_size: Batch size for optimization
        controlnet_id: Optional ControlNet model ID/filename for ControlNet support
        model_path: Optional base path to combine with model filename
        controlnet_path: Optional base path to combine with ControlNet filename
    """
    device = "cuda"
    dtype = torch.float16

    # Determine the full model path
    if model_path is not None:
        # Combine model_path with model filename
        model_full_path = str(Path(model_path) / model)
        print(f"🖼️  Loading base model: {model} from {model_path}")
        print(f"📁 Full path: {model_full_path}")
    else:
        # Use model as-is (HuggingFace ID or full path)
        model_full_path = model
        print(f"🖼️  Loading base model: {model}")

    # 1.  Load diffusers pipeline
    try:
        # Try loading as local file first if it looks like a path
        if model_path is not None or model.endswith(('.safetensors', '.ckpt', '.pt')):
            pipe = StableDiffusionPipeline.from_single_file(
                model_full_path, 
                torch_dtype=dtype
            ).to(device)
        else:
            # Try as HuggingFace model ID
            pipe = StableDiffusionPipeline.from_pretrained(
                model_full_path, 
                torch_dtype=dtype
            ).to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure the model path exists or the HuggingFace model ID is correct")
        return

    # 2.  Wrap with StreamDiffusion (no ControlNet yet – we only need it for export)
    t_index_list = [32, 45]
    stream = StreamDiffusion(
        pipe=pipe,
        t_index_list=t_index_list,
        torch_dtype=dtype,
        width=width,
        height=height,
        frame_buffer_size=1,
        do_add_noise=True,
        use_denoising_batch=True,
        cfg_type="self",
    )

    # 3.  Hint to the builder that we want ControlNet-aware engines (if requested)
    if controlnet_id is not None:
        # Determine the full ControlNet path
        if controlnet_path is not None:
            # Combine controlnet_path with controlnet filename
            controlnet_full_path = str(Path(controlnet_path) / controlnet_id)
            print(f"🎛️  Enabling ControlNet support for: {controlnet_id} from {controlnet_path}")
            print(f"📁 Full ControlNet path: {controlnet_full_path}")
        else:
            # Use controlnet_id as-is (HuggingFace ID or full path)
            controlnet_full_path = controlnet_id
            print(f"🎛️  Enabling ControlNet support for: {controlnet_id}")
        
        # The builder only checks for the presence/length of this attribute
        # – the contents are irrelevant for compilation.
        setattr(stream, "controlnets", [controlnet_full_path])

    # 4.  Kick off engine build (skipped automatically if *.engine* already exists)
    print("🚀 Building TensorRT engines – this might take several minutes …")
    accelerate_with_tensorrt(
        stream,
        str(engine_dir),
        max_batch_size=batch_size,
        min_batch_size=batch_size,
        use_cuda_graph=False,
    )
    print("✅ Done.  Engines stored in", engine_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile TensorRT engines for StreamDiffusion models.")
    parser.add_argument("--model", type=str, required=False, default="runwayml/stable-diffusion-v1-5",
                        help="Model filename (e.g., kohaku-v2.1.safetensors) or HuggingFace model ID.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Base directory containing model files (e.g., /path/to/ComfyUI/models/checkpoints). "
                             "If provided, will be combined with --model filename.")
    parser.add_argument("--controlnet-id", type=str, default=None,
                        help="ControlNet filename (e.g., control_v11p_sd15_canny.safetensors) or HuggingFace repo ID. "
                             "If provided, compile UNet with ControlNet inputs.")
    parser.add_argument("--controlnet-path", type=str, default=None,
                        help="Base directory containing ControlNet files (e.g., /path/to/ComfyUI/models/controlnet). "
                             "If provided, will be combined with --controlnet-id filename.")
    parser.add_argument("--engine-dir", type=str, default="engines",
                        help="Directory where the resulting *.engine files will be placed.")
    parser.add_argument("--width", type=int, default=512, help="Image width used for optimisation.")
    parser.add_argument("--height", type=int, default=512, help="Image height used for optimisation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Optimised batch size (default: 1).")
    args = parser.parse_args()

    build_engines(
        model=args.model,
        engine_dir=Path(args.engine_dir),
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        controlnet_id=args.controlnet_id,
        model_path=args.model_path,
        controlnet_path=args.controlnet_path,
    ) 