# StreamDiffusion TensorRT + ControlNet Scripts

This directory contains scripts to build TensorRT engines and run ControlNet inference with your own model paths, overriding the hardcoded Windows paths in Ryan's configs.

## Quick Start

### 1. Build TensorRT Engines

First, compile the TensorRT engines for your model:

```bash
# Build engines for a specific model with ControlNet support (local ControlNet)
python scripts/build_trt_engine.py \
       --model kohaku-v2.1.safetensors \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --controlnet-id control_v11p_sd15_canny.safetensors \
       --controlnet-path /path/to/your/ComfyUI/models/controlnet \
       --engine-dir ./my_engines

# Or use HuggingFace ControlNet (no --controlnet-path needed)
python scripts/build_trt_engine.py \
       --model kohaku-v2.1.safetensors \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --controlnet-id lllyasviel/control_v11p_sd15_canny \
       --engine-dir ./my_engines

# Or build without ControlNet support (faster, smaller engines)
python scripts/build_trt_engine.py \
       --model sd_turbo.safetensors \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --engine-dir ./my_engines
```

### 2. Run ControlNet Inference

Then use the existing configs but override the model paths:

```bash
# Use Ryan's config but with your ComfyUI model paths
python scripts/controlnet_infer.py \
       --config configs/controlnet_examples/canny_example.yaml \
       --input /path/to/your/input/image.jpg \
       --output generated_image.png \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --controlnet-path /path/to/your/ComfyUI/models/controlnet \
       --engine-dir ./my_engines
```

The inference script will automatically reuse existing engines if they match the config parameters, or build new ones if needed.

### 3. Optional: Check Engine Status

If you want to verify which engines exist:

```bash
# Check if engine filenames match expectations
python scripts/check_engines.py \
       --config configs/controlnet_examples/canny_example.yaml \
       --model-path /path/to/your/ComfyUI/models/checkpoints \
       --engine-dir ./my_engines
```

## How Path Override Works

The scripts automatically extract the model filenames from Ryan's configs and combine them with your provided paths:

**Example**: If the config contains:
```yaml
model_id: "C:\\_dev\\comfy\\ComfyUI\\models\\checkpoints\\kohaku-v2.1.safetensors"
```

And you run with `--model-path /home/user/ComfyUI/models/checkpoints`, it becomes:
```
/home/user/ComfyUI/models/checkpoints/kohaku-v2.1.safetensors
```

The same logic applies to ControlNet paths - local file paths get overridden, but HuggingFace repo names are left unchanged.

## Engine Reuse 

**✅ Fixed**: The inference script now uses the exact same parameters from your config file that were used during engine building. This means:

- **Engines are automatically reused** if they match your config parameters
- **No more rebuilding from scratch** unless you change model or parameters
- **Faster subsequent runs** since engines are loaded instead of compiled

The engine paths are constructed from:
- Model name (e.g., `kohaku-v2.1`)
- `use_lcm_lora` setting from config
- `use_tiny_vae` setting from config  
- `acceleration` method from config
- Batch sizes calculated from `t_index_list` and other parameters
- `cfg_type` from config

Since the inference script now reads all these from your config file, the paths match exactly with what was built.

## Available Configs

The repository includes several pre-made configs in `configs/controlnet_examples/`:

- `canny_example.yaml` - SD 1.5 with Canny edge detection
- `sdturbo_canny_example.yaml` - SD Turbo with Canny 
- `depth_trt_example.yaml` - SD 1.5 with depth estimation
- `sdturbo_depth_trt_example.yaml` - SD Turbo with depth
- `lineart_example.yaml` - SD 1.5 with line art detection
- `multi_controlnet_example.yaml` - Multiple ControlNets at once

## Script Options

### build_trt_engine.py

```
--model              Model filename or HuggingFace ID (default: runwayml/stable-diffusion-v1-5)
--model-path         Base directory for model files (optional)
--controlnet-id      ControlNet filename or HuggingFace repo ID (optional)
--controlnet-path    Base directory for ControlNet files (optional)
--engine-dir         Where to save engines (default: engines)
--width              Image width for optimization (default: 512)
--height             Image height for optimization (default: 512)
--batch-size         Batch size for optimization (default: 1)
```

### controlnet_infer.py

```
--config             YAML/JSON ControlNet configuration file (required)
--input              Input image path (required)
--output             Output image path (default: output.png)
--model-path         Override base model directory (optional)
--controlnet-path    Override ControlNet model directory (optional)
--engine-dir         TensorRT engines directory (default: engines)
```

### check_engines.py

```
--config             YAML/JSON ControlNet configuration file (required)
--model-path         Override base model directory (optional)
--engine-dir         TensorRT engines directory (default: engines)
```

## Notes

- **TensorRT engines are model-specific**: You need to rebuild engines if you switch to a different base model
- **Config consistency**: The inference script uses parameters from your config file to ensure engine paths match exactly
- **HuggingFace vs Local**: If using HuggingFace ControlNet repos (like `lllyasviel/control_v11p_sd15_canny`), you don't need `--controlnet-path`
- **Engine debugging**: Use `check_engines.py` if you want to verify which engines exist and what build commands are needed 