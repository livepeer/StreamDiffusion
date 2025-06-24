# StreamDiffusion Configuration System

The StreamDiffusion configuration system provides a simple, elegant way to configure StreamDiffusion pipelines using YAML or JSON files. This system works for both general StreamDiffusion usage and ControlNet-enabled pipelines.

## Overview

Instead of hardcoding parameters in your scripts, you can now:
- Store all StreamDiffusion parameters in config files
- Share configurations easily across projects
- Override parameters at runtime
- Support both ControlNet and non-ControlNet workflows

## Quick Start

### Basic Usage (Non-ControlNet)

```python
from streamdiffusion import load_config, create_wrapper_from_config

# Load configuration from file
config = load_config("my_config.yaml")

# Create wrapper directly from config
wrapper = create_wrapper_from_config(config)

# Generate image (txt2img or img2img based on config)
output_image = wrapper()
```

### Runtime Parameter Overrides

```python
# Override any config parameters at runtime
wrapper = create_wrapper_from_config(
    config,
    prompt="a different prompt",
    seed=999,
    width=768,
    height=768
)
```

### ControlNet Usage (Backward Compatible)

```python
# Works exactly the same for ControlNet configs
config = load_config("controlnet_config.yaml")
wrapper = create_wrapper_from_config(config)

# ControlNet-specific methods still work
wrapper.update_control_image_efficient(input_image)
output_image = wrapper(input_image)
```

## Configuration Format

### Basic Configuration

```yaml
# Model and pipeline
model_id: "stabilityai/sd-turbo"
pipeline_type: "sdturbo"

# Generation mode
mode: "txt2img"  # or "img2img"
output_type: "pil"

# Image dimensions
width: 512
height: 512

# Device settings
device: "cuda"
dtype: "float16"

# Generation parameters
prompt: "a beautiful landscape"
negative_prompt: "blurry, low quality"
guidance_scale: 1.0
num_inference_steps: 4

# StreamDiffusion settings
t_index_list: [0, 1, 2, 3]
frame_buffer_size: 1
use_denoising_batch: false

# Performance settings
acceleration: "tensorrt"
use_lcm_lora: false
use_tiny_vae: true
cfg_type: "none"
seed: 42
```

### ControlNet Configuration

```yaml
# Basic model settings (same as above)
model_id: "stabilityai/sd-turbo"
mode: "img2img"
# ... other settings ...

# ControlNet configuration
controlnets:
  - model_id: "lllyasviel/control_v11p_sd15_canny"
    preprocessor: "canny"
    conditioning_scale: 1.0
    preprocessor_params:
      low_threshold: 100
      high_threshold: 200
    enabled: true
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | *required* | Model ID or local path |
| `pipeline_type` | str | `"sd1.5"` | Pipeline type: `"sd1.5"`, `"sdturbo"`, `"sdxlturbo"` |
| `mode` | str | `"img2img"` | Generation mode: `"img2img"` or `"txt2img"` |
| `width` | int | `512` | Output image width |
| `height` | int | `512` | Output image height |
| `device` | str | `"cuda"` | Device: `"cuda"` or `"cpu"` |
| `dtype` | str | `"float16"` | Data type: `"float16"` or `"float32"` |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | `""` | Generation prompt |
| `negative_prompt` | str | `""` | Negative prompt |
| `guidance_scale` | float | `1.2` | CFG guidance scale |
| `num_inference_steps` | int | `50` | Number of denoising steps |
| `seed` | int | `2` | Random seed for reproducibility |

### StreamDiffusion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `t_index_list` | List[int] | `[0, 16, 32, 45]` | Timestep indices for denoising |
| `frame_buffer_size` | int | `1` | Frame buffer size for batch processing |
| `use_denoising_batch` | bool | `true` | Enable denoising batch |
| `cfg_type` | str | `"self"` | CFG type: `"none"`, `"full"`, `"self"`, `"initialize"` |
| `delta` | float | `1.0` | Virtual residual noise multiplier |

### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `acceleration` | str | `"tensorrt"` | Acceleration: `"none"`, `"xformers"`, `"tensorrt"` |
| `use_lcm_lora` | bool | `true` | Use LCM LoRA |
| `use_tiny_vae` | bool | `true` | Use TinyVAE for speed |
| `warmup` | int | `10` | Number of warmup steps |
| `engine_dir` | str | `"engines"` | TensorRT engine directory |

### ControlNet Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `controlnets` | List[Dict] | `[]` | List of ControlNet configurations |

Each ControlNet config contains:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | *required* | ControlNet model ID |
| `preprocessor` | str | `"passthrough"` | Preprocessor type |
| `conditioning_scale` | float | `1.0` | ControlNet strength |
| `enabled` | bool | `true` | Enable/disable this ControlNet |
| `preprocessor_params` | Dict | `{}` | Preprocessor-specific parameters |

## API Reference

### Configuration Functions

```python
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML or JSON file"""

def create_wrapper_from_config(config: Dict[str, Any], **overrides) -> StreamDiffusionWrapper:
    """Create StreamDiffusionWrapper from configuration with optional overrides"""
```

### Backward Compatibility

```python
# These functions are deprecated but still work
def load_controlnet_config(config_path: str) -> Dict[str, Any]:
    """DEPRECATED: Use load_config() instead"""

def save_controlnet_config(config: Dict[str, Any], config_path: str) -> None:
    """DEPRECATED: Use save_config() instead"""
```

## Examples

### Example 1: Simple Text-to-Image

```python
from streamdiffusion import load_config, create_wrapper_from_config

config = load_config("configs/general_examples/sd_turbo_txt2img.yaml.example")
wrapper = create_wrapper_from_config(config)

# Generate image
output_image = wrapper()
output_image.save("output.png")
```

### Example 2: Image-to-Image with Overrides

```python
from streamdiffusion import load_config, create_wrapper_from_config
from PIL import Image

config = load_config("configs/general_examples/sd15_img2img.yaml.example")

# Override prompt and resolution
wrapper = create_wrapper_from_config(
    config,
    prompt="cyberpunk city, neon lights",
    width=768,
    height=768
)

# Load input image
input_image = Image.open("input.jpg")

# Generate output
output_image = wrapper(input_image)
output_image.save("output.png")
```

### Example 3: Creating Custom Configs

```python
from streamdiffusion import save_config, load_config, create_wrapper_from_config

# Create custom configuration
custom_config = {
    "model_id": "stabilityai/sd-turbo",
    "mode": "txt2img",
    "width": 1024,
    "height": 1024,
    "prompt": "a majestic dragon",
    "acceleration": "tensorrt",
    "seed": 12345
}

# Save config
save_config(custom_config, "my_custom_config.yaml")

# Load and use
config = load_config("my_custom_config.yaml")
wrapper = create_wrapper_from_config(config)
output = wrapper()
```

## Migration Guide

### From Hardcoded Parameters

**Before:**
```python
wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[0, 1, 2, 3],
    width=512,
    height=512,
    # ... many parameters
)
wrapper.prepare(prompt="my prompt", guidance_scale=1.0)
```

**After:**
```yaml
# config.yaml
model_id: "stabilityai/sd-turbo"
t_index_list: [0, 1, 2, 3]
width: 512
height: 512
prompt: "my prompt"
guidance_scale: 1.0
```

```python
config = load_config("config.yaml")
wrapper = create_wrapper_from_config(config)
```

### From ControlNet Configs

Existing ControlNet configs work unchanged:

```python
# Still works
from streamdiffusion.controlnet import load_controlnet_config

# New way (recommended)
from streamdiffusion import load_config
```

## Configuration Examples

See the `configs/general_examples/` directory for complete configuration examples:

- `sd_turbo_txt2img.yaml.example` - Basic text-to-image
- `sd15_img2img.yaml.example` - Image-to-image with filters
- Existing ControlNet configs in `configs/` continue to work

## Benefits

1. **Simplicity**: Single function call to create configured wrapper
2. **Flexibility**: Override any parameter at runtime
3. **Reproducibility**: Share exact configurations
4. **Maintainability**: Clean separation of code and configuration
5. **Backward Compatibility**: Existing ControlNet configs work unchanged
6. **Extensibility**: Easy to add new parameters as StreamDiffusion evolves

This configuration system provides a clean, elegant solution for managing StreamDiffusion parameters while maintaining full backward compatibility with existing ControlNet workflows. 