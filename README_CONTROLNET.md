# ControlNet Support for StreamDiffusion

This implementation adds comprehensive ControlNet support to StreamDiffusion, enabling real-time image generation conditioned on various types of control inputs like edges, depth maps, pose information, and more.

NOTE: SDXL Support is forthcoming

## Features

- Dynamic multi-controlnet support
- Full tensorrt acceleration, with dynamic runtime conditioning strength
- Extensible preprocessing system for different ControlNet types
- Drop-in replacement for standard StreamDiffusion with ControlNet capabilities
- Use YAML/JSON configs to define your ControlNet pipelines

## Quick Start

### Web demo
```bash
cd demo/realtime-img2img
python main.py
```
Select a YAML configuration file, and click start.  The demo is built upon the existing streamdiffusion demo, but now includes:
- Real-time parameter adjustment
- Multi-ControlNet strength sliders
- Live performance monitoring
- Side-by-side input/output display

### Standalone Demo

```bash
# Standalone example
python examples/controlnet/standalone_controlnet_pipeline.py --config "../../configs/sd15_depth_trt_example.yaml" --input-image "../../assets/img2img_example.png"
```

## Usage

### Configuration-Based (Recommended)

See [Configuration System Documentation](README_CONFIG.md) for detailed configuration instructions.

#### New Simplified Approach

```python
from streamdiffusion import load_config, create_wrapper_from_config
from PIL import Image

# Load configuration and create wrapper in one step
config = load_config("configs/my_config.yaml")
wrapper = create_wrapper_from_config(config)

# Process images
input_image = Image.open("your_image.jpg")

# Update control image for all ControlNets
wrapper.update_control_image_efficient(input_image)

# Generate output
output_image = wrapper(input_image)
```

#### Manual Approach

```python
from streamdiffusion import load_config
from utils.wrapper import StreamDiffusionWrapper
import torch

# Load configuration
config = load_config("configs/my_config.yaml")

# Create wrapper with ControlNet support
wrapper = StreamDiffusionWrapper(
    model_id_or_path=config['model_id'],
    t_index_list=config['t_index_list'],
    mode="img2img",
    output_type="pil",
    device="cuda",
    dtype=torch.float16,
    width=config['width'],
    height=config['height'],
    frame_buffer_size=config.get('frame_buffer_size', 1),
    acceleration=config.get('acceleration', 'none'),
    use_lcm_lora=config.get('use_lcm_lora', True),
    use_tiny_vae=config.get('use_tiny_vae', True),
    cfg_type=config.get('cfg_type', 'self'),
    seed=config.get('seed', 2),
    use_safety_checker=False,
    # ControlNet configuration
    use_controlnet=True,
    controlnet_config=config['controlnets'],
)

# Prepare pipeline
wrapper.prepare(
    prompt=config['prompt'],
    negative_prompt=config.get('negative_prompt', ''),
    num_inference_steps=config.get('num_inference_steps', 50),
    guidance_scale=config.get('guidance_scale', 1.0),
    delta=config.get('delta', 1.0),
)

# Process images
from PIL import Image
input_image = Image.open("your_image.jpg")

# Update control image for all ControlNets
wrapper.update_control_image_efficient(input_image)

# Generate output
output_image = wrapper(input_image)
```

### Programmatic Setup

#### Recommended: Config Dictionary + create_wrapper_from_config()

```python
from streamdiffusion import create_wrapper_from_config

# Define configuration programmatically
config = {
    'model_id': "stabilityai/sd-turbo",
    't_index_list': [32, 45],
    'mode': "img2img",
    'width': 512,
    'height': 512,
    'acceleration': "none",
    'use_lcm_lora': False,
    'cfg_type': "none",
    'seed': 789,
    'prompt': "a beautiful landscape, highly detailed",
    'negative_prompt': "blurry, low quality",
    'guidance_scale': 1.0,
    'controlnets': [
        {
            'model_id': "lllyasviel/control_v11p_sd15_canny",
            'conditioning_scale': 1.0,
            'preprocessor': "canny",
            'preprocessor_params': {
                'low_threshold': 100,
                'high_threshold': 200
            },
            'enabled': True,
            'pipeline_type': 'sd1.5',
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
    ]
}

# Create wrapper from config
wrapper = create_wrapper_from_config(config)

# Process image
wrapper.update_control_image_efficient(input_image)
output = wrapper(input_image)
```

#### Manual Approach: Direct StreamDiffusionWrapper

```python
from utils.wrapper import StreamDiffusionWrapper
import torch

# Define ControlNet configurations
controlnet_configs = [
    {
        'model_id': "lllyasviel/control_v11p_sd15_canny",
        'conditioning_scale': 1.0,
        'preprocessor': "canny",
        'preprocessor_params': {
            'low_threshold': 100,
            'high_threshold': 200
        },
        'enabled': True,
        'pipeline_type': 'sd1.5',
        'control_guidance_start': 0.0,
        'control_guidance_end': 1.0,
    }
]

# Create wrapper
wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[32, 45],
    mode="img2img",
    output_type="pil",
    device="cuda",
    dtype=torch.float16,
    width=512,
    height=512,
    frame_buffer_size=1,
    acceleration="none",
    use_lcm_lora=False,
    use_tiny_vae=True,
    cfg_type="none",
    seed=789,
    use_safety_checker=False,
    # ControlNet options
    use_controlnet=True,
    controlnet_config=controlnet_configs,
)

# Prepare and use
wrapper.prepare(
    prompt="a beautiful landscape, highly detailed",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=1.0,
    delta=1.0,
)

# Process image
wrapper.update_control_image_efficient(input_image)
output = wrapper(input_image)
```

## Configuration Format

**For comprehensive configuration documentation, see [Configuration System Documentation](README_CONFIG.md).**

### ControlNet-Specific Examples

#### Basic ControlNet Configuration

```yaml
# Basic SD-Turbo + Canny ControlNet
model_id: "stabilityai/sd-turbo"
pipeline_type: "sdturbo"
mode: "img2img"
prompt: "a beautiful landscape, highly detailed"

controlnets:
  - model_id: "lllyasviel/control_v11p_sd15_canny"
    conditioning_scale: 1.0
    preprocessor: "canny"
    preprocessor_params:
      low_threshold: 100
      high_threshold: 200
    enabled: true
```

#### Multi-ControlNet Configuration

```yaml
# Multiple ControlNets with different strengths
model_id: "stabilityai/sd-turbo"
pipeline_type: "sdturbo"
prompt: "a person in a detailed environment"

controlnets:
  - model_id: "thibaud/controlnet-sd21-depth-diffusers"
    conditioning_scale: 0.5
    preprocessor: "depth_tensorrt"
    preprocessor_params:
      engine_path: "path/to/depth_anything_vits14-fp16.engine"
      detect_resolution: 518
      image_resolution: 512
    enabled: true
    
  - model_id: "thibaud/controlnet-sd21-canny-diffusers"
    conditioning_scale: 0.5
    preprocessor: "canny"
    preprocessor_params:
      low_threshold: 50
      high_threshold: 100
    enabled: true
```

See `configs/` directory for more complete configuration examples.

## Runtime Control

### Dynamic ControlNet Strength Adjustment

```python
# Update ControlNet strength during runtime
wrapper.update_controlnet_scale(0, 0.8)  # Set first ControlNet to 80% strength
wrapper.update_controlnet_scale(1, 0.3)  # Set second ControlNet to 30% strength
```

### Efficient Control Image Updates

```python
# Update control images for all ControlNets efficiently
# This processes the input through all preprocessors
wrapper.update_control_image_efficient(new_input_image)

# Then generate with current control conditioning
output = wrapper(new_input_image)
```

## Supported Architectures

- **SD 1.5**: Full support with extensive ControlNet ecosystem
- **SD 2.1**: Full support with TensorRT acceleration  
- **SDXL**: Full support via `SDXLTurboControlNetPipeline` (forthcoming)

## Supported Preprocessors

| Preprocessor | Description | Parameters |
|--------------|-------------|------------|
| `canny` | Canny edge detection | `low_threshold`, `high_threshold` |
| `depth` | MiDaS depth estimation | None |
| `depth_tensorrt` | TensorRT-accelerated depth | `engine_path`, `detect_resolution`, `image_resolution` |
| `openpose` | Human pose detection | None |
| `lineart` | Line art extraction | None |
| `passthrough` | No preprocessing (use raw image) | None |

## API Reference

### StreamDiffusionWrapper ControlNet Parameters

```python
StreamDiffusionWrapper(
    # ... other parameters ...
    use_controlnet=True,                    # Enable ControlNet support
    controlnet_config=controlnet_configs,   # List of ControlNet configurations
)
```

### Runtime Control Methods

```python
# Update ControlNet strength dynamically
wrapper.update_controlnet_scale(index: int, scale: float)

# Update control image for all ControlNets efficiently  
wrapper.update_control_image_efficient(image: PIL.Image)

# Main inference call
output_image = wrapper(input_image: PIL.Image)
```

### Configuration Functions

```python
from streamdiffusion import load_config, save_config, create_wrapper_from_config

# Load configuration from YAML/JSON
config = load_config("path/to/config.yaml")

# Create wrapper directly from config
wrapper = create_wrapper_from_config(config)

# Save configuration  
save_config(config, "path/to/save.yaml")
```

#### Backward Compatibility

```python
# These still work but are deprecated
from streamdiffusion.controlnet import load_controlnet_config, save_controlnet_config
```

### ControlNet Configuration Dictionary

```python
controlnet_config = {
    'model_id': str,                    # ControlNet model ID or path
    'conditioning_scale': float,        # Strength (0.0 to 2.0)
    'preprocessor': str,               # Preprocessor name
    'preprocessor_params': dict,       # Preprocessor-specific parameters
    'enabled': bool,                   # Enable/disable this ControlNet
    'pipeline_type': str,              # Pipeline type (sd1.5, sdturbo, soon sdxlturbo)
    'control_guidance_start': float,   # Guidance start timestep (0.0-1.0)
    'control_guidance_end': float,     # Guidance end timestep (0.0-1.0)
}
```

## Adding Custom Preprocessors

1. Create a new preprocessor class inheriting from `BasePreprocessor`:

```python
from streamdiffusion.controlnet.preprocessors import BasePreprocessor, register_preprocessor

class MyCustomPreprocessor(BasePreprocessor):
    def process(self, image, **params):
        # Your preprocessing logic here
        processed_image = your_processing_function(image, **params)
        return processed_image

# Register the preprocessor
register_preprocessor("my_custom", MyCustomPreprocessor)
```

2. Use in configuration:

```yaml
controlnets:
  - model_id: "your/controlnet-model"
    preprocessor: "my_custom"
    preprocessor_params:
      your_param: value
```

## Performance Notes

- **TensorRT**: Provides significant speedup but requires pre-built engines and fixed batch sizes
- **Frame Buffer Size**: Higher values improve temporal consistency but use more VRAM
- **Multiple ControlNets**: Each additional ControlNet increases memory usage and processing time
- **Real-time Usage**: Use `update_control_image_efficient()` for optimal performance in video/webcam scenarios

## Example Scripts

- `examples/controlnet/controlnet_webcam_gui_demo.py` - Interactive GUI demo with real-time parameter adjustment
- `examples/controlnet/standalone_controlnet_pipeline.py` - Production-ready reference implementation
- `configs/` - Various configuration examples for different use cases
