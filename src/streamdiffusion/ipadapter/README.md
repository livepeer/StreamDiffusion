# StreamDiffusion IPAdapter Integration

This module provides an extensible IPAdapter integration for StreamDiffusion, following the same architecture pattern as ControlNet. It leverages the existing `Diffusers_IPAdapter` implementation for maximum compatibility and reliability.

## Features

- **Uses Existing Code**: Built on top of the proven `Diffusers_IPAdapter` implementation
- **Extensible Architecture**: Supports multiple IPAdapters with different style images and scales
- **Configuration System**: Full integration with StreamDiffusion's YAML/JSON config system
- **Runtime Updates**: Change style images and scales dynamically
- **Simple API**: Clean, intuitive interface following ControlNet patterns

## Quick Start

### Using the API Directly

```python
from src.streamdiffusion.ipadapter import IPAdapterPipeline
from utils.wrapper import StreamDiffusionWrapper

# Create base StreamDiffusion wrapper
wrapper = StreamDiffusionWrapper(
    model_id_or_path="KBlueLeaf/kohaku-v2.1",
    mode="txt2img",
    # ... other parameters
)

# Create IPAdapter pipeline
ipadapter_pipeline = IPAdapterPipeline(
    stream_diffusion=wrapper.stream,
    device="cuda",
    dtype=torch.float16
)

# Add IPAdapter using HuggingFace model ID (auto-downloads)
ipadapter_pipeline.add_ipadapter(
    ipadapter_model_path="h94/IP-Adapter",  # HuggingFace model ID
    image_encoder_path="h94/IP-Adapter",    # HuggingFace model ID
    style_image="path/to/style.png",
    scale=1.0
)

# Or use local paths if you have downloaded models
# ipadapter_pipeline.add_ipadapter(
#     ipadapter_model_path="/local/path/to/ip-adapter_sd15.bin",
#     image_encoder_path="/local/path/to/image_encoder",
#     style_image="path/to/style.png",
#     scale=1.0
# )

# Generate with IPAdapter conditioning
wrapper.prepare(prompt="your prompt here")
output = ipadapter_pipeline()
```

### Using Configuration Files

```yaml
# config.yaml
model_id: "KBlueLeaf/kohaku-v2.1"
device: "cuda"
mode: "txt2img"
prompt: "a beautiful portrait"

ipadapters:
  - ipadapter_model_path: "h94/IP-Adapter"      # HuggingFace ID or local path
    image_encoder_path: "h94/IP-Adapter"        # HuggingFace ID or local path
    style_image: "path/to/style.png"
    scale: 1.0
    enabled: true
```

```python
from src.streamdiffusion.controlnet.config import create_wrapper_from_config, load_config

config = load_config("config.yaml")
wrapper = create_wrapper_from_config(config)
output = wrapper()  # Automatically includes IPAdapter conditioning
```

## API Reference

### IPAdapterPipeline

The main class for IPAdapter integration.

#### Methods

- `add_ipadapter(ipadapter_model_path, image_encoder_path, style_image=None, scale=1.0)`: Add an IPAdapter
- `remove_ipadapter(index)`: Remove an IPAdapter by index
- `clear_ipadapters()`: Remove all IPAdapters
- `update_style_image(style_image, index=None)`: Update style image(s)
- `update_scale(index, scale)`: Update conditioning scale

### Configuration Schema

```yaml
ipadapters:
  - ipadapter_model_path: str       # Required: Path to IPAdapter weights
    image_encoder_path: str         # Required: Path to CLIP image encoder
    style_image: str               # Optional: Path to style image
    scale: float                   # Optional: Conditioning scale (default: 1.0)
    enabled: bool                  # Optional: Enable/disable (default: true)
```

## Examples

See the `examples/txt2img/` directory for complete examples:

- `ipadapter_simple_example.py`: Basic API usage
- `ipadapter_config_example.py`: Configuration-based usage
- `ipadapter_config_example.yaml`: Example configuration file

## Model Downloads

The system automatically downloads IPAdapter models from HuggingFace:

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download IPAdapter weights
ipadapter_model_path = hf_hub_download(
    repo_id="h94/IP-Adapter", 
    filename="models/ip-adapter_sd15.bin"
)

# Download image encoder
repo_path = snapshot_download(
    repo_id="h94/IP-Adapter",
    allow_patterns=["models/image_encoder/*"]
)
image_encoder_path = os.path.join(repo_path, "models", "image_encoder")
```

## Architecture

The IPAdapter integration follows the same extensible pattern as ControlNet:

- `BaseIPAdapterPipeline`: Core integration logic with embedding injection
- `IPAdapterPipeline`: Main user-facing class
- Configuration system integration in `../controlnet/config.py`
- Leverages existing `Diffusers_IPAdapter` for proven functionality

This architecture ensures consistency with the existing codebase while maintaining the simplicity and elegance requested. 