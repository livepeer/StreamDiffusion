# LoRA Module for StreamDiffusion

This document describes the new LoRA module system implemented for StreamDiffusion, providing comprehensive LoRA management and hotswapping capabilities.

## Overview

The LoRA module replaces the legacy LoRA handling system with a full module-based approach that supports:

- **Hotswapping**: Add, remove, and modify LoRAs at runtime without restarting the pipeline
- **State Management**: Track loaded LoRAs, their configurations, and current states
- **Type Detection**: Automatically detect LoRA types (standard, LCM, text encoder, etc.)
- **Runtime Updates**: Modify LoRA scales and enabled states in real-time
- **Thread Safety**: Safe concurrent access to LoRA collections
- **Configuration Integration**: Full YAML configuration support

## Architecture

### Core Components

1. **LoRAConfig**: Configuration dataclass for individual LoRAs
2. **LoRAModuleConfig**: Configuration for the LoRA module itself
3. **LoRAModule**: Main module class implementing the OrchestratorUser pattern

### Integration Points

- **StreamDiffusionWrapper**: Integrated into the wrapper constructor and _load_model method
- **Pipeline**: Accessible via `stream.lora_module` after installation
- **Demo Interface**: Full UI controls in the realtime-img2img demo

## Usage

### Basic Configuration

```yaml
# Enable LoRA module in your config
use_lora_module: true

lora_config:
  module_config:
    device: "cuda"
    dtype: "float16"
    auto_detect_type: true
    enable_offline_fallback: true
    default_scale: 1.0
  
  loras:
    - lora_path: "path/to/lora.safetensors"
      scale: 0.8
      target: "both"
      enabled: true
      display_name: "My LoRA"
```

### Programmatic Usage

```python
from streamdiffusion import StreamDiffusionWrapper
from streamdiffusion.modules import LoRAConfig

# Initialize with LoRA module
wrapper = StreamDiffusionWrapper(
    model_id_or_path="runwayml/stable-diffusion-v1-5",
    t_index_list=[35, 45],
    use_lora_module=True,
    lora_config={
        "loras": [
            {
                "lora_path": "path/to/lora.safetensors",
                "scale": 0.8,
                "target": "both"
            }
        ]
    }
)

# Runtime LoRA management
wrapper.add_lora("new/lora/path.safetensors", scale=0.6)
wrapper.update_lora_scale(0, 1.0)
wrapper.remove_lora(1)

# Get LoRA information
loras = wrapper.get_loaded_loras()
print(f"Loaded {len(loras)} LoRAs")
```

### Demo Interface

The realtime-img2img demo includes a full LoRA configuration panel with:

- **Add LoRAs**: By path/URL or file upload
- **Remove LoRAs**: One-click removal
- **Scale Control**: Real-time scale adjustment sliders
- **Enable/Disable**: Toggle LoRAs on/off
- **Status Display**: View loaded LoRAs and their states

## API Reference

### LoRAConfig

```python
@dataclass
class LoRAConfig:
    lora_path: str                    # Path to LoRA file or HuggingFace model ID
    adapter_name: Optional[str]       # Custom adapter name (auto-generated if None)
    scale: float = 1.0               # LoRA strength/scale
    target: Literal["unet", "text_encoder", "both"] = "both"  # Application target
    enabled: bool = True             # Whether LoRA is active
    lora_type: Optional[str] = None  # LoRA type (auto-detected if None)
    display_name: Optional[str]      # Human-readable name
    description: Optional[str]       # Description
```

### LoRAModule Methods

#### Core Management
- `add_lora(config: LoRAConfig) -> bool`: Add a new LoRA
- `remove_lora(index: int) -> bool`: Remove LoRA by index
- `update_lora_scale(index: int, scale: float) -> bool`: Update LoRA scale
- `update_lora_enabled(index: int, enabled: bool) -> bool`: Enable/disable LoRA

#### State Access
- `get_loaded_loras_info() -> List[Dict]`: Get detailed LoRA information
- `get_lora_state() -> Dict`: Get complete module state
- `update_config(config: List[Dict]) -> None`: Update from configuration

### Wrapper Integration

#### Constructor Parameters
- `use_lora_module: bool = False`: Enable LoRA module system
- `lora_config: Optional[Dict] = None`: LoRA configuration

#### Runtime Methods
- `add_lora(lora_path: str, scale: float = 1.0, target: str = "both") -> bool`
- `remove_lora(index: int) -> bool`
- `update_lora_scale(index: int, scale: float) -> bool`
- `update_lora_enabled(index: int, enabled: bool) -> bool`
- `get_loaded_loras() -> List[Dict]`

## Features

### Automatic Type Detection

The module automatically detects LoRA types based on:
- File content analysis for local files
- Naming patterns for HuggingFace model IDs
- Weight key patterns (LCM, text encoder, UNet-specific)

### Offline Fallback Support

When HuggingFace is offline, the module tries common weight filenames:
- `pytorch_lora_weights.safetensors`
- `pytorch_lora_weights.bin`
- `diffusion_pytorch_model.safetensors`
- `adapter_model.safetensors`
- `lora.safetensors`

### Thread Safety

All LoRA operations are protected by threading locks to ensure safe concurrent access from multiple threads.

### Error Handling

Comprehensive error handling with detailed logging:
- Invalid file paths
- Loading failures
- Runtime update errors
- State inconsistencies

## Migration from Legacy System

### Old System (Deprecated)
```python
wrapper = StreamDiffusionWrapper(
    model_id_or_path="model",
    lora_dict={"lora1": 0.8, "lora2": 0.6},
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
    use_lcm_lora=True
)
```

### New System
```python
wrapper = StreamDiffusionWrapper(
    model_id_or_path="model",
    use_lora_module=True,
    lora_config={
        "loras": [
            {"lora_path": "lora1", "scale": 0.8},
            {"lora_path": "lora2", "scale": 0.6},
            {"lora_path": "latent-consistency/lcm-lora-sdv1-5", "scale": 1.0, "lora_type": "lcm"}
        ]
    }
)
```

## Demo API Endpoints

The realtime-img2img demo exposes these LoRA endpoints:

- `GET /api/lora/list`: Get loaded LoRAs
- `POST /api/lora/add`: Add new LoRA
- `POST /api/lora/remove`: Remove LoRA
- `POST /api/lora/update-scale`: Update LoRA scale
- `POST /api/lora/update-enabled`: Enable/disable LoRA
- `POST /api/lora/upload`: Upload LoRA file

## Configuration Examples

### Multiple LoRAs with Different Targets
```yaml
lora_config:
  loras:
    - lora_path: "style_lora.safetensors"
      scale: 0.8
      target: "unet"
      display_name: "Style Enhancement"
    
    - lora_path: "text_lora.safetensors"
      scale: 0.6
      target: "text_encoder"
      display_name: "Text Understanding"
    
    - lora_path: "combined_lora.safetensors"
      scale: 1.0
      target: "both"
      display_name: "Combined Enhancement"
```

### LCM LoRA Configuration
```yaml
lora_config:
  loras:
    - lora_path: "latent-consistency/lcm-lora-sdv1-5"
      scale: 1.0
      target: "both"
      lora_type: "lcm"
      display_name: "LCM Acceleration"
```

### HuggingFace Model Integration
```yaml
lora_config:
  loras:
    - lora_path: "username/awesome-style-lora"
      scale: 0.7
      target: "both"
      display_name: "Awesome Style"
      description: "Downloaded from HuggingFace Hub"
```

## Troubleshooting

### Common Issues

1. **LoRA not loading**: Check file path and permissions
2. **Scale not updating**: Ensure LoRA is enabled and pipeline supports set_adapters
3. **Type detection failing**: Manually specify lora_type in configuration
4. **Memory issues**: Reduce number of loaded LoRAs or lower scales

### Debug Information

Enable debug logging to see detailed LoRA operations:
```python
import logging
logging.getLogger('streamdiffusion').setLevel(logging.DEBUG)
```

### State Inspection

Check LoRA module state:
```python
if hasattr(wrapper.stream, 'lora_module'):
    state = wrapper.stream.lora_module.get_lora_state()
    print(f"LoRA State: {state}")
```

## Performance Considerations

- **Memory Usage**: Each LoRA consumes additional GPU memory
- **Loading Time**: Initial LoRA loading may take several seconds
- **Runtime Updates**: Scale and enabled state updates are fast
- **Hotswapping**: Adding/removing LoRAs may cause brief processing delays

## Future Enhancements

Planned improvements include:
- LoRA blending and mixing capabilities
- Advanced scheduling and automation
- Performance optimizations
- Enhanced type detection
- Batch operations support
