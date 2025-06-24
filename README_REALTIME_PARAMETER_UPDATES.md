# Real-Time Parameter Updates for StreamDiffusion

StreamDiffusion supports updating core streaming parameters in real-time during active generation without recreating the pipeline. This enables dynamic adjustment of generation quality, speed, and behavior while streaming.

## Overview

The `update_stream_params()` method allows efficient updates of:
- **Inference steps** (`num_inference_steps`) - Quality vs speed tradeoff
- **Guidance scale** (`guidance_scale`) - Prompt adherence strength  
- **Delta** (`delta`) - Virtual residual noise multiplier
- **Timestep indices** (`t_index_list`) - Custom denoising schedule

## Usage

### Basic Parameter Updates

```python
from streamdiffusion import create_wrapper_from_config, load_config

# Create wrapper
config = load_config("your_config.yaml")
wrapper = create_wrapper_from_config(config)

# During streaming, update parameters in real-time
wrapper.update_stream_params(
    num_inference_steps=8,      # Increase quality
    guidance_scale=1.5,         # Stronger prompt adherence
    delta=0.8,                  # Adjust noise characteristics
    t_index_list=[0, 2, 4, 6]   # Custom timestep schedule
)

# Continue streaming with new parameters
output = wrapper(input_image)
```

### Individual Parameter Updates

```python
# Update single parameters as needed
wrapper.update_stream_params(guidance_scale=2.0)  # Only guidance scale
wrapper.update_stream_params(num_inference_steps=50)  # Only inference steps
wrapper.update_stream_params(delta=1.2)  # Only delta
```

### Dynamic Quality Adjustment

```python
# Example: Performance-based quality scaling
for frame in video_stream:
    if performance_monitor.fps < target_fps:
        # Reduce quality for better performance
        wrapper.update_stream_params(
            num_inference_steps=2,
            t_index_list=[0, 1]
        )
    elif performance_monitor.fps > target_fps * 1.2:
        # Increase quality when performance allows
        wrapper.update_stream_params(
            num_inference_steps=6,
            t_index_list=[0, 2, 4, 6, 8, 10]
        )
    
    output = wrapper(frame)
```

## Implementation Details

### StreamParameterUpdater Class

The functionality is implemented through the `StreamParameterUpdater` class:

```python
# Located in: src/streamdiffusion/stream_parameter_updater.py
class StreamParameterUpdater:
    def update_stream_params(self, ...):
        # Efficiently recalculates only affected parameters
        # - Updates scheduler timesteps if num_inference_steps changes
        # - Validates and adjusts t_index_list for new step count
        # - Recalculates LCM scheduler scaling parameters
        # - Updates noise schedule parameters
```

### Automatic Timestep Validation

When `num_inference_steps` changes without providing a new `t_index_list`, the system automatically adjusts the current timestep indices to fit the new step count:

```python
# If you change steps from 50 to 10, t_index_list [5, 15, 30, 45] 
# becomes [5, 9, 9, 9] (clamped to max valid index)
wrapper.update_stream_params(num_inference_steps=10)
```

### Performance Characteristics

- **Minimal overhead**: Only affected parameters are recalculated
- **No pipeline recreation**: Existing models and engines remain loaded
- **GPU memory efficient**: No additional memory allocation required
- **Thread-safe**: Updates can be called from different threads

## Parameter Descriptions

| Parameter | Type | Description | Performance Impact |
|-----------|------|-------------|-------------------|
| `num_inference_steps` | int | Total denoising steps available | Higher = slower, better quality |
| `guidance_scale` | float | CFG strength (ignored if cfg_type="none") | Minimal impact |
| `delta` | float | Virtual residual noise multiplier for self-CFG | Minimal impact |
| `t_index_list` | List[int] | Indices into timestep schedule for denoising | Longer list = slower |

## Integration with Other Features

### With ControlNet

```python
# Works seamlessly with ControlNet pipelines
wrapper.update_stream_params(guidance_scale=1.8)
wrapper.update_controlnet_scale(0, 0.9)  # Adjust ControlNet strength
wrapper.update_control_image_efficient(new_control_image)
output = wrapper(input_image)
```

### With Configuration System

```python
# Runtime parameter updates work with any config-based setup
config = load_config("my_config.yaml")
wrapper = create_wrapper_from_config(config)

# Override config values dynamically
wrapper.update_stream_params(
    guidance_scale=config['guidance_scale'] * 1.5,
    num_inference_steps=config['num_inference_steps'] // 2
)
```

## Best Practices

1. **Gradual changes**: Avoid drastic parameter jumps that could cause visual artifacts
2. **Performance monitoring**: Use FPS/latency metrics to guide parameter adjustments
3. **Quality presets**: Define parameter combinations for different quality levels
4. **Validation**: Parameters are automatically validated, but invalid values will be logged

## Available in

- `StreamDiffusionWrapper.update_stream_params()` (utils/wrapper.py)
- `StreamDiffusion.update_stream_params()` (src/streamdiffusion/pipeline.py)
- Both standard and ControlNet-enabled pipelines

This feature enables truly dynamic streaming generation with real-time quality and performance optimization. 