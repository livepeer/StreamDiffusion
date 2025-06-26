import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load StreamDiffusion configuration from YAML or JSON file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"load_config: Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"load_config: Unsupported configuration file format: {config_path.suffix}")

    _validate_config(config_data)

    return config_data

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save StreamDiffusion configuration to YAML or JSON file"""
    config_path = Path(config_path)

    _validate_config(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"save_config: Unsupported configuration file format: {config_path.suffix}")

def create_wrapper_from_config(config: Dict[str, Any], **overrides) -> Any:
    """Create StreamDiffusionWrapper from configuration dictionary"""
    from streamdiffusion import StreamDiffusionWrapper
    import torch

    final_config = {**config, **overrides}
    wrapper_params = _extract_wrapper_params(final_config)
    wrapper = StreamDiffusionWrapper(**wrapper_params)
    prepare_params = _extract_prepare_params(final_config)

    if prepare_params.get('prompt'):
        wrapper.prepare(**prepare_params)

    return wrapper

def _extract_wrapper_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for StreamDiffusionWrapper.__init__() from config"""
    import torch

    param_map = {
        'model_id_or_path': config.get('model_id', 'stabilityai/sd-turbo'),
        't_index_list': config.get('t_index_list', [0, 16, 32, 45]),
        'lora_dict': config.get('lora_dict'),
        'mode': config.get('mode', 'img2img'),
        'output_type': config.get('output_type', 'pil'),
        'lcm_lora_id': config.get('lcm_lora_id'),
        'vae_id': config.get('vae_id'),
        'device': config.get('device', 'cuda'),
        'dtype': _parse_dtype(config.get('dtype', 'float16')),
        'frame_buffer_size': config.get('frame_buffer_size', 1),
        'width': config.get('width', 512),
        'height': config.get('height', 512),
        'warmup': config.get('warmup', 10),
        'acceleration': config.get('acceleration', 'tensorrt'),
        'do_add_noise': config.get('do_add_noise', True),
        'device_ids': config.get('device_ids'),
        'use_lcm_lora': config.get('use_lcm_lora', True),
        'use_tiny_vae': config.get('use_tiny_vae', True),
        'enable_similar_image_filter': config.get('enable_similar_image_filter', False),
        'similar_image_filter_threshold': config.get('similar_image_filter_threshold', 0.98),
        'similar_image_filter_max_skip_frame': config.get('similar_image_filter_max_skip_frame', 10),
        'use_denoising_batch': config.get('use_denoising_batch', True),
        'cfg_type': config.get('cfg_type', 'self'),
        'seed': config.get('seed', 2),
        'use_safety_checker': config.get('use_safety_checker', False),
        'engine_dir': config.get('engine_dir', 'engines'),
    }

    if 'controlnets' in config and config['controlnets']:
        param_map['use_controlnet'] = True
        param_map['controlnet_config'] = _prepare_controlnet_configs(config)
    else:
        param_map['use_controlnet'] = config.get('use_controlnet', False)
        param_map['controlnet_config'] = config.get('controlnet_config')

    return {k: v for k, v in param_map.items() if v is not None}

def _extract_prepare_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for wrapper.prepare() from config"""
    return {
        'prompt': config.get('prompt', ''),
        'negative_prompt': config.get('negative_prompt', ''),
        'num_inference_steps': config.get('num_inference_steps', 50),
        'guidance_scale': config.get('guidance_scale', 1.2),
        'delta': config.get('delta', 1.0),
    }

def _prepare_controlnet_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare ControlNet configurations for wrapper"""
    controlnet_configs = []
    pipeline_type = config.get('pipeline_type', 'sd1.5')

    for cn_config in config['controlnets']:
        controlnet_config = {
            'model_id': cn_config['model_id'],
            'preprocessor': cn_config.get('preprocessor', 'passthrough'),
            'conditioning_scale': cn_config.get('conditioning_scale', 1.0),
            'enabled': cn_config.get('enabled', True),
            'preprocessor_params': cn_config.get('preprocessor_params'),
            'pipeline_type': pipeline_type,
            'control_guidance_start': cn_config.get('control_guidance_start', 0.0),
            'control_guidance_end': cn_config.get('control_guidance_end', 1.0),
        }
        controlnet_configs.append(controlnet_config)

    return controlnet_configs

def _parse_dtype(dtype_str: str) -> Any:
    """Parse dtype string to torch dtype"""
    import torch

    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'half': torch.float16,
        'float': torch.float32,
    }

    if isinstance(dtype_str, str):
        return dtype_map.get(dtype_str.lower(), torch.float16)
    return dtype_str  # Assume it's already a torch dtype

def _validate_config(config: Dict[str, Any]) -> None:
    """Basic validation of configuration dictionary"""
    if not isinstance(config, dict):
        raise ValueError("_validate_config: Configuration must be a dictionary")

    if 'model_id' not in config:
        raise ValueError("_validate_config: Missing required field: model_id")

    if 'controlnets' in config:
        if not isinstance(config['controlnets'], list):
            raise ValueError("_validate_config: 'controlnets' must be a list")

        for i, controlnet in enumerate(config['controlnets']):
            if not isinstance(controlnet, dict):
                raise ValueError(f"_validate_config: ControlNet {i} must be a dictionary")

            if 'model_id' not in controlnet:
                raise ValueError(f"_validate_config: ControlNet {i} missing required 'model_id'")
