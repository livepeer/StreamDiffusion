import os
import sys
import yaml
import json
from typing import Dict, List, Optional, Union, Any, Tuple
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
    """Create StreamDiffusionWrapper from configuration dictionary
    
    Prompt Interface:
    - Legacy: Use 'prompt' field for single prompt
    - New: Use 'prompt_blending' with 'prompt_list' for multiple weighted prompts
    - If both are provided, 'prompt_blending' takes precedence and 'prompt' is ignored
    - negative_prompt: Currently a single string (not list) for all prompt types
    """
    from streamdiffusion import StreamDiffusionWrapper
    import torch

    final_config = {**config, **overrides}
    wrapper_params = _extract_wrapper_params(final_config)
    wrapper = StreamDiffusionWrapper(**wrapper_params)
    
    # Setup IPAdapter if configured
    if 'ipadapters' in final_config and final_config['ipadapters']:
        wrapper = _setup_ipadapter_from_config(wrapper, final_config)
    
    prepare_params = _extract_prepare_params(final_config)

    # Handle prompt configuration with clear precedence
    if 'prompt_blending' in final_config:
        # Use prompt blending (new interface) - ignore legacy 'prompt' field
        blend_config = final_config['prompt_blending']
        
        # Prepare with prompt blending directly using unified interface
        prepare_params_with_blending = {k: v for k, v in prepare_params.items() 
                                       if k not in ['prompt_blending', 'seed_blending']}
        prepare_params_with_blending['prompt'] = blend_config.get('prompt_list', [])
        prepare_params_with_blending['prompt_interpolation_method'] = blend_config.get('interpolation_method', 'slerp')
        
        # Add seed blending if configured
        if 'seed_blending' in final_config:
            seed_blend_config = final_config['seed_blending']
            prepare_params_with_blending['seed_list'] = seed_blend_config.get('seed_list', [])
            prepare_params_with_blending['seed_interpolation_method'] = seed_blend_config.get('interpolation_method', 'linear')
        
        wrapper.prepare(**prepare_params_with_blending)
    elif prepare_params.get('prompt'):
        # Use legacy single prompt interface
        clean_prepare_params = {k: v for k, v in prepare_params.items() 
                               if k not in ['prompt_blending', 'seed_blending']}
        wrapper.prepare(**clean_prepare_params)

    # Apply seed blending if configured and not already handled in prepare
    if 'seed_blending' in final_config and 'prompt_blending' not in final_config:
        seed_blend_config = final_config['seed_blending']
        wrapper.update_seed_blending(
            seed_list=seed_blend_config.get('seed_list', []),
            interpolation_method=seed_blend_config.get('interpolation_method', 'linear')
        )

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
        'normalize_prompt_weights': config.get('normalize_prompt_weights', True),
        'normalize_seed_weights': config.get('normalize_seed_weights', True),
        'enable_pytorch_fallback': config.get('enable_pytorch_fallback', False),
    }
    if 'controlnets' in config and config['controlnets']:
        param_map['use_controlnet'] = True
        param_map['controlnet_config'] = _prepare_controlnet_configs(config)
    else:
        param_map['use_controlnet'] = config.get('use_controlnet', False)
        param_map['controlnet_config'] = config.get('controlnet_config')
    
    # Set IPAdapter usage if IPAdapters are configured
    if 'ipadapters' in config and config['ipadapters']:
        param_map['use_ipadapter'] = True
        param_map['ipadapter_config'] = _prepare_ipadapter_configs(config)
    else:
        param_map['use_ipadapter'] = config.get('use_ipadapter', False)
        param_map['ipadapter_config'] = config.get('ipadapter_config')
    
    # Set StreamV2V usage if configured
    if 'streamv2v' in config and config['streamv2v']:
        param_map['use_streamv2v'] = True
        param_map['streamv2v_config'] = config['streamv2v']
    else:
        param_map['use_streamv2v'] = config.get('use_streamv2v', False)
        param_map['streamv2v_config'] = config.get('streamv2v_config')
    
    return {k: v for k, v in param_map.items() if v is not None}


def _extract_prepare_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters for wrapper.prepare() from config"""
    prepare_params = {
        'prompt': config.get('prompt', ''),
        'negative_prompt': config.get('negative_prompt', ''),
        'num_inference_steps': config.get('num_inference_steps', 50),
        'guidance_scale': config.get('guidance_scale', 1.2),
        'delta': config.get('delta', 1.0),
    }
    
    # Handle prompt blending configuration
    if 'prompt_blending' in config:
        blend_config = config['prompt_blending']
        prepare_params['prompt_blending'] = {
            'prompt_list': blend_config.get('prompt_list', []),
            'interpolation_method': blend_config.get('interpolation_method', 'slerp'),
            'enable_caching': blend_config.get('enable_caching', True)
        }
    
    # Handle seed blending configuration
    if 'seed_blending' in config:
        seed_blend_config = config['seed_blending']
        prepare_params['seed_blending'] = {
            'seed_list': seed_blend_config.get('seed_list', []),
            'interpolation_method': seed_blend_config.get('interpolation_method', 'linear'),
            'enable_caching': seed_blend_config.get('enable_caching', True)
        }
    
    return prepare_params

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


def _prepare_ipadapter_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare IPAdapter configurations for wrapper"""
    ipadapter_configs = []
    
    for ip_config in config['ipadapters']:
        ipadapter_config = {
            'ipadapter_model_path': ip_config['ipadapter_model_path'],
            'image_encoder_path': ip_config['image_encoder_path'],
            'style_image': ip_config.get('style_image'),
            'scale': ip_config.get('scale', 1.0),
            'enabled': ip_config.get('enabled', True),
        }
        ipadapter_configs.append(ipadapter_config)
    
    return ipadapter_configs


def _setup_ipadapter_from_config(wrapper, config: Dict[str, Any]):
    """Setup IPAdapter pipeline from configuration"""
    try:
        from .ipadapter import BaseIPAdapterPipeline
        
        # Create pipeline
        device = config.get('device', 'cuda')
        dtype = _parse_dtype(config.get('dtype', 'float16'))
        pipeline = BaseIPAdapterPipeline(wrapper.stream, device, dtype)
        
        # Handle preloaded models vs fresh setup
        if _has_preloaded_models(wrapper):
            _configure_preloaded_pipeline(pipeline, config)
        else:
            _configure_fresh_pipeline(pipeline, config)
        
        # Setup pipeline attributes
        pipeline.batch_size = getattr(wrapper, 'batch_size', 1)
        pipeline._original_wrapper = wrapper
        
        return pipeline
        
    except ImportError as e:
        raise ImportError(f"_setup_ipadapter_from_config: IPAdapter module not found: {e}") from e
    except Exception as e:
        print(f"_setup_ipadapter_from_config: Failed to setup IPAdapter: {e}")
        raise





def _has_preloaded_models(wrapper) -> bool:
    """Check if wrapper has preloaded IPAdapter models"""
    return (hasattr(wrapper, 'stream') and 
            hasattr(wrapper.stream, '_preloaded_with_weights') and 
            wrapper.stream._preloaded_with_weights and
            hasattr(wrapper.stream, '_preloaded_ipadapters') and 
            wrapper.stream._preloaded_ipadapters)


def _configure_preloaded_pipeline(pipeline, config: Dict[str, Any]):
    """Configure pipeline using preloaded models"""
    pipeline.ipadapter = pipeline.stream._preloaded_ipadapters[0]
    
    ipadapter_configs = _prepare_ipadapter_configs(config)
    if ipadapter_configs:
        ip_config = ipadapter_configs[0]
        if ip_config.get('enabled', True):
            _apply_ipadapter_config(pipeline, ip_config)
            
            # Register enhancer for TensorRT compatibility
            pipeline.stream._param_updater.register_embedding_enhancer(
                pipeline._enhance_embeddings_with_ipadapter, name="IPAdapter"
            )
            
            if len(ipadapter_configs) > 1:
                print("_setup_ipadapter_from_config: WARNING - Multiple IPAdapters configured but only first one will be used")


def _configure_fresh_pipeline(pipeline, config: Dict[str, Any]):
    """Configure pipeline with fresh IPAdapter setup"""
    ipadapter_configs = _prepare_ipadapter_configs(config)
    if ipadapter_configs:
        ip_config = ipadapter_configs[0]
        if ip_config.get('enabled', True):
            pipeline.set_ipadapter(
                ipadapter_model_path=ip_config['ipadapter_model_path'],
                image_encoder_path=ip_config['image_encoder_path'],
                style_image=ip_config.get('style_image'),
                scale=ip_config.get('scale', 1.0)
            )
            
            if len(ipadapter_configs) > 1:
                print("_setup_ipadapter_from_config: WARNING - Multiple IPAdapters configured but only first one will be used")


def _apply_ipadapter_config(pipeline, ip_config: Dict[str, Any]):
    """Apply configuration to existing IPAdapter"""
    # Set style image
    style_image_path = ip_config.get('style_image')
    if style_image_path:
        from PIL import Image
        pipeline.style_image = Image.open(style_image_path).convert("RGB")
    
    # Set scale
    scale = ip_config.get('scale', 1.0)
    pipeline.scale = scale
    if pipeline.ipadapter:
        pipeline.ipadapter.set_scale(scale)


def create_prompt_blending_config(
    base_config: Dict[str, Any],
    prompt_list: List[Tuple[str, float]],
    prompt_interpolation_method: str = "slerp",
    enable_caching: bool = True
) -> Dict[str, Any]:
    """Create a configuration with prompt blending settings"""
    config = base_config.copy()
    
    config['prompt_blending'] = {
        'prompt_list': prompt_list,
        'interpolation_method': prompt_interpolation_method,
        'enable_caching': enable_caching
    }
    
    return config


def create_seed_blending_config(
    base_config: Dict[str, Any],
    seed_list: List[Tuple[int, float]],
    interpolation_method: str = "linear",
    enable_caching: bool = True
) -> Dict[str, Any]:
    """Create a configuration with seed blending settings"""
    config = base_config.copy()
    
    config['seed_blending'] = {
        'seed_list': seed_list,
        'interpolation_method': interpolation_method,
        'enable_caching': enable_caching
    }
    
    return config


def set_normalize_weights_config(
    base_config: Dict[str, Any],
    normalize_prompt_weights: bool = True,
    normalize_seed_weights: bool = True
) -> Dict[str, Any]:
    """Create a configuration with separate normalize weight settings"""
    config = base_config.copy()
    
    config['normalize_prompt_weights'] = normalize_prompt_weights
    config['normalize_seed_weights'] = normalize_seed_weights
    
    return config

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
    
    # Validate ipadapters if present
    if 'ipadapters' in config:
        if not isinstance(config['ipadapters'], list):
            raise ValueError("_validate_config: 'ipadapters' must be a list")
        
        for i, ipadapter in enumerate(config['ipadapters']):
            if not isinstance(ipadapter, dict):
                raise ValueError(f"_validate_config: IPAdapter {i} must be a dictionary")
            
            if 'ipadapter_model_path' not in ipadapter:
                raise ValueError(f"_validate_config: IPAdapter {i} missing required 'ipadapter_model_path'")
            
            if 'image_encoder_path' not in ipadapter:
                raise ValueError(f"_validate_config: IPAdapter {i} missing required 'image_encoder_path'")

    # Validate prompt blending configuration if present
    if 'prompt_blending' in config:
        blend_config = config['prompt_blending']
        if not isinstance(blend_config, dict):
            raise ValueError("_validate_config: 'prompt_blending' must be a dictionary")
        
        if 'prompt_list' in blend_config:
            prompt_list = blend_config['prompt_list']
            if not isinstance(prompt_list, list):
                raise ValueError("_validate_config: 'prompt_list' must be a list")
            
            for i, prompt_item in enumerate(prompt_list):
                if not isinstance(prompt_item, (list, tuple)) or len(prompt_item) != 2:
                    raise ValueError(f"_validate_config: Prompt item {i} must be [text, weight] pair")
                
                text, weight = prompt_item
                if not isinstance(text, str):
                    raise ValueError(f"_validate_config: Prompt text {i} must be a string")
                
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise ValueError(f"_validate_config: Prompt weight {i} must be a non-negative number")
        
        interpolation_method = blend_config.get('interpolation_method', 'slerp')
        if interpolation_method not in ['linear', 'slerp']:
            raise ValueError("_validate_config: interpolation_method must be 'linear' or 'slerp'")

    # Validate seed blending configuration if present
    if 'seed_blending' in config:
        seed_blend_config = config['seed_blending']
        if not isinstance(seed_blend_config, dict):
            raise ValueError("_validate_config: 'seed_blending' must be a dictionary")
        
        if 'seed_list' in seed_blend_config:
            seed_list = seed_blend_config['seed_list']
            if not isinstance(seed_list, list):
                raise ValueError("_validate_config: 'seed_list' must be a list")
            
            for i, seed_item in enumerate(seed_list):
                if not isinstance(seed_item, (list, tuple)) or len(seed_item) != 2:
                    raise ValueError(f"_validate_config: Seed item {i} must be [seed, weight] pair")
                
                seed_value, weight = seed_item
                if not isinstance(seed_value, int) or seed_value < 0:
                    raise ValueError(f"_validate_config: Seed value {i} must be a non-negative integer")
                
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"_validate_config: Seed weight {i} must be a non-negative number")
        
        interpolation_method = seed_blend_config.get('interpolation_method', 'linear')
        if interpolation_method not in ['linear', 'slerp']:
            raise ValueError("_validate_config: seed blending interpolation_method must be 'linear' or 'slerp'")

    # Validate separate normalize settings if present
    if 'normalize_prompt_weights' in config:
        normalize_prompt_weights = config['normalize_prompt_weights']
        if not isinstance(normalize_prompt_weights, bool):
            raise ValueError("_validate_config: 'normalize_prompt_weights' must be a boolean value")
    
    if 'normalize_seed_weights' in config:
        normalize_seed_weights = config['normalize_seed_weights']
        if not isinstance(normalize_seed_weights, bool):
            raise ValueError("_validate_config: 'normalize_seed_weights' must be a boolean value")
    
    if 'enable_pytorch_fallback' in config:
        enable_pytorch_fallback = config['enable_pytorch_fallback']
        if not isinstance(enable_pytorch_fallback, bool):
            raise ValueError("_validate_config: 'enable_pytorch_fallback' must be a boolean value")
