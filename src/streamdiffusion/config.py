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
    print("_setup_ipadapter_from_config: Starting IPAdapter setup...")
    print(f"_setup_ipadapter_from_config: Python path: {sys.path[:3]}...")  # Show first 3 entries
    print(f"_setup_ipadapter_from_config: Current working directory: {os.getcwd()}")
    
    # Check if IPAdapter models were already pre-loaded by the wrapper (for TensorRT)
    if hasattr(wrapper, 'stream') and hasattr(wrapper.stream, '_preloaded_with_weights') and wrapper.stream._preloaded_with_weights:
        print("_setup_ipadapter_from_config: Detected pre-loaded IPAdapter models from wrapper")
        print("_setup_ipadapter_from_config: Setting up IPAdapter pipeline to reuse pre-loaded models...")
        
        try:
            # Import here to avoid circular imports
            from .ipadapter import IPAdapterPipeline
            
            # Create IPAdapter pipeline
            device = config.get('device', 'cuda')
            dtype = _parse_dtype(config.get('dtype', 'float16'))
            
            ipadapter_pipeline = IPAdapterPipeline(
                stream_diffusion=wrapper.stream,
                device=device,
                dtype=dtype
            )
            
            # Reuse the pre-loaded IPAdapter instead of creating new one
            if hasattr(wrapper.stream, '_preloaded_ipadapters') and wrapper.stream._preloaded_ipadapters:
                print("_setup_ipadapter_from_config: Reusing pre-loaded IPAdapter...")
                ipadapter_pipeline.ipadapter = wrapper.stream._preloaded_ipadapters[0]  # Use first (should only be one)
                
                # Set up style image and scale from config (use first config only)
                ipadapter_configs = _prepare_ipadapter_configs(config)
                if ipadapter_configs and len(ipadapter_configs) > 0:
                    ip_config = ipadapter_configs[0]  # Use only first IPAdapter config
                    if ip_config.get('enabled', True):
                        # Process style image if provided
                        style_image_path = ip_config.get('style_image')
                        if style_image_path:
                            from PIL import Image
                            style_image = Image.open(style_image_path).convert("RGB")
                            ipadapter_pipeline.style_image = style_image
                        
                        # Set scale
                        scale = ip_config.get('scale', 1.0)
                        ipadapter_pipeline.scale = scale
                        if hasattr(ipadapter_pipeline, 'ipadapter') and ipadapter_pipeline.ipadapter:
                            ipadapter_pipeline.ipadapter.set_scale(scale)
                        
                        print(f"_setup_ipadapter_from_config: Configured pre-loaded IPAdapter with scale {scale}")
                        
                        if len(ipadapter_configs) > 1:
                            print(f"_setup_ipadapter_from_config: WARNING - Multiple IPAdapters configured but only first one will be used")
            
            # Copy wrapper attributes to maintain compatibility
            ipadapter_pipeline.batch_size = getattr(wrapper, 'batch_size', 1)
            
            # Store reference to original wrapper for attribute forwarding
            ipadapter_pipeline._original_wrapper = wrapper
            print("_setup_ipadapter_from_config: IPAdapter pipeline setup completed using pre-loaded models")
            
            return ipadapter_pipeline
            
        except Exception as e:
            print(f"_setup_ipadapter_from_config: Error setting up pipeline with pre-loaded models: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"_setup_ipadapter_from_config: Failed to setup IPAdapter pipeline with pre-loaded models: {e}") from e
    
    try:
        print("_setup_ipadapter_from_config: Attempting to import IPAdapterPipeline...")
        
        # Add Diffusers_IPAdapter to path before importing
        import pathlib
        current_file = pathlib.Path(__file__)
        # Diffusers_IPAdapter is now located in the ipadapter directory
        diffusers_ipadapter_path = current_file.parent / "ipadapter" / "Diffusers_IPAdapter"
        print(f"_setup_ipadapter_from_config: Adding Diffusers_IPAdapter to path: {diffusers_ipadapter_path}")
        print(f"_setup_ipadapter_from_config: Diffusers_IPAdapter exists: {diffusers_ipadapter_path.exists()}")
        
        if diffusers_ipadapter_path.exists():
            sys.path.insert(0, str(diffusers_ipadapter_path))
            print("_setup_ipadapter_from_config: Successfully added Diffusers_IPAdapter to Python path")
        else:
            print("_setup_ipadapter_from_config: WARNING: Diffusers_IPAdapter directory not found!")
            print(f"_setup_ipadapter_from_config: Expected location: {diffusers_ipadapter_path}")
        
        # Import here to avoid circular imports
        from .ipadapter import IPAdapterPipeline
        print("_setup_ipadapter_from_config: Successfully imported IPAdapterPipeline")
        
        import torch
        print("_setup_ipadapter_from_config: Successfully imported torch")
        
        # Create IPAdapter pipeline
        device = config.get('device', 'cuda')
        dtype = _parse_dtype(config.get('dtype', 'float16'))
        print(f"_setup_ipadapter_from_config: Creating IPAdapterPipeline with device={device}, dtype={dtype}")
        
        ipadapter_pipeline = IPAdapterPipeline(
            stream_diffusion=wrapper.stream,
            device=device,
            dtype=dtype
        )
        print("_setup_ipadapter_from_config: Successfully created IPAdapterPipeline")
        
        # Set up the IPAdapter (use first config only)
        ipadapter_configs = _prepare_ipadapter_configs(config)
        print(f"_setup_ipadapter_from_config: Found {len(ipadapter_configs)} IPAdapter configs")
        
        if ipadapter_configs and len(ipadapter_configs) > 0:
            ip_config = ipadapter_configs[0]  # Use only first IPAdapter config
            if ip_config.get('enabled', True):
                print(f"_setup_ipadapter_from_config: Setting IPAdapter: {ip_config['ipadapter_model_path']}")
                ipadapter_pipeline.set_ipadapter(
                    ipadapter_model_path=ip_config['ipadapter_model_path'],
                    image_encoder_path=ip_config['image_encoder_path'],
                    style_image=ip_config.get('style_image'),
                    scale=ip_config.get('scale', 1.0)
                )
                print(f"_setup_ipadapter_from_config: Successfully set IPAdapter")
                
                if len(ipadapter_configs) > 1:
                    print(f"_setup_ipadapter_from_config: WARNING - Multiple IPAdapters configured but only first one will be used")
            else:
                print(f"_setup_ipadapter_from_config: IPAdapter is disabled")
        else:
            print(f"_setup_ipadapter_from_config: No IPAdapter configs found")
        
        # Replace wrapper with IPAdapter-enabled pipeline
        # Copy wrapper attributes to maintain compatibility
        ipadapter_pipeline.batch_size = getattr(wrapper, 'batch_size', 1)
        
        # Store reference to original wrapper for attribute forwarding
        ipadapter_pipeline._original_wrapper = wrapper
        print("_setup_ipadapter_from_config: IPAdapter setup completed successfully")
        
        return ipadapter_pipeline
        
    except ImportError as e:
        print(f"_setup_ipadapter_from_config: ImportError - {e}")
        print(f"_setup_ipadapter_from_config: Failed to import IPAdapter module")
        
        # Check if the ipadapter directory exists
        import pathlib
        current_file = pathlib.Path(__file__)
        ipadapter_path = current_file.parent / "ipadapter"
        print(f"_setup_ipadapter_from_config: Looking for IPAdapter at: {ipadapter_path}")
        print(f"_setup_ipadapter_from_config: IPAdapter directory exists: {ipadapter_path.exists()}")
        
        if ipadapter_path.exists():
            print(f"_setup_ipadapter_from_config: Contents of IPAdapter directory:")
            try:
                for item in ipadapter_path.iterdir():
                    print(f"_setup_ipadapter_from_config:   - {item.name}")
            except Exception as dir_e:
                print(f"_setup_ipadapter_from_config: Error listing directory: {dir_e}")
        
        raise ImportError(f"_setup_ipadapter_from_config: IPAdapter module not found. IPAdapter directory exists: {ipadapter_path.exists()}. Original error: {e}") from e
    except Exception as e:
        print(f"_setup_ipadapter_from_config: Unexpected error - {type(e).__name__}: {e}")
        import traceback
        print("_setup_ipadapter_from_config: Full traceback:")
        traceback.print_exc()
        raise RuntimeError(f"_setup_ipadapter_from_config: Failed to setup IPAdapter: {e}") from e


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


# For backwards compatibility, provide simple functions that match expected usage patterns
def get_controlnet_config(config_dict: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Get a specific ControlNet configuration by index
    
    Args:
        config_dict: Full configuration dictionary
        index: Index of the ControlNet to get
        
    Returns:
        ControlNet configuration dictionary
    """
    if 'controlnets' not in config_dict or index >= len(config_dict['controlnets']):
        raise IndexError(f"get_controlnet_config: ControlNet index {index} out of range")
    
    return config_dict['controlnets'][index]


def get_pipeline_type(config_dict: Dict[str, Any]) -> str:
    """
    Get pipeline type from configuration, with fallback to SD 1.5
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Pipeline type string
    """
    return config_dict.get('pipeline_type', 'sd1.5')


def get_ipadapter_config(config_dict: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Get a specific IPAdapter configuration by index
    
    Args:
        config_dict: Full configuration dictionary
        index: Index of the IPAdapter to get
        
    Returns:
        IPAdapter configuration dictionary
    """
    if 'ipadapters' not in config_dict or index >= len(config_dict['ipadapters']):
        raise IndexError(f"get_ipadapter_config: IPAdapter index {index} out of range")
    
    return config_dict['ipadapters'][index]


def load_ipadapter_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load IPAdapter configuration from YAML or JSON file
    
    Alias for load_config() for consistency with ControlNet naming
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    return load_config(config_path)


def save_ipadapter_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save IPAdapter configuration to YAML or JSON file
    
    Alias for save_config() for consistency with ControlNet naming
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    save_config(config, config_path)
