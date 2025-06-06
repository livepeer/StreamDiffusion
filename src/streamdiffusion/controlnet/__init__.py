from .base_controlnet_pipeline import BaseControlNetPipeline
from .controlnet_pipeline import ControlNetPipeline
from .controlnet_sdxlturbo_pipeline import SDXLTurboControlNetPipeline
from .config import ControlNetConfig, StreamDiffusionControlNetConfig, load_controlnet_config
from .preprocessors import (
    BasePreprocessor,
    CannyPreprocessor,
    DepthPreprocessor,
    OpenPosePreprocessor,
    LineartPreprocessor,
    get_preprocessor,
)


def create_controlnet_pipeline(config: StreamDiffusionControlNetConfig):
    """
    Create a ControlNet-enabled StreamDiffusion pipeline from configuration
    
    Args:
        config: StreamDiffusionControlNetConfig object
        
    Returns:
        ControlNet pipeline (ControlNetPipeline or SDXLTurboControlNetPipeline)
    """
    import torch
    from diffusers import StableDiffusionPipeline, AutoencoderTiny
    from ..pipeline import StreamDiffusion
    
    # Determine pipeline type
    is_sdxl = config.pipeline_type == "sdxlturbo"
    
    # Load base pipeline
    if is_sdxl:
        from diffusers import StableDiffusionXLPipeline
        if config.model_id.endswith(('.safetensors', '.ckpt', '.pt')):
            pipe = StableDiffusionXLPipeline.from_single_file(
                config.model_id, torch_dtype=getattr(torch, config.dtype)
            ).to(config.device)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                config.model_id, torch_dtype=getattr(torch, config.dtype)
            ).to(config.device)
    else:
        if config.model_id.endswith(('.safetensors', '.ckpt', '.pt')):
            pipe = StableDiffusionPipeline.from_single_file(
                config.model_id, torch_dtype=getattr(torch, config.dtype)
            ).to(config.device)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                config.model_id, torch_dtype=getattr(torch, config.dtype)
            ).to(config.device)
    
    # Replace VAE if using tiny VAE
    if config.use_tiny_vae:
        if is_sdxl:
            taesd_model = "madebyollin/taesdxl"
        else:
            taesd_model = "madebyollin/taesd"
        pipe.vae = AutoencoderTiny.from_pretrained(taesd_model).to(
            device=pipe.device, dtype=pipe.dtype
        )
    
    # Create StreamDiffusion instance
    stream = StreamDiffusion(
        pipe=pipe,
        t_index_list=config.t_index_list,
        torch_dtype=getattr(torch, config.dtype),
        width=config.width,
        height=config.height,
        do_add_noise=True,
        frame_buffer_size=1,
        use_denoising_batch=True,
        cfg_type=config.cfg_type,
    )
    
    # Apply TensorRT acceleration if requested
    if config.acceleration == "tensorrt":
        from ..acceleration.tensorrt import accelerate_with_tensorrt
        
        # Set dummy controlnets attribute for TensorRT compilation
        if config.controlnets:
            setattr(stream, "controlnets", [cn.model_id for cn in config.controlnets])
        
        stream = accelerate_with_tensorrt(
            stream,
            engine_dir="engines",
            max_batch_size=1,
            min_batch_size=1,
            use_cuda_graph=False,
        )
    elif config.acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    
    # Create appropriate ControlNet pipeline
    if is_sdxl:
        controlnet_pipeline = SDXLTurboControlNetPipeline(
            stream, config.device, getattr(torch, config.dtype)
        )
    else:
        controlnet_pipeline = ControlNetPipeline(
            stream, config.device, getattr(torch, config.dtype)
        )
    
    # Add ControlNets from config
    for cn_config in config.controlnets:
        if cn_config.enabled:
            controlnet_pipeline.add_controlnet(cn_config)
    
    # Prepare with prompt and settings
    stream.prepare(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        seed=config.seed,
    )
    
    return controlnet_pipeline


__all__ = [
    # Pipeline classes
    "BaseControlNetPipeline",
    "ControlNetPipeline",
    "SDXLTurboControlNetPipeline",
    
    # Configuration classes
    "ControlNetConfig",
    "StreamDiffusionControlNetConfig", 
    "load_controlnet_config",
    
    # Pipeline creation
    "create_controlnet_pipeline",
    
    # Preprocessors
    "BasePreprocessor",
    "CannyPreprocessor", 
    "DepthPreprocessor",
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "get_preprocessor",
] 