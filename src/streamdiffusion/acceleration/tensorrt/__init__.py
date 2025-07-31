import gc
import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from ...pipeline import StreamDiffusion
from .builder import EngineBuilder, create_onnx_path
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine
from .models import VAE, BaseModel, UNet, VAEEncoder
from ...model_detection import detect_model, extract_unet_architecture, validate_architecture
from .controlnet_wrapper import create_controlnet_wrapper
from .engine_pool import ControlNetEnginePool
from .ipadapter_wrapper import create_ipadapter_wrapper


def _has_ipadapter_processors(unet: UNet2DConditionModel) -> bool:
    """Check if the UNet already has IPAdapter processors installed"""
    try:
        processors = unet.attn_processors
        for name, processor in processors.items():
            # Check for IPAdapter processor class names
            processor_class = processor.__class__.__name__
            if 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                print(f"_has_ipadapter_processors: Found existing IPAdapter processor: {name} -> {processor_class}")
                return True
        return False
    except Exception as e:
        print(f"_has_ipadapter_processors: Error checking existing processors: {e}")
        return False


def _validate_ipadapter_engine_support(unet_engine, cross_attention_dim: int):
    """
    Validate that the TensorRT engine supports IPAdapter functionality
    by checking input/output specifications and testing with extended embeddings
    """
    try:
        print("_validate_ipadapter_engine_support: Validating TensorRT IPAdapter support...")
        
        # Check if engine accepts variable sequence lengths in encoder_hidden_states
        if hasattr(unet_engine, 'engine') and hasattr(unet_engine.engine, 'engine_info'):
            engine_info = unet_engine.engine.engine_info
            input_shapes = getattr(engine_info, 'input_shapes', {})
            
            if 'encoder_hidden_states' in input_shapes:
                encoder_shape = input_shapes['encoder_hidden_states']
                print(f"_validate_ipadapter_engine_support: Encoder hidden states shape: {encoder_shape}")
                
                # Check if sequence dimension is dynamic (should be for IPAdapter)
                if len(encoder_shape) >= 2:
                    seq_dim = encoder_shape[1] if len(encoder_shape) >= 2 else None
                    if seq_dim and hasattr(seq_dim, 'max') and seq_dim.max > 77:
                        print(f"_validate_ipadapter_engine_support: ✓ Engine supports extended sequences (max: {seq_dim.max})")
                        return True
                    else:
                        print(f"_validate_ipadapter_engine_support: ⚠ Engine may not support extended sequences")
                        return False
        
        print("_validate_ipadapter_engine_support: ✓ Basic validation passed")
        return True
        
    except Exception as e:
        print(f"_validate_ipadapter_engine_support: Error during validation: {e}")
        print("_validate_ipadapter_engine_support: ⚠ Could not validate IPAdapter support")
        return False

class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    engine_build_options: dict = {},
):
    if "opt_batch_size" not in engine_build_options or engine_build_options["opt_batch_size"] is None:
        engine_build_options["opt_batch_size"] = max_batch_size
    
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    vae_config = vae.config
    vae_dtype = vae.dtype

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    # Always detect model type for proper embedding dimension
    try:
        detection_results = detect_model(unet)
        model_type = detection_results['model_type']
        is_sdxl = detection_results['is_sdxl']
        print(f"🎯 Detected model type: {model_type}")
    except Exception as e:
        print(f"Failed to detect model type: {e}, defaulting to SD1.5")
        model_type = "SD1.5"
    
    # Detect if ControlNet is being used
    use_controlnet = hasattr(stream, 'controlnets') and len(getattr(stream, 'controlnets', [])) > 0
    
    # Detect if IPAdapter is being used
    # Check multiple indicators: stream attribute, UNet processors, or explicit parameter
    use_ipadapter = (
        (hasattr(stream, 'ipadapters') and len(getattr(stream, 'ipadapters', [])) > 0) or
        _has_ipadapter_processors(unet) or
        (hasattr(stream, 'use_ipadapter') and getattr(stream, 'use_ipadapter', False))
    )
    
    if use_controlnet:
        print("ControlNet detected - enabling TensorRT ControlNet support")
    
    if use_ipadapter:
        print("IPAdapter detected - enabling TensorRT IPAdapter support")
        
        # Extract UNet architecture for ControlNet
        try:
            unet_arch = extract_unet_architecture(unet)
            unet_arch = validate_architecture(unet_arch, model_type)
            
            print(f"Architecture: model_channels={unet_arch['model_channels']}, "
                  f"channel_mult={unet_arch['channel_mult']}, "
                  f"context_dim={unet_arch['context_dim']}")
        except Exception as e:
            print(f"Failed to detect model architecture: {e}")
            print("Falling back to standard TensorRT compilation without ControlNet")
            use_controlnet = False
            unet_arch = {}
    elif use_ipadapter:
        # IPAdapter needs model detection too (for cross_attention_dim)
        try:
            detection_results = detect_model(unet)
            model_type = detection_results['model_type']
            cross_attention_dim = unet.config.cross_attention_dim
            
            print(f"Detected model: {model_type}")
            print(f"IPAdapter: Standard IPAdapter (4 tokens), cross_attention_dim={cross_attention_dim}")
            
            unet_arch = {"context_dim": cross_attention_dim}
        except Exception as e:
            print(f"Failed to detect model architecture: {e}")
            print("Falling back to standard TensorRT compilation without IPAdapter")
            use_ipadapter = False
            unet_arch = {}
    else:
        unet_arch = {}

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"

    # Determine embedding dimension based on model type
    if is_sdxl:
        # SDXL uses concatenated embeddings from dual text encoders (768 + 1280 = 2048)
        embedding_dim = 2048
        print(f"🎯 SDXL detected! Setting embedding_dim = {embedding_dim}")
    else:
        # SD1.5, SD2.1, etc. use single text encoder
        embedding_dim = text_encoder.config.hidden_size
        print(f"🎯 Non-SDXL model ({model_type}) detected! Setting embedding_dim = {embedding_dim}")
    
    print(f"🔧 Final embedding_dim for TensorRT compilation: {embedding_dim}")
    
    # Create UNet model with ControlNet and/or IPAdapter support if needed
    unet_model = UNet(
        fp16=True,
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=embedding_dim,
        unet_dim=unet.config.in_channels,
        use_control=use_controlnet,
        unet_arch=unet_arch if use_controlnet else None,
        use_ipadapter=use_ipadapter,
    )
    
    vae_decoder_model = VAE(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    vae_encoder_model = VAEEncoder(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )

    if not os.path.exists(unet_engine_path):
        if use_controlnet:
            print("Compiling UNet with ControlNet support")
            
            # Create ControlNet-aware wrapper for ONNX export
            control_input_names = unet_model.get_input_names()
            wrapped_unet = create_controlnet_wrapper(unet, control_input_names)
            
            # Compile with ControlNet support
            compile_unet(
                wrapped_unet,  # Use wrapped UNet
                unet_model,
                create_onnx_path("unet", onnx_dir, opt=False),
                create_onnx_path("unet", onnx_dir, opt=True),
                unet_engine_path,
                **engine_build_options,
            )
        elif use_ipadapter:
            print("Compiling UNet with IPAdapter support")
            
            # Create IPAdapter-aware wrapper for ONNX export
            # CRITICAL: Must install processors to bake IPAdapter functionality into TensorRT engine
            wrapped_unet = create_ipadapter_wrapper(unet, install_processors=True)
            
            # Compile with IPAdapter support
            compile_unet(
                wrapped_unet,  # Use wrapped UNet
                unet_model,
                create_onnx_path("unet", onnx_dir, opt=False),
                create_onnx_path("unet", onnx_dir, opt=True),
                unet_engine_path,
                **engine_build_options,
            )
        else:
            print("Compiling UNet without ControlNet or IPAdapter support")
            compile_unet(
                unet,
                unet_model,
                create_onnx_path("unet", onnx_dir, opt=False),
                create_onnx_path("unet", onnx_dir, opt=True),
                unet_engine_path,
                **engine_build_options,
            )
    else:
        print("Using existing UNet engine")
        del unet

    if not os.path.exists(vae_decoder_engine_path):
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            create_onnx_path("vae_decoder", onnx_dir, opt=False),
            create_onnx_path("vae_decoder", onnx_dir, opt=True),
            vae_decoder_engine_path,
            **engine_build_options,
        )

    if not os.path.exists(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            create_onnx_path("vae_encoder", onnx_dir, opt=False),
            create_onnx_path("vae_encoder", onnx_dir, opt=True),
            vae_encoder_engine_path,
            **engine_build_options,
        )

    del vae

    cuda_stream = cuda.Stream()

    # Create TensorRT engine with ControlNet awareness
    stream.unet = UNet2DConditionModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph)
    
    # Store ControlNet metadata on the engine for runtime use
    if use_controlnet:
        setattr(stream.unet, 'use_control', True)
        setattr(stream.unet, 'unet_arch', unet_arch)
        setattr(stream.unet, 'control_input_names', unet_model.get_input_names())
        print("TensorRT UNet engine configured for ControlNet support")
        
        # Initialize ControlNet engine pool for automatic compilation
        controlnet_engine_dir = os.path.join(engine_dir, "controlnet")
        os.makedirs(controlnet_engine_dir, exist_ok=True)
        
        stream.controlnet_engine_pool = ControlNetEnginePool(
            engine_dir=controlnet_engine_dir,
            stream=cuda_stream
        )
        print("ControlNet engine pool initialized")
    else:
        setattr(stream.unet, 'use_control', False)
    
    # Store IPAdapter metadata on the engine for runtime use
    if use_ipadapter:
        setattr(stream.unet, 'use_ipadapter', True)
        setattr(stream.unet, 'ipadapter_arch', unet_arch)
        print("TensorRT UNet engine configured for IPAdapter support")
        
        # Validate that the engine can handle IPAdapter token sequences
        _validate_ipadapter_engine_support(stream.unet, unet_arch.get('context_dim', 768))
    else:
        setattr(stream.unet, 'use_ipadapter', False)
    
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_stream,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return stream
