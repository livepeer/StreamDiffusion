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
from .model_detection import detect_model_from_diffusers_unet, extract_unet_architecture, validate_architecture
from .controlnet_wrapper import create_controlnet_wrapper
from .engine_pool import ControlNetEnginePool


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
    engine_build_options: dict = {},
):
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    engine_build_options: dict = {},
):
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    use_dynamic_batch: bool = False,
    frame_buffer_size: int = 1,
    use_denoising_batch: bool = True,
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

    # Detect if ControlNet is being used
    use_controlnet = hasattr(stream, 'controlnets') and len(getattr(stream, 'controlnets', [])) > 0
    
    if use_controlnet:
        print("ControlNet detected - enabling TensorRT ControlNet support")
        
        # Detect model architecture
        try:
            model_type = detect_model_from_diffusers_unet(unet)
            unet_arch = extract_unet_architecture(unet)
            unet_arch = validate_architecture(unet_arch, model_type)
            
            print(f"Detected model: {model_type}")
            print(f"Architecture: model_channels={unet_arch['model_channels']}, "
                  f"channel_mult={unet_arch['channel_mult']}, "
                  f"context_dim={unet_arch['context_dim']}")
        except Exception as e:
            print(f"Failed to detect model architecture: {e}")
            print("Falling back to standard TensorRT compilation without ControlNet")
            use_controlnet = False
            unet_arch = {}
    else:
        unet_arch = {}

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"

    # Import dynamic models if using dynamic batching
    if use_dynamic_batch and use_denoising_batch:
        from .dynamic_models import DynamicUNet, DynamicVAE, DynamicVAEEncoder
        from .builder import compile_dynamic_unet, compile_dynamic_vae
        
        print(f"\n=== DYNAMIC BATCH CONFIGURATION ===")
        print(f"Original parameters:")
        print(f"  frame_buffer_size: {frame_buffer_size}")
        print(f"  min_batch_size: {min_batch_size}")
        print(f"  max_batch_size: {max_batch_size}")
        print(f"  use_denoising_batch: {use_denoising_batch}")
        
        # Calculate optimal batch size based on frame buffer size
        opt_batch_size = frame_buffer_size
        
        # UNet uses 2x batch size for CFG, so we need to account for this in our batch size ranges
        unet_min_batch = 2 * min_batch_size
        unet_opt_batch = 2 * opt_batch_size
        unet_max_batch = 2 * max_batch_size
        
        # VAE uses regular batch sizes
        vae_min_batch = min_batch_size
        vae_opt_batch = opt_batch_size
        vae_max_batch = max_batch_size
        
        # Update engine_build_options to reflect the UNet's actual optimal batch size
        engine_build_options["opt_batch_size"] = unet_opt_batch
        
        print(f"Calculated batch sizes:")
        print(f"  UNet batch sizes: min={unet_min_batch}, opt={unet_opt_batch}, max={unet_max_batch}")
        print(f"  VAE batch sizes: min={vae_min_batch}, opt={vae_opt_batch}, max={vae_max_batch}")
        print(f"Updated engine_build_options:")
        for key, value in engine_build_options.items():
            print(f"  {key}: {value}")
        print("=====================================\n")
        
        # Create dynamic UNet model with ControlNet support if needed
        unet_model = DynamicUNet(
            fp16=True,
            device=stream.device,
            max_batch_size=unet_max_batch,    # Use UNet-specific max batch
            min_batch_size=unet_min_batch,    # Use UNet-specific min batch
            opt_batch_size=unet_opt_batch,    # Use UNet-specific opt batch
            embedding_dim=text_encoder.config.hidden_size,
            unet_dim=unet.config.in_channels,
            use_control=use_controlnet,
            unet_arch=unet_arch if use_controlnet else None,
            enable_dynamic_batch=True,
        )
        
        vae_decoder_model = DynamicVAE(
            device=stream.device,
            max_batch_size=vae_max_batch,     # Use VAE-specific max batch
            min_batch_size=vae_min_batch,     # Use VAE-specific min batch  
            opt_batch_size=vae_opt_batch,     # Use VAE-specific opt batch
            enable_dynamic_batch=True,
        )
        vae_encoder_model = DynamicVAEEncoder(
            device=stream.device,
            max_batch_size=vae_max_batch,     # Use VAE-specific max batch
            min_batch_size=vae_min_batch,     # Use VAE-specific min batch
            opt_batch_size=vae_opt_batch,     # Use VAE-specific opt batch
            enable_dynamic_batch=True,
        )
    else:
        # Use standard models for fixed batch size
        unet_model = UNet(
            fp16=True,
            device=stream.device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=text_encoder.config.hidden_size,
            unet_dim=unet.config.in_channels,
            use_control=use_controlnet,
            unet_arch=unet_arch if use_controlnet else None,
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
        if use_dynamic_batch and use_denoising_batch:
            print("Compiling dynamic UNet with batch denoising support")
            if use_controlnet:
                print("- With ControlNet support")
                # Create ControlNet-aware wrapper for ONNX export
                control_input_names = unet_model.get_input_names()
                wrapped_unet = create_controlnet_wrapper(unet, control_input_names)
                
                # Compile with dynamic batch and ControlNet support
                compile_dynamic_unet(
                    wrapped_unet,
                    unet_model,
                    create_onnx_path("unet", onnx_dir, opt=False),
                    create_onnx_path("unet", onnx_dir, opt=True),
                    unet_engine_path,
                    min_batch_size=unet_min_batch,    # Use UNet-specific batch sizes
                    max_batch_size=unet_max_batch,    # Use UNet-specific batch sizes  
                    engine_build_options=engine_build_options,
                )
            else:
                # Compile with dynamic batch support only
                compile_dynamic_unet(
                    unet,
                    unet_model,
                    create_onnx_path("unet", onnx_dir, opt=False),
                    create_onnx_path("unet", onnx_dir, opt=True),
                    unet_engine_path,
                    min_batch_size=unet_min_batch,    # Use UNet-specific batch sizes
                    max_batch_size=unet_max_batch,    # Use UNet-specific batch sizes
                    engine_build_options=engine_build_options,
                )
        else:
            # Standard fixed batch compilation
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
            else:
                print("Compiling UNet without ControlNet support")
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
        if use_dynamic_batch and use_denoising_batch:
            print("Compiling dynamic VAE decoder with batch denoising support")
            compile_dynamic_vae(
                vae,
                vae_decoder_model,
                create_onnx_path("vae_decoder", onnx_dir, opt=False),
                create_onnx_path("vae_decoder", onnx_dir, opt=True),
                vae_decoder_engine_path,
                min_batch_size=vae_min_batch,     # Use VAE-specific batch sizes
                max_batch_size=vae_max_batch,     # Use VAE-specific batch sizes
                engine_build_options=engine_build_options,
            )
        else:
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
        if use_dynamic_batch and use_denoising_batch:
            print("Compiling dynamic VAE encoder with batch denoising support")
            compile_dynamic_vae(
                vae_encoder,
                vae_encoder_model,
                create_onnx_path("vae_encoder", onnx_dir, opt=False),
                create_onnx_path("vae_encoder", onnx_dir, opt=True),
                vae_encoder_engine_path,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                engine_build_options=engine_build_options,
            )
        else:
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

    # Create TensorRT engine with dynamic batch support if enabled
    if use_dynamic_batch and use_denoising_batch:
        from .dynamic_engine import DynamicUNet2DConditionModelEngine, DynamicAutoencoderKLEngine
        
        print("Creating dynamic TensorRT engines for batched denoising")
        stream.unet = DynamicUNet2DConditionModelEngine(
            unet_engine_path, 
            cuda_stream, 
            use_cuda_graph=use_cuda_graph,
            enable_dynamic_batch=True,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )
        
        stream.vae = DynamicAutoencoderKLEngine(
            vae_encoder_engine_path,
            vae_decoder_engine_path,
            cuda_stream,
            stream.pipe.vae_scale_factor,
            use_cuda_graph=use_cuda_graph,
            enable_dynamic_batch=True,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )
        
        # Store dynamic batch metadata
        setattr(stream.unet, 'dynamic_batch_enabled', True)
        setattr(stream.vae, 'dynamic_batch_enabled', True)
        print(f"Dynamic batch size range: [{min_batch_size}, {max_batch_size}], optimal: {opt_batch_size}")
    else:
        # Use standard engines
        stream.unet = UNet2DConditionModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph)
        
        stream.vae = AutoencoderKLEngine(
            vae_encoder_engine_path,
            vae_decoder_engine_path,
            cuda_stream,
            stream.pipe.vae_scale_factor,
            use_cuda_graph=use_cuda_graph,
        )
        
        setattr(stream.unet, 'dynamic_batch_enabled', False)
        setattr(stream.vae, 'dynamic_batch_enabled', False)
    
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
    
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return stream
