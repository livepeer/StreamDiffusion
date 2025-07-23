"""
NVIDIA-optimized TensorRT builders based on demo_diffusion
Implements latest TensorRT optimizations and best practices
"""

import os
import gc
import torch
import tensorrt as trt
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# ONNX and TensorRT imports
try:
    import onnx
    import onnx_graphsurgeon as gs
    from polygraphy.backend.trt import CreateConfig, Profile
    from polygraphy.backend.onnx import OnnxFromPath
    from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnx, TrtRunner
    from polygraphy import cuda
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing TensorRT dependencies: {e}")
    print("Install: pip install tensorrt onnx onnx-graphsurgeon polygraphy[all]")
    DEPENDENCIES_AVAILABLE = False


class SDXLTurboUNetWrapper(torch.nn.Module):
    """Wrapper for SDXL-Turbo to handle missing conditioning"""
    
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        """Forward pass that handles optional SDXL conditioning"""
        try:
            # Try with added_cond_kwargs first
            if text_embeds is not None and time_ids is not None:
                added_cond_kwargs = {
                    'text_embeds': text_embeds,
                    'time_ids': time_ids
                }
                return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
            else:
                # For SDXL models, always provide an empty dict instead of None
                return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs={})
        except Exception as e:
            # Final fallback - basic UNet call with empty added_cond_kwargs
            print(f"⚠️ Using basic UNet call due to: {e}")
            try:
                return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs={})
            except:
                # Absolute fallback without any conditioning
                return self.unet(sample, timestep, encoder_hidden_states)


class NVIDIATensorRTBuilder:
    """NVIDIA-optimized TensorRT engine builder"""
    
    def __init__(self, use_fp16: bool = True, use_fp8: bool = False):
        self.use_fp16 = use_fp16
        self.use_fp8 = use_fp8
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Initialize TensorRT plugins
        trt.init_libnvinfer_plugins(self.trt_logger, "")
    
    def build_engine_from_onnx(self, 
                               onnx_path: str,
                               engine_path: str,
                               input_profiles: Dict[str, Tuple],
                               max_workspace_size: int = 8 << 30):  # 8GB
        """
        Build TensorRT engine from ONNX model using NVIDIA's optimized settings
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Output engine path
            input_profiles: Input profile specifications
            max_workspace_size: Maximum workspace size in bytes
        """
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("TensorRT dependencies not available")
            
        print(f"🔧 Building TensorRT engine: {Path(engine_path).name}")
        
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parser Error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Create builder config with NVIDIA optimizations
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        
        # Enable optimizations
        if self.use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 precision enabled")
        
        if self.use_fp8 and hasattr(trt.BuilderFlag, 'FP8'):
            config.set_flag(trt.BuilderFlag.FP8)
            print("✅ FP8 precision enabled")
        
        # Additional NVIDIA optimizations
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        # Set optimization profiles
        profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in input_profiles.items():
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        print(f"⚙️ Building engine with workspace size: {max_workspace_size // (1024**3)}GB")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ Engine saved: {engine_path}")
        
        # Cleanup
        del builder, network, parser, config, profile
        gc.collect()


def export_onnx_unet(unet_model, 
                     onnx_path: str,
                     is_sdxl: bool = False,
                     batch_size: int = 2,
                     num_channels: int = 4,
                     height: int = 64,
                     width: int = 64):
    """Export UNet to ONNX with proper inputs for SDXL or SD1.5"""
    print(f"📤 Exporting UNet to ONNX: {onnx_path}")
    
    # Prepare sample inputs
    sample_shape = (batch_size, num_channels, height, width)
    timestep_shape = (batch_size,)
    
    if is_sdxl:
        # SDXL uses larger context dimensions and additional conditioning
        encoder_hidden_states_shape = (batch_size, 77, 2048)  # Concatenated embeddings
        added_cond_kwargs = {
            'text_embeds': torch.randn(batch_size, 1280, dtype=torch.float16, device='cuda'),  # Pooled embeddings
            'time_ids': torch.randn(batch_size, 6, dtype=torch.float16, device='cuda')  # Size conditioning
        }
    else:
        encoder_hidden_states_shape = (batch_size, 77, 768)
        added_cond_kwargs = None
    
    # Create dummy inputs
    sample = torch.randn(*sample_shape, dtype=torch.float16, device='cuda')
    timestep = torch.randint(0, 1000, timestep_shape, dtype=torch.long, device='cuda')
    encoder_hidden_states = torch.randn(*encoder_hidden_states_shape, dtype=torch.float16, device='cuda')
    
    # Prepare model for export
    unet_model.eval()
    unet_model = unet_model.to(device='cuda', dtype=torch.float16)
    
    # Fix SDXL-Turbo issues: wrap model to handle conditioning properly
    if is_sdxl:
        # For SDXL models, use our wrapper that handles conditioning gracefully
        wrapped_model = SDXLTurboUNetWrapper(unet_model)
        
        # Test what the model expects
        try:
            with torch.no_grad():
                _ = wrapped_model(sample[:1], timestep[:1], encoder_hidden_states[:1], 
                                added_cond_kwargs['text_embeds'][:1], added_cond_kwargs['time_ids'][:1])
            use_added_cond = True
            print("✅ Model supports SDXL conditioning")
        except Exception as e:
            print(f"⚠️ Model test with conditioning failed, using basic mode: {e}")
            use_added_cond = False
    else:
        wrapped_model = unet_model
        use_added_cond = False
    
    # Dynamic axes for flexibility
    dynamic_axes = {
        'sample': {0: 'batch_size'},
        'timestep': {0: 'batch_size'}, 
        'encoder_hidden_states': {0: 'batch_size'},
        'noise_pred': {0: 'batch_size'}
    }
    
    if is_sdxl and use_added_cond and added_cond_kwargs:
        dynamic_axes.update({
            'text_embeds': {0: 'batch_size'},
            'time_ids': {0: 'batch_size'}
        })
    
    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    with torch.no_grad():
        if is_sdxl and use_added_cond and added_cond_kwargs:
            # SDXL with additional conditioning
            print("Exporting SDXL UNet with additional conditioning...")
            torch.onnx.export(
                wrapped_model,
                (sample, timestep, encoder_hidden_states, added_cond_kwargs['text_embeds'], added_cond_kwargs['time_ids']),
                onnx_path,
                input_names=['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids'],
                output_names=['noise_pred'],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True
            )
        else:
            # SD1.5 or SDXL-Turbo without additional conditioning
            print("Exporting UNet without additional conditioning...")
            torch.onnx.export(
                wrapped_model,
                (sample, timestep, encoder_hidden_states),
                onnx_path,
                input_names=['sample', 'timestep', 'encoder_hidden_states'],
                output_names=['noise_pred'],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True
            )
        
        # Convert to external data format for SDXL models after export
        if is_sdxl:
            print("🔧 Converting SDXL model to external data format...")
            import os
            onnx_model = onnx.load(onnx_path)
            
            if onnx_model.ByteSize() > 2147483648:  # 2GB
                print(f"   Model size: {onnx_model.ByteSize() / (1024**3):.2f} GB - using external data format")
                onnx.save_model(
                    onnx_model,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.pb",
                    convert_attribute=False,
                )
                print(f"✅ Converted to external data format")
            
            del onnx_model
    
    print(f"✅ UNet ONNX export complete: {onnx_path}")


def build_unet_engine(unet_model,
                      engine_path: str,
                      max_batch_size: int = 2,
                      is_sdxl: bool = False,
                      use_fp16: bool = True,
                      use_fp8: bool = False):
    """Build optimized UNet TensorRT engine"""
    
    # Create ONNX path in temp directory to avoid conflicts
    import tempfile
    temp_dir = tempfile.mkdtemp()
    onnx_path = os.path.join(temp_dir, "unet_temp.onnx")
    
    # Export to ONNX
    export_onnx_unet(unet_model, onnx_path, is_sdxl, max_batch_size)
    
    # Define input profiles based on model type
    if is_sdxl:
        latent_height, latent_width = 128, 128  # 1024x1024 images
        context_dim = 2048
        input_profiles = {
            'sample': ((1, 4, latent_height//2, latent_width//2), 
                      (max_batch_size, 4, latent_height, latent_width),
                      (max_batch_size*2, 4, latent_height*2, latent_width*2)),
            'timestep': ((1,), (max_batch_size,), (max_batch_size*2,)),
            'encoder_hidden_states': ((1, 77, context_dim), 
                                    (max_batch_size, 77, context_dim),
                                    (max_batch_size*2, 77, context_dim)),
            'text_embeds': ((1, 1280), (max_batch_size, 1280), (max_batch_size*2, 1280)),
            'time_ids': ((1, 6), (max_batch_size, 6), (max_batch_size*2, 6))
        }
    else:
        latent_height, latent_width = 64, 64  # 512x512 images
        context_dim = 768
        input_profiles = {
            'sample': ((1, 4, latent_height//2, latent_width//2),
                      (max_batch_size, 4, latent_height, latent_width),
                      (max_batch_size*2, 4, latent_height*2, latent_width*2)),
            'timestep': ((1,), (max_batch_size,), (max_batch_size*2,)),
            'encoder_hidden_states': ((1, 77, context_dim),
                                    (max_batch_size, 77, context_dim), 
                                    (max_batch_size*2, 77, context_dim))
        }
    
    # Build engine
    builder = NVIDIATensorRTBuilder(use_fp16=use_fp16, use_fp8=use_fp8)
    builder.build_engine_from_onnx(onnx_path, engine_path, input_profiles)
    
    # Cleanup temporary files
    import shutil
    shutil.rmtree(temp_dir)


def build_vae_encoder_engine(vae_model,
                            engine_path: str,
                            max_batch_size: int = 2,
                            use_fp16: bool = True):
    """Build VAE Encoder TensorRT engine"""
    
    onnx_path = engine_path.replace('.engine', '_encoder.onnx')
    
    print(f"📤 Exporting VAE Encoder to ONNX: {onnx_path}")
    
    # Prepare encoder inputs (RGB images)
    sample_input = torch.randn(max_batch_size, 3, 512, 512, dtype=torch.float16, device='cuda')
    
    # Wrap encoder for ONNX export
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, x):
            return self.vae.encode(x).latent_dist.sample()
    
    encoder_wrapper = VAEEncoderWrapper(vae_model).eval().cuda()
    
    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            sample_input,
            onnx_path,
            input_names=['images'],
            output_names=['latents'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'latents': {0: 'batch_size'}
            },
            opset_version=17
        )
    
    # Build engine
    input_profiles = {
        'images': ((1, 3, 512, 512), (max_batch_size, 3, 512, 512), (max_batch_size*2, 3, 512, 512))
    }
    
    builder = NVIDIATensorRTBuilder(use_fp16=use_fp16, use_fp8=False)
    builder.build_engine_from_onnx(onnx_path, engine_path, input_profiles)
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)


def build_vae_decoder_engine(vae_model,
                            engine_path: str,
                            max_batch_size: int = 2,
                            use_fp16: bool = True):
    """Build VAE Decoder TensorRT engine"""
    
    onnx_path = engine_path.replace('.engine', '_decoder.onnx')
    
    print(f"📤 Exporting VAE Decoder to ONNX: {onnx_path}")
    
    # Prepare decoder inputs (latents)
    sample_input = torch.randn(max_batch_size, 4, 64, 64, dtype=torch.float16, device='cuda')
    
    # Use decoder directly
    vae_model.eval().cuda()
    
    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            vae_model.decode,
            sample_input,
            onnx_path,
            input_names=['latents'],
            output_names=['images'],
            dynamic_axes={
                'latents': {0: 'batch_size'},
                'images': {0: 'batch_size'}
            },
            opset_version=17
        )
    
    # Build engine  
    input_profiles = {
        'latents': ((1, 4, 64, 64), (max_batch_size, 4, 64, 64), (max_batch_size*2, 4, 128, 128))
    }
    
    builder = NVIDIATensorRTBuilder(use_fp16=use_fp16, use_fp8=False)
    builder.build_engine_from_onnx(onnx_path, engine_path, input_profiles)
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)


def build_text_encoder_engine(text_encoder_model,
                             engine_path: str,
                             max_batch_size: int = 2,
                             use_fp16: bool = True):
    """Build Text Encoder TensorRT engine"""
    
    onnx_path = engine_path.replace('.engine', '_text_encoder.onnx')
    
    print(f"📤 Exporting Text Encoder to ONNX: {onnx_path}")
    
    # Prepare text encoder inputs
    input_ids = torch.randint(0, 1000, (max_batch_size, 77), dtype=torch.long, device='cuda')
    
    text_encoder_model.eval().cuda()
    
    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            text_encoder_model,
            input_ids,
            onnx_path,
            input_names=['input_ids'],
            output_names=['last_hidden_state', 'pooler_output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size'},
                'pooler_output': {0: 'batch_size'}
            },
            opset_version=17
        )
    
    # Build engine
    input_profiles = {
        'input_ids': ((1, 77), (max_batch_size, 77), (max_batch_size*2, 77))
    }
    
    builder = NVIDIATensorRTBuilder(use_fp16=use_fp16, use_fp8=False)
    builder.build_engine_from_onnx(onnx_path, engine_path, input_profiles)
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path) 