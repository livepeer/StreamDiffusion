"""
NVIDIA TensorRT Runtime for StreamDiffusion
Optimized engine loading and execution based on NVIDIA's demo_diffusion
"""

import torch
import tensorrt as trt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import gc

def check_tensorrt_dependencies():
    """Check if all required TensorRT dependencies are available"""
    missing_deps = []
    
    try:
        import tensorrt as trt
    except ImportError:
        missing_deps.append("tensorrt")
    
    try:
        import pycuda.driver as cuda_driver
        import pycuda.autoinit
    except ImportError:
        missing_deps.append("pycuda")
    
    try:
        from polygraphy import cuda
    except ImportError:
        missing_deps.append("polygraphy[all]")
    
    try:
        from polygraphy.backend.trt import NetworkFromOnnx
    except ImportError:
        missing_deps.append("polygraphy[all] (TensorRT backend)")
    
    try:
        import onnx
        import onnx_graphsurgeon as gs
    except ImportError:
        missing_deps.append("onnx onnx-graphsurgeon")
    
    if missing_deps:
        print(f"🔍 Missing TensorRT dependencies: {', '.join(missing_deps)}")
        print("📥 Install with: pip install tensorrt>=8.6.0 pycuda polygraphy[all] onnx onnx-graphsurgeon")
        print("💡 Note: TensorRT requires CUDA toolkit and compatible GPU drivers")
        return False
    
    return True

# Check dependencies at import time
TENSORRT_AVAILABLE = check_tensorrt_dependencies()

# Import dependencies if available, otherwise create placeholders
if TENSORRT_AVAILABLE:
    try:
        from polygraphy import cuda
        import pycuda.driver as cuda_driver
        import pycuda.autoinit
    except ImportError as e:
        print(f"Secondary import failed: {e}")
        TENSORRT_AVAILABLE = False
        # Create placeholder for type hints
        cuda = None
else:
    # Create placeholder for type hints when dependencies are missing
    cuda = None


class TensorRTEngine:
    """Optimized TensorRT engine wrapper"""
    
    def __init__(self, engine_path: str, stream: Optional['cuda.Stream'] = None):
        """Initialize TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT dependencies not available")
            
        self.engine_path = engine_path
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine with better OOM error handling
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
                engine_size_gb = len(engine_data) / (1024**3)
                print(f"🔧 Loading TensorRT engine ({engine_size_gb:.2f} GB): {engine_path}")
                
                # Check available GPU memory before loading
                if cuda and hasattr(cuda, 'mem_get_info'):
                    try:
                        free_mem, total_mem = cuda.mem_get_info()
                        free_gb = free_mem / (1024**3)
                        total_gb = total_mem / (1024**3)
                        print(f"   📊 GPU Memory: {free_gb:.2f}GB free / {total_gb:.2f}GB total")
                        
                        if free_gb < engine_size_gb * 1.2:  # Need ~20% overhead
                            print(f"   ⚠️ Warning: Engine size ({engine_size_gb:.2f}GB) may not fit in available memory ({free_gb:.2f}GB)")
                    except:
                        pass
                
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_data)
                
        except Exception as e:
            error_msg = str(e).lower()
            if 'out of memory' in error_msg or 'outofmemory' in error_msg:
                # This is a CUDA OOM error during engine loading
                print(f"\n❌ TensorRT Engine Out of Memory Error!")
                print(f"   Engine path: {engine_path}")
                
                if cuda and hasattr(cuda, 'mem_get_info'):
                    try:
                        free_mem, total_mem = cuda.mem_get_info()
                        free_gb = free_mem / (1024**3)
                        total_gb = total_mem / (1024**3)
                        print(f"   📊 Available GPU Memory: {free_gb:.2f}GB / {total_gb:.2f}GB")
                    except:
                        pass
                
                print(f"\n💡 Suggestions to fix this:")
                print(f"   1. Clean up GPU memory first: wrapper.cleanup_gpu_memory()")
                print(f"   2. Reduce batch size in your config (try batch_size=1)")
                print(f"   3. Use smaller model resolution (e.g., 512x512 instead of 1024x1024)")
                print(f"   4. Delete existing engines and rebuild with smaller settings:")
                print(f"      rmdir /s engines  # Windows")
                print(f"      rm -rf engines    # Linux/Mac")
                print(f"   5. Use --lowvram or --medvram if available")
                print(f"   6. Close other GPU applications")
                
                raise RuntimeError(f"TensorRT engine too large for available GPU memory. Engine requires ~{engine_size_gb:.2f}GB but only {free_gb:.2f}GB available. See suggestions above.")
            else:
                raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}. Error: {e}")
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        self.stream = stream or cuda.Stream() if cuda else None
        
        # Initialize bindings
        self._setup_bindings()
        
        print(f"✅ Loaded TensorRT engine: {engine_path}")
    
    def _setup_bindings(self):
        """Setup input/output bindings for efficient execution"""
        self.inputs = {}
        self.outputs = {}
        self.bindings = [None] * self.engine.num_io_tensors
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs[tensor_name] = {
                    'index': i,
                    'dtype': dtype,
                    'buffer': None
                }
            else:
                self.outputs[tensor_name] = {
                    'index': i, 
                    'dtype': dtype,
                    'buffer': None
                }
    
    def allocate_buffers(self, input_shapes: Dict[str, Tuple]):
        """Allocate GPU buffers for inputs and outputs"""
        # Allocate input buffers
        for name, shape in input_shapes.items():
            if name in self.inputs:
                self.context.set_input_shape(name, shape)
                self.inputs[name]['buffer'] = cuda.DeviceArray(
                    shape=shape, 
                    dtype=self.inputs[name]['dtype']
                )
                self.bindings[self.inputs[name]['index']] = self.inputs[name]['buffer'].ptr
        
        # Allocate output buffers based on inferred shapes
        for name, info in self.outputs.items():
            output_shape = self.context.get_tensor_shape(name)
            info['buffer'] = cuda.DeviceArray(
                shape=output_shape,
                dtype=info['dtype']
            )
            self.bindings[info['index']] = info['buffer'].ptr
    
    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference with the TensorRT engine"""
        # Copy inputs to GPU buffers
        input_shapes = {}
        for name, tensor in inputs.items():
            if name in self.inputs:
                input_shapes[name] = tensor.shape
                # Copy tensor to device buffer
                cuda.memcpy_htod_async(
                    self.inputs[name]['buffer'].ptr,
                    tensor.detach().cpu().numpy().ascontiguousarray(),
                    self.stream
                )
        
        # Allocate buffers if shapes changed
        if not all(self.inputs[name]['buffer'] is not None for name in input_shapes.keys()):
            self.allocate_buffers(input_shapes)
        
        # Execute inference
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Copy outputs back to CPU/GPU tensors
        outputs = {}
        for name, info in self.outputs.items():
            output_array = np.empty(info['buffer'].shape, dtype=info['dtype'])
            cuda.memcpy_dtoh_async(output_array, info['buffer'].ptr, self.stream)
            self.stream.synchronize()
            
            # Convert to torch tensor
            outputs[name] = torch.from_numpy(output_array).cuda()
        
        return outputs
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


class TensorRTUNet(torch.nn.Module):
    """TensorRT-optimized UNet replacement"""
    
    def __init__(self, engine_path: str, is_sdxl: bool = False):
        super().__init__()
        self.engine = TensorRTEngine(engine_path)
        self.is_sdxl = is_sdxl
    
    def forward(self, 
                sample: torch.Tensor,
                timestep: torch.Tensor, 
                encoder_hidden_states: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward pass through TensorRT UNet"""
        
        # Prepare inputs
        inputs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states
        }
        
        # Add SDXL-specific inputs
        if self.is_sdxl:
            if 'added_cond_kwargs' in kwargs:
                added_cond = kwargs['added_cond_kwargs']
                if 'text_embeds' in added_cond:
                    inputs['text_embeds'] = added_cond['text_embeds']
                if 'time_ids' in added_cond:
                    inputs['time_ids'] = added_cond['time_ids']
        
        # Run inference
        outputs = self.engine.infer(inputs)
        
        # Return noise prediction
        return outputs['noise_pred']


class TensorRTVAE:
    """TensorRT-optimized VAE replacement"""
    
    def __init__(self, encoder_path: str, decoder_path: str, scale_factor: int = 8):
        self.encoder_engine = TensorRTEngine(encoder_path)
        self.decoder_engine = TensorRTEngine(decoder_path)
        self.scale_factor = scale_factor
    
    def encode(self, images: torch.Tensor):
        """Encode images to latents"""
        outputs = self.encoder_engine.infer({'images': images})
        
        # Return mock distribution for compatibility
        class MockDistribution:
            def __init__(self, latents):
                self.latents = latents
            
            def sample(self):
                return self.latents
        
        return MockDistribution(outputs['latents'])
    
    def decode(self, latents: torch.Tensor):
        """Decode latents to images"""
        outputs = self.decoder_engine.infer({'latents': latents})
        
        # Return mock output for compatibility  
        class MockOutput:
            def __init__(self, sample):
                self.sample = sample
        
        return MockOutput(outputs['images'])


class TensorRTTextEncoder:
    """TensorRT-optimized Text Encoder replacement"""
    
    def __init__(self, engine_path: str):
        self.engine = TensorRTEngine(engine_path)
        
        # Mock config for compatibility
        class MockConfig:
            hidden_size = 768  # Will be updated based on actual model
        
        self.config = MockConfig()
    
    def __call__(self, input_ids: torch.Tensor, **kwargs):
        """Encode text tokens"""
        outputs = self.engine.infer({'input_ids': input_ids})
        
        # Return compatible output format
        return type('TextEncoderOutput', (), {
            'last_hidden_state': outputs['last_hidden_state'],
            'pooler_output': outputs.get('pooler_output', None)
        })()


def load_tensorrt_engines(engine_paths: Dict[str, str], is_sdxl: bool = False) -> Dict[str, Any]:
    """
    Load all TensorRT engines and create optimized components
    
    Args:
        engine_paths: Dictionary mapping component names to engine paths
        is_sdxl: Whether this is an SDXL model
        
    Returns:
        Dictionary of TensorRT-optimized components
    """
    print("🔄 Loading TensorRT engines...")
    
    components = {}
    
    # Load UNet
    if 'unet' in engine_paths:
        components['unet'] = TensorRTUNet(engine_paths['unet'], is_sdxl=is_sdxl)
        print("✅ UNet engine loaded")
    
    # Load VAE (encoder + decoder)
    if 'vae_encoder' in engine_paths and 'vae_decoder' in engine_paths:
        components['vae'] = TensorRTVAE(
            engine_paths['vae_encoder'], 
            engine_paths['vae_decoder']
        )
        print("✅ VAE engines loaded")
    
    # Load Text Encoder(s)
    if 'text_encoder' in engine_paths:
        components['text_encoder'] = TensorRTTextEncoder(engine_paths['text_encoder'])
        print("✅ Text Encoder engine loaded")
    
    if is_sdxl and 'text_encoder_2' in engine_paths:
        components['text_encoder_2'] = TensorRTTextEncoder(engine_paths['text_encoder_2'])
        print("✅ Text Encoder 2 engine loaded")
    
    print(f"🚀 All {len(components)} TensorRT engines loaded successfully!")
    return components


class StreamDiffusionTensorRTAccelerator:
    """Complete TensorRT acceleration for StreamDiffusion"""
    
    @staticmethod
    def accelerate(stream, 
                   model_path: str,
                   engine_dir: str = "./engines",
                   max_batch_size: int = 2,
                   use_fp16: bool = True,
                   use_fp8: bool = False,
                   force_rebuild: bool = False):
        """
        Accelerate StreamDiffusion with TensorRT using NVIDIA's latest optimizations
        
        Returns the accelerated stream object
        """
        # Check dependencies first
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT dependencies not available. "
                "Install with: pip install tensorrt>=8.6.0 pycuda polygraphy[all] onnx onnx-graphsurgeon"
            )
        
        from .modern_pipeline import accelerate_streamdiffusion_with_modern_tensorrt
        
        return accelerate_streamdiffusion_with_modern_tensorrt(
            stream=stream,
            model_path=model_path,
            engine_dir=engine_dir,
            max_batch_size=max_batch_size,
            use_fp16=use_fp16,
            use_fp8=use_fp8,
            force_rebuild=force_rebuild
        ) 