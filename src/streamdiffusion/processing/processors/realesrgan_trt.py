# NOTE: ported from https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import requests
from tqdm import tqdm
import hashlib
import logging
from pathlib import Path

from .base import BasePreprocessor

# Try to import spandrel for model loading
try:
    from spandrel import ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False

# Try to import TensorRT dependencies
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealESRGANProcessor(BasePreprocessor):
    """
    RealESRGAN 2x upscaling processor with automatic model download, ONNX export, and TensorRT acceleration.
    """
    
    MODEL_URL = "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth?download=true"
    
    @classmethod 
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "RealESRGAN 2x",
            "description": "High-quality 2x image upscaling using RealESRGAN with TensorRT acceleration",
            "parameters": {
                "enable_tensorrt": {
                    "type": "bool",
                    "default": True,
                    "description": "Use TensorRT acceleration for faster inference"
                },
                "force_rebuild": {
                    "type": "bool", 
                    "default": False,
                    "description": "Force rebuild TensorRT engine even if it exists"
                }
            },
            "use_cases": ["High-quality upscaling", "Real-time 2x enlargement", "Image enhancement"]
        }
    
    def __init__(self, enable_tensorrt: bool = True, force_rebuild: bool = False, **kwargs):
        super().__init__(enable_tensorrt=enable_tensorrt, force_rebuild=force_rebuild, **kwargs)
        self.enable_tensorrt = enable_tensorrt and TRT_AVAILABLE
        self.force_rebuild = force_rebuild
        self.scale_factor = 2  # RealESRGAN 2x model
        
        # Model paths
        self.models_dir = Path("models") / "realesrgan"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "RealESRGAN_x2.pth"
        self.onnx_path = self.models_dir / "RealESRGAN_x2.onnx"
        self.engine_path = self.models_dir / f"RealESRGAN_x2_{trt.__version__ if TRT_AVAILABLE else 'notrt'}.trt"
        
        # Model state
        self.pytorch_model = None
        self.trt_engine = None
        self.trt_context = None
        
        # Initialize
        self._ensure_model_ready()
    
    def _download_file(self, url: str, save_path: Path):
        """Download file with progress bar"""
        if save_path.exists():
            logger.info(f"_download_file: Model file already exists: {save_path}")
            return
        
        logger.info(f"_download_file: Downloading {url} to {save_path}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(
            desc=f"Downloading {save_path.name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            colour='green'
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        logger.info(f"_download_file: Successfully downloaded {save_path}")
    
    def _ensure_model_ready(self):
        """Ensure PyTorch model is downloaded and loaded"""
        # Download model if needed
        if not self.model_path.exists():
            self._download_file(self.MODEL_URL, self.model_path)
        
        # Load PyTorch model
        if self.pytorch_model is None:
            self._load_pytorch_model()
        
        # Setup TensorRT if enabled
        if self.enable_tensorrt:
            self._setup_tensorrt()
    
    def _load_pytorch_model(self):
        """Load PyTorch model from file"""
        if not SPANDREL_AVAILABLE:
            logger.warning("_load_pytorch_model: Spandrel not available, using basic torch.load")
            # Fallback loading without spandrel
            state_dict = torch.load(self.model_path, map_location=self.device)
            # This is a simplified approach - real implementation would need model architecture
            logger.warning("_load_pytorch_model: Basic loading not fully implemented - need spandrel")
            return
        
        logger.info(f"_load_pytorch_model: Loading PyTorch model from {self.model_path}")
        model_descriptor = ModelLoader().load_from_file(str(self.model_path))
        self.pytorch_model = model_descriptor.model.eval().to(device=self.device, dtype=self.dtype)
        logger.info(f"_load_pytorch_model: PyTorch model loaded successfully with dtype {self.dtype}")
    
    def _export_to_onnx(self):
        """Export PyTorch model to ONNX format"""
        if self.onnx_path.exists() and not self.force_rebuild:
            logger.info(f"_export_to_onnx: ONNX model already exists: {self.onnx_path}")
            return
        
        if self.pytorch_model is None:
            self._load_pytorch_model()
        
        if self.pytorch_model is None:
            logger.error("_export_to_onnx: PyTorch model not available")
            return
        
        logger.info(f"_export_to_onnx: Exporting PyTorch model to ONNX: {self.onnx_path}")
        
        # Test with small input for export
        test_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
        
        with torch.no_grad():
            torch.onnx.export(
                self.pytorch_model,
                test_input,
                str(self.onnx_path),
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
            )
        
        logger.info(f"_export_to_onnx: Successfully exported ONNX model to {self.onnx_path}")
    
    def _setup_tensorrt(self):
        """Setup TensorRT engine"""
        if not TRT_AVAILABLE:
            logger.warning("_setup_tensorrt: TensorRT not available")
            return
        
        # Export to ONNX first if needed
        if not self.onnx_path.exists():
            self._export_to_onnx()
        
        # Build/load TensorRT engine
        self._load_tensorrt_engine()
    
    def _load_tensorrt_engine(self):
        """Load or build TensorRT engine"""
        if self.engine_path.exists() and not self.force_rebuild:
            logger.info(f"_load_tensorrt_engine: Loading existing TensorRT engine: {self.engine_path}")
            self._load_existing_engine()
        else:
            logger.info("_load_tensorrt_engine: Building new TensorRT engine")
            self._build_tensorrt_engine()
    
    def _load_existing_engine(self):
        """Load existing TensorRT engine"""
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.engine_path, 'rb') as f:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.trt_engine is not None:
                self.trt_context = self.trt_engine.create_execution_context()
                logger.info("_load_existing_engine: TensorRT engine loaded successfully")
            else:
                logger.error("_load_existing_engine: Failed to deserialize TensorRT engine")
        except Exception as e:
            logger.error(f"_load_existing_engine: Error loading TensorRT engine: {e}")
    
    def _build_tensorrt_engine(self):
        """Build TensorRT engine from ONNX model"""
        if not self.onnx_path.exists():
            logger.error("_build_tensorrt_engine: ONNX model not found")
            return
        
        logger.info("_build_tensorrt_engine: Building TensorRT engine... this may take several minutes")
        
        try:
            # Create builder and network
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("_build_tensorrt_engine: Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(f"_build_tensorrt_engine: {parser.get_error(error)}")
                    return
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for better performance
            
            # Set optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()
            min_shape = (1, 3, 256, 256)
            opt_shape = (1, 3, 512, 512)
            max_shape = (1, 3, 1024, 1024)
            profile.set_shape("input", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                logger.error("_build_tensorrt_engine: Failed to build TensorRT engine")
                return
            
            # Save engine
            with open(self.engine_path, 'wb') as f:
                f.write(engine)
            
            # Load the built engine
            self._load_existing_engine()
            logger.info(f"_build_tensorrt_engine: Successfully built and saved TensorRT engine: {self.engine_path}")
        
        except Exception as e:
            logger.error(f"_build_tensorrt_engine: Error building TensorRT engine: {e}")
    
    def _allocate_trt_buffers(self, input_shape):
        """Allocate TensorRT buffers for given input shape"""
        if not hasattr(self, 'trt_tensors'):
            self.trt_tensors = {}
        
        batch_size, channels, height, width = input_shape
        
        # Set input shape
        input_name = "input"
        self.trt_context.set_input_shape(input_name, input_shape)
        
        # Allocate tensors for all bindings
        for idx in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(idx)
            shape = self.trt_context.get_tensor_shape(name)
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
            
            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=self.device)
            self.trt_tensors[name] = tensor
    
    def _process_with_tensorrt(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor using TensorRT engine"""
        if self.trt_engine is None or self.trt_context is None:
            raise RuntimeError("_process_with_tensorrt: TensorRT engine not loaded")
        
        batch_size, channels, height, width = tensor.shape
        input_shape = (batch_size, channels, height, width)
        
        # Allocate buffers for this shape
        self._allocate_trt_buffers(input_shape)
        
        # Copy input data
        input_name = "input"
        input_tensor = tensor.contiguous()
        if input_tensor.dtype != self.trt_tensors[input_name].dtype:
            input_tensor = input_tensor.to(dtype=self.trt_tensors[input_name].dtype)
        
        self.trt_tensors[input_name].copy_(input_tensor)
        
        # Set tensor addresses
        for name, tensor_buf in self.trt_tensors.items():
            self.trt_context.set_tensor_address(name, tensor_buf.data_ptr())
        
        # Execute
        stream = torch.cuda.current_stream()
        success = self.trt_context.execute_async_v3(stream.cuda_stream)
        if not success:
            raise RuntimeError("_process_with_tensorrt: TensorRT execution failed")
        
        stream.synchronize()
        
        # Return output tensor
        output_name = "output"
        return self.trt_tensors[output_name].clone()
    
    def _process_with_pytorch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor using PyTorch model"""
        if self.pytorch_model is None:
            raise RuntimeError("_process_with_pytorch: PyTorch model not loaded")
        
        # Ensure model and input tensor have compatible dtypes
        model_dtype = next(self.pytorch_model.parameters()).dtype
        if tensor.dtype != model_dtype:
            logger.info(f"_process_with_pytorch: Converting tensor from {tensor.dtype} to {model_dtype}")
            tensor = tensor.to(dtype=model_dtype)
        
        with torch.no_grad():
            result = self.pytorch_model(tensor)
            # Convert back to original dtype if needed
            return result.to(dtype=self.dtype)
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """Core processing using PIL Image"""
        # Convert to tensor for processing
        tensor = self.pil_to_tensor(image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Process with available backend
        if self.enable_tensorrt and self.trt_engine is not None:
            try:
                output_tensor = self._process_with_tensorrt(tensor)
            except Exception as e:
                logger.warning(f"_process_core: TensorRT processing failed: {e}, falling back to PyTorch")
                output_tensor = self._process_with_pytorch(tensor)
        elif self.pytorch_model is not None:
            output_tensor = self._process_with_pytorch(tensor)
        else:
            # Fallback to simple upscaling if no model is available
            logger.warning("_process_core: No model available, using simple resize")
            target_width, target_height = self.get_target_dimensions()
            return image.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert back to PIL
        if output_tensor.dim() == 4:
            output_tensor = output_tensor.squeeze(0)
        
        return self.tensor_to_pil(output_tensor)
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """Core tensor processing"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Process with available backend
        if self.enable_tensorrt and self.trt_engine is not None:
            try:
                output_tensor = self._process_with_tensorrt(tensor)
            except Exception as e:
                logger.warning(f"_process_tensor_core: TensorRT processing failed: {e}, falling back to PyTorch")
                output_tensor = self._process_with_pytorch(tensor)
        elif self.pytorch_model is not None:
            output_tensor = self._process_with_pytorch(tensor)
        else:
            # Fallback using interpolation
            logger.warning("_process_tensor_core: No model available, using interpolation")
            output_tensor = torch.nn.functional.interpolate(
                tensor, 
                scale_factor=self.scale_factor,
                mode='bicubic',
                align_corners=False
            )
        
        if squeeze_output:
            output_tensor = output_tensor.squeeze(0)
        
        return output_tensor
    
    def get_target_dimensions(self) -> Tuple[int, int]:
        """Get target output dimensions (width, height) - 2x upscaled"""
        width = self.params.get('image_width')
        height = self.params.get('image_height')
        
        if width is not None and height is not None:
            return (width * self.scale_factor, height * self.scale_factor)
        
        # Fallback to square resolution
        resolution = self.params.get('image_resolution', 512)
        target_resolution = resolution * self.scale_factor
        return (target_resolution, target_resolution)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'trt_context') and self.trt_context is not None:
            del self.trt_context
        if hasattr(self, 'trt_engine') and self.trt_engine is not None:
            del self.trt_engine
