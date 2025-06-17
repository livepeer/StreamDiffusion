import os
from typing import Dict, List, Optional, Union, Any
import tensorrt as trt
import torch
import numpy as np
from polygraphy import cuda

from .engine import TensorRTEngine
from .utilities import build_engine


class DynamicTensorRTEngine(TensorRTEngine):
    """
    TensorRT engine with dynamic batch size support for StreamDiffusion batched denoising
    """
    
    def __init__(
        self,
        engine_path: str,
        stream: cuda.Stream,
        use_cuda_graph: bool = True,
        enable_dynamic_batch: bool = True,
        min_batch_size: int = 1,
        max_batch_size: int = 4,
        opt_batch_size: Optional[int] = None,
    ):
        super().__init__(engine_path, stream, use_cuda_graph)
        
        self.enable_dynamic_batch = enable_dynamic_batch
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size or max_batch_size
        
        # Current batch size (for optimization)
        self.current_batch_size = self.opt_batch_size
        
        # Cache for different batch sizes (if using CUDA graphs)
        self.cuda_graph_cache = {}
        
    def set_input_shape(self, binding_name: str, shape: List[int]):
        """Set input shape for dynamic batch inference"""
        if self.enable_dynamic_batch:
            # Validate batch size is within allowed range
            batch_size = shape[0]
            if batch_size < self.min_batch_size or batch_size > self.max_batch_size:
                raise ValueError(
                    f"Batch size {batch_size} is outside allowed range "
                    f"[{self.min_batch_size}, {self.max_batch_size}]"
                )
        
            # Set the input shape for this binding
            binding_idx = self.engine.get_binding_index(binding_name)
            self.context.set_binding_shape(binding_idx, shape)
            
            # Update current batch size for optimization
            self.current_batch_size = batch_size
        else:
            # For static batch, just set the shape
            super().set_input_shape(binding_name, shape)
    
    def infer(self, feed_dict: Dict[str, torch.Tensor], stream: cuda.Stream) -> Dict[str, torch.Tensor]:
        """
        Run inference with dynamic batch size support
        """
        if not self.enable_dynamic_batch:
            return super().infer(feed_dict, stream)
        
        # Determine batch size from the first input
        first_input = next(iter(feed_dict.values()))
        batch_size = first_input.shape[0]
        
        # Validate batch size
        if batch_size < self.min_batch_size or batch_size > self.max_batch_size:
            raise ValueError(
                f"Input batch size {batch_size} is outside engine's allowed range "
                f"[{self.min_batch_size}, {self.max_batch_size}]"
            )
        
        # Set input shapes for all bindings
        for input_name, tensor in feed_dict.items():
            self.set_input_shape(input_name, list(tensor.shape))
        
        # Prepare output tensors with correct batch size
        output_dict = {}
        for binding_name in self.output_names:
            binding_idx = self.engine.get_binding_index(binding_name)
            shape = self.context.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # Create output tensor with correct batch size
            output_tensor = torch.empty(
                shape, 
                dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype,
                device=first_input.device
            )
            output_dict[binding_name] = output_tensor
        
        # Use CUDA graphs for performance if enabled and batch size is cached
        if self.use_cuda_graph and batch_size in self.cuda_graph_cache:
            return self._infer_with_cuda_graph(feed_dict, output_dict, batch_size, stream)
        else:
            return self._infer_direct(feed_dict, output_dict, stream)
    
    def _infer_direct(self, feed_dict: Dict[str, torch.Tensor], output_dict: Dict[str, torch.Tensor], stream: cuda.Stream) -> Dict[str, torch.Tensor]:
        """Direct inference without CUDA graphs"""
        # Prepare bindings
        bindings = [None] * self.engine.num_bindings
        
        # Set input bindings
        for input_name, tensor in feed_dict.items():
            binding_idx = self.engine.get_binding_index(input_name)
            bindings[binding_idx] = tensor.data_ptr()
        
        # Set output bindings
        for output_name, tensor in output_dict.items():
            binding_idx = self.engine.get_binding_index(output_name)
            bindings[binding_idx] = tensor.data_ptr()
        
        # Execute inference
        self.context.execute_async_v2(bindings, stream.ptr)
        stream.synchronize()
        
        return output_dict
    
    def _infer_with_cuda_graph(self, feed_dict: Dict[str, torch.Tensor], output_dict: Dict[str, torch.Tensor], batch_size: int, stream: cuda.Stream) -> Dict[str, torch.Tensor]:
        """Inference using cached CUDA graph for specific batch size"""
        if batch_size not in self.cuda_graph_cache:
            # Create and cache CUDA graph for this batch size
            self._create_cuda_graph(feed_dict, output_dict, batch_size, stream)
        
        # Get cached graph info
        graph_info = self.cuda_graph_cache[batch_size]
        
        # Copy inputs to cached tensors
        for input_name, tensor in feed_dict.items():
            graph_info['inputs'][input_name].copy_(tensor)
        
        # Launch cached graph
        graph_info['graph'].launch(stream.ptr)
        stream.synchronize()
        
        # Copy outputs from cached tensors
        result = {}
        for output_name, cached_tensor in graph_info['outputs'].items():
            result[output_name] = cached_tensor.clone()
        
        return result
    
    def _create_cuda_graph(self, feed_dict: Dict[str, torch.Tensor], output_dict: Dict[str, torch.Tensor], batch_size: int, stream: cuda.Stream):
        """Create CUDA graph for specific batch size"""
        # Create tensors for this batch size
        cached_inputs = {}
        cached_outputs = {}
        
        for input_name, tensor in feed_dict.items():
            cached_inputs[input_name] = torch.empty_like(tensor)
        
        for output_name, tensor in output_dict.items():
            cached_outputs[output_name] = torch.empty_like(tensor)
        
        # Prepare bindings
        bindings = [None] * self.engine.num_bindings
        
        for input_name, tensor in cached_inputs.items():
            binding_idx = self.engine.get_binding_index(input_name)
            bindings[binding_idx] = tensor.data_ptr()
        
        for output_name, tensor in cached_outputs.items():
            binding_idx = self.engine.get_binding_index(output_name)
            bindings[binding_idx] = tensor.data_ptr()
        
        # Record CUDA graph
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            self.context.execute_async_v2(bindings, stream.ptr)
        
        # Cache the graph
        self.cuda_graph_cache[batch_size] = {
            'graph': graph,
            'inputs': cached_inputs,
            'outputs': cached_outputs,
        }


class DynamicUNet2DConditionModelEngine(DynamicTensorRTEngine):
    """
    Dynamic UNet engine for StreamDiffusion with batched denoising support
    """
    
    def __init__(
        self,
        engine_path: str, 
        stream: cuda.Stream,
        use_cuda_graph: bool = True,
        enable_dynamic_batch: bool = True,
        min_batch_size: int = 1,
        max_batch_size: int = 4,
        opt_batch_size: Optional[int] = None,
    ):
        super().__init__(
            engine_path, 
            stream, 
            use_cuda_graph, 
            enable_dynamic_batch,
            min_batch_size,
            max_batch_size, 
            opt_batch_size
        )
        
    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through dynamic UNet
        """
        # Prepare input dictionary
        feed_dict = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        
        # Add ControlNet inputs if present
        if hasattr(self, 'use_control') and self.use_control:
            for key, value in kwargs.items():
                if key.startswith('controlnet_') or key.startswith('conditioning_scale_'):
                    feed_dict[key] = value
        
        # Run inference
        with self.stream:
            output_dict = self.infer(feed_dict, self.stream)
        
        return output_dict["latent"]


class DynamicAutoencoderKLEngine:
    """
    Dynamic VAE engine for StreamDiffusion
    """
    
    def __init__(
        self,
        encoder_engine_path: str,
        decoder_engine_path: str,
        stream: cuda.Stream,
        vae_scale_factor: int = 8,
        use_cuda_graph: bool = True,
        enable_dynamic_batch: bool = True,
        min_batch_size: int = 1,
        max_batch_size: int = 4,
        opt_batch_size: Optional[int] = None,
    ):
        """
        Args:
            encoder_engine_path: Path to VAE encoder TensorRT engine
            decoder_engine_path: Path to VAE decoder TensorRT engine
            stream: CUDA stream for execution
            vae_scale_factor: VAE scale factor (default 8)
            use_cuda_graph: Whether to use CUDA graphs for performance
            enable_dynamic_batch: Whether to enable dynamic batching
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size  
            opt_batch_size: Optimal batch size
        """
        self.vae_scale_factor = vae_scale_factor
        self.stream = stream
        
        # Create encoder and decoder engines
        self.encoder = DynamicTensorRTEngine(
            encoder_engine_path,
            stream,
            use_cuda_graph,
            enable_dynamic_batch,
            min_batch_size,
            max_batch_size,
            opt_batch_size,
        )
        
        self.decoder = DynamicTensorRTEngine(
            decoder_engine_path,
            stream,
            use_cuda_graph,
            enable_dynamic_batch,
            min_batch_size,
            max_batch_size,
            opt_batch_size,
        )
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latents"""
        feed_dict = {"images": images}
        
        with self.stream:
            output_dict = self.encoder.infer(feed_dict, self.stream)
        
        return output_dict["latent"]
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images"""
        feed_dict = {"latent": latents}
        
        with self.stream:
            output_dict = self.decoder.infer(feed_dict, self.stream)
        
        return output_dict["images"] 