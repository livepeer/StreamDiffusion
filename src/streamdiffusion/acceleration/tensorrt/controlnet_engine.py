"""ControlNet TensorRT Engine with PyTorch fallback"""

import torch
import tensorrt as trt
import traceback
from typing import List, Optional, Tuple, Dict, Any
from polygraphy import cuda

from .utilities import Engine


class ControlNetModelEngine:
    """TensorRT-accelerated ControlNet inference engine"""
    
    def __init__(self, engine_path: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        """Initialize ControlNet TensorRT engine"""
        self.engine = Engine(engine_path)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        
        self.engine.load()
        self.engine.activate()
        
        self._input_names = None
        self._output_names = None
    
    def _resolve_output_shapes(self, batch_size: int) -> Dict[str, Tuple[int, ...]]:
        """Resolve dynamic output shapes from TensorRT engine"""
        output_shapes = {}
        
        # Iterate through all engine tensors to find outputs
        for idx in range(self.engine.engine.num_io_tensors):
            name = self.engine.engine.get_tensor_name(idx)
            
            # Check if this is an output tensor
            if self.engine.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                # Get the shape with dynamic dimensions
                dynamic_shape = self.engine.engine.get_tensor_shape(name)
                
                # Replace -1 dimensions with actual batch size
                resolved_shape = tuple(
                    batch_size if dim == -1 else dim 
                    for dim in dynamic_shape
                )
                
                output_shapes[name] = resolved_shape
        
        return output_shapes

    def __call__(self, 
                 sample: torch.Tensor,
                 timestep: torch.Tensor, 
                 encoder_hidden_states: torch.Tensor,
                 controlnet_cond: torch.Tensor,
                 conditioning_scale: float = 1.0,
                 text_embeds: Optional[torch.Tensor] = None,
                 time_ids: Optional[torch.Tensor] = None,
                 **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through TensorRT ControlNet engine"""
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        
        input_dict = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": torch.tensor(conditioning_scale, dtype=torch.float32, device=sample.device)
        }
        
        if text_embeds is not None:
            input_dict["text_embeds"] = text_embeds
        if time_ids is not None:
            input_dict["time_ids"] = time_ids
        
        shape_dict = {name: tensor.shape for name, tensor in input_dict.items()}
        
        batch_size = sample.shape[0]
        output_shapes = self._resolve_output_shapes(batch_size)
        shape_dict.update(output_shapes)
        
        self.engine.allocate_buffers(shape_dict=shape_dict, device=sample.device)
        
        outputs = self.engine.infer(
            input_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        if hasattr(self.stream, 'synchronize'):
            self.stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()
        
        down_blocks, mid_block = self._extract_controlnet_outputs(outputs)
        
        return down_blocks, mid_block
    
    def _extract_controlnet_outputs(self, outputs: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract and organize ControlNet outputs from engine results"""
        down_blocks = []
        for i in range(12):
            output_name = f"down_block_{i:02d}"
            if output_name in outputs:
                tensor = outputs[output_name]
                down_blocks.append(tensor)
        
        mid_block = None
        if "mid_block" in outputs:
            mid_block = outputs["mid_block"]
        
        return down_blocks, mid_block
    


class HybridControlNet:
    """Wrapper that handles TensorRT/PyTorch fallback for ControlNet"""
    
    def __init__(self, 
                 model_id: str,
                 engine_path: Optional[str] = None,
                 pytorch_model: Optional[Any] = None,
                 stream: Optional[cuda.Stream] = None):
        """Initialize hybrid ControlNet wrapper"""
        self.model_id = model_id
        self.engine_path = engine_path
        self.pytorch_model = pytorch_model
        self.stream = stream
        
        self.trt_engine: Optional[ControlNetModelEngine] = None
        self.use_tensorrt = False
        self.fallback_reason = None
        
        if engine_path:
            self._try_load_tensorrt_engine()
    
    def _try_load_tensorrt_engine(self) -> bool:
        """Attempt to load TensorRT engine"""
        try:
            if self.engine_path and self.stream:
                self.trt_engine = ControlNetModelEngine(self.engine_path, self.stream)
                self.use_tensorrt = True
                return True
        except Exception as e:
            self.fallback_reason = f"TensorRT engine load failed: {e}"
        
        return False
    
    def __call__(self, *args, **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass with automatic TensorRT/PyTorch fallback"""
        if self.use_tensorrt and self.trt_engine:
            try:
                return self.trt_engine(*args, **kwargs)
            except Exception as e:
                self.use_tensorrt = False
                self.fallback_reason = f"Runtime error: {e}"
        
        if self.pytorch_model is None:
            raise RuntimeError(f"No PyTorch fallback available for ControlNet {self.model_id}")
        
        return self._call_pytorch_model(*args, **kwargs)
    
    def _call_pytorch_model(self, *args, **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Call PyTorch ControlNet model with proper output formatting"""
        result = self.pytorch_model(*args, **kwargs)
        
        if isinstance(result, tuple) and len(result) == 2:
            return result
        elif hasattr(result, 'down_block_res_samples') and hasattr(result, 'mid_block_res_sample'):
            return result.down_block_res_samples, result.mid_block_res_sample
        else:
            if isinstance(result, (list, tuple)) and len(result) >= 13:
                return list(result[:12]), result[12]
            else:
                raise ValueError(f"Unexpected PyTorch ControlNet output format: {type(result)}")
    
    @property
    def is_using_tensorrt(self) -> bool:
        """Check if currently using TensorRT engine"""
        return self.trt_engine is not None
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current status information"""
        return {
            "model_id": self.model_id,
            "using_tensorrt": self.is_using_tensorrt,
            "engine_path": self.engine_path,
            "fallback_reason": self.fallback_reason,
            "has_pytorch_fallback": self.pytorch_model is not None
        } 