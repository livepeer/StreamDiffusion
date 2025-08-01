"""ControlNet TensorRT Engine with PyTorch fallback"""

import torch
import tensorrt as trt
import traceback
import logging
from typing import List, Optional, Tuple, Dict, Any
from polygraphy import cuda

from ..utilities import Engine
from ....model_detection import detect_model, detect_model_from_diffusers_unet

# Set up logger for this module
logger = logging.getLogger(__name__)


class ControlNetModelEngine:
    """TensorRT-accelerated ControlNet inference engine"""
    
    def __init__(self, engine_path: str, stream: 'cuda.Stream', use_cuda_graph: bool = False, model_type: str = "sd15"):
        """Initialize ControlNet TensorRT engine"""
        self.engine = Engine(engine_path)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.model_type = model_type.lower()
        
        self.engine.load()
        self.engine.activate()
        
        self._input_names = None
        self._output_names = None
        
        # Pre-compute model-specific values to eliminate runtime branching
        if self.model_type in ["sdxl", "sdxl_turbo"]:
            self.max_blocks = 9
            self.down_block_configs = [
                (320, 1), (320, 1), (320, 1), (320, 2),
                (640, 2), (640, 2), (640, 4),
                (1280, 4), (1280, 4)
            ]
            self.mid_block_channels = 1280
            self.mid_downsample_factor = 4
        else:
            self.max_blocks = 12
            self.down_block_configs = [
                (320, 1), (320, 1), (320, 1), (320, 2), (640, 2), (640, 2),
                (640, 4), (1280, 4), (1280, 4), (1280, 8), (1280, 8), (1280, 8)
            ]
            self.mid_block_channels = 1280
            self.mid_downsample_factor = 8
        
        # Cache for computed shapes
        self._shape_cache = {}
    
    def _resolve_output_shapes(self, batch_size: int, latent_height: int, latent_width: int) -> Dict[str, Tuple[int, ...]]:
        """Optimized shape resolution using pre-computed configurations"""
        cache_key = (batch_size, latent_height, latent_width)
        if cache_key in self._shape_cache:
            return self._shape_cache[cache_key]
        
        output_shapes = {}
        
        # Generate down block shapes using pre-computed configs
        for i, (channels, factor) in enumerate(self.down_block_configs):
            output_name = f"down_block_{i:02d}"
            h = max(1, latent_height // factor)
            w = max(1, latent_width // factor)
            output_shapes[output_name] = (batch_size, channels, h, w)
        
        # Generate mid block shape
        mid_h = max(1, latent_height // self.mid_downsample_factor)
        mid_w = max(1, latent_width // self.mid_downsample_factor)
        output_shapes["mid_block"] = (batch_size, self.mid_block_channels, mid_h, mid_w)
        
        self._shape_cache[cache_key] = output_shapes
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
        latent_height = sample.shape[2]
        latent_width = sample.shape[3]
        
        output_shapes = self._resolve_output_shapes(batch_size, latent_height, latent_width)
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
        
        for i in range(self.max_blocks):
            output_name = f"down_block_{i:02d}"
            if output_name in outputs:
                tensor = outputs[output_name]
                down_blocks.append(tensor)
        
        mid_block = outputs.get("mid_block")
        return down_blocks, mid_block
    



class HybridControlNet:
    """Wrapper that handles TensorRT/PyTorch fallback for ControlNet"""
    
    def __init__(self, 
                 model_id: str,
                 engine_path: Optional[str] = None,
                 pytorch_model: Optional[Any] = None,
                 stream: Optional['cuda.Stream'] = None,
                 enable_pytorch_fallback: bool = False,
                 model_type: str = "sd15"):
        """Initialize hybrid ControlNet wrapper"""
        self.model_id = model_id
        self.engine_path = engine_path
        self.pytorch_model = pytorch_model
        self.stream = stream
        self.enable_pytorch_fallback = enable_pytorch_fallback
        
        logger.debug(f"HybridControlNet.__init__: Initializing for model_id='{model_id}'")
        logger.debug(f"HybridControlNet.__init__: Provided model_type='{model_type}'")
        logger.debug(f"HybridControlNet.__init__: Has pytorch_model={pytorch_model is not None}")
        
        # Use existing model detection if pytorch_model is available
        if pytorch_model is not None:
            try:
                detected_type = detect_model_from_diffusers_unet(pytorch_model)
                self.model_type = detected_type.lower()
                logger.info(f"HybridControlNet.__init__: Model type detected from pytorch_model: '{self.model_type}' for {self.model_id}")
                logger.info(f"ControlNet model type detected from pytorch_model: {self.model_type} for {self.model_id}")
            except Exception as e:
                logger.warning(f"HybridControlNet.__init__: Model detection failed for {self.model_id}: {e}, using provided type: {model_type}")
                logger.warning(f"Model detection failed for {self.model_id}: {e}, using provided type: {model_type}")
                self.model_type = model_type.lower()
        else:
            self.model_type = model_type.lower()
            logger.info(f"HybridControlNet.__init__: Using provided model type: '{self.model_type}' for {self.model_id}")
            logger.info(f"ControlNet using provided model type: {self.model_type} for {self.model_id}")
        
        logger.debug(f"HybridControlNet.__init__: Final model_type='{self.model_type}' for {self.model_id}")
        
        self.trt_engine: Optional[ControlNetModelEngine] = None
        self.use_tensorrt = False
        self.fallback_reason = None
        
        if engine_path:
            self._try_load_tensorrt_engine()
    
    def _try_load_tensorrt_engine(self) -> bool:
        """Attempt to load TensorRT engine"""
        try:
            if self.engine_path and self.stream:
                logger.info(f"HybridControlNet._try_load_tensorrt_engine: Loading TensorRT ControlNet engine: {self.engine_path}")
                logger.debug(f"HybridControlNet._try_load_tensorrt_engine: Passing model_type='{self.model_type}' to ControlNetModelEngine")
                logger.info(f"Loading TensorRT ControlNet engine: {self.engine_path}")
                logger.info(f"ControlNet model type detected: {self.model_type} for {self.model_id}")
                self.trt_engine = ControlNetModelEngine(self.engine_path, self.stream, model_type=self.model_type)
                self.use_tensorrt = True
                logger.info(f"HybridControlNet._try_load_tensorrt_engine: Successfully loaded TensorRT ControlNet engine for {self.model_id}")
                logger.info(f"Successfully loaded TensorRT ControlNet engine for {self.model_id}")
                return True
        except Exception as e:
            self.fallback_reason = f"TensorRT engine load failed: {e}"
            logger.warning(f"HybridControlNet._try_load_tensorrt_engine: Failed to load TensorRT ControlNet engine for {self.model_id}: {e}")
            logger.warning(f"Failed to load TensorRT ControlNet engine for {self.model_id}: {e}")
            logger.debug(f"TensorRT ControlNet engine load failure details:", exc_info=True)
        
        return False
    
    def __call__(self, *args, **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass with automatic TensorRT/PyTorch fallback"""
        logger.debug(f"HybridControlNet.__call__: Starting inference for {self.model_id}")
        logger.debug(f"HybridControlNet.__call__: use_tensorrt={self.use_tensorrt}, trt_engine is not None={self.trt_engine is not None}")
        
        if self.use_tensorrt and self.trt_engine:
            try:
                logger.debug(f"HybridControlNet.__call__: Attempting TensorRT inference for {self.model_id}")
                result = self.trt_engine(*args, **kwargs)
                logger.debug(f"HybridControlNet.__call__: TensorRT inference successful for {self.model_id}")
                return result
            except Exception as e:
                self.use_tensorrt = False
                self.fallback_reason = f"Runtime error: {e}"
                logger.warning(f"HybridControlNet.__call__: TensorRT ControlNet runtime error for {self.model_id}, falling back to PyTorch: {e}")
                logger.warning(f"TensorRT ControlNet runtime error for {self.model_id}, falling back to PyTorch: {e}")
                logger.debug(f"TensorRT ControlNet runtime error details:", exc_info=True)
        
        if not self.enable_pytorch_fallback:
            raise RuntimeError(f"TensorRT acceleration failed for ControlNet {self.model_id} and PyTorch fallback is disabled. Error: {self.fallback_reason}")
        
        if self.pytorch_model is None:
            logger.error(f"HybridControlNet.__call__: No PyTorch fallback available for ControlNet {self.model_id}")
            logger.error(f"No PyTorch fallback available for ControlNet {self.model_id}")
            raise RuntimeError(f"No PyTorch fallback available for ControlNet {self.model_id}")
        
        logger.debug(f"HybridControlNet.__call__: Using PyTorch fallback for {self.model_id}")
        logger.debug(f"Using PyTorch ControlNet for {self.model_id}")
        return self._call_pytorch_model(*args, **kwargs)
    
    def _call_pytorch_model(self, *args, **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Call PyTorch ControlNet model with proper output formatting"""
        logger.debug(f"Executing PyTorch ControlNet model for {self.model_id}")
        result = self.pytorch_model(*args, **kwargs)
        
        if isinstance(result, tuple) and len(result) == 2:
            logger.debug(f"PyTorch ControlNet returned standard tuple format")
            return result
        elif hasattr(result, 'down_block_res_samples') and hasattr(result, 'mid_block_res_sample'):
            logger.debug(f"PyTorch ControlNet returned attribute-based format")
            return result.down_block_res_samples, result.mid_block_res_sample
        else:
            if isinstance(result, (list, tuple)) and len(result) >= 13:
                logger.debug(f"PyTorch ControlNet returned list format with {len(result)} elements")
                return list(result[:12]), result[12]
            else:
                logger.error(f"Unexpected PyTorch ControlNet output format for {self.model_id}: {type(result)}")
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
            "model_type": self.model_type,
            "using_tensorrt": self.is_using_tensorrt,
            "engine_path": self.engine_path,
            "fallback_reason": self.fallback_reason,
            "has_pytorch_fallback": self.pytorch_model is not None,
            "enable_pytorch_fallback": self.enable_pytorch_fallback
        } 