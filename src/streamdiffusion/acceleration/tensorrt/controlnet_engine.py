"""ControlNet TensorRT Engine with PyTorch fallback"""

import torch
import tensorrt as trt
import traceback
import logging
from typing import List, Optional, Tuple, Dict, Any
from polygraphy import cuda

from .utilities import Engine
from ...model_detection import detect_model, detect_model_from_diffusers_unet

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
        
        # Cache for output shapes to avoid recalculation
        self._output_shapes_cache = {}
    
    def _resolve_output_shapes(self, batch_size: int, latent_height: int, latent_width: int) -> Dict[str, Tuple[int, ...]]:
        """Resolve dynamic output shapes from TensorRT engine with proper spatial dimensions"""
        # Check cache first
        cache_key = (batch_size, latent_height, latent_width)
        if cache_key in self._output_shapes_cache:
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: Returning cached shapes for {cache_key}")
            return self._output_shapes_cache[cache_key]
        
        logger.debug(f"ControlNetModelEngine._resolve_output_shapes: Starting shape resolution for batch_size={batch_size}, latent_height={latent_height}, latent_width={latent_width}")
        logger.debug(f"ControlNetModelEngine._resolve_output_shapes: model_type='{self.model_type}'")
        output_shapes = {}
        
        if self.model_type in ["sdxl", "sdxl_turbo"]:
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: Using SDXL architecture branch")
            # SDXL architecture: 9 down blocks + 1 mid block
            # Pattern: [88x88] + [88x88, 88x88, 44x44] + [44x44, 44x44, 22x22] + [22x22, 22x22]
            # Following UNet pattern from controlnet_wrapper.py and models.py
            
            sdxl_down_blocks = [
                # Initial sample
                (320, 1),      # down_block_00: 320 channels, 88x88
                # Block 0 residuals
                (320, 1),      # down_block_01: 320 channels, 88x88
                (320, 1),      # down_block_02: 320 channels, 88x88
                (320, 2),      # down_block_03: 320 channels, 44x44 (downsampled)
                # Block 1 residuals
                (640, 2),      # down_block_04: 640 channels, 44x44
                (640, 2),      # down_block_05: 640 channels, 44x44
                (640, 4),      # down_block_06: 640 channels, 22x22 (downsampled)
                # Block 2 residuals
                (1280, 4),     # down_block_07: 1280 channels, 22x22
                (1280, 4),     # down_block_08: 1280 channels, 22x22
            ]
            
            mid_block_channels = 1280
            mid_downsample_factor = 4  # SDXL middle block at 4x downsampling
            
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SDXL config - 9 down blocks with channel pattern")
            
            # Generate output shapes for SDXL down blocks
            for i, (channels, factor) in enumerate(sdxl_down_blocks):
                output_name = f"down_block_{i:02d}"
                
                # Calculate spatial dimensions for this block
                h = max(1, latent_height // factor)
                w = max(1, latent_width // factor)
                
                logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SDXL down_block_{i:02d}: latent_height={latent_height} // factor={factor} = {latent_height // factor} -> h={h}")
                logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SDXL down_block_{i:02d}: latent_width={latent_width} // factor={factor} = {latent_width // factor} -> w={w}")
                
                output_shapes[output_name] = (batch_size, channels, h, w)
                logger.debug(f"ControlNetModelEngine._resolve_output_shapes: {output_name}: (batch_size={batch_size}, channels={channels}, h={h}, w={w})")
            
            # Generate output shape for SDXL mid block
            mid_h = max(1, latent_height // mid_downsample_factor)
            mid_w = max(1, latent_width // mid_downsample_factor)
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SDXL mid_block: latent_height={latent_height} // factor={mid_downsample_factor} = {latent_height // mid_downsample_factor} -> h={mid_h}")
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SDXL mid_block: latent_width={latent_width} // factor={mid_downsample_factor} = {latent_width // mid_downsample_factor} -> w={mid_w}")
            output_shapes["mid_block"] = (batch_size, mid_block_channels, mid_h, mid_w)
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: mid_block: (batch_size={batch_size}, channels={mid_block_channels}, h={mid_h}, w={mid_w})")
        
        else:
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: Using SD1.5/SD2.1 architecture branch")
            # SD 1.5/SD 2.1 architecture: 12 down blocks (original implementation)
            down_block_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
            downsampling_factors = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
            mid_block_channels = 1280
            mid_downsample_factor = 8  # SD1.5 middle block at 8x downsampling
            
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SD1.5 config - {len(down_block_channels)} down blocks")
            
            # Generate output shapes for SD1.5 down blocks
            for i, (channels, factor) in enumerate(zip(down_block_channels, downsampling_factors)):
                output_name = f"down_block_{i:02d}"
                
                # Calculate spatial dimensions for this block
                h = max(1, latent_height // factor)
                w = max(1, latent_width // factor)
                
                if i < 3:  # Only log first few to avoid spam
                    logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SD1.5 down_block_{i:02d}: latent_height={latent_height} // factor={factor} = {latent_height // factor} -> h={h}")
                    logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SD1.5 down_block_{i:02d}: latent_width={latent_width} // factor={factor} = {latent_width // factor} -> w={w}")
                
                output_shapes[output_name] = (batch_size, channels, h, w)
                logger.debug(f"ControlNetModelEngine._resolve_output_shapes: {output_name}: (batch_size={batch_size}, channels={channels}, h={h}, w={w})")
            
            # Generate output shape for SD1.5 mid block
            mid_h = max(1, latent_height // mid_downsample_factor)
            mid_w = max(1, latent_width // mid_downsample_factor)
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SD1.5 mid_block: latent_height={latent_height} // factor={mid_downsample_factor} = {latent_height // mid_downsample_factor} -> h={mid_h}")
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: SD1.5 mid_block: latent_width={latent_width} // factor={mid_downsample_factor} = {latent_width // mid_downsample_factor} -> w={mid_w}")
            output_shapes["mid_block"] = (batch_size, mid_block_channels, mid_h, mid_w)
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: mid_block: (batch_size={batch_size}, channels={mid_block_channels}, h={mid_h}, w={mid_w})")
        
        logger.debug(f"ControlNetModelEngine._resolve_output_shapes: Final resolved output shapes:")
        for name, shape in output_shapes.items():
            logger.debug(f"ControlNetModelEngine._resolve_output_shapes: {name}: {shape}")
        
        # Cache the result
        self._output_shapes_cache[cache_key] = output_shapes
        logger.debug(f"ControlNetModelEngine._resolve_output_shapes: Cached shapes for {cache_key}")
        
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
        logger.debug(f"ControlNetModelEngine.__call__: Starting TensorRT ControlNet inference")
        logger.debug(f"ControlNetModelEngine.__call__: model_type='{self.model_type}'")
        logger.debug(f"ControlNetModelEngine.__call__: Input shapes - sample: {sample.shape}, timestep: {timestep.shape}, encoder_hidden_states: {encoder_hidden_states.shape}, controlnet_cond: {controlnet_cond.shape}")
        logger.debug(f"ControlNetModelEngine.__call__: conditioning_scale: {conditioning_scale}, text_embeds: {text_embeds.shape if text_embeds is not None else None}, time_ids: {time_ids.shape if time_ids is not None else None}")
        
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
        
        logger.debug(f"ControlNetModelEngine.__call__: Calling _resolve_output_shapes with batch_size={batch_size}, latent_height={latent_height}, latent_width={latent_width}, model_type='{self.model_type}'")
        
        output_shapes = self._resolve_output_shapes(batch_size, latent_height, latent_width)
        
        logger.debug(f"ControlNetModelEngine.__call__: Resolved output shapes:")
        for name, shape in output_shapes.items():
            logger.debug(f"ControlNetModelEngine.__call__: {name}: {shape}")
        
        shape_dict.update(output_shapes)
        
        logger.debug(f"ControlNetEngine: Input shapes - sample: {sample.shape}, controlnet_cond: {controlnet_cond.shape}")
        
        logger.debug(f"ControlNetModelEngine.__call__: Calling engine.allocate_buffers...")
        logger.debug(f"ControlNetModelEngine.__call__: About to allocate buffers with shape_dict:")
        for name, shape in shape_dict.items():
            logger.debug(f"ControlNetModelEngine.__call__: {name}: {shape}")
        
        # Debug: Inspect TensorRT engine expectations
        if hasattr(self.engine, 'engine') and hasattr(self.engine.engine, 'get_tensor_shape'):
            logger.debug(f"ControlNetModelEngine.__call__: TensorRT engine tensor info:")
            try:
                for i in range(self.engine.engine.num_io_tensors):
                    tensor_name = self.engine.engine.get_tensor_name(i)
                    tensor_mode = self.engine.engine.get_tensor_mode(tensor_name)
                    tensor_shape = self.engine.engine.get_tensor_shape(tensor_name)
                    logger.debug(f"ControlNetModelEngine.__call__: {tensor_name} ({tensor_mode}): {tensor_shape}")
            except Exception as e:
                logger.debug(f"ControlNetModelEngine.__call__: Failed to inspect engine tensors: {e}")
        
        try:
            self.engine.allocate_buffers(shape_dict=shape_dict, device=sample.device)
        except Exception as e:
            logger.error(f"ControlNetModelEngine.__call__: allocate_buffers failed with error: {e}")
            logger.error(f"ControlNetModelEngine.__call__: Error type: {type(e)}")
            logger.error(f"ControlNetModelEngine.__call__: Failing shape_dict:")
            for name, shape in shape_dict.items():
                logger.error(f"ControlNetModelEngine.__call__: {name}: {shape}")
            raise e
        
        logger.debug(f"ControlNetModelEngine.__call__: Calling engine.infer...")
        outputs = self.engine.infer(
            input_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        logger.debug(f"ControlNetModelEngine.__call__: TensorRT inference completed successfully")
        
        if hasattr(self.stream, 'synchronize'):
            self.stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()
        
        down_blocks, mid_block = self._extract_controlnet_outputs(outputs)
        
        # Log output dimensions
        logger.debug(f"ControlNetModelEngine.__call__: Final output dimensions:")
        for i, block in enumerate(down_blocks):
            if block is not None:
                logger.debug(f"ControlNetModelEngine.__call__: down_block_{i:02d}: {block.shape}")
        if mid_block is not None:
            logger.debug(f"ControlNetModelEngine.__call__: mid_block: {mid_block.shape}")
        
        logger.debug(f"ControlNetEngine: Output dimensions:")
        for i, block in enumerate(down_blocks):
            if block is not None:
                logger.debug(f"ControlNetEngine: down_block_{i:02d}: {block.shape}")
        if mid_block is not None:
            logger.debug(f"ControlNetEngine: mid_block: {mid_block.shape}")
        
        return down_blocks, mid_block
    
    def _extract_controlnet_outputs(self, outputs: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract and organize ControlNet outputs from engine results"""
        down_blocks = []
        
        # Extract down blocks based on model type
        if self.model_type in ["sdxl", "sdxl_turbo"]:
            # SDXL has 9 down blocks
            max_blocks = 9
        else:
            # SD1.5 has 12 down blocks
            max_blocks = 12
            
        for i in range(max_blocks):
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