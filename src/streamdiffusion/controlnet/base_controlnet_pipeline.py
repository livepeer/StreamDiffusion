import torch
import traceback
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path
import logging

from diffusers.models import ControlNetModel
from diffusers.utils import load_image

from ..pipeline import StreamDiffusion
from .preprocessors import get_preprocessor
from .preprocessing_orchestrator import PreprocessingOrchestrator

# Setup logger for parallel processing
logger = logging.getLogger(__name__)

class BaseControlNetPipeline:
    """
    ControlNet-enabled StreamDiffusion pipeline with optional inter-frame pipelining.
    
    Supports both synchronous and pipelined preprocessing modes:
    - Sync mode: Processes each frame completely before moving to the next
    - Pipelined mode: Overlaps preprocessing of next frame with current frame processing
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 use_pipelined_processing: bool = True):
        """
        Initialize ControlNet pipeline.
        
        Args:
            stream_diffusion: StreamDiffusion instance to wrap
            device: Device to run on ("cuda" or "cpu")
            dtype: Tensor dtype for processing
            use_pipelined_processing: Enable inter-frame pipelining for better performance
        """
        self.stream = stream_diffusion
        self.device = device
        self.dtype = dtype
        self.model_type = getattr(self, 'model_type', 'ControlNet')  # Default fallback
        self.use_pipelined_processing = use_pipelined_processing
        
        self.controlnets: List[ControlNetModel] = []
        self.controlnet_images: List[Optional[torch.Tensor]] = []
        self.controlnet_scales: List[float] = []
        self.preprocessors: List[Optional[Any]] = []
        
        self._original_unet_step = None
        self._is_patched = False
        
        # Initialize preprocessing orchestrator
        self._preprocessing_orchestrator = PreprocessingOrchestrator(
            device=self.device, 
            dtype=self.dtype, 
            max_workers=4
        )
        
        # Keep legacy cache for compatibility
        self._active_indices_cache = []
    
    def add_controlnet(self, 
                      controlnet_config: Dict[str, Any],
                      control_image: Optional[Union[str, Image.Image, np.ndarray, torch.Tensor]] = None) -> int:
        """Add a ControlNet to the pipeline"""
        if not controlnet_config.get('enabled', True):
            return -1
        
        # Load ControlNet model
        controlnet = self._load_controlnet_model(controlnet_config['model_id'])
        
        # Load preprocessor if specified
        preprocessor = None
        if controlnet_config.get('preprocessor'):
            preprocessor = get_preprocessor(controlnet_config['preprocessor'])
            # Set preprocessor parameters including device, dtype, and resolution
            preprocessor_params = {
                'device': self.device,
                'dtype': self.dtype,
                'image_width': self.stream.width,    # Pass actual width
                'image_height': self.stream.height,  # Pass actual height
            }
            if controlnet_config.get('preprocessor_params'):
                preprocessor_params.update(controlnet_config['preprocessor_params'])
            preprocessor.params.update(preprocessor_params)
            # Update device and dtype directly
            if hasattr(preprocessor, 'device'):
                preprocessor.device = self.device
            if hasattr(preprocessor, 'dtype'):
                preprocessor.dtype = self.dtype
        
        # Process control image if provided
        processed_image = None
        if control_image is not None:
            processed_image = self._prepare_control_image(control_image, preprocessor)
        elif controlnet_config.get('control_image_path'):
            # Load from configured path
            control_image = load_image(controlnet_config['control_image_path'])
            processed_image = self._prepare_control_image(control_image, preprocessor)
        
        # Add to collections
        self.controlnets.append(controlnet)
        self.controlnet_images.append(processed_image)
        self.controlnet_scales.append(controlnet_config.get('conditioning_scale', 1.0))
        self.preprocessors.append(preprocessor)
        
        # Patch the StreamDiffusion pipeline if this is the first ControlNet
        if len(self.controlnets) == 1:
            self._patch_stream_diffusion()
        
        return len(self.controlnets) - 1
    
    def remove_controlnet(self, index: int) -> None:
        """Remove a ControlNet by index"""
        if 0 <= index < len(self.controlnets):
            self.controlnets.pop(index)
            self.controlnet_images.pop(index)
            self.controlnet_scales.pop(index)
            self.preprocessors.pop(index)
            
            # Unpatch if no ControlNets remain
            if len(self.controlnets) == 0:
                self._unpatch_stream_diffusion()
        else:
            raise IndexError(f"{self.model_type} ControlNet index {index} out of range")
    
    def clear_controlnets(self) -> None:
        """Remove all ControlNets"""
        self.controlnets.clear()
        self.controlnet_images.clear()
        self.controlnet_scales.clear()
        self.preprocessors.clear()
        
        self._unpatch_stream_diffusion()
        
    def cleanup(self) -> None:
        """Cleanup resources including thread pool"""
        if hasattr(self, '_preprocessing_orchestrator'):
            self._preprocessing_orchestrator.cleanup()
            
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    



    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor], index: Optional[int] = None) -> None:
        """Efficiently update ControlNet(s) with cache-aware preprocessing"""
        
        # Single ControlNet case - always use sync processing
        if index is not None:
            processed_images = self._preprocessing_orchestrator.process_control_images_sync(
                control_image=control_image,
                preprocessors=self.preprocessors,
                scales=self.controlnet_scales,
                stream_width=self.stream.width,
                stream_height=self.stream.height,
                index=index
            )
        # Multi-ControlNet case - use pipelined or sync based on configuration  
        elif self.use_pipelined_processing:
            processed_images = self._preprocessing_orchestrator.process_control_images_pipelined(
                control_image=control_image,
                preprocessors=self.preprocessors,
                scales=self.controlnet_scales,
                stream_width=self.stream.width,
                stream_height=self.stream.height
            )
        else:
            processed_images = self._preprocessing_orchestrator.process_control_images_sync(
                control_image=control_image,
                preprocessors=self.preprocessors,
                scales=self.controlnet_scales,
                stream_width=self.stream.width,
                stream_height=self.stream.height,
                index=None
            )
        
        # If empty list returned, no update needed (same frame detected)
        if not processed_images:
            return
        
        # Update controlnet_images with results
        for i, processed_image in enumerate(processed_images):
            if processed_image is not None:
                self.controlnet_images[i] = processed_image
        
        # Update active indices cache for compatibility
        self._active_indices_cache = [
            i for i, scale in enumerate(self.controlnet_scales) if scale > 0
        ]
    
    def update_controlnet_scale(self, index: int, scale: float) -> None:
        """Update the conditioning scale for a specific ControlNet"""
        if 0 <= index < len(self.controlnets):
            self.controlnet_scales[index] = scale
        else:
            raise IndexError(f"{self.model_type} ControlNet index {index} out of range")

    def _load_controlnet_model(self, model_id: str):
        """Load a ControlNet model with TensorRT acceleration support"""
        # First load the PyTorch model as fallback
        pytorch_controlnet = self._load_pytorch_controlnet_model(model_id)
        
        # Check if TensorRT engine pool is available
        if hasattr(self.stream, 'controlnet_engine_pool'):
            model_type = self._detected_model_type
            is_sdxl = self._is_sdxl
            
            logger.info(f"Loading ControlNet {model_id} with TensorRT acceleration support")
            logger.info(f"  Model type: {model_type}, is_sdxl: {is_sdxl}")
            
            # Debug: Check what batch size we're getting
            detected_batch_size = getattr(self.stream, 'trt_unet_batch_size', 1)
            return self.stream.controlnet_engine_pool.get_or_load_engine(
                model_id=model_id,
                pytorch_model=pytorch_controlnet,
                model_type=model_type,
                batch_size=detected_batch_size
            )
        else:
            # Fallback to PyTorch only
            logger.info(f"Loading ControlNet {model_id} (PyTorch only - no TensorRT acceleration)")
            return pytorch_controlnet
    
    def _load_pytorch_controlnet_model(self, model_id: str):
        """Load a ControlNet model from HuggingFace or local path"""
        try:
            # Check if it's a local path
            if Path(model_id).exists():
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    local_files_only=True
                )
            else:
                # Try as HuggingFace model ID
                if "/" in model_id and model_id.count("/") > 1:
                    # Handle subfolder case (e.g., "repo/model/subfolder")
                    parts = model_id.split("/")
                    repo_id = "/".join(parts[:2])
                    subfolder = "/".join(parts[2:])
                    controlnet = ControlNetModel.from_pretrained(
                        repo_id,
                        subfolder=subfolder,
                        torch_dtype=self.dtype
                    )
                else:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype
                    )
            
            # Move to device
            controlnet = controlnet.to(device=self.device, dtype=self.dtype)
            return controlnet
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} ControlNet model '{model_id}': {e}")
            logger.error(f"Full stack trace for model loading failure:")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to load {self.model_type} ControlNet model '{model_id}': {e}")
    

    
    def _prepare_control_image(self, 
                              control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                              preprocessor: Optional[Any] = None) -> torch.Tensor:
        """Prepare a control image for ControlNet input"""
        # Delegate to preprocessing orchestrator
        return self._preprocessing_orchestrator.prepare_control_image(
            control_image=control_image,
            preprocessor=preprocessor,
            target_width=self.stream.width,
            target_height=self.stream.height
        )
    
    def _process_cfg_and_predict(self, model_pred: torch.Tensor, x_t_latent: torch.Tensor, idx=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process CFG logic and scheduler step (shared between TensorRT and PyTorch modes)"""
        # CFG processing
        if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stream.stock_noise = torch.concat(
                [model_pred[0:1], self.stream.stock_noise[1:]], dim=0
            )
        elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        
        if self.stream.guidance_scale > 1.0 and (
            self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stream.stock_noise * self.stream.delta
        
        if self.stream.guidance_scale > 1.0 and self.stream.cfg_type != "none":
            model_pred = noise_pred_uncond + self.stream.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text
        
        # Scheduler step
        if self.stream.use_denoising_batch:
            denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize":
                scaled_noise = self.stream.beta_prod_t_sqrt * self.stream.stock_noise
                delta_x = self.stream.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.stream.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.stream.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.stream.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.stream.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.stream.init_noise[1:], self.stream.init_noise[0:1]], dim=0
                )
                self.stream.stock_noise = init_noise + delta_x
        else:
            denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
        
        return denoised_batch, model_pred

    def _get_controlnet_conditioning(self, 
                                   x_t_latent: torch.Tensor,
                                   timestep: torch.Tensor,
                                   encoder_hidden_states: torch.Tensor,
                                   **kwargs) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """Get ControlNet conditioning for the current latent and timestep"""
        if not self.controlnets:
            logger.debug("ControlNetPipeline: No ControlNets configured, returning None")
            return None, None
        
        logger.debug(f"ControlNetPipeline: Processing {len(self.controlnets)} ControlNets")
        
        # Get active ControlNet indices (ControlNets with scale > 0 and valid images)
        active_indices = [
            i for i, (controlnet, control_image, scale) in enumerate(
                zip(self.controlnets, self.controlnet_images, self.controlnet_scales)
            ) if controlnet is not None and control_image is not None and scale > 0
        ]
        
        if not active_indices:
            logger.debug("ControlNetPipeline: No active ControlNets, returning None")
            return None, None
        
        logger.debug(f"ControlNetPipeline: Active ControlNet indices: {active_indices}")
        
        # Prepare base kwargs for ControlNet calls
        main_batch_size = x_t_latent.shape[0]
        base_kwargs = {
            'sample': x_t_latent,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        base_kwargs.update(self._get_additional_controlnet_kwargs(**kwargs))
        
        down_samples_list = []
        mid_samples_list = []
        
        for i in active_indices:
            controlnet = self.controlnets[i]
            control_image = self.controlnet_images[i]
            scale = self.controlnet_scales[i]
            
            logger.debug(f"ControlNetPipeline: Processing ControlNet {i} with scale {scale}")
            
            # Optimize batch expansion - do once per ControlNet
            current_control_image = control_image
            if (hasattr(controlnet, 'trt_engine') and controlnet.trt_engine is not None and
                control_image.shape[0] != main_batch_size):
                # Only expand if needed for TensorRT and batch sizes don't match
                if control_image.dim() == 4:
                    current_control_image = control_image.repeat(main_batch_size // control_image.shape[0], 1, 1, 1)
                else:
                    current_control_image = control_image.unsqueeze(0).repeat(main_batch_size, 1, 1, 1)
            
            # Optimized kwargs - reuse base dict and only update specific values
            controlnet_kwargs = base_kwargs
            controlnet_kwargs['controlnet_cond'] = current_control_image
            controlnet_kwargs['conditioning_scale'] = scale
            
            # Forward pass through ControlNet
            try:
                logger.debug(f"ControlNetPipeline: Calling ControlNet {i} with input shape: {current_control_image.shape}")
                down_samples, mid_sample = controlnet(**controlnet_kwargs)
                logger.debug(f"ControlNetPipeline: ControlNet {i} returned - down_blocks: {len(down_samples) if down_samples else 0}, mid_block: {mid_sample is not None}")
                
                down_samples_list.append(down_samples)
                mid_samples_list.append(mid_sample)
            except Exception as e:
                logger.error(f"ControlNetPipeline: ControlNet {i} failed: {e}")
                logger.error(f"ControlNetPipeline: Full stack trace for ControlNet {i}:")
                logger.error(traceback.format_exc())
                continue
        
        # Early exit if no outputs
        if not down_samples_list:
            return None, None
        
        # Optimized combination - single pass for single ControlNet
        if len(down_samples_list) == 1:
            return down_samples_list[0], mid_samples_list[0]
        
        # Vectorized combination for multiple ControlNets
        down_block_res_samples = down_samples_list[0]
        mid_block_res_sample = mid_samples_list[0]
        
        # In-place addition for remaining ControlNets
        for down_samples, mid_sample in zip(down_samples_list[1:], mid_samples_list[1:]):
            for j in range(len(down_block_res_samples)):
                down_block_res_samples[j] += down_samples[j]
            mid_block_res_sample += mid_sample
        
        return down_block_res_samples, mid_block_res_sample
    
    def _patch_stream_diffusion(self) -> None:
        """Patch StreamDiffusion's unet_step method to include ControlNet conditioning"""
        if self._is_patched:
            return
        
        # Store original method
        self._original_unet_step = self.stream.unet_step
        
        # Detect if TensorRT acceleration is being used
        is_tensorrt = hasattr(self.stream.unet, 'engine') or hasattr(self.stream.unet, 'use_control')
        
        if is_tensorrt:
            self._patch_tensorrt_mode()
        else:
            self._patch_pytorch_mode()
        
        self._is_patched = True
    
    def _patch_tensorrt_mode(self):
        """Patch for TensorRT mode with ControlNet support"""
        
        def patched_unet_step_tensorrt(x_t_latent, t_list, idx=None):
            logger.debug(f"ControlNetPipeline: TensorRT unet_step called with latent shape: {x_t_latent.shape}")
            
            # Handle CFG expansion (same as original)
            if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
                t_list_expanded = torch.concat([t_list[0:1], t_list], dim=0)
            elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
                t_list_expanded = torch.concat([t_list, t_list], dim=0)
            else:
                x_t_latent_plus_uc = x_t_latent
                t_list_expanded = t_list
            
            # Get pipeline-specific conditioning context
            conditioning_context = self._get_conditioning_context(x_t_latent_plus_uc, t_list_expanded)
            
            # Get ControlNet conditioning
            logger.debug(f"ControlNetPipeline: Getting ControlNet conditioning for {len(self.controlnets)} ControlNets")
            down_block_res_samples, mid_block_res_sample = self._get_controlnet_conditioning(
                x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds[:, :77, :], **conditioning_context
            )
            
            logger.debug(f"ControlNetPipeline: ControlNet conditioning result - down_blocks: {len(down_block_res_samples) if down_block_res_samples else 0}, mid_block: {mid_block_res_sample is not None}")
            
            # Call TensorRT engine with ControlNet inputs
            model_pred = self.stream.unet(
                x_t_latent_plus_uc,
                t_list_expanded,
                self.stream.prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            # Use shared CFG processing
            return self._process_cfg_and_predict(model_pred, x_t_latent, idx)
        
        # Replace the method
        self.stream.unet_step = patched_unet_step_tensorrt
        logger.info("ControlNetPipeline: Successfully patched TensorRT unet_step method")

    def _patch_pytorch_mode(self):
        """Patch for PyTorch mode with ControlNet support (original implementation)"""
        
        def patched_unet_step_pytorch(x_t_latent, t_list, idx=None):
            # Handle CFG expansion
            if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
                t_list_expanded = torch.concat([t_list[0:1], t_list], dim=0)
            elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
                t_list_expanded = torch.concat([t_list, t_list], dim=0)
            else:
                x_t_latent_plus_uc = x_t_latent
                t_list_expanded = t_list
            
            # Get pipeline-specific conditioning context
            conditioning_context = self._get_conditioning_context(x_t_latent_plus_uc, t_list_expanded)
            
            # Get ControlNet conditioning (extract original text embeddings only - no IPAdapter tokens)
            down_block_res_samples, mid_block_res_sample = self._get_controlnet_conditioning(
                x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds[:, :77, :], **conditioning_context
            )
            
            # Prepare UNet kwargs
            unet_kwargs = {
                'sample': x_t_latent_plus_uc,
                'timestep': t_list_expanded,
                'encoder_hidden_states': self.stream.prompt_embeds,
                'return_dict': False,
            }
            
            # Add ControlNet conditioning
            if down_block_res_samples is not None:
                unet_kwargs['down_block_additional_residuals'] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs['mid_block_additional_residual'] = mid_block_res_sample
            
            # Allow subclasses to add additional UNet kwargs (e.g., SDXL added_cond_kwargs)
            unet_kwargs.update(self._get_additional_unet_kwargs(**conditioning_context))
            
            # Call UNet with ControlNet conditioning
            model_pred = self.stream.unet(**unet_kwargs)[0]
            
            # Use shared CFG processing
            return self._process_cfg_and_predict(model_pred, x_t_latent, idx)
        
        # Replace the method  
        self.stream.unet_step = patched_unet_step_pytorch

    def _unpatch_stream_diffusion(self) -> None:
        """Restore original StreamDiffusion unet_step method"""
        if self._is_patched and self._original_unet_step is not None:
            self.stream.unet_step = self._original_unet_step
            self._is_patched = False

    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying StreamDiffusion instance"""
        return self.stream(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying StreamDiffusion instance"""
        return getattr(self.stream, name)

    # Hook methods for subclasses to override
    def _get_conditioning_context(self, x_t_latent: torch.Tensor, t_list: torch.Tensor) -> Dict[str, Any]:
        """Get conditioning context for this pipeline type (hook for subclasses)"""
        return {}

    def _get_additional_controlnet_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get additional kwargs for ControlNet calls (hook for subclasses)"""
        return {}

    def _get_additional_unet_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get additional kwargs for UNet calls (hook for subclasses)"""
        return {} 