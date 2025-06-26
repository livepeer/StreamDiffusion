import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

from diffusers.models import ControlNetModel
from diffusers.utils import load_image

from ..pipeline import StreamDiffusion
from .preprocessors import get_preprocessor

# Setup logger for parallel processing
logger = logging.getLogger(__name__)

class BaseControlNetPipeline:
    """Base ControlNet-enabled StreamDiffusion pipeline"""
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """Initialize base ControlNet pipeline"""
        self.stream = stream_diffusion
        self.device = device
        self.dtype = dtype
        self.model_type = getattr(self, 'model_type', 'ControlNet')  # Default fallback
        
        self.controlnets: List[ControlNetModel] = []
        self.controlnet_images: List[Optional[torch.Tensor]] = []
        self.controlnet_scales: List[float] = []
        self.preprocessors: List[Optional[Any]] = []
        
        self._original_unet_step = None
        self._is_patched = False
        
        import torchvision.transforms as transforms
        self._cached_transform = transforms.ToTensor()
        self._temp_tensor_cache = {}
        self._preprocessed_cache = {}
        self._last_input_frame = None
        self._preprocessor_type_cache = {}
        self._active_indices_cache = []
        self._preprocessor_executor = ThreadPoolExecutor(max_workers=4)
    
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
        if hasattr(self, '_preprocessor_executor'):
            self._preprocessor_executor.shutdown(wait=True)
            
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    

    def _process_single_preprocessor_sync(self, preprocessor_key, group, control_image, control_tensor=None):
        """Process a single preprocessor group synchronously in thread"""
        try:
            preprocessor = group['preprocessor']
            
            # Use tensor processing if available and we have a tensor
            if (preprocessor is not None and 
                hasattr(preprocessor, 'process_tensor') and 
                control_tensor is not None):
                try:
                    processed_image = self._prepare_control_image(control_tensor, preprocessor)
                except Exception:
                    # Fallback to PIL processing
                    processed_image = self._prepare_control_image(control_image, preprocessor)
            else:
                processed_image = self._prepare_control_image(control_image, preprocessor)
            
            return preprocessor_key, group['indices'], processed_image
        except Exception as e:
            logger.error(f"update_control_image_efficient: Preprocessor {preprocessor_key} failed: {e}")
            return preprocessor_key, group['indices'], None

    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor], index: Optional[int] = None) -> None:
        """Efficiently update ControlNet(s) with cache-aware preprocessing"""
        if index is not None:
            if not (0 <= index < len(self.controlnets)):
                raise IndexError(f"{self.model_type} ControlNet index {index} out of range")
            
            if self.controlnet_scales[index] == 0:
                return
                
            preprocessor = self.preprocessors[index]
            processed_image = self._prepare_control_image(control_image, preprocessor)
            self.controlnet_images[index] = processed_image
            return
            
        if not any(scale > 0 for scale in self.controlnet_scales):
            return
        
        if (self._last_input_frame is not None and 
            isinstance(control_image, (torch.Tensor, np.ndarray, Image.Image)) and 
            control_image is self._last_input_frame):
            return
        
        self._last_input_frame = control_image
        self._preprocessed_cache.clear()
        
        control_tensor = None
        if isinstance(control_image, torch.Tensor):
            control_tensor = control_image.to(device=self.device, dtype=self.dtype)
        elif isinstance(control_image, str):
            control_image = load_image(control_image)
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            control_tensor = to_tensor(control_image).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        elif isinstance(control_image, Image.Image):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            control_tensor = to_tensor(control_image).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        
        self._active_indices_cache.clear()
        active_controlnets = []
        preprocessor_groups = {}
        for i, scale in enumerate(self.controlnet_scales):
            if scale > 0:
                self._active_indices_cache.append(i)
                preprocessor = self.preprocessors[i]
                preprocessor_key = id(preprocessor) if preprocessor is not None else 'passthrough'
                
                if preprocessor_key not in preprocessor_groups:
                    preprocessor_groups[preprocessor_key] = {
                        'preprocessor': preprocessor,
                        'indices': []
                    }
                preprocessor_groups[preprocessor_key]['indices'].append(i)
        
        if not self._active_indices_cache:
            return
        
        if len(preprocessor_groups) > 1:
            futures = [
                self._preprocessor_executor.submit(
                    self._process_single_preprocessor_sync, 
                    prep_key, group, control_image, control_tensor
                )
                for prep_key, group in preprocessor_groups.items()
            ]
            
            for future in futures:
                result = future.result()
                if result[2] is not None:
                    prep_key, indices, processed_image = result
                    cache_key = f"prep_{prep_key}"
                    self._preprocessed_cache[cache_key] = processed_image
                    for index in indices:
                        self.controlnet_images[index] = processed_image
        else:
            prep_key, group = next(iter(preprocessor_groups.items()))
            result = self._process_single_preprocessor_sync(prep_key, group, control_image, control_tensor)
            if result[2] is not None:
                prep_key, indices, processed_image = result
                cache_key = f"prep_{prep_key}"
                self._preprocessed_cache[cache_key] = processed_image
                for index in indices:
                    self.controlnet_images[index] = processed_image
    
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
            # Determine ControlNet type and model type from model_id or config
            controlnet_type = self._infer_controlnet_type(model_id)
            model_type = self._infer_model_type()
            
            print(f"Loading ControlNet {model_id} with TensorRT acceleration support")
            print(f"  ControlNet type: {controlnet_type}, Model type: {model_type}")
            
            # Debug: Check what batch size we're getting
            detected_batch_size = getattr(self.stream, 'trt_unet_batch_size', 1)
            return self.stream.controlnet_engine_pool.get_or_load_engine(
                model_id=model_id,
                pytorch_model=pytorch_controlnet,
                controlnet_type=controlnet_type,
                model_type=model_type,
                batch_size=detected_batch_size
            )
        else:
            # Fallback to PyTorch only
            print(f"Loading ControlNet {model_id} (PyTorch only - no TensorRT acceleration)")
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
            raise ValueError(f"Failed to load {self.model_type} ControlNet model '{model_id}': {e}")
    
    def _infer_controlnet_type(self, model_id: str) -> str:
        """Infer ControlNet type from model ID"""
        model_lower = model_id.lower()
        
        # Common ControlNet type mappings
        if "canny" in model_lower:
            return "canny"
        elif "depth" in model_lower:
            return "depth"
        elif "openpose" in model_lower or "pose" in model_lower:
            return "openpose"
        elif "normal" in model_lower:
            return "normal"
        elif "seg" in model_lower:
            return "seg"
        elif "lineart" in model_lower:
            return "lineart"
        elif "softedge" in model_lower:
            return "softedge"
        elif "scribble" in model_lower:
            return "scribble"
        elif "mlsd" in model_lower:
            return "mlsd"
        elif "qr" in model_lower:
            return "qr"
        else:
            # Default fallback
            return "canny"
    
    #TODO: model detection module
    def _infer_model_type(self) -> str:
        """Infer base model type from StreamDiffusion configuration"""
        # Check UNet configuration to determine model type
        if hasattr(self.stream, 'unet') and hasattr(self.stream.unet, 'config'):
            unet_config = self.stream.unet.config
            
            # Check for SDXL characteristics
            if hasattr(unet_config, 'projection_class_embeddings_input_dim'):
                return "sdxl"
            elif hasattr(unet_config, 'time_cond_proj_dim'):
                return "sdxl"
            # Check cross attention dimension for SDXL
            elif hasattr(unet_config, 'cross_attention_dim') and unet_config.cross_attention_dim == 2048:
                return "sdxl"
        
        # Check text encoder for SDXL
        if hasattr(self.stream, 'text_encoder') and hasattr(self.stream.text_encoder, 'config'):
            text_config = self.stream.text_encoder.config
            if hasattr(text_config, 'hidden_size') and text_config.hidden_size == 1024:
                return "sdxl"
        
        # Check model type from pipeline
        if hasattr(self.stream, 'pipe'):
            pipe_class_name = self.stream.pipe.__class__.__name__
            if "SDXL" in pipe_class_name or "Turbo" in pipe_class_name:
                return "sdxl"
        
        # Default to SD 1.5
        return "sd15"
    
    def _prepare_control_image(self, 
                              control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                              preprocessor: Optional[Any] = None) -> torch.Tensor:
        """Prepare a control image for ControlNet input"""
        # Load image if path
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        
        # Direct tensor processing path (fastest)
        if isinstance(control_image, torch.Tensor):
            # Fast path for tensor input with GPU preprocessor
            if preprocessor is not None and hasattr(preprocessor, 'process_tensor'):
                try:
                    processed_tensor = preprocessor.process_tensor(control_image)
                    if processed_tensor.dim() == 3:
                        processed_tensor = processed_tensor.unsqueeze(0)
                    return processed_tensor.to(device=self.device, dtype=self.dtype)
                except Exception:
                    pass  # Fall through to PIL processing
            
            # Direct tensor passthrough (no preprocessor)
            if preprocessor is None:
                target_size = (self.stream.width, self.stream.height)
                
                # Handle dimensions efficiently
                if control_image.dim() == 4:
                    control_image = control_image[0]
                if control_image.dim() == 3 and control_image.shape[0] not in [1, 3]:
                    control_image = control_image.permute(2, 0, 1)
                
                # Resize if needed
                if control_image.shape[-2:] != target_size:
                    if control_image.dim() == 3:
                        control_image = control_image.unsqueeze(0)
                    control_image = torch.nn.functional.interpolate(
                        control_image, size=target_size, mode='bilinear', align_corners=False
                    )
                    if control_image.shape[0] == 1:
                        control_image = control_image.squeeze(0)
                
                if control_image.dim() == 3:
                    control_image = control_image.unsqueeze(0)
                
                return control_image.to(device=self.device, dtype=self.dtype)
        
        # Apply preprocessor to non-tensor inputs
        if preprocessor is not None:
            control_image = preprocessor.process(control_image)
        
        # Optimized PIL to tensor conversion
        if isinstance(control_image, Image.Image):
            target_size = (self.stream.width, self.stream.height)
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.LANCZOS)
            
            control_tensor = self._cached_transform(control_image).unsqueeze(0)
            return control_tensor.to(device=self.device, dtype=self.dtype)
        
        # Handle numpy arrays efficiently
        if isinstance(control_image, np.ndarray):
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image)
            return self._prepare_control_image(control_image, None)  # Recursive call will hit PIL path
        
        # Handle other tensor types
        if isinstance(control_image, torch.Tensor):
            if control_image.dim() == 4:
                control_image = control_image[0]
            if control_image.dim() == 3 and control_image.shape[0] in [1, 3]:
                control_image = control_image.permute(1, 2, 0)
            
            if control_image.is_cuda:
                control_image = control_image.cpu()
            control_image = control_image.numpy()
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image.astype(np.uint8))
            return self._prepare_control_image(control_image, None)  # Recursive call will hit PIL path
        
        raise ValueError(f"Unsupported control image type: {type(control_image)}")
    
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
        """Get combined conditioning from all active ControlNets"""
        if not self.controlnets:
            return None, None
        
        # Use cached active indices if available (from recent update_control_image_efficient call)
        if hasattr(self, '_active_indices_cache') and self._active_indices_cache:
            # Quick validation of cached indices
            active_indices = [
                i for i in self._active_indices_cache 
                if i < len(self.controlnets) and 
                   self.controlnet_scales[i] > 0
            ]
        else:
            # Fallback to full calculation
            active_indices = [
                i for i, (controlnet, control_image, scale) in enumerate(
                    zip(self.controlnets, self.controlnet_images, self.controlnet_scales)
                ) if controlnet is not None and control_image is not None and scale > 0
            ]
        
        if not active_indices:
            return None, None
        
        # Pre-compute batch expansion once for all ControlNets
        main_batch_size = x_t_latent.shape[0]
        
        # Pre-compute base controlnet_kwargs once
        base_kwargs = {
            'sample': x_t_latent,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        base_kwargs.update(self._get_additional_controlnet_kwargs(**kwargs))
        
        # Process all active ControlNets with optimized loop
        down_samples_list = []
        mid_samples_list = []
        
        for i in active_indices:
            controlnet = self.controlnets[i]
            control_image = self.controlnet_images[i]
            scale = self.controlnet_scales[i]
            
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
                down_samples, mid_sample = controlnet(**controlnet_kwargs)
                down_samples_list.append(down_samples)
                mid_samples_list.append(mid_sample)
            except Exception as e:
                print(f"_get_controlnet_conditioning: ControlNet {i} failed: {e}")
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
            down_block_res_samples, mid_block_res_sample = self._get_controlnet_conditioning(
                x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds, **conditioning_context
            )
            
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
            
            # Get ControlNet conditioning
            down_block_res_samples, mid_block_res_sample = self._get_controlnet_conditioning(
                x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds, **conditioning_context
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