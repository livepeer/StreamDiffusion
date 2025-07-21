import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np

from ..pipeline import StreamDiffusion
from .base_controlnet_pipeline import BaseControlNetPipeline

class SDXLTurboControlNetPipeline(BaseControlNetPipeline):
    """SDXL Turbo ControlNet pipeline using StreamDiffusion with inter-frame parallelism"""
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 model_type: str = "SDXL Turbo"):
        """Initialize SDXL Turbo ControlNet pipeline"""
        super().__init__(stream_diffusion, device, dtype, use_pipelined_processing=True)
        self.model_type = model_type

    @property
    def controlnet_configs(self) -> List[Dict[str, Any]]:
        """Get ControlNet configurations for compatibility with demos"""
        configs = []
        for i in range(len(self.controlnets)):
            configs.append({
                'conditioning_scale': self.controlnet_scales[i] if i < len(self.controlnet_scales) else 1.0,
                'enabled': self.controlnet_scales[i] > 0 if i < len(self.controlnet_scales) else False,
                'preprocessor': type(self.preprocessors[i]).__name__ if i < len(self.preprocessors) and self.preprocessors[i] is not None else None
            })
        return configs

    def _post_process_control_image(self, control_tensor: torch.Tensor) -> torch.Tensor:
        """Post-process control image tensor with SDXL-specific handling"""
        target_size = (self.stream.height, self.stream.width)
        current_size = control_tensor.shape[-2:]
        if current_size != target_size:
            import torch.nn.functional as F
            control_tensor = F.interpolate(
                control_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return control_tensor

    def _get_conditioning_context(self, x_t_latent: torch.Tensor, t_list: torch.Tensor) -> Dict[str, Any]:
        """Get SDXL-specific conditioning context"""
        conditioning_context = {}
        
        # Use the conditioning that was set up in StreamDiffusion.prepare()
        if hasattr(self.stream, 'add_text_embeds') and hasattr(self.stream, 'add_time_ids'):
            if self.stream.add_text_embeds is not None and self.stream.add_time_ids is not None:
                # Handle batching for CFG - replicate conditioning to match batch size
                batch_size = x_t_latent.shape[0]
                
                # Replicate add_text_embeds and add_time_ids to match the batch size
                if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                    # For initialize mode: [uncond, cond, cond, ...]
                    add_text_embeds = torch.cat([
                        self.stream.add_text_embeds[0:1],  # uncond
                        self.stream.add_text_embeds[1:2].repeat(batch_size - 1, 1)  # repeat cond
                    ], dim=0)
                    add_time_ids = torch.cat([
                        self.stream.add_time_ids[0:1],  # uncond  
                        self.stream.add_time_ids[1:2].repeat(batch_size - 1, 1)  # repeat cond
                    ], dim=0)
                elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                    # For full mode: repeat both uncond and cond for each latent
                    repeat_factor = batch_size // 2
                    add_text_embeds = self.stream.add_text_embeds.repeat(repeat_factor, 1)
                    add_time_ids = self.stream.add_time_ids.repeat(repeat_factor, 1)
                else:
                    # No CFG: just repeat the conditioning
                    add_text_embeds = self.stream.add_text_embeds[1:2].repeat(batch_size, 1) if self.stream.add_text_embeds.shape[0] > 1 else self.stream.add_text_embeds.repeat(batch_size, 1)
                    add_time_ids = self.stream.add_time_ids[1:2].repeat(batch_size, 1) if self.stream.add_time_ids.shape[0] > 1 else self.stream.add_time_ids.repeat(batch_size, 1)
                
                conditioning_context['text_embeds'] = add_text_embeds
                conditioning_context['time_ids'] = add_time_ids
        
        return conditioning_context

    def _get_additional_controlnet_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get SDXL-specific additional kwargs for ControlNet calls"""
        if 'text_embeds' in kwargs or 'time_ids' in kwargs:
            return {'added_cond_kwargs': kwargs}
        return {}

    def _get_additional_unet_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get SDXL-specific additional kwargs for UNet calls"""
        if kwargs:
            return {'added_cond_kwargs': kwargs}
        return {}

    def __call__(self, 
                 image: Union[str, Image.Image, np.ndarray, torch.Tensor] = None,
                 num_inference_steps: int = None,
                 guidance_scale: float = None,
                 **kwargs) -> torch.Tensor:
        """Generate image using SDXL with ControlNet"""
        
        if image is not None:
            self.update_control_image_efficient(image)
            return self.stream(image)
        else:
            return self.stream() 