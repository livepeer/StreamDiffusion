import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np

from ..pipeline import StreamDiffusion
from .base_controlnet_pipeline import BaseControlNetPipeline

class SDXLTurboControlNetPipeline(BaseControlNetPipeline):
    """SDXL Turbo ControlNet pipeline using StreamDiffusion"""
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 model_type: str = "SDXL Turbo"):
        """Initialize SDXL Turbo ControlNet pipeline"""
        super().__init__(stream_diffusion, device, dtype)
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
        added_cond_kwargs = {}
        
        if hasattr(self.stream, 'add_text_embeds') and hasattr(self.stream, 'add_time_ids'):
            if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                add_text_embeds = torch.concat([self.stream.add_text_embeds[0:1], self.stream.add_text_embeds], dim=0)
                add_time_ids = torch.concat([self.stream.add_time_ids[0:1], self.stream.add_time_ids], dim=0)
            elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                add_text_embeds = torch.concat([self.stream.add_text_embeds, self.stream.add_text_embeds], dim=0)
                add_time_ids = torch.concat([self.stream.add_time_ids, self.stream.add_time_ids], dim=0)
            else:
                add_text_embeds = self.stream.add_text_embeds
                add_time_ids = self.stream.add_time_ids
            
            added_cond_kwargs = {
                'text_embeds': add_text_embeds,
                'time_ids': add_time_ids
            }
        elif hasattr(self.stream.pipe, 'text_encoder_2'):
            batch_size = x_t_latent.shape[0]
            device = self.stream.device
            dtype = self.stream.dtype
            
            time_ids = torch.tensor([
                [self.stream.height, self.stream.width, 0, 0, self.stream.height, self.stream.width]
            ], dtype=dtype, device=device)
            time_ids = time_ids.repeat(batch_size, 1)
            
            text_embeds = torch.zeros((batch_size, 1280), dtype=dtype, device=device)
            
            added_cond_kwargs = {
                'text_embeds': text_embeds,
                'time_ids': time_ids
            }
        
        return added_cond_kwargs

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

    def _setup_sdxl_embeddings(self) -> None:
        """Extract and store SDXL-specific embeddings if using SDXL pipeline"""
        if not hasattr(self.stream.pipe, 'text_encoder_2'):
            return
        
        if hasattr(self.stream, 'add_text_embeds') and hasattr(self.stream, 'add_time_ids'):
            return
        
        try:
            prompt = getattr(self.stream, '_current_prompt', "")
            negative_prompt = getattr(self.stream, '_current_negative_prompt', "")
            
            if not prompt:
                prompt = "a photo"
            
            do_classifier_free_guidance = self.stream.guidance_scale > 1.0
            encoder_output = self.stream.pipe.encode_prompt(
                prompt=prompt,
                device=self.stream.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            
            if len(encoder_output) >= 4:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoder_output[:4]
                
                if do_classifier_free_guidance:
                    self.stream.add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    self.stream.add_text_embeds = pooled_prompt_embeds
                
                original_size = (self.stream.height, self.stream.width)
                target_size = (self.stream.height, self.stream.width)
                crops_coords_top_left = (0, 0)
                
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids], dtype=self.stream.dtype, device=self.stream.device)
                
                if do_classifier_free_guidance:
                    self.stream.add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                else:
                    self.stream.add_time_ids = add_time_ids
                
                print("Set up SDXL embeddings for ControlNet pipeline")
                
        except Exception as e:
            print(f"Warning: Failed to set up SDXL embeddings: {e}")
            batch_size = 2 if self.stream.guidance_scale > 1.0 else 1
            self.stream.add_text_embeds = torch.zeros((batch_size, 1280), dtype=self.stream.dtype, device=self.stream.device)
            
            add_time_ids = torch.tensor([
                [self.stream.height, self.stream.width, 0, 0, self.stream.height, self.stream.width]
            ], dtype=self.stream.dtype, device=self.stream.device)
            self.stream.add_time_ids = add_time_ids.repeat(batch_size, 1)

    def __call__(self, 
                 image: Union[str, Image.Image, np.ndarray, torch.Tensor] = None,
                 num_inference_steps: int = None,
                 guidance_scale: float = None,
                 **kwargs) -> torch.Tensor:
        """Generate image using SDXL Turbo with ControlNet"""
        self._setup_sdxl_embeddings()
        
        if image is not None:
            self.update_control_image_efficient(image)
            return self.stream(image)
        else:
            return self.stream() 