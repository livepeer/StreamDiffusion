"""ControlNet TensorRT model definitions for compilation"""

from typing import List, Dict, Optional
from .models import BaseModel
import torch


class ControlNetTRT(BaseModel):
    """TensorRT model definition for ControlNet compilation"""
    
    def __init__(self, 
                 controlnet_type: str = "canny",
                 fp16: bool = True,
                 device: str = "cuda",
                 max_batch: int = 2,
                 min_batch_size: int = 1,
                 embedding_dim: int = 768,
                 unet_dim: int = 4,
                 **kwargs):
        super().__init__(
            fp16=fp16,
            device=device,
            max_batch=max_batch,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            **kwargs
        )
        self.controlnet_type = controlnet_type
        self.unet_dim = unet_dim
        self.name = f"ControlNet-{controlnet_type}"
        
    def get_input_names(self) -> List[str]:
        """Get input names for ControlNet TensorRT engine"""
        return [
            "sample",
            "timestep",
            "encoder_hidden_states",
            "controlnet_cond"
        ]
        
    def get_output_names(self) -> List[str]:
        """Get output names for ControlNet TensorRT engine"""
        down_names = [f"down_block_{i:02d}" for i in range(12)]
        return down_names + ["mid_block"]
        
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Get dynamic axes configuration for variable input shapes"""
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "B"},
            "timestep": {0: "B"},
            "controlnet_cond": {0: "B", 2: "H_ctrl", 3: "W_ctrl"},
            **{f"down_block_{i:02d}": {0: "B"} for i in range(12)},
            "mid_block": {0: "B"}
        }
    
    def get_input_profile(self, batch_size, image_height, image_width, 
                         static_batch, static_shape):
        """Generate TensorRT input profiles for ControlNet"""
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        
        min_ctrl_h = 256 if not static_shape else image_height
        max_ctrl_h = 1024 if not static_shape else image_height
        min_ctrl_w = 256 if not static_shape else image_width  
        max_ctrl_w = 1024 if not static_shape else image_width
        
        latent_h = image_height // 8
        latent_w = image_width // 8
        min_latent_h = min_ctrl_h // 8
        max_latent_h = max_ctrl_h // 8
        min_latent_w = min_ctrl_w // 8
        max_latent_w = max_ctrl_w // 8
        
        return {
            "sample": [
                (min_batch, self.unet_dim, min_latent_h, min_latent_w),
                (batch_size, self.unet_dim, latent_h, latent_w),
                (max_batch, self.unet_dim, max_latent_h, max_latent_w),
            ],
            "timestep": [
                (min_batch,), (batch_size,), (max_batch,)
            ],
            "encoder_hidden_states": [
                (min_batch, 77, self.embedding_dim),
                (batch_size, 77, self.embedding_dim),
                (max_batch, 77, self.embedding_dim),
            ],
            "controlnet_cond": [
                (min_batch, 3, min_ctrl_h, min_ctrl_w),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_ctrl_h, max_ctrl_w),
            ],
        }
    
    def get_sample_input(self, batch_size, image_height, image_width):
        """Generate sample inputs for ONNX export"""
        latent_height = image_height // 8
        latent_width = image_width // 8
        dtype = torch.float16 if self.fp16 else torch.float32
        
        return (
            torch.randn(batch_size, self.unet_dim, latent_height, latent_width, 
                       dtype=dtype, device=self.device),
            torch.ones(batch_size, dtype=torch.float32, device=self.device),
            torch.randn(batch_size, 77, self.embedding_dim, 
                       dtype=dtype, device=self.device),
            torch.randn(batch_size, 3, image_height, image_width, 
                       dtype=dtype, device=self.device)
        )


class ControlNetSDXLTRT(ControlNetTRT):
    """SDXL-specific ControlNet TensorRT model definition"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('embedding_dim', 2048)
        super().__init__(**kwargs)
        self.name = f"ControlNet-SDXL-{self.controlnet_type}"
    
    def get_input_names(self) -> List[str]:
        """SDXL ControlNet has additional conditioning inputs"""
        base_inputs = super().get_input_names()
        return base_inputs + [
            "text_embeds",
            "time_ids"
        ]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """SDXL dynamic axes include additional inputs"""
        base_axes = super().get_dynamic_axes()
        base_axes.update({
            "text_embeds": {0: "B"},
            "time_ids": {0: "B"}
        })
        return base_axes
    
    def get_input_profile(self, batch_size, image_height, image_width, 
                         static_batch, static_shape):
        """SDXL input profiles with additional conditioning"""
        base_profile = super().get_input_profile(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        
        base_profile.update({
            "text_embeds": [
                (min_batch, 1280), (batch_size, 1280), (max_batch, 1280)
            ],
            "time_ids": [
                (min_batch, 6), (batch_size, 6), (max_batch, 6)
            ]
        })
        
        return base_profile
    
    def get_sample_input(self, batch_size, image_height, image_width):
        """Generate sample inputs for SDXL ControlNet ONNX export"""
        base_inputs = super().get_sample_input(batch_size, image_height, image_width)
        
        dtype = torch.float16 if self.fp16 else torch.float32
        
        sdxl_inputs = (
            torch.randn(batch_size, 1280, dtype=dtype, device=self.device),
            torch.randn(batch_size, 6, dtype=dtype, device=self.device)
        )
        
        return base_inputs + sdxl_inputs


def create_controlnet_model(model_type: str = "sd15", 
                           controlnet_type: str = "canny",
                           **kwargs) -> ControlNetTRT:
    """Factory function to create appropriate ControlNet TensorRT model"""
    if model_type.lower() in ["sdxl", "sdxl-turbo"]:
        return ControlNetSDXLTRT(controlnet_type=controlnet_type, **kwargs)
    else:
        return ControlNetTRT(controlnet_type=controlnet_type, **kwargs) 