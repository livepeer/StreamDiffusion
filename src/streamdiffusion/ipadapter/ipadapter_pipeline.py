import torch
from typing import List, Optional, Union, Dict, Any
from PIL import Image

from ..pipeline import StreamDiffusion
from .base_ipadapter_pipeline import BaseIPAdapterPipeline


class IPAdapterPipeline(BaseIPAdapterPipeline):
    """
    IPAdapter-enabled StreamDiffusion pipeline for SD1.5 and SD Turbo
    
    This class extends StreamDiffusion with IPAdapter support, allowing for
    conditioning the generation process with style images through IPAdapter.
    Supports both SD1.5 and SD Turbo models.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 model_type: str = "SD1.5"):
        """
        Initialize IPAdapter pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run IPAdapter on
            dtype: Data type for IPAdapter models
            model_type: Type of model being used (e.g., "SD1.5", "SD Turbo")
        """
        super().__init__(stream_diffusion, device, dtype)
        self.model_type = model_type 