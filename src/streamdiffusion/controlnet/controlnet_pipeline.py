import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

from ..pipeline import StreamDiffusion
from .base_controlnet_pipeline import BaseControlNetPipeline


class ControlNetPipeline(BaseControlNetPipeline):
    """
    ControlNet-enabled StreamDiffusion pipeline for SD1.5 and SD Turbo with inter-frame parallelism
    
    This class extends StreamDiffusion with ControlNet support, allowing for
    conditioning the generation process with multiple ControlNet models.
    Supports both SD1.5 and SD Turbo models with pipelined preprocessing.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 model_type: str = "SD1.5",
                 model_cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ControlNet pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run ControlNets on
            dtype: Data type for ControlNet models
            model_type: Type of model being used (e.g., "SD1.5", "SD Turbo")
        """
        super().__init__(stream_diffusion, device, dtype, use_pipelined_processing=True, model_cache_dir=model_cache_dir)
        self.model_type = model_type 