import torch
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import numpy as np

from ..pipeline import StreamDiffusion
from .base_streamv2v_pipeline import BaseStreamV2VPipeline


class StreamV2VPipeline(BaseStreamV2VPipeline):
    """
    StreamV2V-enabled StreamDiffusion pipeline for real-time video-to-video translation.
    
    This class extends StreamDiffusion with StreamV2V temporal consistency features,
    maintaining a feature bank for backward-looking frame coherence.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize StreamV2V pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run StreamV2V on
            dtype: Data type for StreamV2V processing
        """
        super().__init__(stream_diffusion, device, dtype)

    def __call__(self, *args, **kwargs):
        """
        Generate frame with StreamV2V temporal consistency.
        Forwards all arguments to the underlying StreamDiffusion object.
        """
        return self.stream(*args, **kwargs)

    def prepare(self, 
               prompt: Union[str, List[str]], 
               negative_prompt: Optional[Union[str, List[str]]] = None,
               **kwargs) -> None:
        """
        Prepare the pipeline for generation.
        
        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s) for generation
            **kwargs: Additional arguments passed to StreamDiffusion.prepare()
        """
        return self.stream.prepare(prompt, negative_prompt, **kwargs)

    def update_prompt(self, 
                     prompt: Union[str, List[str]],
                     negative_prompt: Optional[Union[str, List[str]]] = None) -> None:
        """
        Update prompt during generation.
        
        Args:
            prompt: New prompt(s)
            negative_prompt: New negative prompt(s)
        """
        return self.stream.update_prompt(prompt, negative_prompt)

    def update_stream_params(self, **kwargs) -> None:
        """Update streaming parameters."""
        return self.stream.update_stream_params(**kwargs)

    # Forward any missing attributes/methods to underlying stream
    def __getattr__(self, name):
        if hasattr(self.stream, name):
            return getattr(self.stream, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
