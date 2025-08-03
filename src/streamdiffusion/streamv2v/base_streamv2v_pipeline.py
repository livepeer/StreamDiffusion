import torch
from typing import Dict, Any
from diffusers.models.attention_processor import (
    XFormersAttnProcessor, 
    AttnProcessor2_0
)

from ..pipeline import StreamDiffusion
from .attention_processors import (
    CachedSTXFormersAttnProcessor, 
    CachedSTAttnProcessor2_0
)
from diffusers.utils.import_utils import is_xformers_available


class BaseStreamV2VPipeline:
    """
    Base StreamV2V-enabled StreamDiffusion pipeline.
    
    This class integrates StreamV2V temporal consistency features 
    with StreamDiffusion following the same pattern as ControlNet/IPAdapter.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize base StreamV2V pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run StreamV2V on
            dtype: Data type for StreamV2V models
        """
        self.stream = stream_diffusion
        self.device = device
        self.dtype = dtype
        
        # StreamV2V configuration
        self.use_feature_injection = True
        self.feature_injection_strength = 0.8
        self.feature_similarity_threshold = 0.98
        self.interval = 4
        self.max_frames = 1
        self.use_tome_cache = True
        self.tome_ratio = 0.5
        self.use_grid = False
        
        # Track original processors for restoration
        self.original_processors = None
        self.streamv2v_enabled = False

    def enable_streamv2v(self, config: Dict[str, Any] = None):
        """Enable StreamV2V by replacing attention processors."""
        if self.streamv2v_enabled:
            return
            
        if config:
            self.use_feature_injection = config.get('use_feature_injection', True)
            self.feature_injection_strength = config.get('feature_injection_strength', 0.8)
            self.feature_similarity_threshold = config.get('feature_similarity_threshold', 0.98)
            self.interval = config.get('interval', 4)
            self.max_frames = config.get('max_frames', 1)
            self.use_tome_cache = config.get('use_tome_cache', True)
            self.tome_ratio = config.get('tome_ratio', 0.5)
            self.use_grid = config.get('use_grid', False)
        
        # Store original processors
        self.original_processors = self.stream.unet.attn_processors.copy()
        
        # Replace with StreamV2V processors
        new_processors = {}
        for name, processor in self.original_processors.items():
            if isinstance(processor, XFormersAttnProcessor):
                if not is_xformers_available():
                    raise ImportError("XFormers not available but XFormersAttnProcessor found")
                new_processors[name] = CachedSTXFormersAttnProcessor(
                    name=name,
                    use_feature_injection=self.use_feature_injection,
                    feature_injection_strength=self.feature_injection_strength,
                    feature_similarity_threshold=self.feature_similarity_threshold,
                    interval=self.interval,
                    max_frames=self.max_frames,
                    use_tome_cache=self.use_tome_cache,
                    tome_ratio=self.tome_ratio,
                    use_grid=self.use_grid,
                    attention_op=getattr(processor, 'attention_op', None)
                )
            elif isinstance(processor, AttnProcessor2_0):
                new_processors[name] = CachedSTAttnProcessor2_0(
                    name=name,
                    use_feature_injection=self.use_feature_injection,
                    feature_injection_strength=self.feature_injection_strength,
                    feature_similarity_threshold=self.feature_similarity_threshold,
                    interval=self.interval,
                    max_frames=self.max_frames,
                    use_tome_cache=self.use_tome_cache,
                    tome_ratio=self.tome_ratio,
                    use_grid=self.use_grid
                )
            else:
                # Keep other processors unchanged
                new_processors[name] = processor
        
        self.stream.unet.set_attn_processor(new_processors)
        self.streamv2v_enabled = True

    def disable_streamv2v(self):
        """Disable StreamV2V by restoring original attention processors."""
        if not self.streamv2v_enabled or self.original_processors is None:
            return
            
        self.stream.unet.set_attn_processor(self.original_processors)
        self.streamv2v_enabled = False

    def clear_feature_banks(self):
        """Clear all feature banks to reset temporal state."""
        if not self.streamv2v_enabled:
            return
            
        for processor in self.stream.unet.attn_processors.values():
            if hasattr(processor, 'cached_key'):
                processor.cached_key = None
            if hasattr(processor, 'cached_value'):
                processor.cached_value = None
            if hasattr(processor, 'cached_output'):
                processor.cached_output = None
            if hasattr(processor, 'frame_id'):
                if isinstance(processor.frame_id, torch.Tensor):
                    processor.frame_id = torch.tensor(0)
                else:
                    processor.frame_id = 0

    def update_streamv2v_params(self, **kwargs):
        """Update StreamV2V parameters at runtime."""
        if not self.streamv2v_enabled:
            return
            
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Update processors
                for processor in self.stream.unet.attn_processors.values():
                    if hasattr(processor, key):
                        setattr(processor, key, value) 