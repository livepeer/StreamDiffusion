import torch
from PIL import Image
from typing import Union, Optional, Any
from .base import BasePreprocessor


class FeedbackPreprocessor(BasePreprocessor):
    """
    Feedback preprocessor for ControlNet
    
    Uses the OUTPUT from the previous frame's diffusion as the current frame's annotations.
    This creates a feedback loop where each generated frame influences the next generation,
    enabling temporal coherence and iterative refinement effects.
    
    The preprocessor accesses the pipeline's prev_image_result to get the previous output.
    For the first frame (when no previous output exists), it falls back to the input image.
    """
    
    def __init__(self, 
                 pipeline_ref: Optional[Any] = None,
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize feedback preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (can be set later)
            image_resolution: Output image resolution
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            image_resolution=image_resolution,
            **kwargs
        )
        self.pipeline_ref = pipeline_ref
        self._first_frame = True
    
    def set_pipeline_ref(self, pipeline_ref: Any) -> None:
        """
        Set the pipeline reference after initialization
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance
        """
        self.pipeline_ref = pipeline_ref
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process using previous frame output as annotation
        
        Args:
            image: Current input image (used as fallback for first frame)
            
        Returns:
            Previous frame's output as PIL Image, or input image for first frame
        """
        # Check if we have a pipeline reference and previous output
        if (self.pipeline_ref is not None and 
            hasattr(self.pipeline_ref, 'prev_image_result') and 
            self.pipeline_ref.prev_image_result is not None and
            not self._first_frame):
            
            # Convert previous output tensor to PIL Image
            prev_output_tensor = self.pipeline_ref.prev_image_result
            if prev_output_tensor.dim() == 4:
                prev_output_tensor = prev_output_tensor[0]  # Remove batch dimension
            
            # CRITICAL FIX: Convert from [-1, 1] (VAE output) to [0, 1] (ControlNet input)
            prev_output_tensor = (prev_output_tensor / 2.0 + 0.5).clamp(0, 1)
            
            # Use parent class method to convert tensor to PIL
            prev_output_pil = self.tensor_to_pil(prev_output_tensor)
            return prev_output_pil
        else:
            # First frame, no pipeline ref, or no previous output available - use input image
            self._first_frame = False
            return image
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using previous frame output tensor directly (GPU-optimized path)
        
        Args:
            tensor: Current input tensor (used as fallback for first frame)
            
        Returns:
            Previous frame's output tensor, or input tensor for first frame
        """
        # Check if we have a pipeline reference and previous output
        if (self.pipeline_ref is not None and 
            hasattr(self.pipeline_ref, 'prev_image_result') and 
            self.pipeline_ref.prev_image_result is not None and
            not self._first_frame):
            
            prev_output = self.pipeline_ref.prev_image_result
            
            # CRITICAL FIX: Convert from [-1, 1] (VAE output) to [0, 1] (ControlNet input)
            prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)
            
            # Ensure correct tensor format for ControlNet
            if prev_output.dim() == 4 and prev_output.shape[0] == 1:
                prev_output = prev_output[0]  # Remove batch dimension
            if prev_output.dim() == 3:
                prev_output = prev_output.unsqueeze(0)  # Add batch dimension back
                
            # Ensure correct device and dtype
            prev_output = prev_output.to(device=self.device, dtype=self.dtype)
            return prev_output
        else:
            # First frame, no pipeline ref, or no previous output available - use input tensor
            self._first_frame = False
            # Ensure input tensor has correct format
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(device=self.device, dtype=self.dtype)
    
    def reset(self):
        """
        Reset the preprocessor state (useful for new sequences)
        """
        self._first_frame = True