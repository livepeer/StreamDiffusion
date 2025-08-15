import torch
from PIL import Image
from typing import Union, Optional, Any
from .base import BasePreprocessor


class FeedbackPreprocessor(BasePreprocessor):
    """
    Feedback preprocessor for ControlNet
    
    Creates a configurable blend between the current input image and the previous frame's diffusion output.
    This creates a feedback loop where each generated frame influences the next generation,
    while allowing control over the blend strength for stability and creative effects.
    
    Formula: output = (1 - feedback_strength) * input_image + feedback_strength * previous_output
    
    Examples:
    - feedback_strength = 0.0: Pure passthrough (input only)
    - feedback_strength = 0.5: 50/50 blend (default)
    - feedback_strength = 1.0: Pure feedback (previous output only)
    
    The preprocessor accesses the pipeline's prev_image_result to get the previous output.
    For the first frame (when no previous output exists), it falls back to the input image.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Feedback Loop",
            "description": "Creates a configurable blend between the current input image and the previous frame's diffusion output for temporal consistency.",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Strength of feedback blend (0.0 = pure input, 1.0 = pure feedback)"
                }
            },
            "use_cases": ["Temporal consistency", "Video-like generation", "Smooth transitions", "Deforum", "Blast off"]
        }
    
    def __init__(self, 
                 pipeline_ref: Optional[Any] = None,
                 image_resolution: int = 512,
                 feedback_strength: float = 0.5,
                 **kwargs):
        """
        Initialize feedback preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (can be set later)
            image_resolution: Output image resolution
            feedback_strength: Strength of feedback blend (0.0 = pure input, 1.0 = pure feedback, 0.5 = 50/50)
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            image_resolution=image_resolution,
            feedback_strength=feedback_strength,
            **kwargs
        )
        self.pipeline_ref = pipeline_ref
        self.feedback_strength = max(0.0, min(1.0, feedback_strength))  # Clamp to [0, 1]
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
        Process using configurable blend of input image + previous frame output
        
        Args:
            image: Current input image
            
        Returns:
            Blended PIL Image (blend strength controlled by feedback_strength), or input image for first frame
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
            
            # Convert both to tensors for blending
            prev_output_pil = self.tensor_to_pil(prev_output_tensor)
            input_tensor = self.pil_to_tensor(image).squeeze(0)  # Remove batch dim for blending
            prev_tensor = self.pil_to_tensor(prev_output_pil).squeeze(0)
            
            # Blend with configurable strength
            blended_tensor = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * prev_tensor
            
            # Convert back to PIL
            blended_pil = self.tensor_to_pil(blended_tensor)
            return blended_pil
        else:
            # First frame, no pipeline ref, or no previous output available - use input image
            self._first_frame = False
            return image
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using configurable blend of input tensor + previous frame output (GPU-optimized path)
        
        Args:
            tensor: Current input tensor
            
        Returns:
            Blended tensor (blend strength controlled by feedback_strength), or input tensor for first frame
        """
        # Check if we have a pipeline reference and previous output
        if (self.pipeline_ref is not None and 
            hasattr(self.pipeline_ref, 'prev_image_result') and 
            self.pipeline_ref.prev_image_result is not None and
            not self._first_frame):
            
            prev_output = self.pipeline_ref.prev_image_result
            
            # CRITICAL FIX: Convert from [-1, 1] (VAE output) to [0, 1] (ControlNet input)
            prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)
            
            # Normalize input tensor to [0, 1] if needed
            input_tensor = tensor
            if input_tensor.max() > 1.0:
                input_tensor = input_tensor / 255.0
            
            # Ensure both tensors have same format for blending
            if prev_output.dim() == 4 and prev_output.shape[0] == 1:
                prev_output = prev_output[0]  # Remove batch dimension
            if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
                input_tensor = input_tensor[0]  # Remove batch dimension
                
            # Resize if dimensions don't match
            if prev_output.shape != input_tensor.shape:
                # Use the input tensor's shape as target
                if input_tensor.dim() == 3:
                    target_size = input_tensor.shape[-2:]
                    if prev_output.dim() == 3:
                        prev_output = prev_output.unsqueeze(0)
                    prev_output = torch.nn.functional.interpolate(
                        prev_output, size=target_size, mode='bilinear', align_corners=False
                    )
                    if prev_output.shape[0] == 1:
                        prev_output = prev_output.squeeze(0)
            
            # Blend with configurable strength
            blended_tensor = (1 - self.feedback_strength) * input_tensor + self.feedback_strength * prev_output
            
            # Ensure correct output format
            if blended_tensor.dim() == 3:
                blended_tensor = blended_tensor.unsqueeze(0)  # Add batch dimension back
                
            # Ensure correct device and dtype
            blended_tensor = blended_tensor.to(device=self.device, dtype=self.dtype)
            return blended_tensor
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