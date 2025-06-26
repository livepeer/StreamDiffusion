import numpy as np
from PIL import Image
import torch
from typing import Union, Optional
from .base import BasePreprocessor


class PassthroughPreprocessor(BasePreprocessor):
    """
    Passthrough preprocessor for ControlNet
    
    Simply passes the input image through without any processing.
    Useful for ControlNets that expect the raw input image, such as:
    - Tile ControlNet
    - Reference ControlNet
    - Custom ControlNets that don't need preprocessing
    """
    
    def __init__(self, 
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize passthrough preprocessor
        
        Args:
            image_resolution: Output image resolution
            **kwargs: Additional parameters (ignored for passthrough)
        """
        super().__init__(
            image_resolution=image_resolution,
            **kwargs
        )
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU for passthrough (no CPU transfers needed)
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            Resized tensor suitable for ControlNet conditioning
        """
        # Validate and normalize input tensor
        image_tensor = self.validate_tensor_input(image_tensor)
        
        # Resize if needed using torch operations
        image_resolution = self.params.get('image_resolution', 512)
        current_size = image_tensor.shape[-2:]  # Get H, W
        
        if current_size != (image_resolution, image_resolution):
            # Use torch resize (stay on GPU)
            import torch.nn.functional as F
            # Add batch dim if not present
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = F.interpolate(
                image_tensor, 
                size=(image_resolution, image_resolution),
                mode='bilinear',
                align_corners=False
            )
            
            # Remove batch dim if we added it
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
        
        return image_tensor
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Pass through the input image with minimal processing
        
        Args:
            image: Input image
            
        Returns:
            PIL Image (resized to target resolution if needed)
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Resize to target resolution if specified
        image_resolution = self.params.get('image_resolution', 512)
        if image.size != (image_resolution, image_resolution):
            image = image.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return image 