import cv2
import numpy as np
from PIL import Image
import torch
from typing import Union
from .base import BasePreprocessor


class CannyPreprocessor(BasePreprocessor):
    """
    Canny edge detection preprocessor for ControlNet
    
    Detects edges in the input image using the Canny edge detection algorithm.
    """
    
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200, **kwargs):
        """
        Initialize Canny preprocessor
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            **kwargs: Additional parameters
        """
        super().__init__(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            **kwargs
        )
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU for Canny edge detection (minimize CPU transfers)
        
        Args:
            image_tensor: Input image tensor on GPU
            
        Returns:
            Edge map tensor suitable for ControlNet conditioning
        """
        # Validate and normalize input tensor
        image_tensor = self.validate_tensor_input(image_tensor)
        
        # Use image_resolution parameter instead of hardcoded 512x512
        image_resolution = self.params.get('image_resolution', 512)
        target_size = (image_resolution, image_resolution)
        current_size = image_tensor.shape[-2:]
        if current_size != target_size:
            import torch.nn.functional as F
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = F.interpolate(
                image_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
        
        # Convert to grayscale if needed (stay on GPU)
        if image_tensor.shape[0] == 3:  # RGB
            # Convert RGB to grayscale using standard weights
            gray_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        else:
            gray_tensor = image_tensor[0] if image_tensor.shape[0] == 1 else image_tensor
        
        # For Canny edge detection, we need to use OpenCV which requires CPU
        # This is unavoidable as there's no good GPU Canny implementation in PyTorch
        # But we minimize transfers by doing other operations on GPU
        
        # Convert to CPU only for Canny processing
        gray_cpu = gray_tensor.cpu()
        gray_np = (gray_cpu * 255).clamp(0, 255).to(torch.uint8).numpy()
        
        # Apply Canny edge detection
        low_threshold = self.params.get('low_threshold', 100)
        high_threshold = self.params.get('high_threshold', 200)
        
        edges = cv2.Canny(gray_np, low_threshold, high_threshold)
        
        # Convert back to tensor and move to GPU immediately
        edges_tensor = torch.from_numpy(edges).float() / 255.0
        edges_tensor = edges_tensor.to(device=self.device, dtype=self.dtype)
        
        # Convert to RGB format (stack 3 times for compatibility)
        edges_rgb = edges_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return edges_rgb
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply Canny edge detection to the input image
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with detected edges (black and white) resized to target resolution
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Use image_resolution parameter instead of hardcoded 512x512
        image_resolution = self.params.get('image_resolution', 512)
        target_size = (image_resolution, image_resolution)
        if image.size != target_size:
            image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply Canny edge detection
        low_threshold = self.params.get('low_threshold', 100)
        high_threshold = self.params.get('high_threshold', 200)
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to PIL Image (RGB format for consistency)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = Image.fromarray(edges_rgb)
        
        # Ensure final output matches target resolution
        if result.size != target_size:
            result = result.resize(target_size, Image.LANCZOS)
        
        return result 