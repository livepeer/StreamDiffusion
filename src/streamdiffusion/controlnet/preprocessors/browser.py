import numpy as np
import torch
from PIL import Image
from typing import Union, Optional, Dict, Any
from .base import BasePreprocessor


class BrowserPreprocessor(BasePreprocessor):
    """
    Browser-based preprocessor for client-side processed control data
    
    This preprocessor acts as a passthrough for control images that have already
    been processed client-side using MediaPipe or other browser-based processing.
    It demonstrates the advantages of client-side preprocessing:
    - Reduced server load
    - Lower latency (no server-side processing)
    - Better real-time performance
    - Utilizes client device capabilities
    """
    
    def __init__(self,
                 image_resolution: int = 512,
                 validate_input: bool = True,
                 normalize_brightness: bool = False,
                 **kwargs):
        """
        Initialize browser preprocessor
        
        Args:
            image_resolution: Target output resolution
            validate_input: Whether to validate the control image format
            normalize_brightness: Whether to normalize brightness for consistency
            **kwargs: Additional parameters
        """
        super().__init__(
            image_resolution=image_resolution,
            validate_input=validate_input,
            normalize_brightness=normalize_brightness,
            **kwargs
        )
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Process client-preprocessed control image
        
        This method accepts control images that have already been processed
        client-side (e.g., pose skeletons, hand landmarks, face meshes, etc.)
        and applies minimal server-side validation and formatting.
        
        Args:
            image: Pre-processed control image from client
            
        Returns:
            PIL Image ready for ControlNet conditioning
        """
        # Convert to PIL Image if needed
        control_image = self.validate_input(image)
        
        # Resize to target resolution if needed
        image_resolution = self.params.get('image_resolution', 512)
        if control_image.size != (image_resolution, image_resolution):
            control_image = control_image.resize(
                (image_resolution, image_resolution), 
                Image.LANCZOS
            )
        
        # Optional validation of control image format
        if self.params.get('validate_input', True):
            control_image = self._validate_control_image(control_image)
        
        # Optional brightness normalization for consistency
        if self.params.get('normalize_brightness', False):
            control_image = self._normalize_brightness(control_image)
        
        return control_image
    
    def _validate_control_image(self, image: Image.Image) -> Image.Image:
        """
        Validate that the control image is in proper format
        
        Args:
            image: Control image to validate
            
        Returns:
            Validated control image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic validation - check if image has content
        # (not completely black, which might indicate processing failure)
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        if brightness < 1.0:  # Very dark image, might be processing error
            print("BrowserPreprocessor._validate_control_image: Warning - control image appears very dark")
        
        return image
    
    def _normalize_brightness(self, image: Image.Image) -> Image.Image:
        """
        Normalize brightness for consistent control strength
        
        Args:
            image: Control image to normalize
            
        Returns:
            Brightness-normalized image
        """
        img_array = np.array(image)
        
        # Calculate current brightness
        brightness = np.mean(img_array)
        
        # Only normalize if image has content
        if brightness > 5.0:  # Avoid division by very small numbers
            # Normalize to target brightness (adjust as needed)
            target_brightness = 128  # Mid-range brightness
            scale_factor = target_brightness / brightness
            
            # Apply scaling with clipping
            normalized = np.clip(img_array * scale_factor, 0, 255).astype(np.uint8)
            return Image.fromarray(normalized)
        
        return image
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly (optimized path)
        
        Args:
            image_tensor: Pre-processed control tensor from client
            
        Returns:
            Validated control tensor
        """
        # For browser preprocessor, tensor input likely comes from client WebGL/Canvas
        # which is already in the right format, so minimal processing needed
        
        validated_tensor = self.validate_tensor_input(image_tensor)
        
        # Apply any tensor-level validation or normalization here
        # For now, just ensure proper format and return
        
        return validated_tensor.unsqueeze(0) if validated_tensor.dim() == 3 else validated_tensor
    
    def get_client_config(self) -> Dict[str, Any]:
        """
        Get configuration for client-side processing
        
        Returns:
            Dictionary with client configuration parameters
        """
        return {
            'target_resolution': self.params.get('image_resolution', 512),
            'output_format': 'RGB',
            'background_color': 'black',  # Standard for most control types
            'expected_types': [
                'pose',      # Pose skeleton
                'hands',     # Hand landmarks  
                'face',      # Face mesh
                'depth',     # Depth map
                'canny',     # Edge detection
                'custom'     # Any other control type
            ]
        }
    
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor], **kwargs) -> Image.Image:
        """
        Process control image (convenience method)
        
        Args:
            image: Pre-processed control image from client
            **kwargs: Additional parameters
            
        Returns:
            Processed control image
        """
        # Store any client metadata if provided
        client_metadata = kwargs.get('client_metadata', {})
        if client_metadata:
            print(f"BrowserPreprocessor: Received {client_metadata.get('type', 'unknown')} control from client")
        
        return super().__call__(image, **kwargs) 