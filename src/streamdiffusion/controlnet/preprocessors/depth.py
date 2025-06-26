import numpy as np
from PIL import Image
import torch
from typing import Union, Optional
from .base import BasePreprocessor

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DepthPreprocessor(BasePreprocessor):
    """
    Depth estimation preprocessor for ControlNet using MiDaS
    
    Estimates depth maps from input images using the MiDaS depth estimation model.
    """
    
    def __init__(self, 
                 model_name: str = "Intel/dpt-large",
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize depth preprocessor
        
        Args:
            model_name: Name of the depth estimation model to use
            detect_resolution: Resolution for depth detection
            image_resolution: Output image resolution
            **kwargs: Additional parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for depth preprocessing. "
                "Install it with: pip install transformers"
            )
        
        super().__init__(
            model_name=model_name,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            **kwargs
        )
        
        self._depth_estimator = None
    
    @property
    def depth_estimator(self):
        """Lazy loading of the depth estimation model"""
        if self._depth_estimator is None:
            model_name = self.params.get('model_name', 'Intel/dpt-large')
            print(f"Loading depth estimation model: {model_name}")
            self._depth_estimator = pipeline(
                'depth-estimation', 
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        return self._depth_estimator
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU for depth estimation (minimize CPU transfers)
        
        Args:
            image_tensor: Input image tensor on GPU
            
        Returns:
            Depth map tensor suitable for ControlNet conditioning
        """
        # Validate and normalize input tensor
        image_tensor = self.validate_tensor_input(image_tensor)
        
        # Resize for detection using torch operations
        detect_resolution = self.params.get('detect_resolution', 512)
        current_size = image_tensor.shape[-2:]
        
        if current_size != (detect_resolution, detect_resolution):
            import torch.nn.functional as F
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            resized_tensor = F.interpolate(
                image_tensor,
                size=(detect_resolution, detect_resolution),
                mode='bilinear',
                align_corners=False
            )
            
            if image_tensor.shape[0] == 1:
                resized_tensor = resized_tensor.squeeze(0)
        else:
            resized_tensor = image_tensor
        
        # Convert to PIL for depth estimation (required by transformers)
        # This is unavoidable as transformers pipeline expects PIL
        pil_image = self.tensor_to_pil(resized_tensor)
        
        # Estimate depth
        depth_result = self.depth_estimator(pil_image)
        depth_map = depth_result['depth']
        
        # Keep depth processing on GPU when possible
        if hasattr(depth_map, 'to'):
            # It's already a tensor, keep on GPU
            depth_tensor = depth_map.to(device=self.device, dtype=self.dtype)
        else:
            # Convert numpy to tensor, move to GPU
            depth_np = np.array(depth_map)
            depth_tensor = torch.from_numpy(depth_np).to(device=self.device, dtype=self.dtype)
        
        # Normalize depth map to 0-1 range on GPU
        depth_min = depth_tensor.min()
        depth_max = depth_tensor.max()
        if depth_max > depth_min:
            depth_normalized = (depth_tensor - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = torch.zeros_like(depth_tensor)
        
        # Convert to RGB for ControlNet compatibility (stay on GPU)
        if depth_normalized.dim() == 2:
            # Grayscale to RGB: stack 3 times
            depth_rgb = depth_normalized.unsqueeze(0).repeat(3, 1, 1)
        else:
            depth_rgb = depth_normalized
        
        # Resize to target resolution using torch operations
        image_resolution = self.params.get('image_resolution', 512)
        if depth_rgb.shape[-2:] != (image_resolution, image_resolution):
            import torch.nn.functional as F
            if depth_rgb.dim() == 3:
                depth_rgb = depth_rgb.unsqueeze(0)
            
            depth_rgb = F.interpolate(
                depth_rgb,
                size=(image_resolution, image_resolution),
                mode='bilinear',
                align_corners=False
            )
            
            if depth_rgb.shape[0] == 1:
                depth_rgb = depth_rgb.squeeze(0)
        
        return depth_rgb
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply depth estimation to the input image
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with depth map (grayscale)
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Resize for detection
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        # Estimate depth
        depth_result = self.depth_estimator(image_resized)
        depth_map = depth_result['depth']
        
        # Convert depth map to numpy array
        if hasattr(depth_map, 'cpu'):
            depth_np = depth_map.cpu().numpy()
        else:
            depth_np = np.array(depth_map)
        
        # Normalize depth map to 0-255 range
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
        
        # Convert to RGB for ControlNet compatibility
        depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
        result = Image.fromarray(depth_rgb)
        
        # Resize to target resolution
        image_resolution = self.params.get('image_resolution', 512)
        result = result.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return result 