from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import torch
import numpy as np
from PIL import Image


class BasePreprocessor(ABC):
    """
    Base class for ControlNet preprocessors
    
    All preprocessors should inherit from this class and implement the process method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the preprocessor
        
        Args:
            **kwargs: Preprocessor-specific parameters
        """
        self.params = kwargs
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.float16)
    
    @abstractmethod
    def process(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Process an image for ControlNet input
        
        Args:
            image: Input image in PIL, numpy array, or torch tensor format
            
        Returns:
            Processed PIL Image suitable for ControlNet conditioning
        """
        pass
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process a tensor image directly on GPU (when supported by preprocessor)
        
        Args:
            image_tensor: Input image tensor on GPU
            
        Returns:
            Processed image tensor suitable for ControlNet conditioning
        """
        # Default implementation: convert to PIL and back (subclasses should override)
        pil_image = self.tensor_to_pil(image_tensor)
        processed_pil = self.process(pil_image)
        return self.pil_to_tensor(processed_pil)
    
    def validate_tensor_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate and normalize tensor input for processing
        
        Args:
            image_tensor: Input tensor
            
        Returns:
            Normalized tensor in CHW format, range [0,1], on correct device
        """
        # Handle batch dimension
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take first image from batch
        
        # Convert to CHW format if needed
        if image_tensor.dim() == 3 and image_tensor.shape[0] not in [1, 3]:
            # Likely HWC format, convert to CHW
            image_tensor = image_tensor.permute(2, 0, 1)
        
        # Ensure correct device and dtype
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        
        # Normalize to [0,1] range if needed
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        return image_tensor
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image (minimize CPU transfers)
        
        Args:
            tensor: Input tensor
            
        Returns:
            PIL Image
        """
        # Ensure tensor is in correct format
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy (unavoidable for PIL)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert to uint8
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        else:
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        
        array = tensor.numpy()
        
        if array.shape[-1] == 3:
            return Image.fromarray(array, 'RGB')
        elif array.shape[-1] == 1:
            return Image.fromarray(array.squeeze(-1), 'L')
        else:
            return Image.fromarray(array)
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to tensor on GPU
        
        Args:
            image: PIL Image
            
        Returns:
            Tensor on correct device
        """
        # Convert to numpy first
        array = np.array(image)
        
        # Convert to tensor
        if len(array.shape) == 2:  # Grayscale
            tensor = torch.from_numpy(array).float() / 255.0
            tensor = tensor.unsqueeze(0)  # Add channel dimension
        else:  # RGB
            tensor = torch.from_numpy(array).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        
        # Move to device
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def validate_input(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Convert input to PIL Image for processing
        
        Args:
            image: Input image in various formats
            
        Returns:
            PIL Image
        """
        if isinstance(image, torch.Tensor):
            # Use tensor_to_pil method for better handling
            return self.tensor_to_pil(image)
                
        if isinstance(image, np.ndarray):
            # Ensure uint8 format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Convert to PIL Image
            if len(image.shape) == 3:
                image = Image.fromarray(image, 'RGB')
            else:
                image = Image.fromarray(image, 'L')
                
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        return image
    
    def resize_image(self, image: Image.Image, target_width: int = 512, target_height: int = 512) -> Image.Image:
        """
        Resize image to target dimensions
        
        Args:
            image: PIL Image to resize
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized PIL Image
        """
        return image.resize((target_width, target_height), Image.LANCZOS)
    
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor], **kwargs) -> Image.Image:
        """
        Process an image (convenience method)
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Processed PIL Image
        """
        # Update parameters for this call
        params = {**self.params, **kwargs}
        
        # Store original params and update
        original_params = self.params
        self.params = params
        
        try:
            result = self.process(image)
        finally:
            # Restore original params
            self.params = original_params
            
        return result 