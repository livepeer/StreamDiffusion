import numpy as np
import cv2
from PIL import Image
from typing import Union, Optional
import time
from .base import BasePreprocessor
import torch
import torch.nn.functional as F


class StandardLineartPreprocessor(BasePreprocessor):
    """
    Real-time optimized Standard Lineart detection preprocessor for ControlNet
    
    Extracts line art from input images using traditional computer vision techniques.
    Uses Gaussian blur and intensity calculations to detect lines without requiring
    pre-trained models. GPU-accelerated with PyTorch for optimal real-time performance.
    """
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 gaussian_sigma: float = 6.0,
                 intensity_threshold: int = 8,
                 **kwargs):
        """
        Initialize Standard Lineart preprocessor
        
        Args:
            detect_resolution: Resolution for line art detection
            image_resolution: Output image resolution
            gaussian_sigma: Standard deviation for Gaussian blur
            intensity_threshold: Threshold for intensity calculation
            **kwargs: Additional parameters
        """
        
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            gaussian_sigma=gaussian_sigma,
            intensity_threshold=intensity_threshold,
            **kwargs
        )
        
        # Initialize GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float, device=None) -> torch.Tensor:
        """Create 2D Gaussian kernel - based on existing codebase pattern"""
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, kernel_size, device=device), 
            torch.linspace(-1, 1, kernel_size, device=device), 
            indexing="ij"
        )
        d = torch.sqrt(x * x + y * y)
        g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
        return g / g.sum()
    
    def _gaussian_blur_torch(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur using PyTorch - GPU accelerated"""
        # Calculate kernel size from sigma (odd number)
        kernel_size = int(2 * torch.ceil(torch.tensor(3 * sigma)) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = self._gaussian_kernel(kernel_size, sigma, device=image.device)
        
        # Handle different input shapes
        if image.dim() == 3:  # HWC format
            H, W, C = image.shape
            # Convert to BCHW format for conv2d
            image = image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            needs_reshape = True
        elif image.dim() == 4:  # Already BCHW
            B, C, H, W = image.shape
            needs_reshape = False
        else:
            raise ValueError(f"standardlineart_gaussian_blur_torch: Unsupported image shape: {image.shape}")
        
        # Expand kernel for all channels
        kernel = kernel.repeat(image.shape[1], 1, 1).unsqueeze(1)
        
        # Apply blur with reflection padding
        padding = kernel_size // 2
        padded_image = F.pad(image, (padding, padding, padding, padding), 'reflect')
        blurred = F.conv2d(padded_image, kernel, padding=0, groups=image.shape[1])
        
        # Convert back to original format if needed
        if needs_reshape:
            blurred = blurred.squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
        
        return blurred
    
    def _ensure_hwc3_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure image has 3 channels (HWC3 format) - PyTorch version"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add channel dimension
        
        if x.dim() != 3:
            raise ValueError(f"standardlineart_ensure_hwc3_torch: Expected 2D or 3D tensor, got {x.dim()}D")
        
        H, W, C = x.shape
        
        if C == 3:
            return x
        elif C == 1:
            return x.repeat(1, 1, 3)
        elif C == 4:
            color = x[:, :, 0:3]
            alpha = x[:, :, 3:4] / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            return torch.clamp(y, 0, 255)
        else:
            raise ValueError(f"standardlineart_ensure_hwc3_torch: Unsupported channel count: {C}")
    
    def _pad64(self, x: int) -> int:
        """Pad to nearest multiple of 64"""
        return int(torch.ceil(torch.tensor(float(x) / 64.0)) * 64 - x)
    
    def _resize_image_with_pad_torch(self, input_image: torch.Tensor, resolution: int) -> tuple:
        """Resize image with padding to target resolution - PyTorch GPU accelerated"""
        img = self._ensure_hwc3_torch(input_image)
        H_raw, W_raw, _ = img.shape
        
        if resolution == 0:
            return img, lambda x: x
            
        k = float(resolution) / float(min(H_raw, W_raw))
        H_target = int(torch.round(torch.tensor(float(H_raw) * k)))
        W_target = int(torch.round(torch.tensor(float(W_raw) * k)))
        
        # Convert to BCHW for interpolation
        img_bchw = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        # Use PyTorch's interpolate for GPU-accelerated resize
        mode = 'bicubic' if k > 1 else 'area'
        img_resized_bchw = F.interpolate(
            img_bchw, 
            size=(H_target, W_target), 
            mode=mode, 
            align_corners=False if mode == 'bicubic' else None
        )
        
        # Convert back to HWC
        img_resized = img_resized_bchw.squeeze(0).permute(1, 2, 0)
        
        # Apply padding
        H_pad, W_pad = self._pad64(H_target), self._pad64(W_target)
        img_padded = F.pad(img_resized.permute(2, 0, 1), (0, W_pad, 0, H_pad), mode='replicate').permute(1, 2, 0)

        def remove_pad(x):
            return x[:H_target, :W_target, ...]

        return img_padded, remove_pad
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply standard line art detection to the input image - PyTorch GPU accelerated
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with detected line art (black lines on white background)
        """
        start_time = time.time()
        
        # Convert to PIL Image if needed, then to numpy, then immediately to GPU tensor
        image = self.validate_input(image)
        if isinstance(image, Image.Image):
            input_image_cpu = np.array(image, dtype=np.uint8)
        else:
            input_image_cpu = image.astype(np.uint8)
            
        # Move to GPU immediately to minimize CPU-GPU transfers
        input_image = torch.from_numpy(input_image_cpu).float().to(self.device)
            
        validation_time = time.time()
        
        # Get parameters
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resolution = self.params.get('image_resolution', 512)
        gaussian_sigma = self.params.get('gaussian_sigma', 6.0)
        intensity_threshold = self.params.get('intensity_threshold', 8)
        
        # Resize for detection with padding - all on GPU
        resize_start = time.time()
        input_image, remove_pad = self._resize_image_with_pad_torch(input_image, detect_resolution)
        resize_time = time.time() - resize_start
        
        # Apply standard lineart algorithm - all operations on GPU
        detection_start = time.time()
        
        # Input is already float32, ensure it's the right range
        x = input_image
        
        # Apply GPU-accelerated Gaussian blur
        g = self._gaussian_blur_torch(x, gaussian_sigma)
        
        # Calculate intensity differences - all GPU operations
        intensity = torch.min(g - x, dim=2)[0]  # min along channel dimension
        intensity = torch.clamp(intensity, 0, 255)
        
        # Normalize intensity - GPU operations  
        threshold_mask = intensity > intensity_threshold
        if torch.any(threshold_mask):
            median_val = torch.median(intensity[threshold_mask])
            normalization_factor = max(16, float(median_val))
        else:
            normalization_factor = 16
            
        intensity = intensity / normalization_factor
        intensity = intensity * 127
        
        # Convert back to uint8 and ensure 3 channels
        detected_map = torch.clamp(intensity, 0, 255).byte()
        detected_map = detected_map.unsqueeze(-1)  # Add channel dimension
        detected_map = self._ensure_hwc3_torch(detected_map.float())
        
        detection_time = time.time() - detection_start
        
        # Remove padding - stay on GPU
        postprocess_start = time.time()
        detected_map = remove_pad(detected_map)
        
        # Resize to target resolution if needed - GPU accelerated
        current_H, current_W = detected_map.shape[:2]
        if (current_H, current_W) != (image_resolution, image_resolution):
            # Convert to BCHW for interpolation
            detected_bchw = detected_map.permute(2, 0, 1).unsqueeze(0)
            resized_bchw = F.interpolate(
                detected_bchw, 
                size=(image_resolution, image_resolution), 
                mode='bicubic', 
                align_corners=False
            )
            detected_map = resized_bchw.squeeze(0).permute(1, 2, 0)
        
        # Only transfer to CPU at the very end for PIL conversion
        detected_map_cpu = detected_map.byte().cpu().numpy()
        lineart_image = Image.fromarray(detected_map_cpu)
        
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        
        return lineart_image 