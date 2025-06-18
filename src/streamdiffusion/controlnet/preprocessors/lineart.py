import numpy as np
from PIL import Image
from typing import Union, Optional
import time
from .base import BasePreprocessor

try:
    from controlnet_aux import LineartDetector, LineartAnimeDetector
    CONTROLNET_AUX_AVAILABLE = True
    print("LineartPreprocessor: controlnet_aux successfully imported")
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    raise ImportError("LineartPreprocessor: controlnet_aux is required for real-time optimization. Install with: pip install controlnet_aux")


class LineartPreprocessor(BasePreprocessor):
    """
    Real-time optimized Lineart detection preprocessor for ControlNet
    
    Extracts line art from input images using controlnet_aux line art detection models.
    Supports both realistic and anime-style line art extraction.
    Optimized for real-time performance - no fallbacks.
    """
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 coarse: bool = True,
                 anime_style: bool = False,
                 **kwargs):
        """
        Initialize Lineart preprocessor
        
        Args:
            detect_resolution: Resolution for line art detection
            image_resolution: Output image resolution
            coarse: Whether to use coarse line art detection
            anime_style: Whether to use anime-style line art detection
            **kwargs: Additional parameters
        """
        print(f"LineartPreprocessor.__init__: Initializing with detect_resolution={detect_resolution}, image_resolution={image_resolution}, coarse={coarse}, anime_style={anime_style}")
        
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            coarse=coarse,
            anime_style=anime_style,
            **kwargs
        )
        
        self._detector = None
        print("LineartPreprocessor.__init__: Initialization complete")
    
    @property
    def detector(self):
        """Lazy loading of the line art detector - controlnet_aux only"""
        if self._detector is None:
            start_time = time.time()
            anime_style = self.params.get('anime_style', False)
            
            print(f"LineartPreprocessor.detector: Loading {'Anime' if anime_style else 'Realistic'} Lineart detector from controlnet_aux")
            
            if anime_style:
                self._detector = LineartAnimeDetector.from_pretrained('lllyasviel/Annotators')
                print("LineartPreprocessor.detector: LineartAnimeDetector loaded successfully")
            else:
                self._detector = LineartDetector.from_pretrained('lllyasviel/Annotators')
                print("LineartPreprocessor.detector: LineartDetector loaded successfully")
            
            load_time = time.time() - start_time
            print(f"LineartPreprocessor.detector: Detector loaded in {load_time:.3f}s")
            
        return self._detector
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply line art detection to the input image - real-time optimized
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with detected line art (black lines on white background)
        """
        start_time = time.time()
        print("LineartPreprocessor.process: Starting line art detection")
        
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        validation_time = time.time()
        print(f"LineartPreprocessor.process: Input validation completed in {validation_time - start_time:.3f}s")
        
        # Get parameters
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resolution = self.params.get('image_resolution', 512)
        coarse = self.params.get('coarse', False)
        
        print(f"LineartPreprocessor.process: Using detect_resolution={detect_resolution}, image_resolution={image_resolution}, coarse={coarse}")
        
        # Resize for detection if needed
        if image.size != (detect_resolution, detect_resolution):
            image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
            resize_time = time.time()
            print(f"LineartPreprocessor.process: Image resized from {image.size} to {image_resized.size} in {resize_time - validation_time:.3f}s")
        else:
            image_resized = image
            print("LineartPreprocessor.process: No resizing needed")
        
        # Detect line art using controlnet_aux
        detection_start = time.time()
        print("LineartPreprocessor.process: Starting controlnet_aux line art detection")
        
        lineart_image = self.detector(
            image_resized,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            coarse=coarse
        )
        
        detection_time = time.time() - detection_start
        print(f"LineartPreprocessor.process: Line art detection completed in {detection_time:.3f}s")
        
        # Resize to target resolution if needed
        if lineart_image.size != (image_resolution, image_resolution):
            final_resize_start = time.time()
            lineart_image = lineart_image.resize((image_resolution, image_resolution), Image.LANCZOS)
            final_resize_time = time.time() - final_resize_start
            print(f"LineartPreprocessor.process: Final resize completed in {final_resize_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"LineartPreprocessor.process: Total processing time: {total_time:.3f}s")
        
        return lineart_image 