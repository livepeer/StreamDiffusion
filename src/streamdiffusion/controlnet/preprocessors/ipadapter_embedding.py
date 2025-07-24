from typing import Union, Tuple, Optional, Any
import torch
from PIL import Image
from .base import BasePreprocessor


class IPAdapterEmbeddingPreprocessor(BasePreprocessor):
    """
    Preprocessor that generates IPAdapter embeddings instead of spatial conditioning.
    Leverages existing preprocessing infrastructure for parallel IPAdapter embedding generation.
    """
    
    def __init__(self, ipadapter: Any, **kwargs):
        super().__init__(**kwargs)
        self.ipadapter = ipadapter
        # Verify the ipadapter has the required method
        if not hasattr(ipadapter, 'get_image_embeds'):
            raise ValueError("IPAdapterEmbeddingPreprocessor: ipadapter must have 'get_image_embeds' method")
        print(f"IPAdapterEmbeddingPreprocessor.__init__: Created with IPAdapter {id(ipadapter)}")
        
    def _process_core(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (positive_embeds, negative_embeds) instead of processed image"""
        print(f"IPAdapterEmbeddingPreprocessor._process_core: Processing image {image.size}")
        image_embeds, negative_embeds = self.ipadapter.get_image_embeds(images=[image])
        print(f"IPAdapterEmbeddingPreprocessor._process_core: Generated embeddings {image_embeds.shape}, {negative_embeds.shape}")
        return image_embeds, negative_embeds
        
    def _process_tensor_core(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized path for tensor inputs"""
        print(f"IPAdapterEmbeddingPreprocessor._process_tensor_core: Processing tensor {tensor.shape}")
        # Convert tensor to PIL for IPAdapter processing
        pil_image = self.tensor_to_pil(tensor)
        return self._process_core(pil_image)
    
    def process(self, image: Union[Image.Image, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override base process to return embeddings tuple instead of PIL Image"""
        print(f"IPAdapterEmbeddingPreprocessor.process: Processing {type(image)} - IPAdapter: {id(self.ipadapter)}")
        if isinstance(image, torch.Tensor):
            result = self._process_tensor_core(image)
        else:
            image = self.validate_input(image)
            result = self._process_core(image)
        
        print(f"IPAdapterEmbeddingPreprocessor.process: Completed - result shapes: {result[0].shape}, {result[1].shape}")
        return result
    
    def process_tensor(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override base process_tensor to return embeddings tuple"""
        print(f"IPAdapterEmbeddingPreprocessor.process_tensor: Processing tensor {image_tensor.shape}")
        tensor = self.validate_tensor_input(image_tensor)
        return self._process_tensor_core(tensor)
