from typing import Union, Tuple, Optional, Any
import torch
from PIL import Image
from .ipadapter_embedding import IPAdapterEmbeddingPreprocessor


class FaceIDEmbeddingPreprocessor(IPAdapterEmbeddingPreprocessor):
    """
    Specialized preprocessor for FaceID IPAdapter models that handles face detection 
    and embedding extraction. Inherits from IPAdapterEmbeddingPreprocessor but adds
    FaceID-specific processing capabilities.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "FaceID Embedding", 
            "description": "Generates FaceID embeddings using face detection and InsightFace for identity-preserving image generation.",
            "parameters": {
                "faceid_v2_weight": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Weight for FaceID v2 models (higher values = stronger identity preservation)"
                }
            },
            "use_cases": ["Face identity transfer", "Portrait generation", "Face-conditioned synthesis", "Identity preservation"]
        }
    
    def __init__(self, ipadapter: Any, faceid_v2_weight: float = 1.0, **kwargs):
        """
        Initialize FaceID embedding preprocessor
        
        Args:
            ipadapter: IPAdapter instance with FaceID capabilities
            faceid_v2_weight: Weight for FaceID v2 models
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ipadapter=ipadapter, **kwargs)
        self.faceid_v2_weight = faceid_v2_weight
        
        # Verify this is a FaceID-capable IPAdapter
        if not hasattr(ipadapter, 'is_faceid') or not ipadapter.is_faceid:
            raise ValueError("FaceIDEmbeddingPreprocessor: ipadapter must be a FaceID-capable IPAdapter with is_faceid=True")
        
        if not hasattr(ipadapter, 'insightface_model') or ipadapter.insightface_model is None:
            raise ValueError("FaceIDEmbeddingPreprocessor: ipadapter must have an initialized InsightFace model")
    
    def _process_core(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process face image and extract FaceID embeddings with face detection
        
        Args:
            image: Input image containing a face
            
        Returns:
            Tuple of (positive_embeds, negative_embeds) for FaceID conditioning
        """
        try:
            # Use the IPAdapter's FaceID-specific embedding extraction
            # This handles face detection, alignment, and embedding generation internally
            image_embeds, negative_embeds = self.ipadapter.get_image_embeds(
                images=[image], 
                faceid_v2_weight=self.faceid_v2_weight
            )
            
            print(f"FaceIDEmbeddingPreprocessor._process_core: Generated FaceID embeddings - positive: {image_embeds.shape}, negative: {negative_embeds.shape}")
            
            return image_embeds, negative_embeds
            
        except Exception as e:
            print(f"FaceIDEmbeddingPreprocessor._process_core: Error processing face image: {e}")
            # For FaceID, if face detection fails, we should raise the error rather than return zeros
            # since the whole point is face identity preservation
            raise RuntimeError(f"FaceIDEmbeddingPreprocessor: Failed to extract face embeddings: {e}")
    
    def update_faceid_v2_weight(self, weight: float) -> None:
        """Update the FaceID v2 weight parameter"""
        self.faceid_v2_weight = weight
        print(f"FaceIDEmbeddingPreprocessor.update_faceid_v2_weight: Updated weight to {weight}")