from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ControlNetConfig(BaseModel):
    model_id: str = Field(..., description="HuggingFace model ID or local path to ControlNet model")
    preprocessor: Optional[str] = Field(None, description="Preprocessor name (e.g., 'canny', 'depth', 'pose')")
    conditioning_scale: float = Field(1.0, ge=0.0, le=2.0, description="Conditioning strength (0.0 = disabled, 1.0 = normal, 2.0 = strong)")
    enabled: bool = Field(True, description="Whether this ControlNet is active")
    preprocessor_params: Optional[Dict[str, Any]] = Field(None, description="Parameters passed to the preprocessor")
    
    class Config:
        extra = "forbid"  # Prevent unknown fields

class IPAdapterConfig(BaseModel):
    """Minimal config for constructing an IP-Adapter module instance.

    This module focuses only on embedding composition (step 2 of migration).
    Runtime installation and wrapper wiring will come in later steps.
    """
    style_image_key: Optional[str] = Field("ipadapter_main", description="Key for style image in embedding cache")
    num_image_tokens: int = Field(4, ge=1, le=64, description="Number of image tokens (4 for standard, 16 for plus)")
    ipadapter_model_path: Optional[str] = Field(None, description="Path to IPAdapter model file")
    image_encoder_path: Optional[str] = Field(None, description="Path to image encoder model")
    style_image: Optional[Any] = Field(None, description="Style image (PIL Image, path, or tensor)")
    scale: float = Field(1.0, ge=0.0, le=2.0, description="IPAdapter strength (0.0 = disabled, 1.0 = normal, 2.0 = strong)")
    weight_type: Optional[str] = Field(None, description="Weight distribution type for per-layer scaling")
    enabled: bool = Field(True, description="Whether this IPAdapter is active")
    # FaceID support
    is_faceid: bool = Field(False, description="Whether this is a FaceID-style IPAdapter")
    insightface_model_name: Optional[str] = Field(None, description="InsightFace model name for FaceID")
    
    class Config:
        extra = "forbid"  # Prevent unknown fields

