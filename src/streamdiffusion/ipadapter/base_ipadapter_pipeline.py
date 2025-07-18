import torch
import sys
import os
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import numpy as np
from pathlib import Path

# Using relative import - no sys.path modification needed

from .Diffusers_IPAdapter.ip_adapter.ip_adapter import IPAdapter
from ..pipeline import StreamDiffusion

class BaseIPAdapterPipeline:
    """
    Base IPAdapter-enabled StreamDiffusion pipeline
    
    This class integrates the existing Diffusers_IPAdapter implementation
    with StreamDiffusion following the same pattern as ControlNet.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize base IPAdapter pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run IPAdapter on
            dtype: Data type for IPAdapter models
        """
        self.stream = stream_diffusion
        self.device = device
        self.dtype = dtype
        
        # IPAdapter storage
        self.ipadapters: List[IPAdapter] = []
        self.style_images: List[Optional[Image.Image]] = []
        self.scales: List[float] = []
        
        # Cache for performance optimization
        self._last_prompt = None
        self._last_negative_prompt = None
        self._last_style_images = None
        
        # No patching needed - we use direct embedding assignment like the working script
    
    def add_ipadapter(self, 
                     ipadapter_model_path: str,
                     image_encoder_path: str,
                     style_image: Optional[Union[str, Image.Image]] = None,
                     scale: float = 1.0) -> int:
        """
        Add an IPAdapter to the pipeline
        
        Args:
            ipadapter_model_path: HuggingFace model ID (e.g. "h94/IP-Adapter") or local path to IPAdapter weights
            image_encoder_path: HuggingFace model ID (e.g. "h94/IP-Adapter") or local path to CLIP image encoder
            style_image: Style image for conditioning (optional)
            scale: Conditioning scale
            
        Returns:
            Index of the added IPAdapter
        """
        # Resolve model paths (download if HuggingFace IDs)
        resolved_ipadapter_path = self._resolve_model_path(ipadapter_model_path, "ipadapter")
        resolved_encoder_path = self._resolve_model_path(image_encoder_path, "image_encoder")
        
        # Create IPAdapter instance using existing code
        ipadapter = IPAdapter(
            pipe=self.stream.pipe,
            ipadapter_ckpt_path=resolved_ipadapter_path,
            image_encoder_path=resolved_encoder_path,
            device=self.device,
            dtype=self.dtype
        )
        
        # Process style image if provided
        processed_image = None
        if style_image is not None:
            if isinstance(style_image, str):
                processed_image = Image.open(style_image).convert("RGB")
            else:
                processed_image = style_image
        
        # Add to collections
        self.ipadapters.append(ipadapter)
        self.style_images.append(processed_image)
        self.scales.append(scale)
        
        # Set the initial scale for this IPAdapter
        ipadapter.set_scale(scale)
        
        return len(self.ipadapters) - 1
    
    def remove_ipadapter(self, index: int) -> None:
        """
        Remove an IPAdapter by index
        
        Args:
            index: Index of the IPAdapter to remove
        """
        if 0 <= index < len(self.ipadapters):
            self.ipadapters.pop(index)
            self.style_images.pop(index)
            self.scales.pop(index)
            
            # No unpatching needed
        else:
            raise IndexError(f"IPAdapter index {index} out of range")
    
    def clear_ipadapters(self) -> None:
        """Remove all IPAdapters"""
        self.ipadapters.clear()
        self.style_images.clear()
        self.scales.clear()
    
    def update_style_image(self, style_image: Union[str, Image.Image], index: Optional[int] = None) -> None:
        """
        Update style image for IPAdapter(s)
        
        Args:
            style_image: New style image
            index: Optional IPAdapter index. If None, updates all IPAdapters
        """
        if isinstance(style_image, str):
            style_image = Image.open(style_image).convert("RGB")
        
        if index is not None:
            if 0 <= index < len(self.ipadapters):
                self.style_images[index] = style_image
            else:
                raise IndexError(f"IPAdapter index {index} out of range")
        else:
            # Update all IPAdapters
            for i in range(len(self.ipadapters)):
                self.style_images[i] = style_image
    
    def update_scale(self, index: int, scale: float) -> None:
        """
        Update the conditioning scale for a specific IPAdapter
        
        Args:
            index: Index of the IPAdapter
            scale: New conditioning scale
        """
        if 0 <= index < len(self.ipadapters):
            self.scales[index] = scale
            # Update the IPAdapter's scale directly
            self.ipadapters[index].set_scale(scale)
        else:
            raise IndexError(f"IPAdapter index {index} out of range")
    
    def _resolve_model_path(self, model_path: str, model_type: str) -> str:
        """
        Resolve model path - download from HuggingFace if it's a model ID, or use local path
        
        Args:
            model_path: Either a HuggingFace model ID (e.g. "h94/IP-Adapter") or local file path
            model_type: Type of model ("ipadapter" or "image_encoder")
            
        Returns:
            Resolved local path to the model
        """
        from huggingface_hub import hf_hub_download, snapshot_download
        
        # Check if it's a local path that exists
        if os.path.exists(model_path):
            print(f"_resolve_model_path: Using local {model_type} path: {model_path}")
            return model_path
        
        # Check if it looks like a HuggingFace model ID (contains slash but not local path)
        if "/" in model_path and not os.path.isabs(model_path):
            print(f"_resolve_model_path: Downloading {model_type} from HuggingFace: {model_path}")
            
            if model_type == "ipadapter":
                # Download IPAdapter model file
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=model_path,
                        filename="models/ip-adapter_sd15.bin"
                    )
                    print(f"_resolve_model_path: IPAdapter downloaded to: {downloaded_path}")
                    return downloaded_path
                except Exception as e:
                    # Try alternative filename patterns
                    for filename in ["models/ip-adapter_sd15.bin", "ip-adapter_sd15.bin", "models/ip-adapter-plus_sd15.bin"]:
                        try:
                            downloaded_path = hf_hub_download(repo_id=model_path, filename=filename)
                            print(f"_resolve_model_path: IPAdapter downloaded to: {downloaded_path}")
                            return downloaded_path
                        except:
                            continue
                    raise ValueError(f"_resolve_model_path: Could not download IPAdapter from {model_path}: {e}")
                    
            elif model_type == "image_encoder":
                # Download image encoder directory
                try:
                    repo_path = snapshot_download(
                        repo_id=model_path,
                        allow_patterns=["models/image_encoder/*"]
                    )
                    encoder_path = os.path.join(repo_path, "models", "image_encoder")
                    print(f"_resolve_model_path: Image encoder downloaded to: {encoder_path}")
                    return encoder_path
                except Exception as e:
                    raise ValueError(f"_resolve_model_path: Could not download image encoder from {model_path}: {e}")
        
        # If we get here, it's neither a valid local path nor a HuggingFace ID
        raise ValueError(f"_resolve_model_path: Invalid model path: {model_path}. Must be either a local path or HuggingFace model ID.")

    def preload_models_for_tensorrt(self, ipadapter_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> None:
        """
        Pre-load IPAdapter models and install processors with weights before TensorRT compilation.
        
        This ensures that when TensorRT compilation occurs, the UNet already has IPAdapter 
        processors with actual model weights installed.
        
        Args:
            ipadapter_config: Optional IPAdapter configuration
        """
        print("preload_models_for_tensorrt: Loading IPAdapter models with weights...")
        
        try:
            # Use the config if provided, otherwise use default h94/IP-Adapter
            if ipadapter_config:
                if isinstance(ipadapter_config, list):
                    config = ipadapter_config[0]  # Use first IPAdapter config
                else:
                    config = ipadapter_config
                
                model_path = config.get('ipadapter_model_path', 'h94/IP-Adapter')
                encoder_path = config.get('image_encoder_path', 'h94/IP-Adapter')
                scale = config.get('scale', 1.0)
            else:
                # Default configuration
                model_path = 'h94/IP-Adapter'
                encoder_path = 'h94/IP-Adapter'
                scale = 1.0
            
            # Resolve model paths using existing resolution logic
            resolved_ipadapter_path = self._resolve_model_path(model_path, "ipadapter")
            resolved_encoder_path = self._resolve_model_path(encoder_path, "image_encoder")
            
            print(f"preload_models_for_tensorrt: Resolved IPAdapter path: {resolved_ipadapter_path}")
            print(f"preload_models_for_tensorrt: Resolved image encoder path: {resolved_encoder_path}")
            
            # Create IPAdapter instance - this will install processors with weights
            ipadapter = IPAdapter(
                pipe=self.stream.pipe,
                ipadapter_ckpt_path=resolved_ipadapter_path,
                image_encoder_path=resolved_encoder_path,
                device=self.device,
                dtype=self.dtype,
            )
            
            # Set the correct scale from config BEFORE TensorRT compilation
            ipadapter.set_scale(scale)
            print(f"preload_models_for_tensorrt: Set IPAdapter scale to {scale} before TensorRT compilation")
            
            # Store reference to pre-loaded IPAdapters for later use
            if not hasattr(self.stream, '_preloaded_ipadapters'):
                self.stream._preloaded_ipadapters = []
            self.stream._preloaded_ipadapters.append(ipadapter)
            
            # Add to our own collections too
            self.ipadapters.append(ipadapter)
            self.style_images.append(None)  # No style image during preload
            self.scales.append(scale)
            
            # Mark that stream was pre-loaded with weights
            self.stream._preloaded_with_weights = True
            
            print("preload_models_for_tensorrt: IPAdapter models loaded successfully with weights")
            print("preload_models_for_tensorrt: UNet now has IPAdapter processors with weights installed")
            
        except Exception as e:
            print(f"preload_models_for_tensorrt: Error loading IPAdapter models: {e}")
            print("preload_models_for_tensorrt: Falling back to TensorRT compilation without IPAdapter weights")
            # Mark that pre-loading failed
            self.stream._preloaded_with_weights = False

    def get_tensorrt_info(self) -> Dict[str, Any]:
        """
        Get information needed for TensorRT compilation.
        
        Returns:
            Dictionary with TensorRT-relevant IPAdapter information
        """
        tensorrt_info = {
            'has_preloaded_models': getattr(self.stream, '_preloaded_with_weights', False),
            'num_image_tokens': 4,  # Default
            'scale': 1.0,  # Default
            'cross_attention_dim': None
        }
        
        if self.ipadapters and len(self.ipadapters) > 0:
            first_ipadapter = self.ipadapters[0]
            tensorrt_info['num_image_tokens'] = getattr(first_ipadapter, 'num_tokens', 4)
            if len(self.scales) > 0:
                tensorrt_info['scale'] = self.scales[0]
            
            # Get cross attention dimension
            if hasattr(self.stream, 'unet') and hasattr(self.stream.unet, 'config'):
                tensorrt_info['cross_attention_dim'] = self.stream.unet.config.cross_attention_dim
        
        return tensorrt_info

    def _update_stream_embeddings(self, prompt: str = "", negative_prompt: str = "") -> None:
        """
        Update StreamDiffusion embeddings by integrating IPAdapter conditioning
        
        This method properly combines IPAdapter image conditioning with StreamDiffusion's
        existing text embeddings, maintaining compatibility with both txt2img and img2img modes.
        Now includes TensorRT mode detection and separate embedding storage.
        
        Args:
            prompt: Text prompt  
            negative_prompt: Negative text prompt
        """
        if not self.ipadapters or not any(img is not None for img in self.style_images):
            return  # No IPAdapters or style images
        
        # Skip if nothing changed (performance optimization)
        if (prompt == self._last_prompt and 
            negative_prompt == self._last_negative_prompt and 
            self.style_images == self._last_style_images):
            return
        
        # Use the first IPAdapter to generate image embeddings
        first_ipadapter = self.ipadapters[0] 
        
        # Collect all style images
        active_images = [img for img in self.style_images if img is not None]
        if not active_images:
            return
        
        # Get IPAdapter image embeddings only (without text)
        image_prompt_embeds, negative_image_prompt_embeds = first_ipadapter.get_image_embeds(
            images=active_images
        )
        
        # Get the original text embeddings from StreamDiffusion
        # Store original embeddings on first call to prevent accumulation
        if not hasattr(self.stream, '_original_text_prompt_embeds'):
            self.stream._original_text_prompt_embeds = self.stream.prompt_embeds.clone()
            self.stream._original_text_negative_prompt_embeds = getattr(self.stream, 'negative_prompt_embeds', None)
            if self.stream._original_text_negative_prompt_embeds is not None:
                self.stream._original_text_negative_prompt_embeds = self.stream._original_text_negative_prompt_embeds.clone()
        
        # Always use the original text embeddings (not the combined ones from previous calls)
        original_prompt_embeds = self.stream._original_text_prompt_embeds
        original_negative_prompt_embeds = self.stream._original_text_negative_prompt_embeds
        
        if original_prompt_embeds is None:
            print("_update_stream_embeddings: Warning - No original prompt embeddings found")
            return
        
        print(f"_update_stream_embeddings: Original prompt embeds shape: {original_prompt_embeds.shape}")
        print(f"_update_stream_embeddings: Image prompt embeds shape: {image_prompt_embeds.shape}")
        if original_negative_prompt_embeds is not None:
            print(f"_update_stream_embeddings: Original negative embeds shape: {original_negative_prompt_embeds.shape}")
        print(f"_update_stream_embeddings: Negative image embeds shape: {negative_image_prompt_embeds.shape}")
        
        # Detect TensorRT mode (same pattern as ControlNet)
        is_tensorrt = hasattr(self.stream.unet, 'engine') or hasattr(self.stream, 'unet_engine')
        
        print(f"_update_stream_embeddings: TensorRT mode detected: {is_tensorrt}")
        
        # Ensure image embeddings have the same batch size as original embeddings
        batch_size = original_prompt_embeds.shape[0]
        
        # Repeat image embeddings to match batch size if needed
        if image_prompt_embeds.shape[0] == 1 and batch_size > 1:
            image_prompt_embeds = image_prompt_embeds.repeat(batch_size, 1, 1)
            negative_image_prompt_embeds = negative_image_prompt_embeds.repeat(batch_size, 1, 1)
        
        # For IPAdapter, both TensorRT and PyTorch modes use concatenated embeddings
        print("_update_stream_embeddings: IPAdapter mode - concatenating embeddings (same for TensorRT and PyTorch)")
        
        # Concatenate text and image embeddings along the sequence dimension (dim=1)
        # This is how IPAdapter is designed to work - text tokens + image tokens
        combined_prompt_embeds = torch.cat([original_prompt_embeds, image_prompt_embeds], dim=1)
        
        if original_negative_prompt_embeds is not None:
            combined_negative_prompt_embeds = torch.cat([original_negative_prompt_embeds, negative_image_prompt_embeds], dim=1)
        else:
            # If no negative embeddings, create them from positive embeddings with image conditioning
            combined_negative_prompt_embeds = torch.cat([original_prompt_embeds, negative_image_prompt_embeds], dim=1)
        
        # Update StreamDiffusion embeddings with the combined embeddings
        print(f"_update_stream_embeddings: Combined prompt embeds shape: {combined_prompt_embeds.shape}")
        print(f"_update_stream_embeddings: Combined negative embeds shape: {combined_negative_prompt_embeds.shape}")
        
        self.stream.prompt_embeds = combined_prompt_embeds
        self.stream.negative_prompt_embeds = combined_negative_prompt_embeds
        
        # Update token count for attention processors
        total_tokens = combined_prompt_embeds.shape[1]
        first_ipadapter.set_tokens(image_prompt_embeds.shape[0] * first_ipadapter.num_tokens)
        print(f"_update_stream_embeddings: Set tokens to: {image_prompt_embeds.shape[0] * first_ipadapter.num_tokens}")
        
        if is_tensorrt:
            print("_update_stream_embeddings: TensorRT mode - using concatenated embeddings (same as PyTorch)")
        else:
            print("_update_stream_embeddings: PyTorch mode - using concatenated embeddings")
        
        # Update cache
        self._last_prompt = prompt
        self._last_negative_prompt = negative_prompt
        self._last_style_images = self.style_images.copy()
    
    def prepare(self, *args, **kwargs):
        """Forward prepare calls to the underlying StreamDiffusion"""
        return self.stream.prepare(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the original wrapper, updating IPAdapter embeddings first"""
        # Extract prompt from call arguments
        prompt = kwargs.get('prompt', '')
        negative_prompt = kwargs.get('negative_prompt', getattr(self.stream, 'negative_prompt', ''))
        
        # Update stream embeddings with IPAdapter conditioning
        try:
            self._update_stream_embeddings(prompt, negative_prompt)
        except Exception as e:
            print(f"__call__: Error updating IPAdapter embeddings: {e}")
            import traceback
            traceback.print_exc()
        
        # If we have the original wrapper, use its __call__ method (handles image= parameter correctly)
        if hasattr(self, '_original_wrapper'):
            return self._original_wrapper(*args, **kwargs)
        
        # Fallback to underlying stream
        return self.stream(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to the original wrapper first, then to the underlying StreamDiffusion"""
        # Try original wrapper first (for methods like preprocess_image)
        if hasattr(self, '_original_wrapper') and hasattr(self._original_wrapper, name):
            return getattr(self._original_wrapper, name)
        
        # Fallback to underlying stream
        return getattr(self.stream, name) 