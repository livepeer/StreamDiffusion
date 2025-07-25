import torch
import sys
import os
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

# Using relative import - no sys.path modification needed

try:
    from .Diffusers_IPAdapter.ip_adapter.ip_adapter import IPAdapter
except Exception as e:
    print(f"base_ipadapter_pipeline: Failed to import IPAdapter: {e}")
    raise

try:
    from ..pipeline import StreamDiffusion
except Exception as e:
    print(f"base_ipadapter_pipeline: Failed to import StreamDiffusion: {e}")
    raise

try:
    from ..controlnet.preprocessors.ipadapter_embedding import IPAdapterEmbeddingPreprocessor
except Exception as e:
    print(f"base_ipadapter_pipeline: Failed to import IPAdapterEmbeddingPreprocessor: {e}")
    raise

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
        
        # IPAdapter storage (single IPAdapter for now)
        # TODO: Add support for multiple IPAdapters and multiple style images in future phase
        self.ipadapter: Optional[IPAdapter] = None
        self.style_image: Optional[Image.Image] = None
        self.scale: float = 1.0
        
        # Style image key for embedding preprocessing
        self._style_image_key = "ipadapter_main"
        
        # No caching needed - StreamParameterUpdater handles that
        
        # No patching needed - we use direct embedding assignment like the working script
    
    def set_ipadapter(self, 
                     ipadapter_model_path: str,
                     image_encoder_path: str,
                     style_image: Optional[Union[str, Image.Image]] = None,
                     scale: float = 1.0) -> None:
        """
        Set the IPAdapter for the pipeline (replaces any existing IPAdapter)
        
        Args:
            ipadapter_model_path: Full path to IPAdapter weights file (local path or HuggingFace repo/file path)
            image_encoder_path: Full path to CLIP image encoder (local path or HuggingFace repo/file path)
            style_image: Style image for conditioning (optional)
            scale: Conditioning scale
        """
        # Clear any existing IPAdapter first
        self.clear_ipadapter()
        
        # Resolve model paths (download if HuggingFace paths)
        resolved_ipadapter_path = self._resolve_model_path(ipadapter_model_path)
        resolved_encoder_path = self._resolve_model_path(image_encoder_path)
        
        # Create IPAdapter instance using existing code
        self.ipadapter = IPAdapter(
            pipe=self.stream.pipe,
            ipadapter_ckpt_path=resolved_ipadapter_path,
            image_encoder_path=resolved_encoder_path,
            device=self.device,
            dtype=self.dtype
        )
        
        # Create embedding preprocessor for parallel processing (if not already registered)
        if not self._has_registered_preprocessor():
            embedding_preprocessor = IPAdapterEmbeddingPreprocessor(
                ipadapter=self.ipadapter,
                device=self.device,
                dtype=self.dtype
            )
            
            # Register with StreamParameterUpdater for integrated processing
            self.stream._param_updater.register_embedding_preprocessor(
                embedding_preprocessor, 
                self._style_image_key
            )
        
        # Process style image if provided
        if style_image is not None:
            if isinstance(style_image, str):
                self.style_image = Image.open(style_image).convert("RGB")
            else:
                self.style_image = style_image
            
            # Immediately process embeddings synchronously to ensure they're cached
            self.stream._param_updater.update_style_image(
                self._style_image_key, 
                self.style_image
            )
        else:
            self.style_image = None
        
        # Set scale
        self.scale = scale
        self.ipadapter.set_scale(scale)
        
        # Register IPAdapter enhancer with StreamParameterUpdater
        self.stream._param_updater.register_embedding_enhancer(
            self._enhance_embeddings_with_ipadapter, 
            name="IPAdapter"
        )
    
    def _has_registered_preprocessor(self) -> bool:
        """Check if an embedding preprocessor is already registered for our style image key"""
        if not hasattr(self.stream._param_updater, '_embedding_preprocessors'):
            return False
        
        for preprocessor, key in self.stream._param_updater._embedding_preprocessors:
            if key == self._style_image_key:
                return True
        return False

    def clear_ipadapter(self) -> None:
        """Remove the IPAdapter"""
        # Unregister enhancer from StreamParameterUpdater
        if hasattr(self, '_enhance_embeddings_with_ipadapter'):
            self.stream._param_updater.unregister_embedding_enhancer(
                self._enhance_embeddings_with_ipadapter
            )
        
        # Unregister embedding preprocessor from StreamParameterUpdater
        self.stream._param_updater.unregister_embedding_preprocessor(self._style_image_key)
        
        self.ipadapter = None
        self.style_image = None
        self.scale = 1.0
    
    def update_style_image(self, style_image: Union[str, Image.Image]) -> None:
        """
        Update style image for the IPAdapter
        
        Args:
            style_image: New style image
        """
        if isinstance(style_image, str):
            self.style_image = Image.open(style_image).convert("RGB")
        else:
            self.style_image = style_image
        
        # Trigger parallel embedding preprocessing via StreamParameterUpdater
        if self.style_image is not None:
            self.stream._param_updater.update_style_image(
                self._style_image_key, 
                self.style_image
            )
    
    def update_scale(self, scale: float) -> None:
        """
        Update the conditioning scale for the IPAdapter
        
        Args:
            scale: New conditioning scale
        """
        if self.ipadapter is not None:
            self.scale = scale
            self.ipadapter.set_scale(scale)
    
    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolve model path - download from HuggingFace if it's a repo/file path, or use local path
        
        Args:
            model_path: Either a local file path or HuggingFace repo/file path (e.g. "h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors")
            
        Returns:
            Resolved local path to the model
        """
        from huggingface_hub import hf_hub_download, snapshot_download
        
        print(f"_resolve_model_path: Resolving path: {model_path}")
        
        # Check if it's a local path that exists
        if os.path.exists(model_path):
            print(f"_resolve_model_path: Using local path: {model_path}")
            return model_path
        
        # Check if it looks like a HuggingFace repo/file path
        if "/" in model_path and not os.path.isabs(model_path):
            parts = model_path.split("/")
            if len(parts) >= 3:
                # Format: "repo/owner/path/to/file.bin" or "repo/owner/directory"
                repo_id = "/".join(parts[:2])  # "h94/IP-Adapter"
                file_path = "/".join(parts[2:])  # "models/ip-adapter-plus_sd15.bin" or "models/image_encoder"
                
                # Check if it's a file (has extension) or directory
                if "." in parts[-1]:
                    # It's a file
                    print(f"_resolve_model_path: Downloading file {file_path} from {repo_id}")
                    try:
                        downloaded_path = hf_hub_download(repo_id=repo_id, filename=file_path)
                        print(f"_resolve_model_path: Downloaded to: {downloaded_path}")
                        return downloaded_path
                    except Exception as e:
                        raise ValueError(f"_resolve_model_path: Could not download {file_path} from {repo_id}: {e}")
                else:
                    # It's a directory
                    print(f"_resolve_model_path: Downloading directory {file_path} from {repo_id}")
                    try:
                        repo_path = snapshot_download(
                            repo_id=repo_id,
                            allow_patterns=[f"{file_path}/*"]
                        )
                        full_path = os.path.join(repo_path, file_path)
                        print(f"_resolve_model_path: Downloaded directory to: {full_path}")
                        return full_path
                    except Exception as e:
                        raise ValueError(f"_resolve_model_path: Could not download directory {file_path} from {repo_id}: {e}")
        
        # If we get here, it's neither a valid local path nor a valid HuggingFace path
        raise ValueError(f"_resolve_model_path: Invalid model path: {model_path}. Must be either a local path or HuggingFace repo/file path (e.g. 'h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors').")

    def preload_models_for_tensorrt(self, ipadapter_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> None:
        """
        Pre-load IPAdapter models and install processors with weights before TensorRT compilation.
        
        This ensures that when TensorRT compilation occurs, the UNet already has IPAdapter 
        processors with actual model weights installed.
        
        Args:
            ipadapter_config: Optional IPAdapter configuration
        """

        
        try:
            # Use the config if provided, otherwise use default h94/IP-Adapter
            if ipadapter_config:
                if isinstance(ipadapter_config, list):
                    config = ipadapter_config[0]  # Use first IPAdapter config
                else:
                    config = ipadapter_config
                
                model_path = config.get('ipadapter_model_path', 'h94/IP-Adapter/models/ip-adapter-plus_sd15.bin')
                encoder_path = config.get('image_encoder_path', 'h94/IP-Adapter/models/image_encoder')
                scale = config.get('scale', 1.0)
            else:
                # Default configuration
                model_path = 'h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors'
                encoder_path = 'h94/IP-Adapter/models/image_encoder'
                scale = 1.0
            
            # Resolve model paths using existing resolution logic
            resolved_ipadapter_path = self._resolve_model_path(model_path)
            resolved_encoder_path = self._resolve_model_path(encoder_path)
            

            
            # Create IPAdapter instance - this will install processors with weights
            self.ipadapter = IPAdapter(
                pipe=self.stream.pipe,
                ipadapter_ckpt_path=resolved_ipadapter_path,
                image_encoder_path=resolved_encoder_path,
                device=self.device,
                dtype=self.dtype,
            )
            
            # Set the correct scale from config BEFORE TensorRT compilation
            self.ipadapter.set_scale(scale)
            
            # Create and register embedding preprocessor for parallel processing (if not already registered)
            if not self._has_registered_preprocessor():
                embedding_preprocessor = IPAdapterEmbeddingPreprocessor(
                    ipadapter=self.ipadapter,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # Register with StreamParameterUpdater for integrated processing
                self.stream._param_updater.register_embedding_preprocessor(
                    embedding_preprocessor, 
                    self._style_image_key
                )
            
            # Store reference to pre-loaded IPAdapter for later use
            if not hasattr(self.stream, '_preloaded_ipadapters'):
                self.stream._preloaded_ipadapters = []
            self.stream._preloaded_ipadapters.append(self.ipadapter)
            
            # Set our own properties
            self.style_image = None  # No style image during preload
            self.scale = scale
            
            # Mark that stream was pre-loaded with weights
            self.stream._preloaded_with_weights = True
            

            
        except Exception as e:
            raise RuntimeError(f"Failed to load IPAdapter models: {e}. Check model paths and file formats.")

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
        
        if self.ipadapter is not None:
            tensorrt_info['num_image_tokens'] = getattr(self.ipadapter, 'num_tokens', 4)
            tensorrt_info['scale'] = self.scale
            
            # Get cross attention dimension
            if hasattr(self.stream, 'unet') and hasattr(self.stream.unet, 'config'):
                tensorrt_info['cross_attention_dim'] = self.stream.unet.config.cross_attention_dim
        
        return tensorrt_info

    def _enhance_embeddings_with_ipadapter(self, prompt_embeds: torch.Tensor, negative_prompt_embeds: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhance embeddings with IPAdapter conditioning using the hook system.
        
        This method integrates IPAdapter image conditioning with text embeddings,
        maintaining compatibility with both single prompts and prompt blending.
        Now uses cached parallel-processed embeddings when available.
        
        Args:
            prompt_embeds: Text prompt embeddings from StreamParameterUpdater
            negative_prompt_embeds: Negative prompt embeddings (may be None)
            
        Returns:
            Tuple of (enhanced_prompt_embeds, enhanced_negative_prompt_embeds)
        """
        # If no IPAdapter or style image, return original embeddings
        if self.ipadapter is None or self.style_image is None:
            return prompt_embeds, negative_prompt_embeds
        
        # Get cached embeddings from StreamParameterUpdater (must be available)
        cached_embeddings = self.stream._param_updater.get_cached_embeddings(self._style_image_key)
        if cached_embeddings is None:
            raise RuntimeError(f"_enhance_embeddings_with_ipadapter: No cached embeddings found for key '{self._style_image_key}'. Embedding preprocessing must complete before enhancement.")
        
        image_prompt_embeds, negative_image_prompt_embeds = cached_embeddings
        
        # Ensure image embeddings have the same batch size as text embeddings
        batch_size = prompt_embeds.shape[0]
        if image_prompt_embeds.shape[0] == 1 and batch_size > 1:
            image_prompt_embeds = image_prompt_embeds.repeat(batch_size, 1, 1)
            negative_image_prompt_embeds = negative_image_prompt_embeds.repeat(batch_size, 1, 1)
        
        # Concatenate text and image embeddings along sequence dimension (dim=1)
        # This is how IPAdapter works - text tokens + image tokens
        enhanced_prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        
        if negative_prompt_embeds is not None:
            enhanced_negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_image_prompt_embeds], dim=1)
        else:
            # Create negative embeddings if none provided
            enhanced_negative_prompt_embeds = torch.cat([prompt_embeds, negative_image_prompt_embeds], dim=1)
        
        # Update token count for attention processors
        old_tokens = getattr(self.ipadapter, '_current_tokens', None)
        new_tokens = image_prompt_embeds.shape[0] * self.ipadapter.num_tokens
        
        if old_tokens != new_tokens:
            self.ipadapter.set_tokens(new_tokens)
            self.ipadapter._current_tokens = new_tokens
        
        return enhanced_prompt_embeds, enhanced_negative_prompt_embeds
    
    def prepare(self, *args, **kwargs):
        """Forward prepare calls to the underlying StreamDiffusion"""        
        return self._original_wrapper.prepare(*args, **kwargs)
        
     
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the original wrapper, IPAdapter enhancement happens automatically via hook system"""
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