import torch
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any, List

from ...model_detection import detect_model, detect_model_from_diffusers_unet

class IPAdapterUNetWrapper(torch.nn.Module):
    """
    Wrapper that bakes IPAdapter attention processors into the UNet for ONNX export.
    
    This approach installs IPAdapter attention processors before ONNX export,
    allowing the specialized attention logic to be compiled into TensorRT.
    The UNet expects concatenated embeddings (text + image) as encoder_hidden_states.
    """
    
    def __init__(self, unet: UNet2DConditionModel, cross_attention_dim: int, num_tokens: int = 4, install_processors: bool = True):
        super().__init__()
        self.unet = unet
        self.num_image_tokens = num_tokens  # 4 for standard, 16 for plus
        self.cross_attention_dim = cross_attention_dim  # 768 for SD1.5, 2048 for SDXL
        self.install_processors = install_processors
        
        # Convert to float32 BEFORE installing processors (to avoid resetting them)
        self.unet = self.unet.to(dtype=torch.float32)
        
        # Check if IPAdapter processors are already installed (from pre-loading)
        if self._has_ipadapter_processors():
            self._ensure_processor_dtype_consistency()
        elif install_processors:
            # Install IPAdapter processors AFTER dtype conversion
            self._install_ipadapter_processors()
        else:
            print("IPAdapterUNetWrapper: WARNING - UNet will not have IPAdapter functionality without processors!")
    
    def _has_ipadapter_processors(self) -> bool:
        """Check if the UNet already has IPAdapter processors installed"""
        try:
            processors = self.unet.attn_processors
            for name, processor in processors.items():
                # Check for IPAdapter processor class names
                processor_class = processor.__class__.__name__
                if 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                    return True
            return False
        except Exception as e:
            print(f"IPAdapterUNetWrapper: Error checking existing processors: {e}")
            return False
    
    def _ensure_processor_dtype_consistency(self):
        """Ensure existing IPAdapter processors have correct dtype for ONNX export"""
        try:
            processors = self.unet.attn_processors
            updated_processors = {}
            
            for name, processor in processors.items():
                processor_class = processor.__class__.__name__
                if 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                    # Convert IPAdapter processors to float32 for ONNX consistency
                    # This preserves the weights while updating dtype
                    updated_processors[name] = processor.to(dtype=torch.float32)
                else:
                    # Keep standard processors as-is
                    updated_processors[name] = processor
            
            # Update all processors to ensure consistency
            self.unet.set_attn_processor(updated_processors)
                
        except Exception as e:
            print(f"IPAdapterUNetWrapper: Error updating processor dtypes: {e}")
            import traceback
            traceback.print_exc()
    
    def _install_ipadapter_processors(self):
        """
        Install IPAdapter attention processors that will be baked into ONNX.
        These processors handle the internal splitting and processing of concatenated embeddings.
        """
        # Import IPAdapter attention processors
        import sys
        from pathlib import Path
        
        # Add Diffusers_IPAdapter to path
        current_dir = Path(__file__).parent.parent.parent / "ipadapter" / "Diffusers_IPAdapter"
        sys.path.insert(0, str(current_dir))
        
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
            else:
                from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
            
            # Install attention processors with proper configuration
            processor_names = list(self.unet.attn_processors.keys())
            
            attn_procs = {}
            for name in processor_names:
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                
                # Determine hidden_size based on processor location
                hidden_size = None
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                else:
                    # Fallback for any unexpected processor names
                    hidden_size = self.unet.config.block_out_channels[0]  # Use first block size as fallback
                
                if cross_attention_dim is None:
                    # Self-attention layers use standard processors
                    attn_procs[name] = AttnProcessor()
                else:
                    # Cross-attention layers use IPAdapter processors
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size, 
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.num_image_tokens
                    ).to(self.unet.device, dtype=torch.float32)  # Force float32 for ONNX
            
            self.unet.set_attn_processor(attn_procs)
            

            
        except Exception as e:
            print(f"IPAdapterUNetWrapper: ERROR - Could not install IPAdapter processors: {e}")
            print(f"IPAdapterUNetWrapper: Exception type: {type(e).__name__}")
            print("IPAdapterUNetWrapper: IPAdapter functionality will not work without processors!")
            import traceback
            traceback.print_exc()
            raise e
    
    def forward(self, sample, timestep, encoder_hidden_states):
        """
        Forward pass with concatenated embeddings (text + image).
        
        The IPAdapter processors installed in the UNet will automatically:
        1. Split the concatenated embeddings into text and image parts
        2. Process image tokens with separate attention computation
        3. Apply scaling and blending between text and image attention
        
        Args:
            sample: Latent input tensor
            timestep: Timestep tensor  
            encoder_hidden_states: Concatenated embeddings [text_tokens + image_tokens, cross_attention_dim]
            
        Returns:
            UNet output (noise prediction)
        """
        # Validate input shapes
        batch_size, seq_len, embed_dim = encoder_hidden_states.shape
        
        # Check that we have the expected number of image tokens
        if embed_dim != self.cross_attention_dim:
            raise ValueError(f"Embedding dimension {embed_dim} doesn't match expected {self.cross_attention_dim}")
        
        # Ensure dtype consistency for ONNX export
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.to(torch.float32)
        
        # Pass concatenated embeddings to UNet with baked-in IPAdapter processors
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )


def create_ipadapter_wrapper(unet: UNet2DConditionModel, num_tokens: int = 4, install_processors: bool = True) -> IPAdapterUNetWrapper:
    """
    Create an IPAdapter wrapper with automatic architecture detection and baked-in processors.
    
    Handles both cases:
    1. UNet with pre-loaded IPAdapter processors (preserves existing weights)
    2. UNet without IPAdapter processors (installs new ones if install_processors=True)
    
    Args:
        unet: UNet2DConditionModel to wrap
        num_tokens: Number of image tokens (4 for standard, 16 for plus)
        install_processors: Whether to install IPAdapter processors if none exist
        
    Returns:
        IPAdapterUNetWrapper with baked-in IPAdapter attention processors
    """
    # Detect model architecture
    try:
        model_type = detect_model_from_diffusers_unet(unet)
        cross_attention_dim = unet.config.cross_attention_dim
        
        # Check if UNet already has IPAdapter processors installed
        existing_processors = unet.attn_processors
        has_ipadapter = any('IPAttn' in proc.__class__.__name__ or 'IPAttnProcessor' in proc.__class__.__name__ 
                           for proc in existing_processors.values())
        
        # Validate expected dimensions
        expected_dims = {
            "SD15": 768,
            "SDXL": 2048, 
            "SD21": 1024
        }
        
        expected_dim = expected_dims.get(model_type)
        
        return IPAdapterUNetWrapper(unet, cross_attention_dim, num_tokens, install_processors)
        
    except Exception as e:
        print(f"create_ipadapter_wrapper: Error during model detection: {e}")
        return IPAdapterUNetWrapper(unet, 768, num_tokens, install_processors) 