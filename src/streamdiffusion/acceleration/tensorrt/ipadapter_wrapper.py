import torch
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any, List

try:
    from .model_detection import detect_model_from_diffusers_unet
except ImportError:
    # Handle case when running as standalone script
    from model_detection import detect_model_from_diffusers_unet

class IPAdapterUNetWrapper(torch.nn.Module):
    """
    Wrapper that separates image embeddings for ONNX export (Standard IPAdapter only)
    
    This wrapper implements the split embeddings approach where image embeddings
    are passed as separate inputs to the UNet during TensorRT execution, following
    the same pattern as ControlNet but for embeddings.
    """
    
    def __init__(self, unet: UNet2DConditionModel, cross_attention_dim: int, install_processors: bool = True):
        super().__init__()
        self.unet = unet
        self.num_image_tokens = 4  # Standard IPAdapter only (simplified)
        self.cross_attention_dim = cross_attention_dim  # 768 for SD1.5, 2048 for SDXL
        
        print(f"IPAdapterUNetWrapper: Standard IPAdapter (4 tokens), "
              f"cross_attn_dim={self.cross_attention_dim}")
        
        # For ONNX export safety, ensure UNet is in float32
        if not install_processors:
            print("IPAdapterUNetWrapper: Converting UNet to float32 for ONNX export")
            self.unet = self.unet.to(dtype=torch.float32)
        
        # Store original processors for restoration
        self.original_processors = None
        
        # Install IPAdapter processors only if requested
        # For ONNX export, we skip this to avoid dtype issues
        if install_processors:
            self._install_ipadapter_processors()
        else:
            print("IPAdapterUNetWrapper: Skipping attention processor installation for ONNX export")
    
    def _install_ipadapter_processors(self):
        """
        Install IPAdapter attention processors for weight extraction.
        This ensures the UNet has the correct weights when exported to ONNX.
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
            
            # Install attention processors (similar to IPAdapter.set_ip_adapter)
            attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                
                if cross_attention_dim is None:
                    attn_procs[name] = AttnProcessor()
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size, 
                        cross_attention_dim=cross_attention_dim
                    ).to(self.unet.device, dtype=self.unet.dtype)
            
            self.unet.set_attn_processor(attn_procs)
            print(f"IPAdapterUNetWrapper: Installed IPAdapter attention processors")
            
        except ImportError as e:
            print(f"IPAdapterUNetWrapper: Warning - Could not install IPAdapter processors: {e}")
            print("IPAdapterUNetWrapper: Proceeding with default attention processors")
    
    def forward(self, sample, timestep, encoder_hidden_states, image_embeddings):
        """
        Forward pass with separate image embeddings
        
        Args:
            sample: Latent input tensor
            timestep: Timestep tensor  
            encoder_hidden_states: Text embeddings
            image_embeddings: Image embeddings from IPAdapter
            
        Returns:
            UNet output (noise prediction)
        """
        # Validate input shapes match expected standard IPAdapter format
        batch_size = encoder_hidden_states.shape[0]
        expected_image_shape = (batch_size, 4, self.cross_attention_dim)
        
        if image_embeddings.shape != expected_image_shape:
            raise ValueError(f"Image embeddings shape {image_embeddings.shape} doesn't match "
                           f"expected {expected_image_shape} for Standard IPAdapter")
        
        # Combine embeddings for UNet processing (same as current PyTorch implementation)
        # Text embeddings: [batch, text_tokens, cross_attention_dim]
        # Image embeddings: [batch, 4, cross_attention_dim] 
        # Combined: [batch, text_tokens + 4, cross_attention_dim]
        
        # Ensure dtype consistency for ONNX export
        if encoder_hidden_states.dtype != image_embeddings.dtype:
            print(f"IPAdapterUNetWrapper: Converting image_embeddings from {image_embeddings.dtype} to {encoder_hidden_states.dtype}")
            image_embeddings = image_embeddings.to(encoder_hidden_states.dtype)
        
        combined_embeddings = torch.cat([encoder_hidden_states, image_embeddings], dim=1)
        
        print(f"IPAdapterUNetWrapper forward: "
              f"text_embeds={encoder_hidden_states.shape} dtype={encoder_hidden_states.dtype}, "
              f"image_embeds={image_embeddings.shape} dtype={image_embeddings.dtype}, "
              f"combined={combined_embeddings.shape} dtype={combined_embeddings.dtype}")
        
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=combined_embeddings,
            return_dict=False
        )


def create_ipadapter_wrapper(unet: UNet2DConditionModel, install_processors: bool = True) -> IPAdapterUNetWrapper:
    """
    Create an IPAdapter wrapper with automatic architecture detection
    
    Args:
        unet: UNet2DConditionModel to wrap
        install_processors: Whether to install IPAdapter attention processors (set False for ONNX export)
        
    Returns:
        IPAdapterUNetWrapper configured for the detected model architecture
    """
    # Detect model architecture
    try:
        model_type = detect_model_from_diffusers_unet(unet)
        cross_attention_dim = unet.config.cross_attention_dim
        
        print(f"create_ipadapter_wrapper: Detected model type: {model_type}")
        print(f"create_ipadapter_wrapper: Cross attention dim: {cross_attention_dim}")
        
        # Validate expected dimensions
        expected_dims = {
            "SD15": 768,
            "SDXL": 2048, 
            "SD21": 1024
        }
        
        expected_dim = expected_dims.get(model_type)
        if expected_dim and cross_attention_dim != expected_dim:
            print(f"create_ipadapter_wrapper: Warning - Expected {expected_dim} for {model_type}, "
                  f"but got {cross_attention_dim}")
        
        return IPAdapterUNetWrapper(unet, cross_attention_dim, install_processors)
        
    except Exception as e:
        print(f"create_ipadapter_wrapper: Error during model detection: {e}")
        print(f"create_ipadapter_wrapper: Falling back to default cross_attention_dim=768")
        return IPAdapterUNetWrapper(unet, 768, install_processors) 