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
    Wrapper that bakes IPAdapter attention processors into the UNet for ONNX export.
    
    This approach installs IPAdapter attention processors before ONNX export,
    allowing the specialized attention logic to be compiled into TensorRT.
    The UNet expects concatenated embeddings (text + image) as encoder_hidden_states.
    """
    
    def __init__(self, unet: UNet2DConditionModel, cross_attention_dim: int, num_tokens: int = 4):
        super().__init__()
        self.unet = unet
        self.num_image_tokens = num_tokens  # 4 for standard, 16 for plus
        self.cross_attention_dim = cross_attention_dim  # 768 for SD1.5, 2048 for SDXL
        
        print(f"IPAdapterUNetWrapper: Baking IPAdapter processors into UNet")
        print(f"IPAdapterUNetWrapper: {self.num_image_tokens} tokens, cross_attn_dim={self.cross_attention_dim}")
        
        # Convert to float32 BEFORE installing processors (to avoid resetting them)
        print("IPAdapterUNetWrapper: Converting UNet to float32 for ONNX export")
        self.unet = self.unet.to(dtype=torch.float32)
        
        # Check if IPAdapter processors are already installed (from pre-loading)
        if self._has_ipadapter_processors():
            print("IPAdapterUNetWrapper: Detected existing IPAdapter processors with weights - preserving them")
            self._ensure_processor_dtype_consistency()
        else:
            print("IPAdapterUNetWrapper: Installing new IPAdapter processors")
            # Install IPAdapter processors AFTER dtype conversion
            self._install_ipadapter_processors()
    
    def _has_ipadapter_processors(self) -> bool:
        """Check if the UNet already has IPAdapter processors installed"""
        try:
            processors = self.unet.attn_processors
            for name, processor in processors.items():
                # Check for IPAdapter processor class names
                processor_class = processor.__class__.__name__
                if 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                    print(f"IPAdapterUNetWrapper: Found existing IPAdapter processor: {name} -> {processor_class}")
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
                    print(f"IPAdapterUNetWrapper: Updated processor {name} to float32")
                else:
                    # Keep standard processors as-is
                    updated_processors[name] = processor
            
            # Update all processors to ensure consistency
            self.unet.set_attn_processor(updated_processors)
            print("IPAdapterUNetWrapper: Updated IPAdapter processors for ONNX compatibility")
                
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
            print(f"IPAdapterUNetWrapper: Found {len(processor_names)} attention processors")
            if len(processor_names) > 0:
                print(f"IPAdapterUNetWrapper: First few processor names: {processor_names[:3]}")
            
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
                    print(f"IPAdapterUNetWrapper: Warning - Unexpected processor name: {name}")
                    hidden_size = self.unet.config.block_out_channels[0]  # Use first block size as fallback
                
                if cross_attention_dim is None:
                    # Self-attention layers use standard processors
                    attn_procs[name] = AttnProcessor()
                    print(f"IPAdapterUNetWrapper: Added standard processor for {name}")
                else:
                    # Cross-attention layers use IPAdapter processors
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size, 
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.num_image_tokens
                    ).to(self.unet.device, dtype=torch.float32)  # Force float32 for ONNX
                    print(f"IPAdapterUNetWrapper: Added IPAdapter processor for {name} (hidden_size={hidden_size})")
            
            print(f"IPAdapterUNetWrapper: Created {len(attn_procs)} processors total")
            self.unet.set_attn_processor(attn_procs)
            print(f"IPAdapterUNetWrapper: Successfully installed IPAdapter attention processors")
            
            # Count different types of processors and debug class names
            cross_attn_count = len([n for n in attn_procs.keys() if n.endswith('attn2.processor')])
            self_attn_count = len([n for n in attn_procs.keys() if n.endswith('attn1.processor')])
            
            # Debug actual class names
            class_names = [proc.__class__.__name__ for proc in attn_procs.values()]
            unique_classes = set(class_names)
            print(f"IPAdapterUNetWrapper: Processor class types found: {unique_classes}")
            
            # Count IPAdapter processors with more flexible matching
            ipadapter_count = sum(1 for proc in attn_procs.values() 
                                 if 'IPAttn' in proc.__class__.__name__ or 'IPAttnProcessor' in proc.__class__.__name__)
            
            print(f"IPAdapterUNetWrapper: Installed {cross_attn_count} cross-attention processors")
            print(f"IPAdapterUNetWrapper: Installed {self_attn_count} self-attention processors") 
            print(f"IPAdapterUNetWrapper: {ipadapter_count} processors are IPAdapter type")
            
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
        # Note: We can't validate exact sequence length since text length varies
        if embed_dim != self.cross_attention_dim:
            raise ValueError(f"Embedding dimension {embed_dim} doesn't match expected {self.cross_attention_dim}")
        
        # Ensure dtype consistency for ONNX export
        if encoder_hidden_states.dtype != torch.float32:
            print(f"IPAdapterUNetWrapper: Converting embeddings from {encoder_hidden_states.dtype} to float32")
            encoder_hidden_states = encoder_hidden_states.to(torch.float32)
        
        print(f"IPAdapterUNetWrapper forward: "
              f"sample={sample.shape} dtype={sample.dtype}, "
              f"timestep={timestep.shape} dtype={timestep.dtype}, "
              f"encoder_hidden_states={encoder_hidden_states.shape} dtype={encoder_hidden_states.dtype}")
        
        # Pass concatenated embeddings to UNet with baked-in IPAdapter processors
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )


def create_ipadapter_wrapper(unet: UNet2DConditionModel, num_tokens: int = 4) -> IPAdapterUNetWrapper:
    """
    Create an IPAdapter wrapper with automatic architecture detection and baked-in processors.
    
    Handles both cases:
    1. UNet with pre-loaded IPAdapter processors (preserves existing weights)
    2. UNet without IPAdapter processors (installs new ones)
    
    Args:
        unet: UNet2DConditionModel to wrap
        num_tokens: Number of image tokens (4 for standard, 16 for plus)
        
    Returns:
        IPAdapterUNetWrapper with baked-in IPAdapter attention processors
    """
    # Detect model architecture
    try:
        model_type = detect_model_from_diffusers_unet(unet)
        cross_attention_dim = unet.config.cross_attention_dim
        
        print(f"create_ipadapter_wrapper: Detected model type: {model_type}")
        print(f"create_ipadapter_wrapper: Cross attention dim: {cross_attention_dim}")
        print(f"create_ipadapter_wrapper: Image tokens: {num_tokens}")
        
        # Check if UNet already has IPAdapter processors installed
        existing_processors = unet.attn_processors
        has_ipadapter = any('IPAttn' in proc.__class__.__name__ or 'IPAttnProcessor' in proc.__class__.__name__ 
                           for proc in existing_processors.values())
        
        if has_ipadapter:
            print("create_ipadapter_wrapper: UNet already has IPAdapter processors - will preserve existing weights")
        else:
            print("create_ipadapter_wrapper: UNet has no IPAdapter processors - will install new ones")
        
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
        
        return IPAdapterUNetWrapper(unet, cross_attention_dim, num_tokens)
        
    except Exception as e:
        print(f"create_ipadapter_wrapper: Error during model detection: {e}")
        print(f"create_ipadapter_wrapper: Falling back to default cross_attention_dim=768, num_tokens=4")
        return IPAdapterUNetWrapper(unet, 768, num_tokens) 