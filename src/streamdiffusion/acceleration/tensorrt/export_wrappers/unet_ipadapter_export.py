import torch
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any, List

from ....model_detection import detect_model, detect_model_from_diffusers_unet


class TRTIPAttnProcessor(torch.nn.Module):
    """
    TensorRT export-focused IP-Adapter attention processor that consumes a runtime scale tensor.

    Differences from the standard IPAttnProcessor:
    - No fixed instance attribute "scale". Instead, a per-layer tensor is set on the instance
      before forward via wrapper.set_ipadapter_scale(), and used here.
    - This ensures the ONNX graph depends on a real input tensor (ipadapter_scale), enabling
      true runtime control in TensorRT.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self._scale_index: int = -1  # set by installer
        # runtime-provided scalar tensor; must be set before forward
        self._scale_tensor: torch.Tensor = None  # type: ignore

        self.to_k_ip = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        if self._scale_tensor is None:
            raise RuntimeError("TRTIPAttnProcessor: _scale_tensor not set for this layer")

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # split text vs image tokens
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # ip-adapter branch
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)
        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        scale_tensor = self._scale_tensor.to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + scale_tensor * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class TRTIPAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens: int = 4):
        super().__init__()
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("TRTIPAttnProcessor2_0 requires PyTorch 2.0")
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self._scale_index: int = -1
        self._scale_tensor: torch.Tensor = None  # type: ignore

        self.to_k_ip = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = torch.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        if self._scale_tensor is None:
            raise RuntimeError("TRTIPAttnProcessor2_0: _scale_tensor not set for this layer")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # IP branch
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        scale_tensor = self._scale_tensor.to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + scale_tensor * ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class IPAdapterUNetExportWrapper(torch.nn.Module):
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
        
        # Track installed TRT processors
        self._ip_trt_processors: List[torch.nn.Module] = []
        self.num_ip_layers: int = 0

        # Check if IPAdapter processors are already installed (from pre-loading)
        if self._has_ipadapter_processors():
            self._ensure_processor_dtype_consistency()
        elif install_processors:
            # Install IPAdapter processors AFTER dtype conversion
            self._install_ipadapter_processors()
        else:
            print("IPAdapterUNetExportWrapper: WARNING - UNet will not have IPAdapter functionality without processors!")
    
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
            print(f"IPAdapterUNetExportWrapper: Error checking existing processors: {e}")
            return False
    
    def _ensure_processor_dtype_consistency(self):
        """Ensure existing IPAdapter processors have correct dtype for ONNX export"""
        try:
            processors = self.unet.attn_processors
            updated_processors = {}
            self._ip_trt_processors = []
            ip_layer_index = 0
            
            for name, processor in processors.items():
                processor_class = processor.__class__.__name__
                if 'TRTIPAttn' in processor_class:
                    # Already TRT processors: ensure dtype and record
                    proc = processor.to(dtype=torch.float32)
                    proc._scale_index = ip_layer_index
                    self._ip_trt_processors.append(proc)
                    ip_layer_index += 1
                    updated_processors[name] = proc
                elif 'IPAttn' in processor_class or 'IPAttnProcessor' in processor_class:
                    # Replace standard processors with TRT variants, preserving weights where applicable
                    hidden_size = getattr(processor, 'hidden_size', None)
                    cross_attention_dim = getattr(processor, 'cross_attention_dim', None)
                    num_tokens = getattr(processor, 'num_tokens', self.num_image_tokens)
                    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                        from diffusers_ipadapter.ip_adapter.attention_processor import AttnProcessor2_0 as AttnProcessor
                        IPProcClass = TRTIPAttnProcessor2_0
                    else:
                        from diffusers_ipadapter.ip_adapter.attention_processor import AttnProcessor
                        IPProcClass = TRTIPAttnProcessor
                    proc = IPProcClass(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
                    # Copy IP projection weights if present
                    if hasattr(processor, 'to_k_ip') and hasattr(processor, 'to_v_ip') and hasattr(proc, 'to_k_ip'):
                        with torch.no_grad():
                            proc.to_k_ip.weight.copy_(processor.to_k_ip.weight.to(dtype=torch.float32))
                            proc.to_v_ip.weight.copy_(processor.to_v_ip.weight.to(dtype=torch.float32))
                    proc = proc.to(self.unet.device, dtype=torch.float32)
                    proc._scale_index = ip_layer_index
                    self._ip_trt_processors.append(proc)
                    ip_layer_index += 1
                    updated_processors[name] = proc
                else:
                    # Keep standard processors as-is
                    updated_processors[name] = processor
            
            # Update all processors to ensure consistency
            self.unet.set_attn_processor(updated_processors)
            self.num_ip_layers = len(self._ip_trt_processors)
                
        except Exception as e:
            print(f"IPAdapterUNetExportWrapper: Error updating processor dtypes: {e}")
            import traceback
            traceback.print_exc()
    
    def _install_ipadapter_processors(self):
        """
        Install IPAdapter attention processors that will be baked into ONNX.
        These processors handle the internal splitting and processing of concatenated embeddings.
        """
        # Import IPAdapter attention processors from installed package
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                from diffusers_ipadapter.ip_adapter.attention_processor import AttnProcessor2_0 as AttnProcessor
                IPProcClass = TRTIPAttnProcessor2_0
            else:
                from diffusers_ipadapter.ip_adapter.attention_processor import AttnProcessor
                IPProcClass = TRTIPAttnProcessor
            
            # Install attention processors with proper configuration
            processor_names = list(self.unet.attn_processors.keys())
            
            attn_procs = {}
            ip_layer_index = 0
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
                    # Cross-attention layers use TRTIPAttn processors (runtime scale tensor)
                    proc = IPProcClass(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.num_image_tokens,
                    ).to(self.unet.device, dtype=torch.float32)
                    # record mapping index
                    proc._scale_index = ip_layer_index
                    self._ip_trt_processors.append(proc)
                    ip_layer_index += 1
                    attn_procs[name] = proc
            
            self.unet.set_attn_processor(attn_procs)
            self.num_ip_layers = len(self._ip_trt_processors)
            

            
        except Exception as e:
            print(f"IPAdapterUNetExportWrapper: ERROR - Could not install IPAdapter processors: {e}")
            print(f"IPAdapterUNetExportWrapper: Exception type: {type(e).__name__}")
            print("IPAdapterUNetExportWrapper: IPAdapter functionality will not work without processors!")
            import traceback
            traceback.print_exc()
            raise e
    
    def set_ipadapter_scale(self, ipadapter_scale: torch.Tensor) -> None:
        """Assign per-layer scale tensor to installed TRTIPAttn processors."""
        if not isinstance(ipadapter_scale, torch.Tensor):
            raise TypeError("ipadapter_scale must be a torch.Tensor")
        if self.num_ip_layers <= 0 or not self._ip_trt_processors:
            raise RuntimeError("No TRTIPAttn processors installed")
        if ipadapter_scale.ndim != 1 or ipadapter_scale.shape[0] != self.num_ip_layers:
            raise ValueError(f"ipadapter_scale must have shape [{self.num_ip_layers}]")

        # Ensure float32 for ONNX export stability
        scale_vec = ipadapter_scale.to(dtype=torch.float32)
        for proc in self._ip_trt_processors:
            proc._scale_tensor = scale_vec[proc._scale_index]

    def forward(self, sample, timestep, encoder_hidden_states, ipadapter_scale: torch.Tensor = None):
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

        # Set per-layer scale tensor
        if ipadapter_scale is None:
            raise RuntimeError("IPAdapterUNetExportWrapper.forward requires ipadapter_scale tensor")
        self.set_ipadapter_scale(ipadapter_scale)
        
        # Pass concatenated embeddings to UNet with baked-in IPAdapter processors
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )


def create_ipadapter_wrapper(unet: UNet2DConditionModel, num_tokens: int = 4, install_processors: bool = True) -> IPAdapterUNetExportWrapper:
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
        IPAdapterUNetExportWrapper with baked-in IPAdapter attention processors
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
        
        return IPAdapterUNetExportWrapper(unet, cross_attention_dim, num_tokens, install_processors)
        
    except Exception as e:
        print(f"create_ipadapter_wrapper: Error during model detection: {e}")
        return IPAdapterUNetExportWrapper(unet, 768, num_tokens, install_processors) 