from typing import Callable, Optional
from collections import deque
import torch
import torch.nn.functional as F

from .attention_processors import CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0
from diffusers_ipadapter.ip_adapter.attention_processor import IPAttnProcessor2_0, IPAttnProcessor

from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class CombinedIPAdapterStreamV2VXFormersAttnProcessor(CachedSTXFormersAttnProcessor):
    """
    Combined attention processor that handles both IPAdapter image conditioning 
    and StreamV2V temporal consistency with XFormers acceleration.
    
    Inherits StreamV2V caching logic and adds IPAdapter functionality.
    """
    
    def __init__(self, 
                 # StreamV2V parameters
                 attention_op: Optional[Callable] = None, name=None, 
                 use_feature_injection=False, feature_injection_strength=0.8, 
                 feature_similarity_threshold=0.98, interval=4, max_frames=4, 
                 use_tome_cache=False, tome_metric="keys", use_grid=False, tome_ratio=0.5,
                 # IPAdapter parameters
                 hidden_size=None, cross_attention_dim=None, scale=1.0, num_tokens=4):
        """
        Initialize combined processor with both StreamV2V and IPAdapter capabilities.
        
        Args:
            All StreamV2V parameters from CachedSTXFormersAttnProcessor
            hidden_size: IPAdapter hidden size
            cross_attention_dim: IPAdapter cross attention dimension  
            scale: IPAdapter conditioning scale
            num_tokens: IPAdapter number of image tokens
        """
        # Initialize StreamV2V functionality
        super().__init__(
            attention_op=attention_op, name=name,
            use_feature_injection=use_feature_injection,
            feature_injection_strength=feature_injection_strength,
            feature_similarity_threshold=feature_similarity_threshold,
            interval=interval, max_frames=max_frames,
            use_tome_cache=use_tome_cache, tome_metric=tome_metric,
            use_grid=use_grid, tome_ratio=tome_ratio
        )
        
        # Initialize IPAdapter functionality
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        
        # IPAdapter layers (will be set by IPAdapter when loading weights)
        self.to_k_ip = None
        self.to_v_ip = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
        """
        Combined forward pass with both IPAdapter and StreamV2V processing.
        """
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

        # Use PEFT scale parameter 
        args = () if hasattr(attn, 'to_q') and hasattr(attn.to_q, 'base_layer') else (scale,)
        query = attn.to_q(hidden_states, *args)

        # Handle IPAdapter image conditioning if encoder_hidden_states provided
        ip_hidden_states = None
        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        else:
            # IPAdapter: Split text and image embeddings
            if self.to_k_ip is not None and self.to_v_ip is not None:
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :], 
                    encoder_hidden_states[:, end_pos:, :]
                )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        # StreamV2V: Apply temporal caching for self-attention
        if is_selfattn:
            cached_key = key.clone()
            cached_value = value.clone()

            if len(self.cached_key) > 0:
                key = torch.cat([key] + list(self.cached_key), dim=1)
                value = torch.cat([value] + list(self.cached_value), dim=1)

        # Prepare tensors for attention computation
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # Main attention computation with XFormers
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # IPAdapter: Add image conditioning if available
        if ip_hidden_states is not None and self.to_k_ip is not None and self.to_v_ip is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            
            ip_key = attn.head_to_batch_dim(ip_key).contiguous()
            ip_value = attn.head_to_batch_dim(ip_value).contiguous()
            
            ip_hidden_states = xformers.ops.memory_efficient_attention(
                query, ip_key, ip_value, attn_bias=None, op=self.attention_op, scale=attn.scale
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
            
            # Combine with IPAdapter scale
            hidden_states = hidden_states + self.scale * ip_hidden_states

        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # StreamV2V: Apply temporal processing for self-attention
        if is_selfattn:
            cached_output = hidden_states.clone()
            
            # Feature injection for temporal consistency
            if self.use_feature_injection and ("up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name):
                if len(self.cached_output) > 0:
                    from .utils import get_nn_feats
                    nn_hidden_states = get_nn_feats(hidden_states, self.cached_output, threshold=self.threshold)
                    hidden_states = hidden_states * (1-self.fi_strength) + self.fi_strength * nn_hidden_states
            
            # Cache management
            if self.frame_id % self.interval == 0:
                if self.use_tome_cache:
                    self._tome_step_kvout(cached_key, cached_value, cached_output)
                else:
                    self.cached_key.append(cached_key)
                    self.cached_value.append(cached_value)
                    self.cached_output.append(cached_output)
        
        self.frame_id += 1
        return hidden_states


class CombinedIPAdapterStreamV2VAttnProcessor2_0(CachedSTAttnProcessor2_0):
    """
    Combined attention processor for PyTorch 2.0 scaled_dot_product_attention
    that handles both IPAdapter image conditioning and StreamV2V temporal consistency.
    """
    
    def __init__(self, 
                 # StreamV2V parameters
                 name=None, use_feature_injection=False, feature_injection_strength=0.8, 
                 feature_similarity_threshold=0.98, interval=4, max_frames=1, 
                 use_tome_cache=False, tome_metric="keys", use_grid=False, tome_ratio=0.5,
                 # IPAdapter parameters
                 hidden_size=None, cross_attention_dim=None, scale=1.0, num_tokens=4):
        """Initialize combined processor with both StreamV2V and IPAdapter capabilities."""
        # Initialize StreamV2V functionality
        super().__init__(
            name=name, use_feature_injection=use_feature_injection,
            feature_injection_strength=feature_injection_strength,
            feature_similarity_threshold=feature_similarity_threshold,
            interval=interval, max_frames=max_frames,
            use_tome_cache=use_tome_cache, tome_metric=tome_metric,
            use_grid=use_grid, tome_ratio=tome_ratio
        )
        
        # Initialize IPAdapter functionality
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
            
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        
        # IPAdapter layers (will be set by IPAdapter when loading weights)
        self.to_k_ip = None
        self.to_v_ip = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale: float = 1.0):
        """Combined forward pass with both IPAdapter and StreamV2V processing."""
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

        # Use PEFT scale parameter
        args = () if hasattr(attn, 'to_q') and hasattr(attn.to_q, 'base_layer') else (scale,)
        query = attn.to_q(hidden_states, *args)

        # Handle IPAdapter image conditioning if encoder_hidden_states provided
        ip_hidden_states = None
        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        else:
            # IPAdapter: Split text and image embeddings
            if self.to_k_ip is not None and self.to_v_ip is not None:
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :], 
                    encoder_hidden_states[:, end_pos:, :]
                )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        # StreamV2V: Apply temporal caching for self-attention
        if is_selfattn:
            cached_key = key.clone()
            cached_value = value.clone()
            
            if torch.equal(self.frame_id, self.zero_tensor):
                self.cached_key = cached_key
                self.cached_value = cached_value

            key = torch.cat([key, self.cached_key], dim=1)
            value = torch.cat([value, self.cached_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Main attention computation
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # IPAdapter: Add image conditioning if available
        if ip_hidden_states is not None and self.to_k_ip is not None and self.to_v_ip is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
            
            # Combine with IPAdapter scale
            hidden_states = hidden_states + self.scale * ip_hidden_states

        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # StreamV2V: Apply temporal processing for self-attention
        if is_selfattn:
            cached_output = hidden_states.clone()

            if torch.equal(self.frame_id, self.zero_tensor):
                self.cached_output = cached_output

            # Feature injection for temporal consistency
            if self.use_feature_injection and ("up_blocks.0" in self.name or "up_blocks.1" in self.name or 'mid_block' in self.name):
                from .utils import get_nn_feats
                nn_hidden_states = get_nn_feats(hidden_states, self.cached_output, threshold=self.threshold)
                hidden_states = hidden_states * (1-self.fi_strength) + self.fi_strength * nn_hidden_states

        # Cache management
        mod_result = torch.remainder(self.frame_id, self.interval)
        if torch.equal(mod_result, self.zero_tensor) and is_selfattn:
                self._tome_step_kvout(cached_key, cached_value, cached_output)
        
        self.frame_id = self.frame_id + 1
        return hidden_states 