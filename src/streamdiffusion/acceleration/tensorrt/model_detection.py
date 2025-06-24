"""Model detection and architecture extraction for TensorRT ControlNet support"""

from typing import Dict, Tuple, Optional
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


def detect_model_from_diffusers_unet(unet: UNet2DConditionModel) -> str:
    """Detect model type from diffusers UNet configuration"""
    in_channels = unet.config.in_channels
    block_out_channels = tuple(unet.config.block_out_channels)
    cross_attention_dim = unet.config.cross_attention_dim
    attention_head_dim = getattr(unet.config, 'attention_head_dim', None)
    
    if (cross_attention_dim == 768 and 
        block_out_channels == (320, 640, 1280, 1280) and
        in_channels == 4):
        return "SD15"
    
    elif (cross_attention_dim == 2048 and 
          block_out_channels == (320, 640, 1280) and
          in_channels == 4):
        return "SDXL"
    
    elif (cross_attention_dim == 1024 and 
          block_out_channels == (320, 640, 1280, 1280) and
          in_channels == 4):
        return "SD21"
    
    elif cross_attention_dim == 768 and in_channels == 4:
        return "SD15"
    elif cross_attention_dim == 2048 and in_channels == 4:
        return "SDXL"
    elif cross_attention_dim == 1024 and in_channels == 4:
        return "SD21"
    
    if cross_attention_dim == 768:
        print(f"Unknown SD1.5-like model with channels {block_out_channels}, defaulting to SD15")
        return "SD15"
    elif cross_attention_dim == 2048:
        print(f"Unknown SDXL-like model with channels {block_out_channels}, defaulting to SDXL")
        return "SDXL"
    elif cross_attention_dim == 1024:
        print(f"Unknown SD2.1-like model with channels {block_out_channels}, defaulting to SD21")
        return "SD21"
    else:
        raise ValueError(
            f"Unknown model architecture: "
            f"cross_attention_dim={cross_attention_dim}, "
            f"block_out_channels={block_out_channels}, "
            f"in_channels={in_channels}"
        )


def extract_unet_architecture(unet: UNet2DConditionModel) -> Dict:
    """Extract UNet architecture details needed for ControlNet input generation"""
    config = unet.config
    
    model_channels = config.block_out_channels[0]
    block_out_channels = tuple(config.block_out_channels)
    channel_mult = tuple(ch // model_channels for ch in block_out_channels)
    
    if hasattr(config, 'layers_per_block'):
        if isinstance(config.layers_per_block, (list, tuple)):
            num_res_blocks = tuple(config.layers_per_block)
        else:
            num_res_blocks = tuple([config.layers_per_block] * len(block_out_channels))
    else:
        num_res_blocks = tuple([2] * len(block_out_channels))
    
    context_dim = config.cross_attention_dim
    in_channels = config.in_channels
    
    attention_head_dim = getattr(config, 'attention_head_dim', 8)
    if isinstance(attention_head_dim, (list, tuple)):
        attention_head_dim = attention_head_dim[0]
    
    transformer_depth = getattr(config, 'transformer_layers_per_block', 1)
    if isinstance(transformer_depth, (list, tuple)):
        transformer_depth = tuple(transformer_depth)
    else:
        transformer_depth = tuple([transformer_depth] * len(block_out_channels))
    
    time_embed_dim = getattr(config, 'time_embedding_dim', None)
    if time_embed_dim is None:
        time_embed_dim = model_channels * 4

    actual_down_block_types = getattr(config, 'down_block_types', [])
    if actual_down_block_types:
        downsample_blocks = [bt for bt in actual_down_block_types if 'Downsample' in bt or bt == 'DownBlock2D']
    
    architecture_dict = {
        "model_channels": model_channels,
        "in_channels": in_channels,
        "out_channels": getattr(config, 'out_channels', in_channels),
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "context_dim": context_dim,
        "attention_head_dim": attention_head_dim,
        "transformer_depth": transformer_depth,
        "time_embed_dim": time_embed_dim,
        "block_out_channels": block_out_channels,
        
        "use_linear_in_transformer": getattr(config, 'use_linear_in_transformer', False),
        "conv_in_kernel": getattr(config, 'conv_in_kernel', 3),
        "conv_out_kernel": getattr(config, 'conv_out_kernel', 3),
        "resnet_time_scale_shift": getattr(config, 'resnet_time_scale_shift', 'default'),
        "class_embed_type": getattr(config, 'class_embed_type', None),
        "num_class_embeds": getattr(config, 'num_class_embeds', None),
        
        "down_block_types": getattr(config, 'down_block_types', []),
        "up_block_types": getattr(config, 'up_block_types', []),
    }
    
    return architecture_dict


def get_model_architecture_preset(model_type: str) -> Dict:
    """Get architecture preset for known model types"""
    presets = {
        "SD15": {
            "model_channels": 320,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": (2, 2, 2, 2),
            "channel_mult": (1, 2, 4, 4),
            "context_dim": 768,
            "attention_head_dim": 8,
            "transformer_depth": (1, 1, 1, 1),
            "time_embed_dim": 1280,
            "block_out_channels": (320, 640, 1280, 1280),
            "use_linear_in_transformer": False,
        },
        "SDXL": {
            "model_channels": 320,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": (2, 2, 2),
            "channel_mult": (1, 2, 4),
            "context_dim": 2048,
            "attention_head_dim": 64,
            "transformer_depth": (2, 2, 10),
            "time_embed_dim": 1280,
            "block_out_channels": (320, 640, 1280),
            "use_linear_in_transformer": True,
        },
        "SD21": {
            "model_channels": 320,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": (2, 2, 2, 2),
            "channel_mult": (1, 2, 4, 4),
            "context_dim": 1024,
            "attention_head_dim": 64,
            "transformer_depth": (1, 1, 1, 1),
            "time_embed_dim": 1280,
            "block_out_channels": (320, 640, 1280, 1280),
            "use_linear_in_transformer": False,
        }
    }
    
    return presets.get(model_type, presets["SD15"])  # Default to SD15


def validate_architecture(arch_dict: Dict, model_type: str) -> Dict:
    """Validate and fix architecture dictionary"""
    preset = get_model_architecture_preset(model_type)
    
    for key, default_value in preset.items():
        if key not in arch_dict or arch_dict[key] is None:
            arch_dict[key] = default_value
            print(f"Using preset value for {key}: {default_value}")
    
    required_keys = [
        "model_channels", "channel_mult", "num_res_blocks", 
        "context_dim", "in_channels", "block_out_channels"
    ]
    
    for key in required_keys:
        if key not in arch_dict:
            raise ValueError(f"Missing required architecture parameter: {key}")
    
    for key in ["channel_mult", "num_res_blocks", "transformer_depth", "block_out_channels"]:
        if key in arch_dict and not isinstance(arch_dict[key], tuple):
            if isinstance(arch_dict[key], (list, int)):
                if isinstance(arch_dict[key], int):
                    arch_dict[key] = tuple([arch_dict[key]] * len(arch_dict["channel_mult"]))
                else:
                    arch_dict[key] = tuple(arch_dict[key])
            else:
                arch_dict[key] = preset[key]
    
    expected_levels = len(arch_dict["channel_mult"])
    for key in ["num_res_blocks", "transformer_depth"]:
        if key in arch_dict and len(arch_dict[key]) != expected_levels:
            print(f"{key} length mismatch, using preset")
            arch_dict[key] = preset[key]
    
    return arch_dict 