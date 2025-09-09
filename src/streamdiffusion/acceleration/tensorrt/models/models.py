#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/models.py

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            print(f"⚠️ Model size ({onnx_graph.ByteSize() / (1024**3):.2f} GB) exceeds 2GB - this is normal for SDXL models")
            print("🔧 ONNX shape inference will be skipped for large models to avoid memory issues")
            # For large models like SDXL, skip shape inference to avoid memory/size issues
            # The model will still work with TensorRT's own shape inference during engine building
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(
        self,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=4,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        # Make batch size check more flexible for ONNX export
        if hasattr(self, '_allow_export_batch_override') and self._allow_export_batch_override:
            # During ONNX export, allow different batch sizes
            effective_min_batch = min(self.min_batch, batch_size)
            effective_max_batch = max(self.max_batch, batch_size)
        else:
            effective_min_batch = self.min_batch
            effective_max_batch = self.max_batch
            
        assert batch_size >= effective_min_batch and batch_size <= effective_max_batch, \
            f"Batch size {batch_size} not in range [{effective_min_batch}, {effective_max_batch}]"
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        # Following ComfyUI TensorRT approach: ensure proper min ≤ opt ≤ max constraints
        # Even with static_batch=True, we need different min/max to avoid TensorRT constraint violations
        
        if static_batch:
            # For static batch, still provide range to avoid min=opt=max constraint violation
            min_batch = max(1, batch_size - 1)  # At least 1, but allow some range
            max_batch = batch_size
        else:
            min_batch = self.min_batch
            max_batch = self.max_batch
        
        latent_height = image_height // 8
        latent_width = image_width // 8
        
        # Force dynamic shapes for height/width to enable runtime resolution changes
        # Always use 384-1024 range regardless of static_shape flag
        min_image_height = self.min_image_shape
        max_image_height = self.max_image_shape
        min_image_width = self.min_image_shape
        max_image_width = self.max_image_shape
        min_latent_height = self.min_latent_shape
        max_latent_height = self.max_latent_shape
        min_latent_width = self.min_latent_shape
        max_latent_width = self.max_latent_shape
        
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.select_outputs([0])
        opt.cleanup()
        opt.info(self.name + ": remove output[1]")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0], names=["text_embeddings"])
        opt.info(self.name + ": remove output[0]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class SafetyChecker(BaseModel):
    def __init__(self, device, max_batch_size = 1, min_batch_size = 1):
        super(SafetyChecker, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        self.name = "safety_checker"

    def get_input_names(self):
        return ["clip_input"]

    def get_output_names(self):
        return ["has_nsfw_concepts"]

    def get_dynamic_axes(self):
        return {"clip_input": {0: "B"}}

    def get_input_profile(self, batch_size, *args, **kwargs):
        return {
            "clip_input": [
                (self.min_batch, 3, 224, 224),
                (batch_size, 3, 224, 224),
                (self.max_batch, 3, 224, 224),
            ],
        }

    def get_shape_dict(self, batch_size, *args, **kwargs):
        return {
            "clip_input": (batch_size, 3, 224, 224),
            "has_nsfw_concepts": (batch_size,),
        }

    def get_sample_input(self, batch_size, *args, **kwargs):
        return (
            torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=self.device),
        )

class NSFWDetector(BaseModel):
    def __init__(self, device, max_batch_size = 1, min_batch_size = 1):
        super(NSFWDetector, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        self.name = "nsfw_detector"
    
    def get_input_names(self):
        return ["pixel_values"]
    
    def get_output_names(self):
        return ["logits"]
    
    def get_dynamic_axes(self):
        return {"pixel_values": {0: "B"}}
    
    def get_input_profile(self, batch_size, *args, **kwargs):
        return {
            "pixel_values": [
                (self.min_batch, 3, 448, 448),
                (batch_size, 3, 448, 448),
                (self.max_batch, 3, 448, 448),
            ],
        }
    
    def get_shape_dict(self, batch_size, *args, **kwargs):
        return {
            "pixel_values": (batch_size, 3, 448, 448),
            "logits": (batch_size, 2),
        }
    
    def get_sample_input(self, batch_size, *args, **kwargs):
        return (
            torch.randn(batch_size, 3, 448, 448, dtype=torch.float16, device=self.device),
        )

class UNet(BaseModel):
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch_size=4,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        use_control=False,
        unet_arch=None,
        image_height=512,
        image_width=512,
        use_ipadapter=False,
        num_image_tokens=4,
        num_ip_layers: int = None,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.image_height = image_height
        self.image_width = image_width
        
        self.use_control = use_control
        self.unet_arch = unet_arch or {}
        self.use_ipadapter = use_ipadapter
        self.num_image_tokens = num_image_tokens
        self.num_ip_layers = num_ip_layers
        
        # Baked-in IPAdapter configuration
        if self.use_ipadapter:
            # With baked-in processors, we extend text_maxlen to include image tokens
            # TODO: Consider making this dynamic instead of fixed per IPAdapter variant
            # Could use dynamic shapes: min=77 (text only), max=93 (text + 16 tokens)
            # This would allow a single engine to handle all IPAdapter types instead of separate engines
            self.text_maxlen = text_maxlen + self.num_image_tokens
            if self.num_ip_layers is None:
                raise ValueError("UNet model requires num_ip_layers when use_ipadapter=True")

        
        if self.use_control and self.unet_arch:
            self.control_inputs = self.get_control(image_height, image_width)
            self._add_control_inputs()
        else:
            self.control_inputs = {}

    def get_control(self, image_height: int = 512, image_width: int = 512) -> dict:
        """Generate ControlNet input configurations with dynamic spatial dimensions based on input resolution."""
        block_out_channels = self.unet_arch.get('block_out_channels', (320, 640, 1280, 1280))
        
        # Calculate latent space dimensions
        latent_height = image_height // 8
        latent_width = image_width // 8
        
        control_inputs = {}
        
        if len(block_out_channels) == 3:
            # SDXL architecture: Match UNet's exact down_block_res_samples structure
            # UNet down_block_res_samples = [initial_sample] + [block0_residuals] + [block1_residuals] + [block2_residuals]
            # Pattern: [88x88] + [88x88, 88x88, 44x44] + [44x44, 44x44, 22x22] + [22x22, 22x22]
            # Total: 9 control tensors needed
            control_tensors = [
                # Initial sample (after conv_in: 4->320 channels, no downsampling)
                (block_out_channels[0], 1),  # 320 channels, 88x88
                
                # Block 0 residuals (320 channels)
                (block_out_channels[0], 1),  # 320 channels, 88x88 
                (block_out_channels[0], 1),  # 320 channels, 88x88
                (block_out_channels[0], 2),  # 320 channels, 44x44 (downsampled)
                
                # Block 1 residuals (640 channels) 
                (block_out_channels[1], 2),  # 640 channels, 44x44
                (block_out_channels[1], 2),  # 640 channels, 44x44
                (block_out_channels[1], 4),  # 640 channels, 22x22 (downsampled)
                
                # Block 2 residuals (1280 channels)
                (block_out_channels[2], 4),  # 1280 channels, 22x22
                (block_out_channels[2], 4),  # 1280 channels, 22x22
            ]
        else:
            # SD1.5/SD2.1 architecture: 4 down blocks with 12 control tensors
            control_tensors = [
                # Block 0: No downsampling from latent space (factor = 1)
                (320, 1), (320, 1), (320, 1),
                # Block 1: 2x downsampling from latent space (factor = 2) 
                (320, 2), (640, 2), (640, 2),
                # Block 2: 4x downsampling from latent space (factor = 4)
                (640, 4), (1280, 4), (1280, 4),
                # Block 3: 8x downsampling from latent space (factor = 8)
                (1280, 8), (1280, 8), (1280, 8)
            ]
        
        # Generate control inputs with proper spatial dimensions
        for i, (channels, downsample_factor) in enumerate(control_tensors):
            input_name = f"input_control_{i:02d}"
            
            # Calculate spatial dimensions for this level
            control_height = max(1, latent_height // downsample_factor)
            control_width = max(1, latent_width // downsample_factor)
            
            control_inputs[input_name] = {
                'batch': self.min_batch,
                'channels': channels,
                'height': control_height,
                'width': control_width,
                'downsampling_factor': downsample_factor
            }
        
        # Middle block uses the most downsampled resolution based on architecture
        if len(block_out_channels) == 3:
            # SDXL: middle block at 4x downsampling (after 3 down blocks)
            middle_downsample_factor = 4
        else:
            # SD1.5: middle block at 8x downsampling (after 4 down blocks)
            middle_downsample_factor = 8
            
        control_inputs["input_control_middle"] = {
            'batch': self.min_batch,
            'channels': 1280,
            'height': max(1, latent_height // middle_downsample_factor),
            'width': max(1, latent_width // middle_downsample_factor),
            'downsampling_factor': middle_downsample_factor
        }
        
        return control_inputs

    def _add_control_inputs(self):
        """Add ControlNet inputs to the model's input/output specifications"""
        if not self.control_inputs:
            return
        
        self._original_get_input_names = self.get_input_names
        self._original_get_dynamic_axes = self.get_dynamic_axes
        self._original_get_input_profile = self.get_input_profile
        self._original_get_shape_dict = self.get_shape_dict
        self._original_get_sample_input = self.get_sample_input

    def get_input_names(self):
        """Get input names including ControlNet inputs"""
        base_names = ["sample", "timestep", "encoder_hidden_states"]
        if self.use_ipadapter:
            base_names.append("ipadapter_scale")
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: get_input_names with ipadapter -> {base_names}")
            except Exception:
                pass
        if self.use_control and self.control_inputs:
            control_names = sorted(self.control_inputs.keys())
            return base_names + control_names
        return base_names

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        base_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }
        if self.use_ipadapter:
            base_axes["ipadapter_scale"] = {0: "L_ip"}
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: dynamic axes include ipadapter_scale with L_ip={getattr(self, 'num_ip_layers', None)}")
            except Exception:
                pass
        
        if self.use_control and self.control_inputs:
            for name, shape_spec in self.control_inputs.items():
                height = shape_spec["height"]
                width = shape_spec["width"]
                spatial_suffix = f"{height}x{width}"
                base_axes[name] = {
                    0: "2B", 
                    2: f"H_{spatial_suffix}", 
                    3: f"W_{spatial_suffix}"
                }
        
        return base_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        
        # Following TensorRT documentation: ensure proper min ≤ opt ≤ max constraints for ALL dimensions
        # Calculate optimal latent dimensions that fall within min/max range
        opt_latent_height = min(max(latent_height, min_latent_height), max_latent_height)
        opt_latent_width = min(max(latent_width, min_latent_width), max_latent_width)
        
        # Ensure no dimension equality that causes constraint violations
        if opt_latent_height == min_latent_height and min_latent_height < max_latent_height:
            opt_latent_height = min(min_latent_height + 8, max_latent_height)  # Add 8 pixels for separation
        if opt_latent_width == min_latent_width and min_latent_width < max_latent_width:
            opt_latent_width = min(min_latent_width + 8, max_latent_width)
        
        # Image dimensions for ControlNet inputs
        min_image_h, max_image_h = self.min_image_shape, self.max_image_shape
        min_image_w, max_image_w = self.min_image_shape, self.max_image_shape
        opt_image_height = min(max(image_height, min_image_h), max_image_h)
        opt_image_width = min(max(image_width, min_image_w), max_image_w)
        
        # Ensure image dimension separation as well
        if opt_image_height == min_image_h and min_image_h < max_image_h:
            opt_image_height = min(min_image_h + 64, max_image_h)  # Add 64 pixels for separation
        if opt_image_width == min_image_w and min_image_w < max_image_w:
            opt_image_width = min(min_image_w + 64, max_image_w)
        
        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, opt_latent_height, opt_latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }
        if self.use_ipadapter:
            # scalar per-layer vector, length fixed to num_ip_layers
            profile["ipadapter_scale"] = [
                (1,),
                (self.num_ip_layers,),
                (self.num_ip_layers,),
            ]
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: profile ipadapter_scale min/opt/max={(1,),(self.num_ip_layers,),(self.num_ip_layers,)}")
            except Exception:
                pass
        
        if self.use_control and self.control_inputs:
            # Use the actual calculated spatial dimensions for each ControlNet input
            # Each control input has its own specific spatial resolution based on UNet architecture
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                
                # Create optimization profile with proper spatial dimension scaling
                # Scale the spatial dimensions proportionally with the main latent dimensions
                scale_h = opt_latent_height / latent_height if latent_height > 0 else 1.0
                scale_w = opt_latent_width / latent_width if latent_width > 0 else 1.0
                
                min_control_h = max(1, int(control_height * min_latent_height / latent_height))
                max_control_h = max(min_control_h + 1, int(control_height * max_latent_height / latent_height))
                opt_control_h = max(min_control_h, min(int(control_height * scale_h), max_control_h))
                
                min_control_w = max(1, int(control_width * min_latent_width / latent_width))
                max_control_w = max(min_control_w + 1, int(control_width * max_latent_width / latent_width))
                opt_control_w = max(min_control_w, min(int(control_width * scale_w), max_control_w))
                
                profile[name] = [
                    (min_batch, channels, min_control_h, min_control_w),    # min
                    (batch_size, channels, opt_control_h, opt_control_w),   # opt  
                    (max_batch, channels, max_control_h, max_control_w),    # max
                ]
        
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }
        if self.use_ipadapter:
            shape_dict["ipadapter_scale"] = (self.num_ip_layers,)
            try:
                import logging
                logging.getLogger(__name__).debug(f"TRT Models: shape_dict ipadapter_scale={(self.num_ip_layers,)}")
            except Exception:
                pass
        
        if self.use_control and self.control_inputs:
            # Use the actual calculated spatial dimensions for each ControlNet input
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                shape_dict[name] = (2 * batch_size, channels, control_height, control_width)
        
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        # Enable flexible batch size checking for ONNX export
        self._allow_export_batch_override = True
        
        try:
            latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        finally:
            # Clean up the override flag
            if hasattr(self, '_allow_export_batch_override'):
                delattr(self, '_allow_export_batch_override')
        
        dtype = torch.float16 if self.fp16 else torch.float32
        
        # Use smaller batch size for memory efficiency during ONNX export
        export_batch_size = min(batch_size, 1)  # Use batch size 1 for ONNX export to save memory
        
        base_inputs = [
            torch.randn(
                2 * export_batch_size, self.unet_dim, latent_height, latent_width, 
                dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * export_batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * export_batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        ]
        
        
        
        if self.use_control and self.control_inputs:
            control_inputs = []
            
            # Use the ACTUAL calculated spatial dimensions for each control input
            # This ensures each control input matches its expected UNet feature map resolution
            
            for name in sorted(self.control_inputs.keys()):
                shape_spec = self.control_inputs[name]
                channels = shape_spec["channels"]
                
                # KEY FIX: Use the specific spatial dimensions calculated for this control input
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                
                control_input = torch.randn(
                    2 * export_batch_size, channels, control_height, control_width, 
                    dtype=dtype, device=self.device
                )
                control_inputs.append(control_input)
                
                # Clear cache periodically to prevent memory buildup
                if len(control_inputs) % 4 == 0:
                    torch.cuda.empty_cache()
            
            # Append ipadapter_scale if needed
            if self.use_ipadapter:
                base_inputs.append(torch.ones(self.num_ip_layers, dtype=torch.float32, device=self.device))
            return tuple(base_inputs + control_inputs)
        
        if self.use_ipadapter:
            base_inputs.append(torch.ones(self.num_ip_layers, dtype=torch.float32, device=self.device))
        return tuple(base_inputs)


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32,
            device=self.device,
        )
