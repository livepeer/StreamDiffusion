import sys
import os
import logging

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from streamdiffusion import StreamDiffusionWrapper
# Import the config system functions
from streamdiffusion import load_config, create_wrapper_from_config

import torch
import yaml
from pathlib import Path

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold"><a href="https://github.com/livepeer/StreamDiffusion" target="_blank" class="text-blue-500 underline hover:no-underline">StreamDiffusion</a></h1>
<div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4">
    <div class="flex">
        <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
        </div>
        <div class="ml-3">
            <div class="text-sm text-yellow-700">
                <p><strong>Development Tool Notice:</strong> This is an internal development tool. It may change frequently and contain bugs. It is not supported.</p>
                <p>For production-level real-time research tools, use <a href="https://github.com/livepeer/stream-model-lab" target="_blank" class="text-blue-600 underline hover:no-underline">Livepeer Stream Model Lab</a></p>
            </div>
        </div>
    </div>
</div>

"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        resolution: str = Field(
            "512x512 (1:1)",
            title="Resolution",
            field="select",
            id="resolution",
            values=[
                # --- Square (1:1) ---
                "384x384 (1:1)",
                "512x512 (1:1)",
                "640x640 (1:1)",
                "704x704 (1:1)",
                "768x768 (1:1)",
                "896x896 (1:1)",
                "1024x1024 (1:1)",
                # --- Portrait ---
                "384x512 (3:4)",
                "512x768 (2:3)",
                "640x896 (5:7)",
                "768x1024 (3:4)",
                "576x1024 (9:16)",
                # --- Landscape ---
                "512x384 (4:3)",
                "768x512 (3:2)",
                "896x640 (7:5)",
                "1024x768 (4:3)",
                "1024x576 (16:9)"
            ]
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

#TODO update naming convention to reflect the controlnet agnostic nature of the config system (pipeline_config instead of controlnet_config for example)
    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, width: int = 512, height: int = 512):
        # Load configuration if provided
        self.config = None
        self.use_config = False
        self.pipeline_mode = "img2img"  # default mode
        self.has_controlnet = False
        self.has_ipadapter = False

        if args.controlnet_config:
            try:
                self.config = load_config(args.controlnet_config)
                self.use_config = True
                # Check mode from config
                self.pipeline_mode = self.config.get('mode', 'img2img')
                
                # Check what features are enabled
                self.has_controlnet = 'controlnets' in self.config and len(self.config['controlnets']) > 0
                self.has_ipadapter = 'ipadapters' in self.config and len(self.config['ipadapters']) > 0
                

                
            except Exception as e:
                print(f"Failed to load config file {args.controlnet_config}: {e}")
                self.use_config = False

        # Update input_mode based on pipeline mode
        self.info = self.Info()
        if self.pipeline_mode == "txt2img":
            self.info.input_mode = "text"
        else:
            self.info.input_mode = "image"

        params = self.InputParams()

        if self.use_config:
            # Use config-based pipeline creation
            # Set up runtime overrides for args that might differ from config
            overrides = {
                'device': device,
                'dtype': torch_dtype,
                'acceleration': args.acceleration,
                'use_safety_checker': args.safety_checker,
            }

            # Determine engine_dir: use config value if available, otherwise use args
            engine_dir = args.engine_dir  # Default to command-line/environment value
            if 'engine_dir' in self.config:
                engine_dir = self.config['engine_dir']
            if engine_dir:
                overrides['engine_dir'] = engine_dir

            # Override taesd if provided via args and not in config
            if args.taesd and 'use_tiny_vae' not in self.config:
                overrides['use_tiny_vae'] = args.taesd

            # Use passed width/height, falling back to config values, then defaults
            params.width = width if width != 512 else self.config.get('width', 512)
            params.height = height if height != 512 else self.config.get('height', 512)
            
            # Override width/height in config for pipeline creation
            overrides['width'] = params.width
            overrides['height'] = params.height

            # Create wrapper using config system
            self.stream = create_wrapper_from_config(self.config, **overrides)

            # Store config values for later use (excluding prompt which is handled via blending)
            self.negative_prompt = self.config.get('negative_prompt', default_negative_prompt)
            self.guidance_scale = self.config.get('guidance_scale', 1.2)
            self.num_inference_steps = self.config.get('num_inference_steps', 50)

        else:
            # Create StreamDiffusionWrapper without config (original behavior)
            # Use passed width/height parameters
            params.width = width
            params.height = height
            
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                use_tiny_vae=args.taesd,
                device=device,
                dtype=torch_dtype,
                t_index_list=[35, 45],
                frame_buffer_size=1,
                width=params.width,
                height=params.height,
                use_lcm_lora=False,
                output_type="pt",
                warmup=10,
                vae_id=None,
                acceleration=args.acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
                engine_dir=args.engine_dir,
            )

            # Store default values for later use (excluding prompt which is handled via blending)
            self.negative_prompt = default_negative_prompt
            self.guidance_scale = 1.2
            self.num_inference_steps = 50

            # Initial preparation without prompt (will be set via blending interface)
            self.stream.prepare(
                prompt=default_prompt,  # Temporary initial prompt 
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )

        # Initialize pipeline parameters
        self.seed = 2
        self.guidance_scale = 1.1
        self.num_inference_steps = 50
        self.negative_prompt = default_negative_prompt
        
        # Store output type for frame conversion - always force "pt" for optimal performance
        self.output_type = "pt"

        # Model and acceleration setup

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Handle different modes
        if self.pipeline_mode == "txt2img":
            # Text-to-image mode
            if self.has_controlnet:
                # txt2img with ControlNets: push control image via direct method
                try:
                    stream_state = self.stream.get_stream_state()
                    current_cfg = stream_state.get('controlnet_config', [])
                except Exception:
                    current_cfg = []
                if current_cfg:
                    # Update control image for all configured ControlNets using direct method
                    for i in range(len(current_cfg)):
                        self.stream.update_control_image(index=i, image=params.image)
                output_image = self.stream(params.image)
            elif self.has_ipadapter:
                # txt2img with IPAdapter: no input image needed (style image handled separately)
                output_image = self.stream()
            else:
                # Pure txt2img: no image needed
                output_image = self.stream()
        else:
            # Image-to-image mode: use original logic
            if self.has_controlnet:
                # ControlNet mode: push control image via direct method and use PIL image
                try:
                    stream_state = self.stream.get_stream_state()
                    current_cfg = stream_state.get('controlnet_config', [])
                except Exception:
                    current_cfg = []
                if current_cfg:
                    # Update control image for all configured ControlNets using direct method
                    for i in range(len(current_cfg)):
                        self.stream.update_control_image(index=i, image=params.image)
                output_image = self.stream(params.image)
            elif self.has_ipadapter:
                # IPAdapter mode: use PIL image for img2img
                output_image = self.stream(params.image)
            else:
                # Standard mode: handle tensor inputs (always from bytes_to_pt)
                if isinstance(params.image, torch.Tensor):
                    # Direct tensor input - already preprocessed
                    output_image = self.stream(image=params.image)
                else:
                    # Fallback for PIL input - needs preprocessing
                    image_tensor = self.stream.preprocess_image(params.image)
                    output_image = self.stream(image=image_tensor)

        return output_image

    def update_ipadapter_config(self, scale: float = None, style_image: Image.Image = None) -> bool:
        """
        Update IPAdapter configuration in real-time using direct methods
        
        Args:
            scale: New IPAdapter scale value (optional)
            style_image: New style image (PIL Image, optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.has_ipadapter:
            return False
            
        if scale is None and style_image is None:
            return False  # Nothing to update
            
        try:
            # Update scale via unified config system (no direct method needed)
            if scale is not None:
                self.stream.update_stream_params(ipadapter_config={'scale': scale})
            
            # Update style image via direct method
            if style_image is not None:
                self.stream.update_style_image(style_image)
                
            return True
        except Exception as e:
            return False

    def update_ipadapter_scale(self, scale: float) -> bool:
        """Update IPAdapter scale - convenience method"""
        return self.update_ipadapter_config(scale=scale)

    def update_ipadapter_style_image(self, style_image: Image.Image) -> bool:
        """Update IPAdapter style image - convenience method"""
        return self.update_ipadapter_config(style_image=style_image)

    def update_ipadapter_weight_type(self, weight_type: str) -> bool:
        """Update IPAdapter weight type in real-time"""
        if not self.has_ipadapter:
            return False
            
        try:
            # Use unified updater on wrapper
            if hasattr(self.stream, 'update_stream_params'):
                self.stream.update_stream_params(ipadapter_config={ 'weight_type': weight_type })
                return True
            # Should not reach here in normal operation
            return False
        except Exception as e:
            return False

    def get_ipadapter_info(self) -> dict:
        """
        Get current IPAdapter information
        
        Returns:
            dict: IPAdapter information including scale, model info, etc.
        """
        info = {
            "enabled": self.has_ipadapter,
            "scale": 1.0,
            "weight_type": "linear",
            "model_path": None,
            "style_image_set": False
        }
        
        if self.has_ipadapter and self.config and 'ipadapters' in self.config:
            # Get info from first IPAdapter config
            if len(self.config['ipadapters']) > 0:
                ipadapter_config = self.config['ipadapters'][0]
                info["scale"] = ipadapter_config.get('scale', 1.0)
                info["weight_type"] = ipadapter_config.get('weight_type', 'linear')
                info["model_path"] = ipadapter_config.get('ipadapter_model_path')
                info["style_image_set"] = 'style_image' in ipadapter_config
                
        # Get current runtime state from wrapper's public API
        try:
            if hasattr(self.stream, 'get_stream_state'):
                stream_state = self.stream.get_stream_state()
                ipadapter_runtime_config = stream_state.get('ipadapter_config', {})
                if ipadapter_runtime_config:
                    info["scale"] = ipadapter_runtime_config.get('scale', info.get("scale", 1.0))
                    info["weight_type"] = ipadapter_runtime_config.get('weight_type', info.get("weight_type", 'linear'))
        except Exception:
            pass  # Use defaults from config if wrapper method fails
            
        return info

    def update_stream_params(self, **kwargs):
        """
        Update streaming parameters using the consolidated API
        
        Args:
            **kwargs: All parameters supported by StreamDiffusionWrapper.update_stream_params()
                     including controlnet_config, guidance_scale, delta, etc.
        """
        return self.stream.update_stream_params(**kwargs)
