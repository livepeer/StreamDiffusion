import sys
import os

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

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
pipeline using configuration system.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

#TODO update naming convention to reflect the controlnet agnostic nature of the config system (pipeline_config instead of controlnet_config for example)
    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        # Load configuration if provided
        self.config = None
        self.use_config = False
        self.pipeline_mode = "img2img"  # default mode

        if args.controlnet_config:
            try:
                self.config = load_config(args.controlnet_config)
                self.use_config = True
                print("__init__: Using configuration file mode")
                
                # Check mode from config
                self.pipeline_mode = self.config.get('mode', 'img2img')
                print(f"__init__: Pipeline mode set to {self.pipeline_mode}")
                
            except Exception as e:
                print(f"__init__: Failed to load config file {args.controlnet_config}: {e}")
                print("__init__: Falling back to standard mode")
                self.use_config = False

        # Update input_mode based on pipeline mode
        if self.pipeline_mode == "txt2img":
            self.Info.input_mode = "text"
        else:
            self.Info.input_mode = "image"

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

            # Update params with config values
            params.width = self.config.get('width', 512)
            params.height = self.config.get('height', 512)

            # Create wrapper using config system
            self.stream = create_wrapper_from_config(self.config, **overrides)

            # Store config values for later use
            self.prompt = self.config.get('prompt', default_prompt)
            self.negative_prompt = self.config.get('negative_prompt', default_negative_prompt)
            self.guidance_scale = self.config.get('guidance_scale', 1.2)
            self.num_inference_steps = self.config.get('num_inference_steps', 50)

        else:
            # Create StreamDiffusionWrapper without config (original behavior)
            print("__init__: Using standard mode (no config)")
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
                output_type="pil",
                warmup=10,
                vae_id=None,
                acceleration=args.acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
                engine_dir=args.engine_dir,
            )

            # Store default values for later use
            self.prompt = default_prompt
            self.negative_prompt = default_negative_prompt
            self.guidance_scale = 1.2
            self.num_inference_steps = 50

            # Prepare pipeline with default prompts
            self.stream.prepare(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )

        self.last_prompt = self.prompt

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Update prompt if it has changed
        if hasattr(params, 'prompt') and params.prompt != self.last_prompt:
            self.last_prompt = params.prompt
            # Update the pipeline with new prompt
            self.stream.prepare(
                prompt=params.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )

        # Handle different modes
        if self.pipeline_mode == "txt2img":
            # Text-to-image mode
            if self.use_config and self.config and 'controlnets' in self.config:
                # txt2img with ControlNets: need image for control
                self.stream.update_control_image_efficient(params.image)
                output_image = self.stream(params.image)
            else:
                # Pure txt2img: no image needed
                output_image = self.stream()
        else:
            # Image-to-image mode: use original logic
            if self.use_config and self.config and 'controlnets' in self.config:
                # ControlNet mode: update control image and use PIL image
                self.stream.update_control_image_efficient(params.image)
                output_image = self.stream(params.image)
            else:
                # Standard mode: use original logic with preprocessed tensor
                image_tensor = self.stream.preprocess_image(params.image)
                output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
