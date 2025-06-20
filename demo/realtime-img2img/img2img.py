import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

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
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


def load_controlnet_config(config_path: str) -> dict:
    """Load ControlNet configuration from YAML file"""
    if not config_path or not Path(config_path).exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        return None


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
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
        # Load ControlNet config if provided
        self.controlnet_config = load_controlnet_config(args.controlnet_config) if args.controlnet_config else None
        self.use_controlnet = self.controlnet_config is not None
        
        params = self.InputParams()
        
        # Determine engine_dir: use config value if available, otherwise use args
        engine_dir = args.engine_dir  # Default to command-line/environment value
        if self.use_controlnet and 'engine_dir' in self.controlnet_config:
            engine_dir = self.controlnet_config['engine_dir']
        
        # Determine model and parameters based on config
        if self.use_controlnet:
            print("__init__: Using ControlNet mode")
            model_id = self.controlnet_config.get('model_id', base_model)
            pipeline_type = self.controlnet_config.get('pipeline_type', 'sd1.5')
            t_index_list = self.controlnet_config.get('t_index_list', [35, 45])
            frame_buffer_size = self.controlnet_config.get('frame_buffer_size', 1)
            cfg_type = self.controlnet_config.get('cfg_type', 'none')
            use_lcm_lora = self.controlnet_config.get('use_lcm_lora', False)
            use_tiny_vae = self.controlnet_config.get('use_tiny_vae', args.taesd)
            
            # Prepare ControlNet configurations
            controlnet_configs = []
            if 'controlnets' in self.controlnet_config and self.controlnet_config['controlnets']:
                for cn_config in self.controlnet_config['controlnets']:
                    controlnet_config = {
                        'model_id': cn_config['model_id'],
                        'preprocessor': cn_config['preprocessor'],
                        'conditioning_scale': cn_config['conditioning_scale'],
                        'enabled': cn_config.get('enabled', True),
                        'preprocessor_params': cn_config.get('preprocessor_params', None),
                        'pipeline_type': pipeline_type,
                        'control_guidance_start': cn_config.get('control_guidance_start', 0.0),
                        'control_guidance_end': cn_config.get('control_guidance_end', 1.0),
                    }
                    controlnet_configs.append(controlnet_config)
            
            # Update width/height from config
            params.width = self.controlnet_config.get('width', 512)
            params.height = self.controlnet_config.get('height', 512)
            
            # Create StreamDiffusionWrapper with ControlNet
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=model_id,
                use_tiny_vae=use_tiny_vae,
                device=device,
                dtype=torch_dtype,
                t_index_list=t_index_list,
                frame_buffer_size=frame_buffer_size,
                width=params.width,
                height=params.height,
                use_lcm_lora=use_lcm_lora,
                output_type="pil",
                warmup=10,
                vae_id=None,
                acceleration=args.acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type=cfg_type,
                use_safety_checker=args.safety_checker,
                engine_dir=engine_dir,
                # ControlNet options
                use_controlnet=True,
                controlnet_config=controlnet_configs,
            )
        else:
            print("__init__: Using standard mode (no ControlNet)")
            # Create StreamDiffusionWrapper without ControlNet (original behavior)
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
                engine_dir=engine_dir,
            )

        # Prepare pipeline with appropriate prompts
        if self.use_controlnet:
            prompt = self.controlnet_config.get('prompt', default_prompt)
            negative_prompt = self.controlnet_config.get('negative_prompt', default_negative_prompt)
            guidance_scale = self.controlnet_config.get('guidance_scale', 1.2)
            num_inference_steps = self.controlnet_config.get('num_inference_steps', 50)
        else:
            prompt = default_prompt
            negative_prompt = default_negative_prompt
            guidance_scale = 1.2
            num_inference_steps = 50

        self.last_prompt = prompt
        self.stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Update prompt if it has changed (for both controlnet and standard modes)
        if hasattr(params, 'prompt') and params.prompt != self.last_prompt:
            self.last_prompt = params.prompt
            # Update the pipeline with new prompt
            if self.use_controlnet:
                negative_prompt = self.controlnet_config.get('negative_prompt', default_negative_prompt)
                guidance_scale = self.controlnet_config.get('guidance_scale', 1.2)
                num_inference_steps = self.controlnet_config.get('num_inference_steps', 50)
            else:
                negative_prompt = default_negative_prompt
                guidance_scale = 1.2
                num_inference_steps = 50
            
            self.stream.prepare(
                prompt=params.prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        if self.use_controlnet:
            # Update each ControlNet with appropriate image using unified API
            if 'controlnets' in self.controlnet_config:
                for i, cn_config in enumerate(self.controlnet_config['controlnets']):
                    if (cn_config['preprocessor'] == 'browser' and 
                        'pose' in cn_config['model_id'].lower() and
                        hasattr(params, 'pose_image') and params.pose_image is not None):
                        # Send pose image to browser preprocessor
                        self.stream.update_control_image_efficient(params.pose_image, index=i)
                        print(f"predict: Sent pose image to ControlNet {i} (browser preprocessor)")
                    else:
                        # Send webcam image to other preprocessors
                        self.stream.update_control_image_efficient(params.image, index=i)
            else:
                # Fallback for no ControlNet config - update all
                self.stream.update_control_image_efficient(params.image)
                
            output_image = self.stream(params.image)
        else:
            # Standard mode: use original logic with preprocessed tensor
            image_tensor = self.stream.preprocess_image(params.image)
            output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image


