import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper_batch import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold"></h3>
<p class="text-sm">
   Frame Buffer Size Demo
</p>
"""


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

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        self.args = args
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=[25, 35],
            frame_buffer_size=args.frame_buffer_size,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none", # "self", # "none" ## "initialize" and "full" does not work
            use_safety_checker=args.safety_checker,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir=args.engine_dir,
        )

        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def predict(self, params: "Pipeline.InputParams", frame_buffer=None) -> Image.Image:
        # Simplified processing - the pipeline now handles multi-stream internally
        if frame_buffer is not None and len(frame_buffer) > 0:
            # Take the most recent frame from the buffer
            latest_frame = frame_buffer[-1]
            
            # Ensure frame is RGB and correct size
            latest_frame = latest_frame.convert("RGB").resize((params.width, params.height))
            
            # Process single frame through the wrapper
            # The StreamDiffusion pipeline handles multi-stream processing internally
            output_image = self.stream(image=latest_frame, prompt=params.prompt)
            
            return output_image
        elif hasattr(params, 'image') and params.image is not None:
            # Direct image processing
            output_image = self.stream(image=params.image, prompt=params.prompt)
            return output_image
        else:
            return None
