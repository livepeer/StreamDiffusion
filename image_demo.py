import argparse
import os
import json 
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
from diffusers import (
    AutoencoderTiny, LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)
from diffusers.utils import load_image, make_image_grid
from PIL import Image

from streamdiffusion import StreamUNetControlDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt_unetcontrol
from PIL import Image
import torch
import numpy as np

def png_to_torch_tensor(png_path, device='cpu', dtype=torch.float32):
    """Converts a PNG image to a PyTorch tensor (Batch, Channels, Height, Width)."""
    img = Image.open(png_path).convert("RGB")
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(device, dtype) / 255.0
    return tensor


parser = argparse.ArgumentParser()
parser.add_argument('--accel', type=str, choices=['xformers', 'tensorrt'], default='tensorrt')
parser.add_argument('--num_inference_steps', type=int, default=4)
parser.add_argument('--strength', type=float, default=0.8)
parser.add_argument('--cfg_type', type=str, choices=['none', 'self', 'initialize', 'full'], default='none')
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--cond_size', type=int, default=64)
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True) # Changed from video_path to image_path
parser.add_argument(
    '--lcm_id', type=str, default='1e-6',
    choices=[
        '1e-6', '5e-6', '1e-5', '5e-5', '1e-4',
        '1e-6-cfg7.5', '5e-6-cfg7.5', '1e-5-cfg7.5', '5e-5-cfg7.5', '1e-4-cfg7.5'
    ]
)
parser.add_argument('--server', type=str, choices=['athena', 'lighthouse'], default='lighthouse') # Not directly used in the inference logic
args = parser.parse_args()

args.frame_bff_size = 1 # Hardcoded for single image inference

input_image = load_image(args.image_path).convert("RGB")
input_image = input_image.resize((args.size, args.size), Image.LANCZOS)
print(f"Input image size: {input_image.size}")

controlnet = ControlNetModel.from_pretrained(
    'lllyasviel/control_v11f1p_sd15_depth',
    torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

if args.accel == 'xformers':
    pipe.enable_xformers_memory_efficient_attention()

stream = StreamUNetControlDiffusion(
    pipe,
    height=args.size,
    width=args.size,
    num_inference_steps=args.num_inference_steps,
    strength=args.strength,
    torch_dtype=torch.float16,
    cfg_type=args.cfg_type,
    frame_buffer_size=args.frame_bff_size,
)

delay = stream.denoising_steps_num


stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

if args.accel == 'tensorrt':
    stream = accelerate_with_tensorrt_unetcontrol(
        stream,
        f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}" \
        if args.size != 256 else \
        f"engines/unet_controlnet_size{args.size}-{args.cond_size}_strength{args.strength}_steps{args.num_inference_steps}_cfg={args.cfg_type}_framebff{args.frame_bff_size}_lcm{args.lcm_id}",
        max_batch_size=stream.batch_size,
        engine_build_options={
            'opt_image_height': args.size,
            'opt_image_width': args.size,
        }
    )

save_dir = f'./image_outputs/' 
os.makedirs(save_dir, exist_ok=True)

img_batch_size = args.frame_bff_size
negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, blur, blurred, DOF"] * img_batch_size

stream.prepare(
    [args.prompt] * img_batch_size,
    negative_prompt,
    guidance_scale=1.2,
)


input_image_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).unsqueeze(0).to(pipe.device, dtype=pipe.dtype) / 255.0

control_image_tensor = png_to_torch_tensor("/workspace/StreamDiffusion/ComfyUI_00006_.png", device='cuda', dtype=torch.float16)

print("Running inference on the image...")
output_frames = stream(input_image_tensor, control_image_tensor)

print("Debug: output_frames shape:", output_frames.shape)
print("Debug: output_frames type:", type(output_frames))

processed_output_image = postprocess_image(
    output_frames,
    output_type="pil"
)

# If postprocess_image returns a list, take the first image
if isinstance(processed_output_image, list):
    processed_output_image = processed_output_image[0]

output_filename = f"{os.path.splitext(os.path.basename(args.image_path))[0]}_output.png"
processed_output_image.save(os.path.join(save_dir, output_filename))
print(f"Output image saved to {os.path.join(save_dir, output_filename)}")

with open(f"{save_dir}/{os.path.splitext(os.path.basename(args.image_path))[0]}_time.txt", 'w') as f:
    f.write(f'Inference time per image: {stream.inference_time_ema/img_batch_size:.6f} s\n')
    f.write(f'FPS (approx for single image): {1/(stream.inference_time_ema/img_batch_size):.4f}')