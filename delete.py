# from safetensors import safe_open

# # Path to your model file
# file_path = "training/models/checkpoint-27500/model.safetensors"

# with safe_open(file_path, framework="pt", device="cpu") as f:
#     # The metadata is a dictionary
#     metadata = f.metadata()

# if metadata:
#     print("✅ Metadata found:")
#     # Print the metadata in a readable format
#     import json
#     print(json.dumps(metadata, indent=4))
# else:
#     print("❌ No metadata found in this file.")

import torch
from diffusers import ControlNetModel, StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                safety_checker=None,
                torch_dtype=torch.float16,
            )

controlnet = ControlNetModel.from_unet(pipe.unet, conditioning_channels=6)
controlnet.save_pretrained("training/models/checkpoint-27500/")
# print(controlnet.config)