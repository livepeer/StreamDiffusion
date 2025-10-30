import warnings
import torch
from diffusers import ControlNetModel, EulerDiscreteScheduler, DDPMScheduler, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from torch import Tensor
from torchvision.io.video import read_video, write_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, raft_small, Raft_Small_Weights
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import trange

raft_transform = Raft_Small_Weights.DEFAULT.transforms()
generator = torch.Generator(device="cuda").manual_seed(42)


@torch.inference_mode()
def stylize_video(
    input_video: Tensor,
    prompt: str = "elon musk",
    strength: float = 0.40,
    num_steps: int = 30,
    guidance_scale: float = 7.0,
    controlnet_scale: float = 1.0,
    use_temporal_controlnet: bool = True,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
    diffusion_model: str = "stabilityai/stable-diffusion-2-1", # "stabilityai/stable-diffusion-2-1", "Lykon/dreamshaper-8"
    controlnet_model: str = "/home/user/StreamDiffusion/training/models/", # checkpoint-27500/ "wav/TemporalNet2"
) -> Tensor:
    """
    Stylize a video with optional temporal coherence using HuggingFace's Stable Diffusion pipeline.

    Args:
        input_video (Tensor): Input video tensor of shape (T, C, H, W) and range [0, 1].
        prompt (str): Text prompt to condition the diffusion process.
        strength (float, optional): How heavily stylization affects the image.
        num_steps (int, optional): Number of diffusion steps (tradeoff between quality and speed).
        guidance_scale (float, optional): Scale of the text guidance loss (how closely to adhere to text prompt).
        controlnet_scale (float, optional): Scale of the ControlNet conditioning (strength of temporal coherence).
        use_temporal_controlnet (bool, optional): Whether to use temporal controlnet for coherence.
        height (int, optional): Height of the output video.
        width (int, optional): Width of the output video.
        device (str, optional): Device to run stylization process on.

    Returns:
        Tensor: Output video tensor of shape (T, C, H, W) and range [0, 1].
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence annoying TypedStorage warnings

        if use_temporal_controlnet:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                diffusion_model,
                controlnet=ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16),
                safety_checker=None,
                torch_dtype=torch.float16,
            ).to(device)
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                diffusion_model,
                safety_checker=None,
                torch_dtype=torch.float16,
            ).to(device)
        
        pipe.enable_xformers_memory_efficient_attention()
        pipe._progress_bar_config = dict(disable=True)

    if use_temporal_controlnet:
        raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True).eval().to(device)

    output_video = []
    prev_output = None
    
    if use_temporal_controlnet:
        for i in trange(len(input_video), desc="Diffusing...", unit="frame"):
            curr = resize(input_video[i:i + 1], (height, width), antialias=True).to(device)
            
            if i == 0:
                prev = curr
            else:
                prev = prev_output.to(device)

            flow_img = flow_to_image(raft.forward(*raft_transform(prev, curr))[-1]).div(255)
            control_img = torch.cat((prev, flow_img), dim=1)

            output, _ = pipe(
                prompt=prompt,
                image=curr,
                control_image=control_img,
                height=height,
                width=width,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                output_type="pt",
                generator=generator,
                return_dict=False,
            )

            output_video.append(output.permute(0, 2, 3, 1).cpu())
            prev_output = output
    else:
        for i in trange(len(input_video), desc="Diffusing...", unit="frame"):
            curr = resize(input_video[i:i + 1], (height, width), antialias=True).to(device)

            output, _ = pipe(
                prompt=prompt,
                image=curr,
                height=height,
                width=width,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                output_type="pt",
                generator=generator,
                return_dict=False,
            )

            output_video.append(output.permute(0, 2, 3, 1).cpu())

    return torch.cat(output_video)


if __name__ == "__main__":
    input_video, _, info = read_video("images/inputs/temporal_test_resized.mp4", pts_unit="sec", output_format="TCHW")
    input_video = input_video.div(255)
    
    tpnet = True
    output_video = stylize_video(input_video, use_temporal_controlnet=tpnet)

    out_file = f"temporal_net_{tpnet}_test_video.mp4"
    write_video(out_file, output_video.mul(255), fps=int(info["video_fps"]))


# import torch
# from torchvision.models.optical_flow import raft_large
# from torchvision.utils import flow_to_image
# from torchvision.io import read_image, write_jpeg

# def run_raft_and_save_flow(img_path1, img_path2, save_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load images
#     img1 = read_image(img_path1).float() / 255.0
#     img2 = read_image(img_path2).float() / 255.0
    
#     # Add batch dimension and move to device
#     img1 = img1.unsqueeze(0).to(device)
#     img2 = img2.unsqueeze(0).to(device)
    
#     # Load pretrained RAFT large model
#     model = raft_large(pretrained=True).to(device).eval()
    
#     # Run model (returns list of flows from iterations)
#     with torch.no_grad():
#         flow_list = model(img1, img2)
    
#     # Take the final flow (most accurate)
#     flow = flow_list[-1][0]  # Remove batch dimension
    
#     # Convert flow to RGB image
#     flow_img = flow_to_image(flow).cpu()
    
#     # Save as JPEG
#     write_jpeg(flow_img, save_path)
    
#     return save_path

# # Usage
# run_raft_and_save_flow('/home/user/datasets/test/jmD-dnzaxtw_0/8.jpg', '/home/user/datasets/test/jmD-dnzaxtw_0/13.jpg', 'temp_optical.jpg')


# import os
# import cv2
# import json
# import base64
# from tqdm import tqdm
# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
# from PIL import Image

# default_prompt = """You get:

# * image (what you can see)
# * appearance_traits\[] (tokens like "male", "pale_skin", "oval_face", "bald", "bangs", "rosy_cheeks", "young", "chubby", etc.)
# * action_traits\[] (tokens like "drink", "gaze", "kiss", "chew", "close_eyes", "head_wagging", etc.)

# Goal:
# Write TEN Stable Diffusion 2.1-style prompt variations that sound like a normal user wrote them: short, visual, comma-separated.
# The variations should be very different from each other. WHILE THEY SHOULD CONVEY THE SAME INFO THEY SHOULD BE VERY DIFFERENT. DO NOT repeat the same prompt.
# Return ONLY a JSON array of 10 items; each item must have keys: "prompt" and (optional) "negative_prompt".

# How to write (keep it simple):

# * Start with what you actually see in the image (subject). Use appearance_traits and action_traits to guide wording.
# * If a trait conflicts with the image, ignore the trait and trust the image.
# * Convert tokens to natural words:

#   * snake_case → spaced words (pale_skin → "pale skin", rosy_cheeks → "rosy cheeks", oval_face → "oval face", receding_hairline → "receding hairline")
#   * gender/age/build: male→"man", female→"woman", young→"young", old→"older", chubby→"chubby build"
#   * hair: bald→"bald", bangs→"with bangs" (only if visible)
# * Convert action tokens to natural gerunds:

#   * drink→"drinking", eat→"eating", gaze→"gazing", glare→"glaring", kiss→"kissing", cough→"coughing", cry→"crying", blow→"blowing", head_wagging→"shaking head", close_eyes→"eyes closed"
# * Add setting/mood only if obvious (street at night, park in daylight, kitchen counter, etc.).
# * Add light/composition only if obvious (close-up, mid shot, soft light, shallow depth of field). Skip fancy jargon and artist names.
# * Keep the positive prompt \~15–40 words.
# * If any low-quality/defect hints appear (e.g., "blurry"), do NOT put them in the positive prompt—put them in "negative_prompt" along with typical defects like "low resolution, watermark, text". If nothing to suppress, you may omit "negative_prompt".

# Output format (JSON only):
# \[
# {
# "prompt": "<concise SD2.1-style prompt>",
# "negative_prompt": "<optional short list of defects to avoid>"
# },
# ...
# // total of 10 items
# ]

# Examples (illustrative — each example would be one item in the array):

# Example 1
# Inputs summary: image shows a young man with pale skin and rosy cheeks, oval face, slight receding hairline (no bangs), close-up at a cafe table; actions: \["drink","gaze"]
# Return:
# {
# "prompt": "young man with pale skin and rosy cheeks, oval face, slight receding hairline, drinking from a mug at a cafe table, close-up, gazing to the side, warm indoor light, soft focus background, sharp subject",
# "negative_prompt": "blurry, low resolution, watermark, text"
# }

# Example 2
# Inputs summary: image shows a chubby woman with bangs, bright street at dusk, mid shot; actions: \["cry","close_eyes"]
# Return:
# {
# "prompt": "chubby woman with bangs on a city street at dusk, mid shot, eyes closed, crying, soft evening light, gentle bokeh, natural colors, simple documentary feel",
# "negative_prompt": "overprocessed, oversaturated, watermark, text"
# }

# Following are the appearance and action traits:

# Appearance: {{appearance}}
# Action: {{action}}
# """

# celeb_meta = json.load(open("/home/user/datasets/celebvhq_info.json"))
# meta_info = celeb_meta["meta_info"]
# clips = celeb_meta["clips"]
# appearance_mapping = meta_info["appearance_mapping"]
# action_mapping = meta_info["action_mapping"]

# def image_from_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     ret, first_frame = cap.read()
#     cap.release()
#     if ret:
#         return Image.fromarray(first_frame).resize((512, 512))
#     else:
#         return None

# def video_attributes(video_id):
#     appearance = []
#     actions = []
#     for idx, l in enumerate(clips[video_id]["attributes"]["appearance"]):
#         if l:
#             appearance.append(appearance_mapping[idx])
#     for idx, l in enumerate(clips[video_id]["attributes"]["action"]):
#         if l:
#             actions.append(action_mapping[idx])
#     return appearance, actions

# model = 'OpenGVLab/InternVL2_5-4B-AWQ'
# pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1, enable_prefix_caching=True), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
# generation_config = GenerationConfig(do_sample=True, temperature=0.7, max_new_tokens=4096)
# batch_size = 16

# ans = {}
# batched = []
# video_ids = []
# for idx, video in enumerate(tqdm(os.listdir("/home/user/datasets/celebs/"))):
#     video_id = video.replace(".mp4", "")
#     video_ids.append(video_id)
#     image = image_from_video(f"/home/user/datasets/celebs/{video}")
#     if image is None:
#         continue
#     appearance, actions = video_attributes(video_id)
#     prompt = default_prompt.replace("{{appearance}}", str(appearance)).replace("{{action}}", str(actions))
#     batched.append((prompt, image))
#     if len(batched) == batch_size:
#         responses = pipe(batched, gen_config=generation_config)
#         for response, video_id in zip(responses, video_ids):
#             ans[video_id] = response.text
#         with open("prompts.json", "w") as f:
#             json.dump(ans, f)
#         batched = []
#         video_ids = []

# if len(batched) > 0:
#     responses = pipe(batched, gen_config=generation_config)
#     for response, video_id in zip(responses, video_ids):
#         ans[video_id] = response.text
#     with open("prompts.json", "w") as f:
#         json.dump(ans, f)