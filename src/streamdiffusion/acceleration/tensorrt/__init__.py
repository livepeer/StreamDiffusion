import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from .builder import EngineBuilder
from .models.models import BaseModel

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

class StableDiffusionSafetyCheckerWrapper(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)
    
    @torch.no_grad()
    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        return has_nsfw_concepts

class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))

def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # DEBUG: VRAM before VAE encoder compilation
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: Before VAE encoder compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    vae = vae.to(torch.device("cuda"))
    
    # DEBUG: VRAM after VAE encoder to CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After VAE encoder to CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    
    # DEBUG: VRAM after VAE encoder compilation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After VAE encoder compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # DEBUG: VRAM before VAE decoder compilation
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: Before VAE decoder compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    vae = vae.to(torch.device("cuda"))
    
    # DEBUG: VRAM after VAE decoder to CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After VAE decoder to CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    
    # DEBUG: VRAM after VAE decoder compilation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After VAE decoder compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def compile_safety_checker(
    safety_checker: StableDiffusionSafetyCheckerWrapper,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    safety_checker = safety_checker.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, safety_checker, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # DEBUG: VRAM before UNet compilation
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: Before UNet compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    
    # DEBUG: VRAM after UNet to CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After UNet to CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    
    # DEBUG: VRAM after UNet compilation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After UNet compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def compile_controlnet(
    controlnet: ControlNetModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # DEBUG: VRAM before ControlNet compilation
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: Before ControlNet compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    controlnet = controlnet.to(torch.device("cuda"), dtype=torch.float16)
    
    # DEBUG: VRAM after ControlNet to CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After ControlNet to CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    builder = EngineBuilder(model_data, controlnet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    
    # DEBUG: VRAM after ControlNet compilation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"DEBUG_VRAM: After ControlNet compilation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")