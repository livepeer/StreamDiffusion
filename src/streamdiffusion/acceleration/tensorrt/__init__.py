import gc
import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from .builder import EngineBuilder
from .models.models import BaseModel

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
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
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
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
