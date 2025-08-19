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
    print(f"compile_vae_encoder: Starting VAE encoder compilation")
    print(f"compile_vae_encoder: ONNX path: {onnx_path}")
    print(f"compile_vae_encoder: ONNX opt path: {onnx_opt_path}")
    print(f"compile_vae_encoder: Engine path: {engine_path}")
    print(f"compile_vae_encoder: Batch size: {opt_batch_size}")
    print(f"compile_vae_encoder: Build options: {engine_build_options}")
    
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    print(f"compile_vae_encoder: Created EngineBuilder")
    
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    print(f"compile_vae_encoder: VAE encoder compilation completed")


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    print(f"compile_vae_decoder: Starting VAE decoder compilation")
    print(f"compile_vae_decoder: ONNX path: {onnx_path}")
    print(f"compile_vae_decoder: ONNX opt path: {onnx_opt_path}")
    print(f"compile_vae_decoder: Engine path: {engine_path}")
    print(f"compile_vae_decoder: Batch size: {opt_batch_size}")
    print(f"compile_vae_decoder: Build options: {engine_build_options}")
    
    vae = vae.to(torch.device("cuda"))
    print(f"compile_vae_decoder: Moved VAE to CUDA")
    
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    print(f"compile_vae_decoder: Created EngineBuilder")
    
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    print(f"compile_vae_decoder: VAE decoder compilation completed")


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    print(f"compile_unet: Starting UNet compilation")
    print(f"compile_unet: ONNX path: {onnx_path}")
    print(f"compile_unet: ONNX opt path: {onnx_opt_path}")
    print(f"compile_unet: Engine path: {engine_path}")
    print(f"compile_unet: Batch size: {opt_batch_size}")
    print(f"compile_unet: Build options: {engine_build_options}")
    
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    print(f"compile_unet: Moved UNet to CUDA with float16 dtype")
    
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    print(f"compile_unet: Created EngineBuilder")
    
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    print(f"compile_unet: UNet compilation completed")
