import gc
import os
from typing import *

import torch
from diffusers import AutoencoderKL, ControlNetModel
import numpy as np

from .models import BaseModel
from .utilities import (
    build_engine,
    export_onnx,
    optimize_onnx,
)


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        engine_path: str,
        opt_image_height: int = 512,
        opt_image_width: int = 512,
        opt_batch_size: int = 1,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = False,
        build_dynamic_batch: bool = False,
        min_batch_size: int = 1,
        max_batch_size: int = 4,
        build_all_tactics: bool = False,
        onnx_opset: int = 17,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
    ):
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
        else:
            print(f"Exporting model: {onnx_path}")
            export_onnx(
                self.network,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
                enable_dynamic_batch=build_dynamic_batch,
            )
            del self.network
            gc.collect()
            torch.cuda.empty_cache()
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"Found cached model: {onnx_opt_path}")
        else:
            print(f"Generating optimizing model: {onnx_opt_path}")
            optimize_onnx(
                onnx_path=onnx_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
            )
        self.model.min_latent_shape = min_image_resolution // 8
        self.model.max_latent_shape = max_image_resolution // 8
        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
        else:
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_dynamic_batch=build_dynamic_batch,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
            )

        gc.collect()
        torch.cuda.empty_cache()


def compile_controlnet(
    controlnet: ControlNetModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    engine_build_options: dict = {},
):
    """
    Compile ControlNet model to TensorRT
    
    Args:
        controlnet: PyTorch ControlNet model
        model_data: ControlNet TensorRT model definition
        onnx_path: Path for ONNX export
        onnx_opt_path: Path for optimized ONNX
        engine_path: Path for TensorRT engine
        engine_build_options: Additional build options (should include opt_batch_size)
    """
    controlnet = controlnet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, controlnet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def compile_dynamic_unet(
    unet: torch.nn.Module,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    engine_build_options: dict = {},
):
    """
    Compile UNet model to TensorRT with dynamic batch size support
    
    Args:
        unet: PyTorch UNet model
        model_data: UNet TensorRT model definition with dynamic batch support
        onnx_path: Path for ONNX export
        onnx_opt_path: Path for optimized ONNX
        engine_path: Path for TensorRT engine
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        engine_build_options: Additional build options for TensorRT
    """
    print(f"\n=== COMPILE_DYNAMIC_UNET ===")
    print(f"  min_batch_size: {min_batch_size}")
    print(f"  max_batch_size: {max_batch_size}")
    print(f"  engine_build_options: {engine_build_options}")
    print(f"  onnx_path: {onnx_path}")
    print(f"  engine_path: {engine_path}")
    
    # Force regeneration of ONNX files for dynamic batch support
    if os.path.exists(onnx_path):
        print(f"  Removing existing ONNX file for dynamic batch rebuild: {onnx_path}")
        os.remove(onnx_path)
    if os.path.exists(onnx_opt_path):
        print(f"  Removing existing optimized ONNX file for dynamic batch rebuild: {onnx_opt_path}")
        os.remove(onnx_opt_path)
    if os.path.exists(engine_path):
        print(f"  Removing existing TensorRT engine for dynamic batch rebuild: {engine_path}")
        os.remove(engine_path)
    print("============================\n")
    
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        build_dynamic_batch=True,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        **engine_build_options,
    )


def compile_dynamic_vae(
    vae: torch.nn.Module,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    engine_build_options: dict = {},
):
    """
    Compile VAE model to TensorRT with dynamic batch size support
    
    Args:
        vae: PyTorch VAE model
        model_data: VAE TensorRT model definition with dynamic batch support
        onnx_path: Path for ONNX export
        onnx_opt_path: Path for optimized ONNX
        engine_path: Path for TensorRT engine
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        engine_build_options: Additional build options for TensorRT
    """
    # Force regeneration of ONNX files for dynamic batch support
    if os.path.exists(onnx_path):
        print(f"Removing existing ONNX file for dynamic batch rebuild: {onnx_path}")
        os.remove(onnx_path)
    if os.path.exists(onnx_opt_path):
        print(f"Removing existing optimized ONNX file for dynamic batch rebuild: {onnx_opt_path}")
        os.remove(onnx_opt_path)
    if os.path.exists(engine_path):
        print(f"Removing existing TensorRT engine for dynamic batch rebuild: {engine_path}")
        os.remove(engine_path)
    
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        build_dynamic_batch=True,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        **engine_build_options,
    )
