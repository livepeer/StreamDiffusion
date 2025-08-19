import gc
import os
from typing import *

import torch
from diffusers.models import ControlNetModel

from .models.models import BaseModel
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
        build_dynamic_shape: bool = True,
        build_all_tactics: bool = False,
        onnx_opset: int = 17,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
    ):
        print(f"EngineBuilder.build: Starting build process")
        print(f"EngineBuilder.build: Image size: {opt_image_width}x{opt_image_height}")
        print(f"EngineBuilder.build: Batch size: {opt_batch_size}")
        print(f"EngineBuilder.build: Resolution range: {min_image_resolution}-{max_image_resolution}")
        print(f"EngineBuilder.build: Dynamic shape: {build_dynamic_shape}")
        print(f"EngineBuilder.build: Static batch: {build_static_batch}")
        print(f"EngineBuilder.build: Enable refit: {build_enable_refit}")
        print(f"EngineBuilder.build: All tactics: {build_all_tactics}")
        print(f"EngineBuilder.build: Force flags - export: {force_onnx_export}, optimize: {force_onnx_optimize}, engine: {force_engine_build}")
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"EngineBuilder.build: Found cached ONNX model: {onnx_path}")
        else:
            print(f"EngineBuilder.build: Exporting ONNX model: {onnx_path}")
            print(f"EngineBuilder.build: ONNX opset version: {onnx_opset}")
            export_onnx(
                self.network,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            print(f"EngineBuilder.build: ONNX export completed, cleaning up network")
            del self.network
            gc.collect()
            torch.cuda.empty_cache()
            print(f"EngineBuilder.build: Memory cleanup completed")
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"EngineBuilder.build: Found cached optimized ONNX model: {onnx_opt_path}")
        else:
            print(f"EngineBuilder.build: Optimizing ONNX model: {onnx_opt_path}")
            optimize_onnx(
                onnx_path=onnx_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
            )
            print(f"EngineBuilder.build: ONNX optimization completed")
        self.model.min_latent_shape = min_image_resolution // 8
        self.model.max_latent_shape = max_image_resolution // 8
        print(f"EngineBuilder.build: Set latent shape range: {self.model.min_latent_shape}-{self.model.max_latent_shape}")
        
        if not force_engine_build and os.path.exists(engine_path):
            print(f"EngineBuilder.build: Found cached TensorRT engine: {engine_path}")
        else:
            print(f"EngineBuilder.build: Building TensorRT engine: {engine_path}")
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
            )
            print(f"EngineBuilder.build: TensorRT engine building completed")

        print(f"EngineBuilder.build: Final cleanup")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"EngineBuilder.build: Build process completed successfully")


def compile_controlnet(
    controlnet: ControlNetModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
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
        opt_batch_size: Optimal batch size for compilation
        engine_build_options: Additional build options
    """
    print(f"compile_controlnet: Starting ControlNet compilation")
    print(f"compile_controlnet: ONNX path: {onnx_path}")
    print(f"compile_controlnet: ONNX opt path: {onnx_opt_path}")
    print(f"compile_controlnet: Engine path: {engine_path}")
    print(f"compile_controlnet: Batch size: {opt_batch_size}")
    print(f"compile_controlnet: Build options: {engine_build_options}")
    
    controlnet = controlnet.to(torch.device("cuda"), dtype=torch.float16)
    print(f"compile_controlnet: Moved ControlNet to CUDA with float16 dtype")
    
    builder = EngineBuilder(model_data, controlnet, device=torch.device("cuda"))
    print(f"compile_controlnet: Created EngineBuilder")
    
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
    print(f"compile_controlnet: ControlNet compilation completed")
