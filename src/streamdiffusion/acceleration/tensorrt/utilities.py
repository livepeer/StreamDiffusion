#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
import logging
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util

from .models import CLIP, VAE, BaseModel, UNet, VAEEncoder

# Set up logger for this module
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        
        # Buffer reuse optimization tracking
        self._last_shape_dict = None
        self._last_device = None

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        logger.info(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                logger.debug(f"Add Constant {name}")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                logger.warning(f"No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            logger.error("Failed to refit!")
            raise RuntimeError("TensorRT engine refit failed")

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        workspace_size=0,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(
                fp16=fp16, refittable=enable_refit, profiles=[p], load_timing_cache=timing_cache, **config_kwargs
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        # Check if we can reuse existing buffers (OPTIMIZATION)
        if self._can_reuse_buffers(shape_dict, device):
            logger.debug(f"[DEBUG] Engine.allocate_buffers: Reusing existing buffers")
            return
        
        logger.debug(f"[DEBUG] Engine.allocate_buffers: Allocating new buffers")
        logger.debug(f"[DEBUG] Engine.allocate_buffers: Device = {device}")
        logger.debug(f"[DEBUG] Engine.allocate_buffers: Engine has {self.engine.num_io_tensors} I/O tensors")
            
        # Clear existing buffers before reallocating
        self.tensors.clear()
        
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)

            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)

            dtype_np = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)

            logger.debug(f"[DEBUG] Engine.allocate_buffers: Tensor[{idx}] '{name}' - shape: {shape}, dtype: {dtype_np}, mode: {mode}")

            if mode == trt.TensorIOMode.INPUT:
                logger.debug(f"[DEBUG] Engine.allocate_buffers: Setting input shape for '{name}': {shape}")
                self.context.set_input_shape(name, shape)

            tensor = torch.empty(tuple(shape),
                                 dtype=numpy_to_torch_dtype_dict[dtype_np]) \
                          .to(device=device)
            self.tensors[name] = tensor
            
            logger.debug(f"[DEBUG] Engine.allocate_buffers: Allocated tensor '{name}' with shape {tensor.shape} and dtype {tensor.dtype}")
        
        # Cache allocation parameters for reuse check
        self._last_shape_dict = shape_dict.copy() if shape_dict else None
        self._last_device = device
        logger.debug(f"[DEBUG] Engine.allocate_buffers: Buffer allocation completed")
    
    def _can_reuse_buffers(self, shape_dict=None, device="cuda"):
        """
        Check if existing buffers can be reused (avoiding expensive reallocation)
        
        Returns:
            bool: True if buffers can be reused, False if reallocation needed
        """
        # No existing tensors - need to allocate
        if not self.tensors:
            return False
        
        # Device changed - need to reallocate
        if not hasattr(self, '_last_device') or self._last_device != device:
            return False
        
        # No cached shape_dict - need to allocate
        if not hasattr(self, '_last_shape_dict'):
            return False
            
        # Compare current vs cached shape_dict
        if shape_dict is None and self._last_shape_dict is None:
            return True
        elif shape_dict is None or self._last_shape_dict is None:
            return False
        
        # Quick check: if tensor counts differ, can't reuse
        if len(shape_dict) != len(self._last_shape_dict):
            return False
        
        # Compare shapes for all tensors in the new shape_dict
        for name, new_shape in shape_dict.items():
            # Check if tensor exists in cached shapes
            cached_shape = self._last_shape_dict.get(name)
            if cached_shape is None:
                return False
            
            # Compare shapes (handle different types consistently)
            if tuple(cached_shape) != tuple(new_shape):
                return False
        
        return True

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        logger.debug(f"[DEBUG] Engine.infer: Starting inference")
        logger.debug(f"[DEBUG] Engine.infer: feed_dict keys: {list(feed_dict.keys())}")
        logger.debug(f"[DEBUG] Engine.infer: use_cuda_graph: {use_cuda_graph}")
        
        # Copy input data to allocated tensors
        for name, buf in feed_dict.items():
            if name not in self.tensors:
                logger.warning(f"[DEBUG] Engine.infer: *** WARNING: Input '{name}' not found in allocated tensors! ***")
                logger.warning(f"[DEBUG] Engine.infer: Available tensors: {list(self.tensors.keys())}")
                continue
                
            logger.debug(f"[DEBUG] Engine.infer: Copying input '{name}' - shape: {buf.shape}, dtype: {buf.dtype}")
            logger.debug(f"[DEBUG] Engine.infer: Input '{name}' range: [{buf.min().item():.6f}, {buf.max().item():.6f}]")
            
            # Check for NaN/Inf in inputs
            if torch.isnan(buf).any():
                logger.warning(f"[DEBUG] Engine.infer: *** WARNING: NaN detected in input '{name}'! ***")
            if torch.isinf(buf).any():
                logger.warning(f"[DEBUG] Engine.infer: *** WARNING: Inf detected in input '{name}'! ***")
            
            self.tensors[name].copy_(buf)

        # Set tensor addresses for TensorRT context
        logger.debug(f"[DEBUG] Engine.infer: Setting tensor addresses for TensorRT context")
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
            logger.debug(f"[DEBUG] Engine.infer: Set address for tensor '{name}' at {hex(tensor.data_ptr())}")

        # Execute inference
        if use_cuda_graph:
            logger.debug(f"[DEBUG] Engine.infer: Using CUDA Graph execution")
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                logger.debug(f"[DEBUG] Engine.infer: Capturing CUDA Graph (first run)")
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    logger.error(f"[DEBUG] Engine.infer: *** ERROR: Initial inference failed during CUDA graph capture! ***")
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            logger.debug(f"[DEBUG] Engine.infer: Using standard TensorRT execution")
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                logger.error(f"[DEBUG] Engine.infer: *** ERROR: TensorRT inference execution failed! ***")
                logger.error(f"[DEBUG] Engine.infer: This could indicate:")
                logger.error(f"[DEBUG] Engine.infer: - Invalid input shapes or types")
                logger.error(f"[DEBUG] Engine.infer: - Memory allocation issues")
                logger.error(f"[DEBUG] Engine.infer: - Engine compatibility problems")
                raise ValueError("ERROR: inference failed.")
            else:
                logger.debug(f"[DEBUG] Engine.infer: TensorRT execution completed successfully")

        # Check outputs for NaN/Inf
        logger.debug(f"[DEBUG] Engine.infer: Checking output tensors")
        for name, tensor in self.tensors.items():
            mode = self.engine.get_tensor_mode(self.engine.get_tensor_name(
                next(i for i in range(self.engine.num_io_tensors) 
                     if self.engine.get_tensor_name(i) == name)
            ))
            
            if mode == trt.TensorIOMode.OUTPUT:
                logger.debug(f"[DEBUG] Engine.infer: Output '{name}' - shape: {tensor.shape}, dtype: {tensor.dtype}")
                logger.debug(f"[DEBUG] Engine.infer: Output '{name}' range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
                
                if torch.isnan(tensor).any():
                    nan_count = torch.isnan(tensor).sum().item()
                    total_elements = tensor.numel()
                    logger.error(f"[DEBUG] Engine.infer: *** ERROR: NaN detected in output '{name}'! Count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%) ***")
                    
                if torch.isinf(tensor).any():
                    inf_count = torch.isinf(tensor).sum().item()
                    total_elements = tensor.numel()
                    logger.error(f"[DEBUG] Engine.infer: *** ERROR: Inf detected in output '{name}'! Count: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%) ***")

        logger.debug(f"[DEBUG] Engine.infer: Inference completed, returning tensors")
        return self.tensors


def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    unet_in_channels: int = 4,
    embedding_dim: int = 768,
):
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_in_channels,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_all_tactics: bool = False,
    build_enable_refit: bool = False,
):
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30
    if free_mem > 6 * GiB:
        activation_carveout = 4 * GiB
        max_workspace_size = free_mem - activation_carveout
    else:
        max_workspace_size = 0
    engine = Engine(engine_path)
    input_profile = model_data.get_input_profile(
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=build_static_batch,
        static_shape=not build_dynamic_shape,
    )
    engine.build(
        onnx_opt_path,
        fp16=True,
        input_profile=input_profile,
        enable_refit=build_enable_refit,
        enable_all_tactics=build_all_tactics,
        workspace_size=max_workspace_size,
    )

    return engine


def export_onnx(
    model,
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
):
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    onnx_opt_graph = model_data.optimize(onnx.load(onnx_path))
    onnx.save(onnx_opt_graph, onnx_opt_path)
    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
