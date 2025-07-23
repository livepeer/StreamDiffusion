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
                nan_count = torch.isnan(buf).sum().item()
                total_elements = buf.numel()
                logger.warning(f"[DEBUG] Engine.infer: *** NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%) ***")
            if torch.isinf(buf).any():
                logger.warning(f"[DEBUG] Engine.infer: *** WARNING: Inf detected in input '{name}'! ***")
                inf_count = torch.isinf(buf).sum().item()
                total_elements = buf.numel()
                logger.warning(f"[DEBUG] Engine.infer: *** Inf count: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%) ***")
            if (buf == 0).all():
                logger.warning(f"[DEBUG] Engine.infer: *** WARNING: All values in input '{name}' are zero! ***")
            
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

        # Check output tensors
        logger.debug(f"[DEBUG] Engine.infer: Checking output tensors")
        output_tensors = {}
        for name, tensor in self.tensors.items():
            if name not in feed_dict:  # This is an output tensor
                logger.debug(f"[DEBUG] Engine.infer: Output '{name}' - shape: {tensor.shape}, dtype: {tensor.dtype}")
                logger.debug(f"[DEBUG] Engine.infer: Output '{name}' range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
                
                # Check for problematic values in outputs
                if torch.isnan(tensor).any():
                    nan_count = torch.isnan(tensor).sum().item()
                    total_elements = tensor.numel()
                    logger.error(f"[DEBUG] Engine.infer: *** ERROR: NaN detected in output '{name}'! ***")
                    logger.error(f"[DEBUG] Engine.infer: *** NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%) ***")
                if torch.isinf(tensor).any():
                    inf_count = torch.isinf(tensor).sum().item()
                    total_elements = tensor.numel()
                    logger.error(f"[DEBUG] Engine.infer: *** ERROR: Inf detected in output '{name}'! ***")
                    logger.error(f"[DEBUG] Engine.infer: *** Inf count: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%) ***")
                if (tensor == 0).all():
                    logger.error(f"[DEBUG] Engine.infer: *** ERROR: All values in output '{name}' are zero! ***")
                
                output_tensors[name] = tensor
        
        logger.debug(f"[DEBUG] Engine.infer: Inference completed, returning tensors")
        return output_tensors


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


class SDXLUNetWrapper(torch.nn.Module):
    """Wrapper for SDXL UNet to handle optional conditioning in legacy TensorRT"""
    
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.base_unet = self._get_base_unet(unet)
        self.supports_added_cond = self._test_added_cond_support()
        
    def _get_base_unet(self, unet):
        """Extract the base UNet from wrappers like ControlNetUNetWrapper"""
        # Handle ControlNet wrapper
        if hasattr(unet, 'unet_model') and hasattr(unet.unet_model, 'config'):
            return unet.unet_model
        elif hasattr(unet, 'unet') and hasattr(unet.unet, 'config'):
            return unet.unet
        elif hasattr(unet, 'config'):
            return unet
        else:
            # Fallback: try to find any attribute that has config
            for attr_name in dir(unet):
                if not attr_name.startswith('_'):
                    attr = getattr(unet, attr_name, None)
                    if hasattr(attr, 'config') and hasattr(attr.config, 'addition_embed_type'):
                        return attr
            return unet
        
    def _test_added_cond_support(self):
        """Test if this SDXL model supports added_cond_kwargs"""
        try:
            # Create minimal test inputs
            sample = torch.randn(1, 4, 8, 8, device='cuda', dtype=torch.float16)
            timestep = torch.tensor([0.5], device='cuda', dtype=torch.float32)
            encoder_hidden_states = torch.randn(1, 77, 2048, device='cuda', dtype=torch.float16)
            
            # Test with added_cond_kwargs
            test_added_cond = {
                'text_embeds': torch.randn(1, 1280, device='cuda', dtype=torch.float16),
                'time_ids': torch.randn(1, 6, device='cuda', dtype=torch.float16)
            }
            
            with torch.no_grad():
                _ = self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=test_added_cond)
            
            print("✅ SDXL model supports added_cond_kwargs")
            return True
            
        except Exception as e:
            print(f"⚠️ SDXL model does not support added_cond_kwargs: {e}")
            return False
        
    def forward(self, *args, **kwargs):
        """Forward pass that handles SDXL conditioning gracefully"""
        print(f"[SDXL_WRAPPER] forward: Called with {len(args)} args and {len(kwargs)} kwargs")
        print(f"[SDXL_WRAPPER] forward: Args shapes: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}")
        print(f"[SDXL_WRAPPER] forward: Kwargs keys: {list(kwargs.keys())}")
        print(f"[SDXL_WRAPPER] forward: self.supports_added_cond: {self.supports_added_cond}")
        print(f"[SDXL_WRAPPER] forward: Underlying UNet type: {type(self.unet)}")
        
        try:
            # Ensure added_cond_kwargs is never None to prevent TypeError
            if 'added_cond_kwargs' in kwargs and kwargs['added_cond_kwargs'] is None:
                print(f"[SDXL_WRAPPER] forward: Setting added_cond_kwargs from None to empty dict")
                kwargs['added_cond_kwargs'] = {}
            
            # Auto-generate SDXL conditioning if missing and model needs it
            if (len(args) >= 3 and 'added_cond_kwargs' not in kwargs and 
                hasattr(self.base_unet.config, 'addition_embed_type') and 
                self.base_unet.config.addition_embed_type == 'text_time'):
                
                sample = args[0]
                device = sample.device
                batch_size = sample.shape[0]
                
                print("🔧 Auto-generating required SDXL conditioning...")
                kwargs['added_cond_kwargs'] = {
                    'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                    'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
                }
                
            # If model supports added conditioning and we have the kwargs, use them
            if self.supports_added_cond and 'added_cond_kwargs' in kwargs:
                print(f"[SDXL_WRAPPER] forward: Using full SDXL call with added_cond_kwargs")
                print(f"[SDXL_WRAPPER] forward: About to call self.unet(*args, **kwargs)")
                print(f"[SDXL_WRAPPER] forward: 🚀 Starting underlying UNet call...")
                
                import time
                start_time = time.time()
                result = self.unet(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                print(f"[SDXL_WRAPPER] forward: ✅ Underlying UNet call completed in {elapsed_time:.3f}s")
                return result
            elif len(args) >= 3:
                print(f"[SDXL_WRAPPER] forward: Using basic SDXL call (no added_cond_kwargs)")
                print(f"[SDXL_WRAPPER] forward: About to call self.unet(args[0], args[1], args[2])")
                
                import time
                start_time = time.time()
                result = self.unet(args[0], args[1], args[2])
                elapsed_time = time.time() - start_time
                
                print(f"[SDXL_WRAPPER] forward: ✅ Basic UNet call completed in {elapsed_time:.3f}s")
                return result
            else:
                print(f"[SDXL_WRAPPER] forward: Using fallback call")
                # Fallback
                return self.unet(*args, **kwargs)
                
        except (TypeError, AttributeError) as e:
            print(f"[SDXL_WRAPPER] forward: Exception caught: {e}")
            if "NoneType" in str(e) or "iterable" in str(e) or "text_embeds" in str(e):
                # Handle SDXL-Turbo models that need proper conditioning
                print(f"🔧 Providing minimal SDXL conditioning due to: {e}")
                if len(args) >= 3:
                    sample, timestep, encoder_hidden_states = args[0], args[1], args[2]
                    device = sample.device
                    batch_size = sample.shape[0]
                    
                    # Create minimal valid SDXL conditioning
                    minimal_conditioning = {
                        'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                        'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
                    }
                    
                    try:
                        print(f"[SDXL_WRAPPER] forward: Trying with minimal conditioning...")
                        return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=minimal_conditioning)
                    except Exception as final_e:
                        print(f"🔧 Final fallback to basic call: {final_e}")
                        return self.unet(sample, timestep, encoder_hidden_states)
                else:
                    return self.unet(*args)
            else:
                raise e


def export_onnx(
    model,
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
):
    # Use our advanced SDXL support for better handling
    from .sdxl_support import SDXLModelDetector, create_sdxl_tensorrt_wrapper
    
    detector = SDXLModelDetector()
    
    # Check if this is an SDXL model that might need wrapping
    is_sdxl_by_data = (hasattr(model_data, 'embedding_dim') and 
                       model_data.embedding_dim in [2048, 1024])
    
    # Check if this is a UNet model that needs SDXL handling
    is_unet = hasattr(model, 'config') and hasattr(model.config, 'cross_attention_dim')
    
    wrapped_model = model
    
    if is_unet and (is_sdxl_by_data or detector.detect_unet_type(model)['is_sdxl']):
        print("🔧 Detected SDXL model, using advanced wrapper for robust ONNX export...")
        
        # Try to get model path for better detection
        model_path = getattr(model_data, 'model_path', '') or getattr(model, 'model_path', '')
        
        # Use our advanced SDXL wrapper
        wrapped_model = create_sdxl_tensorrt_wrapper(model, model_path)
        
        print(f"   SDXL detection - by data: {is_sdxl_by_data}")
        print(f"   Model path: {model_path}")
    
    # For legacy compatibility, still wrap other potential SDXL models
    elif is_sdxl_by_data and hasattr(model, 'forward'):
        print("🔧 Using legacy SDXL wrapper for ONNX export...")
        wrapped_model = SDXLUNetWrapper(model)
    
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        
        # Determine if we need external data format for large models (like SDXL)
        is_large_model = is_sdxl_by_data or (hasattr(model, 'config') and getattr(model.config, 'sample_size', 32) >= 64)
        
        # Export ONNX normally first
        torch.onnx.export(
            wrapped_model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
        )
        
        # Convert to external data format for large models (SDXL)
        if is_large_model:
            print("🔧 Converting ONNX to external data format for large model (SDXL)...")
            import os
            
            # Load the exported model
            onnx_model = onnx.load(onnx_path)
            
            # Check if model is large enough to need external data
            if onnx_model.ByteSize() > 2147483648:  # 2GB
                print(f"   Model size: {onnx_model.ByteSize() / (1024**3):.2f} GB - converting to external data format")
                
                # Create directory for external data
                onnx_dir = os.path.dirname(onnx_path)
                
                # Re-save with external data format
                onnx.save_model(
                    onnx_model,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.pb",
                    convert_attribute=False,
                )
                print(f"✅ Converted to external data format with weights in weights.pb")
            else:
                print(f"   Model size: {onnx_model.ByteSize() / (1024**3):.2f} GB - keeping standard format")
            
            del onnx_model
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    import os
    import shutil
    
    # Check if external data files exist (indicating external data format was used)
    onnx_dir = os.path.dirname(onnx_path)
    external_data_files = [f for f in os.listdir(onnx_dir) if f.endswith('.pb')]
    uses_external_data = len(external_data_files) > 0
    
    if uses_external_data:
        print(f"🔧 Optimizing ONNX model with external data format...")
        print(f"   Found {len(external_data_files)} external data files")
        
        # Load model with external data
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx_opt_graph = model_data.optimize(onnx_model)
        
        # Create output directory
        opt_dir = os.path.dirname(onnx_opt_path)
        os.makedirs(opt_dir, exist_ok=True)
        
        # Clean up existing files in output directory
        if os.path.exists(opt_dir):
            for f in os.listdir(opt_dir):
                if f.endswith('.pb') or f.endswith('.onnx'):
                    os.remove(os.path.join(opt_dir, f))
        
        # Save optimized model with external data format
        print(f"🔧 Saving optimized model with external data to: {onnx_opt_path}")
        onnx.save_model(
            onnx_opt_graph,
            onnx_opt_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        print(f"✅ ONNX optimization complete with external data")
        
    else:
        print(f"🔧 Optimizing ONNX model (standard format)...")
        # Standard optimization for smaller models
        onnx_opt_graph = model_data.optimize(onnx.load(onnx_path))
        onnx.save(onnx_opt_graph, onnx_opt_path)
    
    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
