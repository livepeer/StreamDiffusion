from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models import ControlNetModel
import logging

from streamdiffusion.hooks import StepCtx, UnetKwargsDelta, UnetHook
from streamdiffusion.preprocessing.preprocessing_orchestrator import (
    PreprocessingOrchestrator,
)
from streamdiffusion.preprocessing.orchestrator_user import OrchestratorUser


@dataclass
class ControlNetConfig:
    model_id: str
    preprocessor: Optional[str] = None
    conditioning_scale: float = 1.0
    enabled: bool = True
    preprocessor_params: Optional[Dict[str, Any]] = None


class ControlNetModule(OrchestratorUser):
    """ControlNet module that provides a UNet hook for residual conditioning.

    Responsibilities in this step (3):
    - Manage a collection of ControlNet models, their scales, and current images
    - Provide a UNet hook that computes down/mid residuals for active ControlNets
    - Reuse the existing preprocessing orchestrator for control images
    - Do not alter the wrapper or pipeline call sites (registration happens via install())
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.device = device
        self.dtype = dtype

        self.controlnets: List[Optional[ControlNetModel]] = []
        self.controlnet_images: List[Optional[torch.Tensor]] = []
        self.controlnet_scales: List[float] = []
        self.preprocessors: List[Optional[Any]] = []
        self.enabled_list: List[bool] = []

        self._collections_lock = threading.RLock()
        self._preprocessing_orchestrator: Optional[PreprocessingOrchestrator] = None

        self._stream = None  # set in install
        # Per-frame prepared tensor cache to avoid per-step device/dtype alignment and batch repeats
        self._prepared_tensors: List[Optional[torch.Tensor]] = []
        self._prepared_device: Optional[torch.device] = None
        self._prepared_dtype: Optional[torch.dtype] = None
        self._prepared_batch: Optional[int] = None
        self._images_version: int = 0

        # Pre-allocated CUDA streams for PyTorch ControlNets (indexed to controlnets)
        self._pt_cn_streams: List[Optional[torch.cuda.Stream]] = []
        # Cache max parallel setting once
        try:
            import os
            self._max_parallel_controlnets = int(os.getenv('STREAMDIFFUSION_CN_MAX_PAR', '0'))
        except Exception:
            self._max_parallel_controlnets = 0
        # Persistent thread pool to avoid per-step creation cost
        self._executor = None
        self._executor_workers = 0

    # ---------- Public API (used by wrapper in a later step) ----------
    def install(self, stream) -> None:
        self._stream = stream
        self.device = stream.device
        self.dtype = stream.dtype
        if self._preprocessing_orchestrator is None:
            # Enforce shared orchestrator via base helper (raises if missing)
            self.attach_orchestrator(stream)
        # Register UNet hook
        stream.unet_hooks.append(self.build_unet_hook())
        # Expose controlnet collections so existing updater can find them
        setattr(stream, 'controlnets', self.controlnets)
        setattr(stream, 'controlnet_scales', self.controlnet_scales)
        setattr(stream, 'preprocessors', self.preprocessors)
        # Reset prepared tensors on install
        self._prepared_tensors = []
        self._prepared_device = None
        self._prepared_dtype = None
        self._prepared_batch = None
        # Reset PT CN streams on install
        self._pt_cn_streams = []

    def add_controlnet(self, cfg: ControlNetConfig, control_image: Optional[Union[str, Any, torch.Tensor]] = None) -> None:
        model = self._load_pytorch_controlnet_model(cfg.model_id)
        model = model.to(device=self.device, dtype=self.dtype)

        preproc = None
        if cfg.preprocessor:
            from streamdiffusion.preprocessing.processors import get_preprocessor
            preproc = get_preprocessor(cfg.preprocessor)
            # Apply provided parameters to the preprocessor instance
            if cfg.preprocessor_params:
                params = cfg.preprocessor_params or {}
                # If the preprocessor exposes a 'params' dict, update it
                if hasattr(preproc, 'params') and isinstance(getattr(preproc, 'params'), dict):
                    preproc.params.update(params)
                # Also set attributes directly when they exist
                for name, value in params.items():
                    try:
                        if hasattr(preproc, name):
                            setattr(preproc, name, value)
                    except Exception:
                        pass

            # Provide pipeline reference for preprocessors that need it (e.g., FeedbackPreprocessor)
            try:
                if hasattr(preproc, 'set_pipeline_ref'):
                    preproc.set_pipeline_ref(self._stream)
            except Exception:
                pass

            # Align preprocessor target size with stream resolution once (avoid double-resize later)
            try:
                if hasattr(preproc, 'params') and isinstance(getattr(preproc, 'params'), dict):
                    preproc.params['image_width'] = int(self._stream.width)
                    preproc.params['image_height'] = int(self._stream.height)
                if hasattr(preproc, 'image_width'):
                    setattr(preproc, 'image_width', int(self._stream.width))
                if hasattr(preproc, 'image_height'):
                    setattr(preproc, 'image_height', int(self._stream.height))
            except Exception:
                pass

        image_tensor: Optional[torch.Tensor] = None
        if control_image is not None and self._preprocessing_orchestrator is not None:
            image_tensor = self._prepare_control_image(control_image, preproc)

        with self._collections_lock:
            self.controlnets.append(model)
            self.controlnet_images.append(image_tensor)
            self.controlnet_scales.append(float(cfg.conditioning_scale))
            self.preprocessors.append(preproc)
            self.enabled_list.append(bool(cfg.enabled))
            # Invalidate prepared tensors and bump version when graph changes
            self._prepared_tensors = []
            self._images_version += 1
            # Maintain stream slots for PyTorch ControlNets
            self._pt_cn_streams.append(None)
            # Initialize/update engine map if present
            try:
                if hasattr(self._stream, 'controlnet_engines'):
                    for eng in list(getattr(self._stream, 'controlnet_engines') or []):
                        if not hasattr(eng, 'model_id'):
                            try:
                                setattr(eng, 'model_id', cfg.model_id)
                            except Exception:
                                pass
            except Exception:
                pass

    def update_control_image_efficient(self, control_image: Union[str, Any, torch.Tensor], index: Optional[int] = None) -> None:
        if self._preprocessing_orchestrator is None:
            return
        with self._collections_lock:
            if not self.controlnets:
                return
            total = len(self.controlnets)
            # Build active scales, respecting enabled_list if present
            scales = [
                (self.controlnet_scales[i] if i < len(self.controlnet_scales) else 1.0)
                for i in range(total)
            ]
            if hasattr(self, 'enabled_list') and self.enabled_list and len(self.enabled_list) == total:
                scales = [sc if bool(self.enabled_list[i]) else 0.0 for i, sc in enumerate(scales)]
            preprocessors = [self.preprocessors[i] if i < len(self.preprocessors) else None for i in range(total)]

        # Single-index fast path
        if index is not None:
            results = self._preprocessing_orchestrator.process_control_images_sync(
                control_image=control_image,
                preprocessors=preprocessors,
                scales=scales,
                stream_width=self._stream.width,
                stream_height=self._stream.height,
                index=index,
            )
            processed = results[index] if results and len(results) > index else None
            with self._collections_lock:
                if processed is not None and index < len(self.controlnet_images):
                    self.controlnet_images[index] = processed
                    # Invalidate prepared tensors and bump version for per-frame reuse
                    self._prepared_tensors = []
                    self._images_version += 1
                    # Pre-prepare tensors if we know the target specs
                    if self._stream and hasattr(self._stream, 'device') and hasattr(self._stream, 'dtype'):
                        # Use default batch size of 1 for now, will be adjusted on first use
                        self.prepare_frame_tensors(self._stream.device, self._stream.dtype, 1)
            return

        # Use intelligent pipelining (automatically detects feedback preprocessors and switches to sync)
        processed_images = self._preprocessing_orchestrator.process_control_images_pipelined(
            control_image=control_image,
            preprocessors=preprocessors,
            scales=scales,
            stream_width=self._stream.width,
            stream_height=self._stream.height,
        )

        # If orchestrator returns empty list, it indicates no update needed for this frame
        if processed_images is None or (isinstance(processed_images, list) and len(processed_images) == 0):
            return

        # Assign results
        with self._collections_lock:
            for i, img in enumerate(processed_images):
                if img is not None and i < len(self.controlnet_images):
                    self.controlnet_images[i] = img
            # Invalidate prepared cache and bump version after bulk update
            self._prepared_tensors = []
            self._images_version += 1
            # Pre-prepare tensors if we know the target specs
            if self._stream and hasattr(self._stream, 'device') and hasattr(self._stream, 'dtype'):
                # Use default batch size of 1 for now, will be adjusted on first use
                self.prepare_frame_tensors(self._stream.device, self._stream.dtype, 1)

    def update_controlnet_scale(self, index: int, scale: float) -> None:
        with self._collections_lock:
            if 0 <= index < len(self.controlnet_scales):
                self.controlnet_scales[index] = float(scale)

    def update_controlnet_enabled(self, index: int, enabled: bool) -> None:
        with self._collections_lock:
            if 0 <= index < len(self.enabled_list):
                self.enabled_list[index] = bool(enabled)

    def remove_controlnet(self, index: int) -> None:
        with self._collections_lock:
            if 0 <= index < len(self.controlnets):
                del self.controlnets[index]
                if index < len(self.controlnet_images):
                    del self.controlnet_images[index]
                if index < len(self.controlnet_scales):
                    del self.controlnet_scales[index]
                if index < len(self.preprocessors):
                    del self.preprocessors[index]
                if index < len(self.enabled_list):
                    del self.enabled_list[index]
                # Invalidate prepared tensors and bump version
                self._prepared_tensors = []
                self._images_version += 1
                if index < len(self._pt_cn_streams):
                    del self._pt_cn_streams[index]

    def reorder_controlnets_by_model_ids(self, desired_model_ids: List[str]) -> None:
        """Reorder internal collections to match the desired model_id order.

        Any controlnet whose model_id is not present in desired_model_ids retains its
        relative order after those that are specified.
        """
        with self._collections_lock:
            # Build current mapping from model_id to index
            current_ids: List[str] = []
            for i, cn in enumerate(self.controlnets):
                model_id = getattr(cn, 'model_id', f'controlnet_{i}')
                current_ids.append(model_id)

            # Compute new index order
            picked = set()
            new_order: List[int] = []
            for mid in desired_model_ids:
                if mid in current_ids:
                    idx = current_ids.index(mid)
                    new_order.append(idx)
                    picked.add(idx)
            # Append remaining indices (not specified) preserving order
            for i in range(len(self.controlnets)):
                if i not in picked:
                    new_order.append(i)

            if new_order == list(range(len(self.controlnets))):
                return  # Already in desired order

            def reindex(lst: List[Any]) -> List[Any]:
                return [lst[i] for i in new_order]

            self.controlnets = reindex(self.controlnets)
            self.controlnet_images = reindex(self.controlnet_images)
            self.controlnet_scales = reindex(self.controlnet_scales)
            self.preprocessors = reindex(self.preprocessors)
            self.enabled_list = reindex(self.enabled_list)

    def get_current_config(self) -> List[Dict[str, Any]]:
        cfg: List[Dict[str, Any]] = []
        with self._collections_lock:
            for i, cn in enumerate(self.controlnets):
                model_id = getattr(cn, 'model_id', f'controlnet_{i}')
                scale = self.controlnet_scales[i] if i < len(self.controlnet_scales) else 1.0
                preproc_params = getattr(self.preprocessors[i], 'params', {}) if i < len(self.preprocessors) and self.preprocessors[i] else {}
                cfg.append({
                    'model_id': model_id,
                    'conditioning_scale': scale,
                    'preprocessor_params': preproc_params,
                    'enabled': (self.enabled_list[i] if i < len(self.enabled_list) else True),
                })
        return cfg

    def prepare_frame_tensors(self, device: torch.device, dtype: torch.dtype, batch_size: int) -> None:
        """Prepare control image tensors for the current frame.
        
        This method is called once per frame to prepare all control images with the correct
        device, dtype, and batch size. This avoids redundant operations during each denoising step.
        
        Args:
            device: Target device for tensors
            dtype: Target dtype for tensors
            batch_size: Target batch size
        """
        with self._collections_lock:
            # Check if we need to re-prepare tensors
            cache_valid = (
                self._prepared_device == device and
                self._prepared_dtype == dtype and
                self._prepared_batch == batch_size and
                len(self._prepared_tensors) == len(self.controlnet_images)
            )
            
            if cache_valid:
                return
            
            # Prepare tensors for current frame
            self._prepared_tensors = []
            for img in self.controlnet_images:
                if img is None:
                    self._prepared_tensors.append(None)
                    continue
                
                # Prepare tensor with correct batch size
                prepared = img
                if prepared.dim() == 4 and prepared.shape[0] != batch_size:
                    if prepared.shape[0] == 1:
                        prepared = prepared.repeat(batch_size, 1, 1, 1)
                    else:
                        repeat_factor = max(1, batch_size // prepared.shape[0])
                        prepared = prepared.repeat(repeat_factor, 1, 1, 1)[:batch_size]
                
                # Move to correct device and dtype
                prepared = prepared.to(device=device, dtype=dtype)
                self._prepared_tensors.append(prepared)
            
            # Update cache state
            self._prepared_device = device
            self._prepared_dtype = dtype
            self._prepared_batch = batch_size

    # ---------- Internal helpers ----------
    def build_unet_hook(self) -> UnetHook:
        def _unet_hook(ctx: StepCtx) -> UnetKwargsDelta:
            # Compute residuals under lock, using only original text tokens for ControlNet encoding
            x_t = ctx.x_t_latent
            t_list = ctx.t_list

            with self._collections_lock:
                if not self.controlnets:
                    return UnetKwargsDelta()

                active_indices = [
                    i
                    for i, (cn, img, scale, enabled) in enumerate(
                        zip(
                            self.controlnets,
                            self.controlnet_images,
                            self.controlnet_scales,
                            self.enabled_list if len(self.enabled_list) == len(self.controlnets) else [True] * len(self.controlnets),
                        )
                    )
                    if cn is not None and img is not None and scale > 0 and bool(enabled)
                ]

                if not active_indices:
                    return UnetKwargsDelta()

                active_controlnets = [self.controlnets[i] for i in active_indices]
                active_images = [self.controlnet_images[i] for i in active_indices]
                active_scales = [self.controlnet_scales[i] for i in active_indices]

            # Prefer TRT engines when available by model_id
            engines_by_id: Dict[str, Any] = {}
            try:
                if hasattr(self._stream, 'controlnet_engines') and isinstance(self._stream.controlnet_engines, list):
                    for eng in self._stream.controlnet_engines:
                        mid = getattr(eng, 'model_id', None)
                        if mid:
                            engines_by_id[mid] = eng
            except Exception:
                pass

            # Use original text token window only for ControlNet encoding
            # Detect expected text length from UNet config if available; fallback to 77
            expected_text_len = 77
            try:
                if hasattr(self._stream.unet, 'config') and hasattr(self._stream.unet.config, 'cross_attention_dim'):
                    # For SDXL TRT with IPAdapter baked, engine may expect 77+num_image_tokens for encoder_hidden_states
                    # However, ControlNet expects just the text portion. Slice accordingly.
                    expected_text_len = 77
            except Exception:
                pass
            encoder_hidden_states = self._stream.prompt_embeds[:, :expected_text_len, :]

            base_kwargs: Dict[str, Any] = {
                'sample': x_t,
                'timestep': t_list,
                'encoder_hidden_states': encoder_hidden_states,
                'return_dict': False,
            }

            down_samples_list: List[List[torch.Tensor]] = []
            mid_samples_list: List[torch.Tensor] = []

            # Optionally prepare tensors for this frame (used by other code paths)
            try:
                if (self._prepared_device != x_t.device or
                    self._prepared_dtype != x_t.dtype or
                    self._prepared_batch != x_t.shape[0]):
                    self.prepare_frame_tensors(x_t.device, x_t.dtype, x_t.shape[0])
            except Exception:
                pass

            # Helper: run sequentially (baseline, safest for PyTorch/xformers)
            def run_sequential():
                local_down: List[List[torch.Tensor]] = []
                local_mid: List[torch.Tensor] = []
                for cn, img, scale in zip(active_controlnets, active_images, active_scales):
                    # Swap to TRT engine if compiled and available for this model_id
                    try:
                        model_id = getattr(cn, 'model_id', None)
                        if model_id and model_id in engines_by_id:
                            cn = engines_by_id[model_id]
                    except Exception:
                        pass
                    current_img = img
                    if current_img is None:
                        continue
                    try:
                        main_batch = x_t.shape[0]
                        if current_img.dim() == 4 and current_img.shape[0] != main_batch:
                            if current_img.shape[0] == 1:
                                current_img = current_img.repeat(main_batch, 1, 1, 1)
                            else:
                                repeat_factor = max(1, main_batch // current_img.shape[0])
                                current_img = current_img.repeat(repeat_factor, 1, 1, 1)
                        current_img = current_img.to(device=x_t.device, dtype=x_t.dtype)
                    except Exception:
                        pass
                    kwargs = base_kwargs.copy()
                    kwargs['controlnet_cond'] = current_img
                    kwargs['conditioning_scale'] = float(scale)
                    try:
                        if getattr(self._stream, 'is_sdxl', False) and ctx.sdxl_cond is not None:
                            kwargs['added_cond_kwargs'] = ctx.sdxl_cond
                    except Exception:
                        pass
                    try:
                        if hasattr(cn, 'engine') and hasattr(cn, 'stream'):
                            ds, ms = cn(
                                sample=kwargs['sample'],
                                timestep=kwargs['timestep'],
                                encoder_hidden_states=kwargs['encoder_hidden_states'],
                                controlnet_cond=kwargs['controlnet_cond'],
                                conditioning_scale=float(scale),
                                **({} if 'added_cond_kwargs' not in kwargs else kwargs['added_cond_kwargs'])
                            )
                        else:
                            ds, ms = cn(**kwargs)
                        local_down.append(ds)
                        local_mid.append(ms)
                    except Exception:
                        continue
                return local_down, local_mid

            # Bounded parallelism for ControlNet forwards
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Cap parallel CN based on cached env override or count
            max_par = self._max_parallel_controlnets if self._max_parallel_controlnets > 0 else len(active_controlnets)

            # Fast path: single active ControlNet → run inline, no thread pool or extra CUDA stream creation
            if len(active_controlnets) == 1:
                cn = active_controlnets[0]
                current_img = active_images[0]
                scale = active_scales[0]
                try:
                    model_id = getattr(cn, 'model_id', None)
                    if model_id and model_id in engines_by_id:
                        cn = engines_by_id[model_id]
                except Exception:
                    pass
                if current_img is not None:
                    try:
                        main_batch = x_t.shape[0]
                        if current_img.dim() == 4 and current_img.shape[0] != main_batch:
                            if current_img.shape[0] == 1:
                                current_img = current_img.repeat(main_batch, 1, 1, 1)
                            else:
                                repeat_factor = max(1, main_batch // current_img.shape[0])
                                current_img = current_img.repeat(repeat_factor, 1, 1, 1)
                        current_img = current_img.to(device=x_t.device, dtype=x_t.dtype)
                    except Exception:
                        pass
                    kwargs = base_kwargs.copy()
                    kwargs['controlnet_cond'] = current_img
                    kwargs['conditioning_scale'] = float(scale)
                    try:
                        if getattr(self._stream, 'is_sdxl', False) and ctx.sdxl_cond is not None:
                            kwargs['added_cond_kwargs'] = ctx.sdxl_cond
                    except Exception:
                        pass
                    try:
                        if hasattr(cn, 'engine') and hasattr(cn, 'stream'):
                            ds, ms = cn(
                                sample=kwargs['sample'],
                                timestep=kwargs['timestep'],
                                encoder_hidden_states=kwargs['encoder_hidden_states'],
                                controlnet_cond=kwargs['controlnet_cond'],
                                conditioning_scale=float(scale),
                                **({} if 'added_cond_kwargs' not in kwargs else kwargs['added_cond_kwargs'])
                            )
                        else:
                            ds, ms = cn(**kwargs)
                        down_samples_list.append(ds)
                        mid_samples_list.append(ms)
                    except Exception:
                        pass
                # Build delta (handles empty gracefully below)
                if not down_samples_list:
                    return UnetKwargsDelta()
                return UnetKwargsDelta(
                    down_block_additional_residuals=down_samples_list[0],
                    mid_block_additional_residual=mid_samples_list[0],
                )

            # If any active CN is PyTorch (no engine.stream), prefer sequential for correctness on xformers
            try:
                all_trt = True
                for cn in active_controlnets:
                    mid = getattr(cn, 'model_id', None)
                    if mid and mid in engines_by_id:
                        cn = engines_by_id[mid]
                    if not (hasattr(cn, 'engine') and hasattr(cn, 'stream')):
                        all_trt = False
                        break
            except Exception:
                all_trt = False

            if not all_trt:
                seq_down, seq_mid = run_sequential()
                down_samples_list.extend(seq_down)
                mid_samples_list.extend(seq_mid)
                if not down_samples_list:
                    return UnetKwargsDelta()
                if len(down_samples_list) == 1:
                    return UnetKwargsDelta(
                        down_block_additional_residuals=down_samples_list[0],
                        mid_block_additional_residual=mid_samples_list[0],
                    )
                merged_down = down_samples_list[0]
                merged_mid = mid_samples_list[0]
                for ds, ms in zip(down_samples_list[1:], mid_samples_list[1:]):
                    for j in range(len(merged_down)):
                        merged_down[j] = merged_down[j] + ds[j]
                    merged_mid = merged_mid + ms
                return UnetKwargsDelta(
                    down_block_additional_residuals=merged_down,
                    mid_block_additional_residual=merged_mid,
                )

            tasks = []
            results: List[Tuple[int, Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[torch.cuda.Stream]]] = []

            def run_one(idx: int, global_idx: int, cn_model: Any, img_tensor: torch.Tensor, scale_val: float):
                cn_local = cn_model
                # Swap to TRT engine if compiled and available for this model_id
                try:
                    model_id_local = getattr(cn_local, 'model_id', None)
                    if model_id_local and model_id_local in engines_by_id:
                        cn_local = engines_by_id[model_id_local]
                except Exception:
                    pass

                current_img_local = img_tensor
                if current_img_local is None:
                    return (idx, None, None)

                # Ensure control image batch matches latent batch; match device/dtype
                try:
                    main_batch = x_t.shape[0]
                    if current_img_local.dim() == 4 and current_img_local.shape[0] != main_batch:
                        if current_img_local.shape[0] == 1:
                            current_img_local = current_img_local.repeat(main_batch, 1, 1, 1)
                        else:
                            repeat_factor = max(1, main_batch // current_img_local.shape[0])
                            current_img_local = current_img_local.repeat(repeat_factor, 1, 1, 1)
                    current_img_local = current_img_local.to(device=x_t.device, dtype=x_t.dtype)
                except Exception:
                    pass

                local_kwargs = base_kwargs.copy()
                local_kwargs['controlnet_cond'] = current_img_local
                local_kwargs['conditioning_scale'] = float(scale_val)
                try:
                    if getattr(self._stream, 'is_sdxl', False) and ctx.sdxl_cond is not None:
                        local_kwargs['added_cond_kwargs'] = ctx.sdxl_cond
                except Exception:
                    pass

                try:
                    if hasattr(cn_local, 'engine') and hasattr(cn_local, 'stream'):
                        # TRT engine path: engine has its own CUDA stream; just call
                        down_s, mid_s = cn_local(
                            sample=local_kwargs['sample'],
                            timestep=local_kwargs['timestep'],
                            encoder_hidden_states=local_kwargs['encoder_hidden_states'],
                            controlnet_cond=local_kwargs['controlnet_cond'],
                            conditioning_scale=float(scale_val),
                            **({} if 'added_cond_kwargs' not in local_kwargs else local_kwargs['added_cond_kwargs'])
                        )
                        # Engine call synchronizes internally; no stream to wait on
                        return (idx, down_s, mid_s, None)
                    else:
                        # PyTorch path: use a per-call CUDA stream for concurrency
                        # Lazily create/reuse a dedicated stream for this controlnet index
                        stream_obj = self._pt_cn_streams[global_idx]
                        if stream_obj is None:
                            stream_obj = torch.cuda.Stream(device=x_t.device)
                            self._pt_cn_streams[global_idx] = stream_obj
                        with torch.cuda.stream(stream_obj):
                            down_s, mid_s = cn_local(**local_kwargs)
                        # Do not synchronize here; main thread will wait on stream before use
                        return (idx, down_s, mid_s, stream_obj)
                except Exception as e:
                    import traceback
                    __import__('logging').getLogger(__name__).error("ControlNetModule: run_one forward failed: %s", e)
                    __import__('logging').getLogger(__name__).error(traceback.format_exc())
                    return (idx, None, None, None)

            # Submit tasks in bounded thread pool
            desired_workers = max(1, min(max_par, len(active_controlnets)))
            # (Re)create persistent executor only when worker count changes
            if self._executor is None or self._executor_workers != desired_workers:
                if self._executor is not None:
                    try:
                        self._executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                self._executor = ThreadPoolExecutor(max_workers=desired_workers)
                self._executor_workers = desired_workers

            ex = self._executor
            # Map active sub-index to global controlnet index to reuse per-cn streams
            for sub_i, (cn_i, img_i, sc_i) in enumerate(zip(active_controlnets, active_images, active_scales)):
                global_i = active_indices[sub_i]
                tasks.append(ex.submit(run_one, sub_i, global_i, cn_i, img_i, sc_i))
            for fut in as_completed(tasks):
                idx, ds, ms, s = fut.result()
                if ds is not None and ms is not None:
                    results.append((idx, ds, ms, s))

            if not results:
                return UnetKwargsDelta()

            # Restore original order
            results.sort(key=lambda x: x[0])
            # Ensure default stream waits on any per-CN PyTorch streams before using tensors
            default_stream = torch.cuda.current_stream(device=x_t.device)
            for _, ds, ms, s in results:
                if isinstance(s, torch.cuda.Stream):
                    default_stream.wait_stream(s)
                down_samples_list.append(ds)  # type: ignore[arg-type]
                mid_samples_list.append(ms)   # type: ignore[arg-type]

            if not down_samples_list:
                return UnetKwargsDelta()

            if len(down_samples_list) == 1:
                return UnetKwargsDelta(
                    down_block_additional_residuals=down_samples_list[0],
                    mid_block_additional_residual=mid_samples_list[0],
                )

            # Merge multiple ControlNet residuals
            merged_down = down_samples_list[0]
            merged_mid = mid_samples_list[0]
            for ds, ms in zip(down_samples_list[1:], mid_samples_list[1:]):
                for j in range(len(merged_down)):
                    merged_down[j] = merged_down[j] + ds[j]
                merged_mid = merged_mid + ms

            return UnetKwargsDelta(
                down_block_additional_residuals=merged_down,
                mid_block_additional_residual=merged_mid,
            )

        return _unet_hook

    def _prepare_control_image(self, control_image: Union[str, Any, torch.Tensor], preprocessor: Optional[Any]) -> torch.Tensor:
        if self._preprocessing_orchestrator is None:
            raise RuntimeError("ControlNetModule: preprocessing orchestrator is not initialized")
        # Reuse orchestrator API used by BaseControlNetPipeline
        images = self._preprocessing_orchestrator.process_control_images_sync(
            control_image=control_image,
            preprocessors=[preprocessor],
            scales=[1.0],
            stream_width=self._stream.width,
            stream_height=self._stream.height,
            index=0,
        )
        # API returns a list; pick first if present
        return images[0] if images else None

    def _load_pytorch_controlnet_model(self, model_id: str) -> ControlNetModel:
        from pathlib import Path
        try:
            if Path(model_id).exists():
                controlnet = ControlNetModel.from_pretrained(
                    model_id, torch_dtype=self.dtype, local_files_only=True
                )
            else:
                if "/" in model_id and model_id.count("/") > 1:
                    parts = model_id.split("/")
                    repo_id = "/".join(parts[:2])
                    subfolder = "/".join(parts[2:])
                    controlnet = ControlNetModel.from_pretrained(
                        repo_id, subfolder=subfolder, torch_dtype=self.dtype
                    )
                else:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id, torch_dtype=self.dtype
                    )
            controlnet = controlnet.to(device=self.device, dtype=self.dtype)
            # Track model_id for updater diffing
            try:
                setattr(controlnet, 'model_id', model_id)
            except Exception:
                pass
            return controlnet
        except Exception as e:
            import logging, traceback
            logger = logging.getLogger(__name__)
            logger.error(f"ControlNetModule: failed to load model '{model_id}': {e}")
            logger.error(traceback.format_exc())
            raise

