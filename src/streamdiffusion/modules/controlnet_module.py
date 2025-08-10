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


@dataclass
class ControlNetConfig:
    model_id: str
    preprocessor: Optional[str] = None
    conditioning_scale: float = 1.0
    enabled: bool = True
    preprocessor_params: Optional[Dict[str, Any]] = None


class ControlNetModule:
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

    # ---------- Public API (used by wrapper in a later step) ----------
    def install(self, stream) -> None:
        self._stream = stream
        self.device = stream.device
        self.dtype = stream.dtype
        if self._preprocessing_orchestrator is None:
            self._preprocessing_orchestrator = PreprocessingOrchestrator(
                device=self.device, dtype=self.dtype, max_workers=4
            )
        # Register UNet hook
        stream.unet_hooks.append(self.build_unet_hook())
        # Expose controlnet collections so existing updater can find them
        setattr(stream, 'controlnets', self.controlnets)
        setattr(stream, 'controlnet_scales', self.controlnet_scales)
        setattr(stream, 'preprocessors', self.preprocessors)

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

        image_tensor: Optional[torch.Tensor] = None
        if control_image is not None and self._preprocessing_orchestrator is not None:
            image_tensor = self._prepare_control_image(control_image, preproc)

        with self._collections_lock:
            self.controlnets.append(model)
            self.controlnet_images.append(image_tensor)
            self.controlnet_scales.append(float(cfg.conditioning_scale))
            self.preprocessors.append(preproc)
            self.enabled_list.append(bool(cfg.enabled))

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
            return

        # Decide whether to enable inter-frame pipelining (disabled if Feedback preprocessor is active)
        allow_pipelining = True
        #TODO FIX THIS
        try:
            from streamdiffusion.preprocessing.processors.feedback import FeedbackPreprocessor  # type: ignore
            for prep in preprocessors:
                if isinstance(prep, FeedbackPreprocessor):
                    allow_pipelining = False
                    break
        except Exception:
            # Fallback on class name check without importing
            for prep in preprocessors:
                if prep is not None and prep.__class__.__name__.lower().startswith('feedback'):
                    allow_pipelining = False
                    break

        # Run processing with intraframe parallelism always; use interframe pipelining when allowed
        if allow_pipelining:
            processed_images = self._preprocessing_orchestrator.process_control_images_pipelined(
                control_image=control_image,
                preprocessors=preprocessors,
                scales=scales,
                stream_width=self._stream.width,
                stream_height=self._stream.height,
            )
        else:
            processed_images = self._preprocessing_orchestrator.process_control_images_sync(
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

            for cn, img, scale in zip(active_controlnets, active_images, active_scales):
                # Swap to TRT engine if compiled and available for this model_id
                try:
                    model_id = getattr(cn, 'model_id', None)
                    if model_id and model_id in engines_by_id:
                        cn = engines_by_id[model_id]
                        # Swapped to TRT engine
                except Exception:
                    pass
                current_img = img
                if current_img is None:
                    continue
                # Ensure control image batch matches latent batch for TRT engines
                try:
                    main_batch = x_t.shape[0]
                    if current_img.dim() == 4 and current_img.shape[0] != main_batch:
                        if current_img.shape[0] == 1:
                            current_img = current_img.repeat(main_batch, 1, 1, 1)
                        else:
                            repeat_factor = max(1, main_batch // current_img.shape[0])
                            current_img = current_img.repeat(repeat_factor, 1, 1, 1)
                    # Align device/dtype with latent for engine inputs
                    current_img = current_img.to(device=x_t.device, dtype=x_t.dtype)
                except Exception:
                    pass
                kwargs = base_kwargs.copy()
                kwargs['controlnet_cond'] = current_img
                kwargs['conditioning_scale'] = float(scale)
                # For SDXL ControlNet, pass added_cond_kwargs (text_embeds, time_ids)
                try:
                    if getattr(self._stream, 'is_sdxl', False) and ctx.sdxl_cond is not None:
                        kwargs['added_cond_kwargs'] = ctx.sdxl_cond
                except Exception:
                    pass
                # For SDXL, preparing CN forward
                # Route to TensorRT engine if this ControlNet is an engine wrapper
                try:
                    if hasattr(cn, 'engine') and hasattr(cn, 'stream'):
                        # Using TRT ControlNet engine
                        # Engine expects positional args and scalar conditioning_scale
                        down_samples, mid_sample = cn(
                            sample=kwargs['sample'],
                            timestep=kwargs['timestep'],
                            encoder_hidden_states=kwargs['encoder_hidden_states'],
                            controlnet_cond=kwargs['controlnet_cond'],
                            conditioning_scale=float(scale),
                            **({} if 'added_cond_kwargs' not in kwargs else kwargs['added_cond_kwargs'])
                        )
                    else:
                        down_samples, mid_sample = cn(**kwargs)
                except Exception as e:
                    import traceback
                    __import__('logging').getLogger(__name__).error("ControlNetModule: controlnet forward failed: %s", e)
                    try:
                        __import__('logging').getLogger(__name__).error("ControlNetModule: kwargs_summary: keys=%s, cond_shape=%s, img_shape=%s, scale=%s, is_sdxl=%s",
                                     list(kwargs.keys()),
                                     (tuple(kwargs.get('encoder_hidden_states').shape) if isinstance(kwargs.get('encoder_hidden_states'), torch.Tensor) else None),
                                     (tuple(current_img.shape) if isinstance(current_img, torch.Tensor) else None),
                                     scale,
                                     getattr(self._stream, 'is_sdxl', False))
                    except Exception:
                        pass
                    __import__('logging').getLogger(__name__).error(traceback.format_exc())
                    continue
                down_samples_list.append(down_samples)
                mid_samples_list.append(mid_sample)

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

