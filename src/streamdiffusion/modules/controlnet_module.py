from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models import ControlNetModel

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
        # Attach facade methods expected by existing wrapper/demo code
        setattr(stream, 'update_control_image_efficient', self.update_control_image_efficient)
        setattr(stream, 'update_controlnet_scale', self.update_controlnet_scale)
        setattr(stream, 'update_controlnet_enabled', self.update_controlnet_enabled)
        setattr(stream, 'remove_controlnet', self.remove_controlnet)
        setattr(stream, 'get_current_controlnet_config', self.get_current_config)
        # Expose controlnet collections so existing updater can find them
        setattr(stream, 'controlnets', self.controlnets)
        setattr(stream, 'controlnet_scales', self.controlnet_scales)
        setattr(stream, 'preprocessors', self.preprocessors)
        # Add shim for add_controlnet with legacy signature
        def _add_controlnet_legacy(cfg_dict: Dict[str, Any], control_image: Optional[Any] = None, immediate: bool = False) -> None:
            try:
                cfg = ControlNetConfig(
                    model_id=cfg_dict.get('model_id'),
                    preprocessor=cfg_dict.get('preprocessor'),
                    conditioning_scale=cfg_dict.get('conditioning_scale', 1.0),
                    enabled=cfg_dict.get('enabled', True),
                    preprocessor_params=cfg_dict.get('preprocessor_params'),
                )
                self.add_controlnet(cfg, control_image=control_image)
            except Exception:
                import logging, traceback
                logging.getLogger(__name__).error("ControlNetModule: add_controlnet legacy shim failed")
                logging.getLogger(__name__).error(traceback.format_exc())
        setattr(stream, 'add_controlnet', _add_controlnet_legacy)

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
            if index is not None:
                indices = [index]
            else:
                indices = list(range(len(self.controlnets)))

        # Process per-index to preserve individual preprocessors / scales
        for i in indices:
            preproc = self.preprocessors[i] if i < len(self.preprocessors) else None
            processed = self._prepare_control_image(control_image, preproc)
            with self._collections_lock:
                if i < len(self.controlnet_images):
                    self.controlnet_images[i] = processed

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
                current_img = img
                if current_img is None:
                    continue
                kwargs = base_kwargs.copy()
                kwargs['controlnet_cond'] = current_img
                kwargs['conditioning_scale'] = float(scale)
                try:
                    down_samples, mid_sample = cn(**kwargs)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"ControlNetModule: controlnet forward failed: {e}")
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

