from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from streamdiffusion.hooks import StepCtx, UnetHook, UnetKwargsDelta
from streamdiffusion.preprocessing.preprocessing_orchestrator import (
    PreprocessingOrchestrator,
)
from streamdiffusion.modules.controlnet_module import ControlNetModule, ControlNetConfig


logger = logging.getLogger(__name__)


# -----------------------------
# Minimal T2I-Adapter backbones
# -----------------------------

def _conv_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"_conv_nd: unsupported dims={dims}")


def _avg_pool_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"_avg_pool_nd: unsupported dims={dims}")


class _Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = _conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            self.op = _avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conv and x.dim() == 4:
            padding = [x.shape[2] % 2, x.shape[3] % 2]
            # type: ignore[attr-defined]
            self.op.padding = padding  # align behavior with Comfy implementation
        return self.op(x)


class _ResnetBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, down: bool, ksize: int = 3, sk: bool = False, use_conv: bool = True):
        super().__init__()
        ps = ksize // 2
        self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps) if (in_c != out_c or not sk) else None
        self.block1 = nn.Conv2d(out_c if self.in_conv is not None else in_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        self.skep = None if sk else nn.Conv2d(in_c, out_c, ksize, 1, ps)

        self.down = down
        if self.down:
            self.down_opt = _Downsample(in_c, use_conv=use_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        return h + x


class _Adapter(nn.Module):
    """
    Adapter backbone (XL-aware) that returns feature maps per stage.
    Mirrors notes/REFERENCE_REPOS/ComfyUI/comfy/t2i_adapter/adapter.py behavior.
    """

    def __init__(
        self,
        channels: List[int],
        nums_rb: int = 2,
        cin: int = 64,
        ksize: int = 3,
        sk: bool = True,
        use_conv: bool = False,
        xl: bool = False,
    ) -> None:
        super().__init__()
        self.unshuffle_amount = 16 if xl else 8
        self.xl = xl

        # Build resnet body
        self.channels = channels
        self.nums_rb = nums_rb
        body: List[nn.Module] = []
        resblock_no_downsample = [1] if self.xl else []
        resblock_downsample = [2] if self.xl else [3, 2, 1]

        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i in resblock_downsample) and (j == 0):
                    body.append(_ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                elif (i in resblock_no_downsample) and (j == 0):
                    body.append(_ResnetBlock(channels[i - 1], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    body.append(_ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, List[Optional[torch.Tensor]]]:
        # Pixel unshuffle
        x = nn.PixelUnshuffle(self.unshuffle_amount)(x)

        # Feature extraction
        features: List[Optional[torch.Tensor]] = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            if self.xl:
                features.append(None)
                if i == 0:
                    features.append(None)
                    features.append(None)
                if i == 2:
                    features.append(None)
            else:
                features.append(None)
                features.append(None)
            features.append(x)

        features = features[::-1]
        if self.xl:
            return {"input": features[1:], "middle": features[:1]}
        return {"input": features}


class _ResnetBlockLight(nn.Module):
    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        return h + x


class _Extractor(nn.Module):
    def __init__(self, in_c: int, inter_c: int, out_c: int, nums_rb: int, down: bool = False) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = nn.Sequential(*[_ResnetBlockLight(inter_c) for _ in range(nums_rb)])
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down:
            self.down_opt = _Downsample(in_c, use_conv=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)
        return x


class _AdapterLight(nn.Module):
    def __init__(self, channels: List[int], nums_rb: int = 3, cin: int = 64) -> None:
        super().__init__()
        self.unshuffle_amount = 8
        self.channels = channels
        self.nums_rb = nums_rb
        body: List[nn.Module] = []
        for i in range(len(channels)):
            if i == 0:
                body.append(_Extractor(in_c=cin, inter_c=channels[i] // 4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                body.append(_Extractor(in_c=channels[i - 1], inter_c=channels[i] // 4, out_c=channels[i], nums_rb=nums_rb, down=True))
        self.body = nn.ModuleList(body)

    def forward(self, x: torch.Tensor) -> Dict[str, List[Optional[torch.Tensor]]]:
        x = nn.PixelUnshuffle(self.unshuffle_amount)(x)
        features: List[Optional[torch.Tensor]] = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(None)
            features.append(None)
            features.append(x)
        return {"input": features[::-1]}


# -----------------------------
# T2I-Adapter module
# -----------------------------


class T2IAdapterModule(ControlNetModule):
    """
    T2I-Adapter module that reuses the ControlNetModule interface and control image
    handling end-to-end. Only T2I model loading and UNet residual construction are
    overridden.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        super().__init__(device=device, dtype=dtype)
        # Cached features per adapter (recomputed when image changes)
        self._cached_controls: List[Optional[Dict[str, List[Optional[torch.Tensor]]]]] = []
        # Stream reference set in install
        self._stream = None

    def install(self, stream) -> None:
        """Install without exposing ControlNet-specific stream attributes."""
        self._stream = stream
        self.device = stream.device
        self.dtype = stream.dtype
        if self._preprocessing_orchestrator is None:
            self._preprocessing_orchestrator = PreprocessingOrchestrator(
                device=self.device, dtype=self.dtype, max_workers=4
            )
        logger.debug("install: Installing T2I-Adapter module; device=%s dtype=%s", str(self.device), str(self.dtype))
        stream.unet_hooks.append(self.build_unet_hook())

    def add_controlnet(self, cfg: ControlNetConfig, control_image: Optional[Union[str, Any, torch.Tensor]] = None) -> None:
        """Unified add method: treat cfg.model_id as a T2I checkpoint path."""
        logger.debug("add_controlnet: Loading T2I adapter from path=%s", str(cfg.model_id))
        model = self._load_t2i_model(cfg.model_id)
        model = model.to(device=self.device, dtype=self.dtype)
        # Attach an identifier for parity with ControlNet
        try:
            setattr(model, 'model_id', cfg.model_id)
        except Exception:
            pass

        preproc = None
        if cfg.preprocessor:
            from streamdiffusion.preprocessing.processors import get_preprocessor
            preproc = get_preprocessor(cfg.preprocessor)
            # Apply provided parameters to the preprocessor instance
            if cfg.preprocessor_params:
                params = cfg.preprocessor_params or {}
                if hasattr(preproc, 'params') and isinstance(getattr(preproc, 'params'), dict):
                    preproc.params.update(params)
                for name, value in params.items():
                    try:
                        if hasattr(preproc, name):
                            setattr(preproc, name, value)
                    except Exception:
                        pass
            # Provide pipeline reference when required by certain preprocessors
            try:
                if hasattr(preproc, 'set_pipeline_ref'):
                    preproc.set_pipeline_ref(self._stream)
            except Exception:
                pass

        image_tensor: Optional[torch.Tensor] = None
        if control_image is not None and self._preprocessing_orchestrator is not None:
            logger.debug("add_controlnet: Preparing initial control image via orchestrator")
            image_tensor = super()._prepare_control_image(control_image, preproc)

        with self._collections_lock:
            # Reuse parent collections directly
            self.controlnets.append(model)
            self.controlnet_images.append(image_tensor)
            self.controlnet_scales.append(float(cfg.conditioning_scale))
            self.preprocessors.append(preproc)
            self.enabled_list.append(bool(cfg.enabled))
            self._cached_controls.append(None)
        logger.debug(
            "add_controlnet: Added T2I adapter idx=%d scale=%.4f enabled=%s has_image=%s",
            len(self.controlnets) - 1,
            float(cfg.conditioning_scale),
            str(bool(cfg.enabled)),
            str(image_tensor is not None),
        )

    # Control image updates, scale/enable toggles, reordering, and removal
    # are inherited from ControlNetModule:
    # - update_control_image_efficient
    # - update_controlnet_scale
    # - update_controlnet_enabled
    # - remove_controlnet
    # - reorder_controlnets_by_model_ids

    # -----------------
    # Internal helpers
    # -----------------

    def build_unet_hook(self) -> UnetHook:
        def _unet_hook(ctx: StepCtx) -> UnetKwargsDelta:
            try:
                logger.debug("build_unet_hook: step_index=%s x_t_shape=%s t_len=%s", str(ctx.step_index), str(tuple(ctx.x_t_latent.shape) if isinstance(ctx.x_t_latent, torch.Tensor) else None), str(len(ctx.t_list) if hasattr(ctx.t_list, '__len__') else None))
            except Exception:
                pass
            with self._collections_lock:
                if not self.controlnets:
                    logger.debug("build_unet_hook: No adapters installed; skipping")
                    return UnetKwargsDelta()

                active_indices = [
                    i
                    for i, (adp, img, scale, enabled) in enumerate(
                        zip(
                            self.controlnets,
                            self.controlnet_images,
                            self.controlnet_scales,
                            self.enabled_list if len(self.enabled_list) == len(self.controlnets) else [True] * len(self.controlnets),
                        )
                    )
                    if adp is not None and img is not None and scale > 0 and bool(enabled)
                ]

                logger.debug("build_unet_hook: total_adapters=%d active_indices=%s", len(self.controlnets), str(active_indices))
                if not active_indices:
                    logger.debug("build_unet_hook: No active adapters for this step; skipping")
                    return UnetKwargsDelta()

                active_adapters = [self.controlnets[i] for i in active_indices]
                active_images = [self.controlnet_images[i] for i in active_indices]
                active_scales = [self.controlnet_scales[i] for i in active_indices]
                cached_controls = [self._cached_controls[i] for i in active_indices]

            # Compute or reuse cached controls per adapter
            down_samples_list: List[List[torch.Tensor]] = []
            mid_samples_list: List[Optional[torch.Tensor]] = []

            for idx, (adapter, image_tensor, scale, cached) in enumerate(zip(active_adapters, active_images, active_scales, cached_controls)):
                if image_tensor is None:
                    logger.debug("build_unet_hook: adapter_local_idx=%d has no image; skipping", idx)
                    continue
                # Align with latent batch/device/dtype
                current_img = image_tensor
                try:
                    main_batch = ctx.x_t_latent.shape[0]
                    if current_img.dim() == 4 and current_img.shape[0] != main_batch:
                        if current_img.shape[0] == 1:
                            current_img = current_img.repeat(main_batch, 1, 1, 1)
                        else:
                            repeat_factor = max(1, main_batch // current_img.shape[0])
                            current_img = current_img.repeat(repeat_factor, 1, 1, 1)
                    current_img = current_img.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype)
                    logger.debug("build_unet_hook: adapter_local_idx=%d image aligned; shape=%s dtype=%s device=%s", idx, str(tuple(current_img.shape)), str(current_img.dtype), str(current_img.device))
                except Exception:
                    pass

                controls = cached
                if controls is None:
                    try:
                        adapter.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype)
                        with torch.no_grad():
                            controls = adapter(current_img)
                        logger.debug("build_unet_hook: adapter_local_idx=%d cache MISS; computed controls keys=%s", idx, ",".join(list(controls.keys())) if isinstance(controls, dict) else str(type(controls)))
                    except Exception as e:
                        logger.error(f"build_unet_hook: adapter forward failed: {e}")
                        continue
                    # Store into cache (under lock)
                    with self._collections_lock:
                        try:
                            orig_index = active_indices[idx]
                            self._cached_controls[orig_index] = controls
                        except Exception:
                            pass
                else:
                    logger.debug("build_unet_hook: adapter_local_idx=%d cache HIT", idx)

                # Extract down and middle lists
                down_list = controls.get("input", []) if isinstance(controls, dict) else []
                mid_list = controls.get("middle", None) if isinstance(controls, dict) else None
                if isinstance(mid_list, list) and len(mid_list) > 0:
                    mid_tensor = mid_list[0]
                else:
                    mid_tensor = None

                # Apply scale and normalize shapes/dtypes
                scaled_down: List[Optional[torch.Tensor]] = []
                for t in down_list:
                    if t is None:
                        scaled_down.append(t)  # keep placeholders
                    else:
                        td = t.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype) * float(scale)
                        scaled_down.append(td)
                if mid_tensor is not None:
                    mid_tensor = mid_tensor.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype) * float(scale)
                logger.debug("build_unet_hook: adapter_local_idx=%d scale=%.4f down_count=%d mid_present=%s", idx, float(scale), len(scaled_down), str(mid_tensor is not None))

                # Adapt spatial shapes to match UNet expectations (SD1.5 vs SDXL)
                try:
                    sample_h, sample_w = ctx.x_t_latent.shape[-2], ctx.x_t_latent.shape[-1]
                    is_sdxl = bool(getattr(self._stream, 'is_sdxl', False))
                    expected_factors = [1, 1, 1, 2, 2, 2, 4, 4, 4] if is_sdxl else [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
                    adapted_down: List[Optional[torch.Tensor]] = []
                    for i, ten in enumerate(scaled_down):
                        if ten is None:
                            adapted_down.append(None)
                            continue
                        if i < len(expected_factors):
                            fac = expected_factors[i]
                            exp_h = sample_h // fac
                            exp_w = sample_w // fac
                            if ten.shape[-2] != exp_h or ten.shape[-1] != exp_w:
                                import torch.nn.functional as F
                                ten = F.interpolate(ten, size=(exp_h, exp_w), mode='bilinear', align_corners=False)
                        adapted_down.append(ten)
                    scaled_down = adapted_down
                    if mid_tensor is not None:
                        mid_fac = 4 if is_sdxl else 8
                        exp_h = sample_h // mid_fac
                        exp_w = sample_w // mid_fac
                        if mid_tensor.shape[-2] != exp_h or mid_tensor.shape[-1] != exp_w:
                            import torch.nn.functional as F
                            mid_tensor = F.interpolate(mid_tensor, size=(exp_h, exp_w), mode='bilinear', align_corners=False)
                except Exception:
                    pass
                # Adjust channel dimensions to match UNet expectations per position
                try:
                    if not is_sdxl:
                        # SD1.5 typical expected channels sequence across 12 residuals
                        expected_channels = [320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280]
                    else:
                        # SDXL rough expected channels across 9 residuals
                        expected_channels = [320, 320, 320, 640, 640, 640, 1280, 1280, 1280]
                    adjusted_down: List[Optional[torch.Tensor]] = []
                    for i, ten in enumerate(scaled_down):
                        if ten is None or i >= len(expected_channels):
                            adjusted_down.append(ten)
                            continue
                        exp_ch = expected_channels[i]
                        cur_ch = ten.shape[1]
                        if cur_ch == exp_ch:
                            adjusted_down.append(ten)
                        elif cur_ch > exp_ch:
                            adjusted_down.append(ten[:, :exp_ch, :, :])
                        else:
                            pad_ch = exp_ch - cur_ch
                            pad = torch.zeros(ten.shape[0], pad_ch, ten.shape[2], ten.shape[3], device=ten.device, dtype=ten.dtype)
                            adjusted_down.append(torch.cat([ten, pad], dim=1))
                    scaled_down = adjusted_down
                except Exception:
                    pass
                # Fill missing entries so UNet receives a tensor for each intrablock residual
                try:
                    expected_len = len(expected_channels)
                    if len(scaled_down) < expected_len:
                        # pad with Nones for uniform handling
                        scaled_down = scaled_down + [None] * (expected_len - len(scaled_down))
                    group_size = 3 if not is_sdxl else 3
                    num_groups = expected_len // group_size
                    filled_down: List[torch.Tensor] = []
                    for g in range(num_groups):
                        group_indices = list(range(g * group_size, (g + 1) * group_size))
                        # pick first available tensor in the group
                        base_ten: Optional[torch.Tensor] = None
                        for gi in group_indices:
                            ten = scaled_down[gi]
                            if isinstance(ten, torch.Tensor):
                                base_ten = ten
                                break
                        # if still None, try previous group's tensor
                        if base_ten is None and filled_down:
                            base_ten = filled_down[-1]
                        # if still None (very first group), create zeros matching expected channels and spatial size
                        if base_ten is None:
                            exp_ch = expected_channels[group_indices[0]]
                            h = sample_h // ([1, 2, 4, 8][g] if not is_sdxl else [1, 2, 4][g])
                            w = sample_w // ([1, 2, 4, 8][g] if not is_sdxl else [1, 2, 4][g])
                            base_ten = torch.zeros(ctx.x_t_latent.shape[0], exp_ch, h, w, device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype)
                        # replicate for all positions in the group
                        for gi in group_indices:
                            filled_down.append(base_ten)
                    scaled_down = filled_down
                except Exception:
                    pass
                try:
                    shapes_down = [tuple(t.shape) if isinstance(t, torch.Tensor) else None for t in scaled_down]
                    logger.debug("build_unet_hook: adapter_local_idx=%d adapted down shapes=%s mid_shape=%s", idx, str(shapes_down), str(tuple(mid_tensor.shape) if isinstance(mid_tensor, torch.Tensor) else None))
                except Exception:
                    pass

                # Reduce intrablock residuals to per-block residuals by summing each group
                try:
                    group_size = 3
                    block_down: List[torch.Tensor] = []
                    num_groups = (len(scaled_down) // group_size)
                    for g in range(num_groups):
                        group = scaled_down[g * group_size:(g + 1) * group_size]
                        base: Optional[torch.Tensor] = None
                        for t in group:
                            if isinstance(t, torch.Tensor):
                                base = t if base is None else (base + t)
                        # If still None, synthesize zero tensor with expected channels/size
                        if base is None:
                            exp_ch = expected_channels[g * group_size]
                            h = sample_h // ([1, 2, 4, 8][g] if not is_sdxl else [1, 2, 4][g])
                            w = sample_w // ([1, 2, 4, 8][g] if not is_sdxl else [1, 2, 4][g])
                            base = torch.zeros(ctx.x_t_latent.shape[0], exp_ch, h, w, device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype)
                        block_down.append(base)
                    scaled_down = block_down
                    logger.debug("build_unet_hook: adapter_local_idx=%d reduced to block residuals count=%d", idx, len(scaled_down))
                except Exception:
                    pass

                down_samples_list.append(scaled_down)
                mid_samples_list.append(mid_tensor)

            if not down_samples_list:
                return UnetKwargsDelta()

            # If single adapter, return directly
            if len(down_samples_list) == 1:
                only_down = down_samples_list[0]
                only_mid = mid_samples_list[0]
                try:
                    logger.debug("build_unet_hook: single adapter: emitting down_count=%d mid_present=%s", len([t for t in only_down if t is not None]) if only_down else 0, str(only_mid is not None))
                except Exception:
                    pass
                return UnetKwargsDelta(
                    down_block_additional_residuals=only_down if only_down else None,
                    mid_block_additional_residual=only_mid if only_mid is not None else None,
                )

            # Multiple adapters: sum element-wise, preserving None placeholders
            merged_down = down_samples_list[0]
            merged_mid = mid_samples_list[0]
            for ds, ms in zip(down_samples_list[1:], mid_samples_list[1:]):
                # Down blocks
                max_len = max(len(merged_down), len(ds))
                if len(merged_down) < max_len:
                    merged_down = merged_down + [None] * (max_len - len(merged_down))
                if len(ds) < max_len:
                    ds = ds + [None] * (max_len - len(ds))
                for j in range(max_len):
                    a = merged_down[j]
                    b = ds[j]
                    if a is None:
                        merged_down[j] = b
                    elif b is None:
                        merged_down[j] = a
                    else:
                        merged_down[j] = a + b
                # Mid block
                if merged_mid is None:
                    merged_mid = ms
                elif ms is None:
                    merged_mid = merged_mid
                else:
                    merged_mid = merged_mid + ms

            filtered_down = merged_down
            try:
                logger.debug("build_unet_hook: merged adapters: emitting down_count=%d mid_present=%s", len([t for t in filtered_down if t is not None]) if filtered_down else 0, str(merged_mid is not None))
            except Exception:
                pass
            return UnetKwargsDelta(
                down_block_additional_residuals=filtered_down if filtered_down else None,
                mid_block_additional_residual=merged_mid if merged_mid is not None else None,
            )

        return _unet_hook

    def _load_t2i_model(self, model_path: str) -> nn.Module:
        """
        Load a T2I-Adapter backbone from a torch checkpoint path.
        Supports two layouts:
        - AdapterLight: contains key 'body.0.in_conv.weight'
        - Adapter (XL-aware): contains key 'conv_in.weight'
        """
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"_load_t2i_model: model not found at {model_path}")

        # Load state dict (safe_load semantics expected from comfy utils; use torch.load here)
        logger.debug("_load_t2i_model: Loading checkpoint from %s", str(p))
        sd = torch.load(str(p), map_location="cpu")

        # Some checkpoints may wrap under 'adapter'
        if isinstance(sd, dict) and 'adapter' in sd and isinstance(sd['adapter'], dict):
            sd = sd['adapter']

        if not isinstance(sd, dict):
            raise ValueError("_load_t2i_model: unexpected checkpoint format (expected state dict)")

        keys = list(sd.keys())
        # AdapterLight branch
        if any(k.endswith("body.0.in_conv.weight") for k in keys):
            logger.debug("_load_t2i_model: Detected AdapterLight layout")
            cin = sd['body.0.in_conv.weight'].shape[1]
            channels = [320, 640, 1280, 1280]
            model = _AdapterLight(channels=channels, nums_rb=4, cin=cin)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"_load_t2i_model: missing keys (light): {missing}")
            if unexpected:
                logger.debug(f"_load_t2i_model: unexpected keys (light): {unexpected}")
            return model

        # Adapter (XL-aware) branch
        if 'conv_in.weight' in sd:
            logger.debug("_load_t2i_model: Detected Adapter (XL-aware) layout")
            cin = sd['conv_in.weight'].shape[1]
            channel = sd['conv_in.weight'].shape[0]
            ksize = sd.get('body.0.block2.weight', torch.empty(1, 1, 3, 3)).shape[2]
            # Heuristic used by Comfy: presence of down_opt convs toggles use_conv
            use_conv = any(k.endswith('down_opt.op.weight') for k in keys)
            xl = cin in (256, 768)
            channels = [channel, channel * 2, channel * 4, channel * 4][:4]
            model = _Adapter(cin=cin, channels=channels, nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv, xl=xl)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"_load_t2i_model: missing keys: {missing}")
            if unexpected:
                logger.debug(f"_load_t2i_model: unexpected keys: {unexpected}")
            return model

        raise ValueError("_load_t2i_model: unsupported T2I-Adapter checkpoint format")


