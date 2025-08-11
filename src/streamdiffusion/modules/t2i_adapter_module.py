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

        stream.unet_hooks.append(self.build_unet_hook())

    def add_controlnet(self, cfg: ControlNetConfig, control_image: Optional[Union[str, Any, torch.Tensor]] = None) -> None:
        """Unified add method: treat cfg.model_id as a T2I checkpoint path."""

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

            image_tensor = super()._prepare_control_image(control_image, preproc)

        with self._collections_lock:
            # Reuse parent collections directly
            self.controlnets.append(model)
            self.controlnet_images.append(image_tensor)
            self.controlnet_scales.append(float(cfg.conditioning_scale))
            self.preprocessors.append(preproc)
            self.enabled_list.append(bool(cfg.enabled))
            self._cached_controls.append(None)


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

            with self._collections_lock:
                if not self.controlnets:
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

                if not active_indices:
                    return UnetKwargsDelta()

                active_adapters = [self.controlnets[i] for i in active_indices]
                active_images = [self.controlnet_images[i] for i in active_indices]
                active_scales = [self.controlnet_scales[i] for i in active_indices]
                cached_controls = [self._cached_controls[i] for i in active_indices]

            # Compute or reuse cached controls per adapter
            down_samples_list: List[List[torch.Tensor]] = []
            mid_samples_list: List[Optional[torch.Tensor]] = []
            # Track intrablock vs block-mode lists for merging
            # PyTorch path: nested per-block intrablock residuals: List[List[Tensor]]
            intra_down_lists: List[List[List[torch.Tensor]]] = []
            # TRT path: per-block aggregated residual: List[Tensor]
            block_down_lists: List[List[torch.Tensor]] = []

            for idx, (adapter, image_tensor, scale, cached) in enumerate(zip(active_adapters, active_images, active_scales, cached_controls)):
                if image_tensor is None:
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
                except Exception:
                    pass

                controls = cached
                if controls is None:
                    try:
                        adapter.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype)
                        with torch.no_grad():
                            controls = adapter(current_img)

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
                    pass

                # Extract down and middle lists
                down_list = controls.get("input", []) if isinstance(controls, dict) else []
                mid_list = controls.get("middle", None) if isinstance(controls, dict) else None
                if isinstance(mid_list, list) and len(mid_list) > 0:
                    mid_tensor = mid_list[0]
                else:
                    mid_tensor = None

                # Apply scale and normalize shapes/dtypes (log raw candidate shapes)
                scaled_candidates: List[torch.Tensor] = []
                for t in down_list:
                    if t is None:
                        continue
                    td = t.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype) * float(scale)
                    scaled_candidates.append(td)
                if mid_tensor is not None:
                    mid_tensor = mid_tensor.to(device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype) * float(scale)

                # verbose candidate shapes logging removed in cleanup

                # Introspect UNet down path to compute expected intrablock slots (shapes and channels)
                try:
                    sample_h, sample_w = int(ctx.x_t_latent.shape[-2]), int(ctx.x_t_latent.shape[-1])
                    unet = getattr(self._stream, 'unet', None)
                    down_blocks = getattr(unet, 'down_blocks', []) if unet is not None else []
                    expected_slots: List[Tuple[int, int, int, int, str]] = []  # (h,w,exp_ch,block_idx,block_class)
                    h, w = sample_h, sample_w
                    resnets_per_block: List[int] = []
                    for b_i, block in enumerate(down_blocks):
                        block_cls = block.__class__.__name__
                        resnets = getattr(block, 'resnets', [])
                        downsamplers = getattr(block, 'downsamplers', None)
                        downsamplers_len = 0
                        try:
                            downsamplers_len = len(downsamplers) if hasattr(downsamplers, '__len__') else (1 if downsamplers is not None else 0)
                        except Exception:
                            downsamplers_len = 0

                        resnets_per_block.append(len(resnets))
                        for r_i, res in enumerate(resnets):
                            exp_ch = getattr(res, 'out_channels', None)
                            if exp_ch is None:
                                try:
                                    exp_ch = getattr(res, 'conv2', getattr(res, 'conv_shortcut', None)).out_channels  # type: ignore
                                except Exception:
                                    exp_ch = scaled_candidates[0].shape[1] if scaled_candidates else 0
                            expected_slots.append((h, w, int(exp_ch), b_i, block_cls))
                        if downsamplers_len > 0:
                            h = max(1, h // 2)
                            w = max(1, w // 2)

                except Exception as e:
                    logger.error(f"build_unet_hook: UNet introspection failed: {e}")
                    expected_slots = []

                # Map candidates to expected slots by HxW match, then adjust channels
                import torch.nn.functional as F
                # Reuse-by-resolution: reuse a candidate for multiple resnets at same resolution
                resolution_to_candidate_idx: Dict[Tuple[int, int], int] = {}
                intrablock_down: List[torch.Tensor] = []
                for slot_i, (eh, ew, ech, b_i, b_cls) in enumerate(expected_slots):
                    key = (int(eh), int(ew))
                    chosen_j = resolution_to_candidate_idx.get(key, None)
                    if chosen_j is None:
                        for j, cand in enumerate(scaled_candidates):
                            if int(cand.shape[-2]) == int(eh) and int(cand.shape[-1]) == int(ew):
                                chosen_j = j
                                resolution_to_candidate_idx[key] = j
                                break
                    if chosen_j is None and scaled_candidates:
                        diffs = [abs(int(c.shape[-2]) - int(eh)) + abs(int(c.shape[-1]) - int(ew)) for c in scaled_candidates]
                        sorted_idxs = sorted([(d, j) for j, d in enumerate(diffs)], key=lambda x: x[0])
                        if sorted_idxs:
                            chosen_j = sorted_idxs[0][1]
                            resolution_to_candidate_idx[key] = chosen_j
                    if chosen_j is None:
                        logger.error("build_unet_hook: adapter_local_idx=%d slot=%d(b%d:%s) no candidate for expected (H=%d,W=%d,C=%d)", idx, slot_i, b_i, b_cls, eh, ew, ech)
                        continue
                    cand = scaled_candidates[chosen_j]
                    pre_shape = tuple(cand.shape)
                    if int(cand.shape[-2]) != int(eh) or int(cand.shape[-1]) != int(ew):
                        cand = F.interpolate(cand, size=(int(eh), int(ew)), mode='bilinear', align_corners=False)
                    cur_ch = int(cand.shape[1])
                    if ech > 0 and cur_ch != ech:
                        if cur_ch > ech:
                            cand = cand[:, :ech, :, :]
                        else:
                            pad = torch.zeros(cand.shape[0], ech - cur_ch, cand.shape[2], cand.shape[3], device=cand.device, dtype=cand.dtype)
                            cand = torch.cat([cand, pad], dim=1)
                    post_shape = tuple(cand.shape)
                    intrablock_down.append(cand)
                    # per-slot mapping log removed in cleanup
                

                # Group intrablock residuals per block in encounter order
                intrablock_by_block: List[List[torch.Tensor]] = [[] for _ in range(len(resnets_per_block))]
                block_counts: Dict[int, int] = {b_i: 0 for b_i in range(len(resnets_per_block))}
                for slot_i, slot in enumerate(expected_slots):
                    b_i = slot[3]
                    if slot_i < len(intrablock_down):
                        intrablock_by_block[b_i].append(intrablock_down[slot_i])
                        block_counts[b_i] += 1

                # per-block tensor shapes logging removed in cleanup
                # Compute per-block list for TensorRT engines; keep intrablock for PyTorch
                try:
                    is_trt = hasattr(self._stream, 'unet') and hasattr(self._stream.unet, 'engine') and hasattr(self._stream.unet, 'stream')
                except Exception:
                    is_trt = False

                if is_trt:
                    try:
                        block_down: List[torch.Tensor] = []
                        for b_i, group in enumerate(intrablock_by_block):
                            base: Optional[torch.Tensor] = None
                            for t in group:
                                if isinstance(t, torch.Tensor):
                                    base = t if base is None else (base + t)
                            if base is None:
                                # Synthesize zero tensor matching group first element
                                ref = group[0]
                                exp_ch = int(ref.shape[1])
                                h = int(ref.shape[-2])
                                w = int(ref.shape[-1])
                                base = torch.zeros(ctx.x_t_latent.shape[0], exp_ch, h, w, device=ctx.x_t_latent.device, dtype=ctx.x_t_latent.dtype)
                            block_down.append(base)
                        block_down_lists.append(block_down)

                    except Exception:
                        pass
                else:
                    intra_down_lists.append(intrablock_by_block)


                # Keep for mid usage alignment (should be None for adapter path)
                down_samples_list.append(intrablock_down)
                mid_samples_list.append(mid_tensor)

            if not down_samples_list:
                return UnetKwargsDelta()

            # If single adapter, return directly (choose intrablock vs block based on engine)
            if len(down_samples_list) == 1:
                try:
                    is_trt = hasattr(self._stream, 'unet') and hasattr(self._stream.unet, 'engine') and hasattr(self._stream.unet, 'stream')
                except Exception:
                    is_trt = False
                only_mid = None  # Adapter path: no mid residual
                if not is_trt and len(intra_down_lists) == 1:
                    only_intra = intra_down_lists[0]

                    return UnetKwargsDelta(
                        down_intrablock_additional_residuals=only_intra if only_intra else None,
                        mid_block_additional_residual=None,
                    )
                else:
                    only_block = block_down_lists[0] if block_down_lists else down_samples_list[0]

                    return UnetKwargsDelta(
                        down_block_additional_residuals=only_block if only_block else None,
                        mid_block_additional_residual=None,
                    )

            # Multiple adapters: sum element-wise, preserving None placeholders
            # Merge multiple adapters element-wise; pick mode by engine
            try:
                is_trt = hasattr(self._stream, 'unet') and hasattr(self._stream.unet, 'engine') and hasattr(self._stream.unet, 'stream')
            except Exception:
                is_trt = False

            if is_trt:
                lists_to_merge_blocks = block_down_lists
                merged_down = lists_to_merge_blocks[0]
                for ds in lists_to_merge_blocks[1:]:
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

                return UnetKwargsDelta(
                    down_block_additional_residuals=merged_down if merged_down else None,
                    mid_block_additional_residual=None,
                )
            else:
                lists_to_merge_nested = intra_down_lists  # List[adapters][blocks][resnets]
                # Initialize merged structure with first adapter
                merged_nested: List[List[torch.Tensor]] = [blk.copy() for blk in lists_to_merge_nested[0]]
                # Merge others
                for intra in lists_to_merge_nested[1:]:
                    # Resize to max blocks
                    if len(merged_nested) < len(intra):
                        merged_nested += [[] for _ in range(len(intra) - len(merged_nested))]
                    for b_i in range(len(intra)):
                        # Ensure per-resnet alignment
                        if b_i >= len(merged_nested):
                            merged_nested.append([])
                        a_block = merged_nested[b_i]
                        b_block = intra[b_i]
                        max_r = max(len(a_block), len(b_block))
                        # Normalize lengths by filling with None-equivalent zeros of proper shape if needed
                        # Since we cannot infer shapes safely here, prefer using existing tensors when available
                        new_block: List[torch.Tensor] = []
                        for r in range(max_r):
                            a_t = a_block[r] if r < len(a_block) else None
                            b_t = b_block[r] if r < len(b_block) else None
                            if a_t is None:
                                new_block.append(b_t)
                            elif b_t is None:
                                new_block.append(a_t)
                            else:
                                new_block.append(a_t + b_t)
                        merged_nested[b_i] = new_block

                return UnetKwargsDelta(
                    down_intrablock_additional_residuals=merged_nested if merged_nested else None,
                    mid_block_additional_residual=None,
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

        sd = torch.load(str(p), map_location="cpu")

        # Some checkpoints may wrap under 'adapter'
        if isinstance(sd, dict) and 'adapter' in sd and isinstance(sd['adapter'], dict):
            sd = sd['adapter']

        if not isinstance(sd, dict):
            raise ValueError("_load_t2i_model: unexpected checkpoint format (expected state dict)")

        keys = list(sd.keys())
        # AdapterLight branch
        if any(k.endswith("body.0.in_conv.weight") for k in keys):

            cin = sd['body.0.in_conv.weight'].shape[1]
            channels = [320, 640, 1280, 1280]
            model = _AdapterLight(channels=channels, nums_rb=4, cin=cin)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"_load_t2i_model: missing keys (light): {missing}")
            if unexpected:
                pass
            return model

        # Adapter (XL-aware) branch
        if 'conv_in.weight' in sd:

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
                pass
            return model

        raise ValueError("_load_t2i_model: unsupported T2I-Adapter checkpoint format")


