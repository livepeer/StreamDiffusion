from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import torch


@dataclass
class EmbedsCtx:
    """Context passed to embedding hooks.

    Fields:
    - prompt_embeds: [batch, seq_len, dim]
    - negative_prompt_embeds: optional [batch, seq_len, dim]
    """
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor] = None


@dataclass
class StepCtx:
    """Context passed to UNet hooks for each denoising step.

    Fields:
    - x_t_latent: latent tensor (possibly CFG-expanded)
    - t_list: timesteps tensor (possibly CFG-expanded)
    - step_index: optional int step index within total steps
    - guidance_mode: one of {"none","full","self","initialize"}
    - sdxl_cond: optional dict with SDXL micro-cond tensors
    """
    x_t_latent: torch.Tensor
    t_list: torch.Tensor
    step_index: Optional[int]
    guidance_mode: str
    sdxl_cond: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class UnetKwargsDelta:
    """Delta produced by UNet hooks to augment UNet call kwargs."""
    down_block_additional_residuals: Optional[List[torch.Tensor]] = None
    mid_block_additional_residual: Optional[torch.Tensor] = None
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
    # Additional kwargs to pass directly to the UNet call (e.g., ipadapter_scale)
    extra_unet_kwargs: Optional[Dict[str, Any]] = None


# Type aliases for clarity
EmbeddingHook = Callable[[EmbedsCtx], EmbedsCtx]
UnetHook = Callable[[StepCtx], UnetKwargsDelta]

