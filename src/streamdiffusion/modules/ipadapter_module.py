from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import torch

from streamdiffusion.hooks import EmbedsCtx, EmbeddingHook


@dataclass
class IPAdapterConfig:
    """Minimal config for constructing an IP-Adapter module instance.

    This module focuses only on embedding composition (step 2 of migration).
    Runtime installation and wrapper wiring will come in later steps.
    """
    style_image_key: Optional[str] = None
    num_image_tokens: int = 4  # e.g., 4 for standard, 16 for plus
    ipadapter_model_path: Optional[str] = None
    image_encoder_path: Optional[str] = None
    style_image: Optional[Any] = None
    scale: float = 1.0


class IPAdapterModule:
    """IP-Adapter embedding hook provider.

    Produces an embedding hook that concatenates cached image tokens (from
    StreamParameterUpdater's embedding cache) to the current text embeddings.
    """

    def __init__(self, config: IPAdapterConfig) -> None:
        self.config = config

    def build_embedding_hook(self, stream) -> EmbeddingHook:
        style_key = self.config.style_image_key or "default"
        num_tokens = int(self.config.num_image_tokens)

        def _embedding_hook(ctx: EmbedsCtx) -> EmbedsCtx:
            # Fetch cached image token embeddings (prompt, negative)
            cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = stream._param_updater.get_cached_embeddings(style_key)
            if cached is None:
                # No style embeddings available; leave embeddings unchanged
                return ctx

            image_prompt_tokens, image_negative_tokens = cached

            # Validate token count if specified
            if image_prompt_tokens is not None and num_tokens is not None:
                if image_prompt_tokens.shape[1] != num_tokens:
                    raise ValueError(
                        f"IPAdapterModule: Expected {num_tokens} image tokens, got {image_prompt_tokens.shape[1]}"
                    )

            # Concatenate image tokens to the right of text tokens
            prompt_with_image = ctx.prompt_embeds
            if image_prompt_tokens is not None:
                # Repeat to match batch size if needed
                if image_prompt_tokens.shape[0] != prompt_with_image.shape[0]:
                    image_prompt_tokens = image_prompt_tokens.repeat_interleave(
                        repeats=prompt_with_image.shape[0] // max(image_prompt_tokens.shape[0], 1), dim=0
                    )
                prompt_with_image = torch.cat([prompt_with_image, image_prompt_tokens], dim=1)

            neg_with_image = ctx.negative_prompt_embeds
            if neg_with_image is not None and image_negative_tokens is not None:
                if image_negative_tokens.shape[0] != neg_with_image.shape[0]:
                    image_negative_tokens = image_negative_tokens.repeat_interleave(
                        repeats=neg_with_image.shape[0] // max(image_negative_tokens.shape[0], 1), dim=0
                    )
                neg_with_image = torch.cat([neg_with_image, image_negative_tokens], dim=1)

            return EmbedsCtx(prompt_embeds=prompt_with_image, negative_prompt_embeds=neg_with_image)

        return _embedding_hook

    def install(self, stream) -> None:
        """Install IP-Adapter processors and register embedding hook and preprocessor.

        - Instantiates IP-Adapter with model and encoder paths
        - Registers IPAdapterEmbeddingPreprocessor with StreamParameterUpdater using style_image_key
        - Optionally processes provided style image to populate the embedding cache
        - Registers the embedding hook onto stream.embedding_hooks
        - Sets the initial scale and mirrors it onto stream.ipadapter_scale
        """
        logger = __import__('logging').getLogger(__name__)
        style_key = self.config.style_image_key or "ipadapter_main"

        # Validate required paths
        if not self.config.ipadapter_model_path or not self.config.image_encoder_path:
            raise ValueError("IPAdapterModule.install: ipadapter_model_path and image_encoder_path are required")

        # Lazy import to avoid hard dependency unless used
        try:
            from diffusers_ipadapter import IPAdapter  # type: ignore
        except Exception as e:
            logger.error(f"IPAdapterModule.install: Failed to import IPAdapter: {e}")
            raise
        try:
            from streamdiffusion.preprocessing.processors.ipadapter_embedding import IPAdapterEmbeddingPreprocessor
        except Exception as e:
            logger.error(f"IPAdapterModule.install: Failed to import IPAdapterEmbeddingPreprocessor: {e}")
            raise

        # Create IP-Adapter and install processors into UNet
        ipadapter = IPAdapter(
            pipe=stream.pipe,
            ipadapter_ckpt_path=self.config.ipadapter_model_path,
            image_encoder_path=self.config.image_encoder_path,
            device=stream.device,
            dtype=stream.dtype,
        )

        # Register embedding preprocessor for this style key
        embedding_preprocessor = IPAdapterEmbeddingPreprocessor(
            ipadapter=ipadapter,
            device=stream.device,
            dtype=stream.dtype,
        )
        stream._param_updater.register_embedding_preprocessor(embedding_preprocessor, style_key)

        # Process initial style image if provided
        if self.config.style_image is not None:
            try:
                stream._param_updater.update_style_image(style_key, self.config.style_image, is_stream=False)
            except Exception as e:
                logger.error(f"IPAdapterModule.install: Failed to process style image: {e}")
                raise

        # Set initial scale and mirror onto stream for TRT runtime vector if needed
        try:
            ipadapter.set_scale(float(self.config.scale))
            setattr(stream, 'ipadapter_scale', float(self.config.scale))
        except Exception:
            pass

        # Register embedding hook for concatenation of image tokens
        stream.embedding_hooks.append(self.build_embedding_hook(stream))

