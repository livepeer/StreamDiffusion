from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import torch

from streamdiffusion.hooks import EmbedsCtx, EmbeddingHook
import os


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
        self.ipadapter: Optional[Any] = None

    def build_embedding_hook(self, stream) -> EmbeddingHook:
        style_key = self.config.style_image_key or "default"
        num_tokens = int(self.config.num_image_tokens)

        def _embedding_hook(ctx: EmbedsCtx) -> EmbedsCtx:
            # Fetch cached image token embeddings (prompt, negative)
            cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = stream._param_updater.get_cached_embeddings(style_key)
            image_prompt_tokens: Optional[torch.Tensor] = None
            image_negative_tokens: Optional[torch.Tensor] = None
            if cached is not None:
                image_prompt_tokens, image_negative_tokens = cached

            # Validate or synthesize tokens when missing to satisfy engine shape (e.g., TRT expects 77+num_tokens)
            hidden_dim = ctx.prompt_embeds.shape[2]
            batch_size = ctx.prompt_embeds.shape[0]
            if image_prompt_tokens is None:
                image_prompt_tokens = torch.zeros(
                    (batch_size, num_tokens, hidden_dim), dtype=ctx.prompt_embeds.dtype, device=ctx.prompt_embeds.device
                )
            else:
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
            if neg_with_image is not None:
                if image_negative_tokens is None:
                    image_negative_tokens = torch.zeros(
                        (neg_with_image.shape[0], num_tokens, hidden_dim), dtype=neg_with_image.dtype, device=neg_with_image.device
                    )
                else:
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

        # Resolve model paths (HF repo file or local path)
        resolved_ip_path = self._resolve_model_path(self.config.ipadapter_model_path)
        resolved_encoder_path = self._resolve_model_path(self.config.image_encoder_path)

        # Create IP-Adapter and install processors into UNet
        ipadapter = IPAdapter(
            pipe=stream.pipe,
            ipadapter_ckpt_path=resolved_ip_path,
            image_encoder_path=resolved_encoder_path,
            device=stream.device,
            dtype=stream.dtype,
        )
        self.ipadapter = ipadapter

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

        # Compatibility: expose expected attributes/methods used by StreamParameterUpdater
        try:
            setattr(stream, 'ipadapter', ipadapter)
            setattr(stream, 'scale', float(self.config.scale))
            def _update_scale(new_scale: float) -> None:
                ipadapter.set_scale(float(new_scale))
                setattr(stream, 'ipadapter_scale', float(new_scale))
                try:
                    setattr(stream, 'scale', float(new_scale))
                except Exception:
                    pass
            def _update_style_image(style_image) -> None:
                stream._param_updater.update_style_image(style_key, style_image, is_stream=False)
            setattr(stream, 'update_scale', _update_scale)
            setattr(stream, 'update_style_image', _update_style_image)
        except Exception:
            pass

        # Register embedding hook for concatenation of image tokens
        stream.embedding_hooks.append(self.build_embedding_hook(stream))

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve a model path.

        Accepts either a local filesystem path or a Hugging Face repo/file spec like
        "h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors" or a directory path
        such as "h94/IP-Adapter/models/image_encoder".
        """
        if not model_path:
            raise ValueError("IPAdapterModule._resolve_model_path: model_path is required")

        if os.path.exists(model_path):
            return model_path

        # Treat as HF repo path
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"IPAdapterModule: huggingface_hub required to resolve '{model_path}': {e}")
            raise

        parts = model_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"IPAdapterModule._resolve_model_path: Invalid Hugging Face spec: '{model_path}'")

        repo_id = "/".join(parts[:2])
        subpath = "/".join(parts[2:])

        # File if last component has an extension; otherwise treat as directory
        if "." in parts[-1]:
            # File download
            local_path = hf_hub_download(repo_id=repo_id, filename=subpath)
            return local_path
        else:
            # Directory download
            repo_root = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subpath}/*"]) 
            full_path = os.path.join(repo_root, subpath)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"IPAdapterModule._resolve_model_path: Downloaded path not found: {full_path}")
            return full_path

