import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.model_detection import detect_model
from streamdiffusion.hooks import EmbedsCtx, StepCtx, UnetKwargsDelta, EmbeddingHook, UnetHook
from streamdiffusion.image_filter import SimilarImageFilter
from streamdiffusion.stream_parameter_updater import StreamParameterUpdater

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        # Detect model type
        detection_result = detect_model(pipe.unet, pipe)
        self.model_type = detection_result['model_type']
        self.is_sdxl = detection_result['is_sdxl']
        self.is_turbo = detection_result['is_turbo']
        self.detection_confidence = detection_result['confidence']
    
        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0

        # Initialize SDXL-specific attributes
        if self.is_sdxl:
            self.add_text_embeds = None
            self.add_time_ids = None

        # Initialize parameter updater
        self._param_updater = StreamParameterUpdater(self, normalize_prompt_weights, normalize_seed_weights)
        # Default IP-Adapter runtime weight mode (None = uniform). Can be set to strings like
        # "ease in", "ease out", "ease in-out", "reverse in-out", "style transfer precise", "composition precise".
        self.ipadapter_weight_type = None

        # Hook containers (step 1: introduced but initially no-op)
        self.embedding_hooks: List[EmbeddingHook] = []
        self.unet_hooks: List[UnetHook] = []

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        # Check for SDXL compatibility
        if self.is_sdxl:
            return
            
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        self.current_seed = seed
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        # Handle SDXL vs SD1.5/SD2.1 text encoding differently
        if self.is_sdxl:
            # SDXL encode_prompt returns 4 values: 
            # (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,  # Use same prompt for both encoders
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,  # Use same negative prompt for both encoders
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            
            if len(encoder_output) >= 4:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoder_output[:4]
                
                # Set up prompt embeddings for the UNet (base before hooks)
                base_prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
                
                # Handle CFG for prompt embeddings
                if self.use_denoising_batch and self.cfg_type == "full":
                    uncond_prompt_embeds = negative_prompt_embeds.repeat(self.batch_size, 1, 1)
                elif self.cfg_type == "initialize":
                    uncond_prompt_embeds = negative_prompt_embeds.repeat(self.frame_bff_size, 1, 1)

                if self.guidance_scale > 1.0 and (
                    self.cfg_type == "initialize" or self.cfg_type == "full"
                ):
                    base_prompt_embeds = torch.cat(
                        [uncond_prompt_embeds, base_prompt_embeds], dim=0
                    )
                
                # Set up SDXL-specific conditioning (added_cond_kwargs)
                if do_classifier_free_guidance:
                    self.add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    self.add_text_embeds = pooled_prompt_embeds
                
                # Create time conditioning for SDXL micro-conditioning
                original_size = (self.height, self.width)
                target_size = (self.height, self.width)
                crops_coords_top_left = (0, 0)
                
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype, device=self.device)
                
                if do_classifier_free_guidance:
                    self.add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                else:
                    self.add_time_ids = add_time_ids
            else:
                raise ValueError(f"SDXL encode_prompt returned {len(encoder_output)} outputs, expected at least 4")
            # Run embedding hooks (no-op unless modules register)
            embeds_ctx = EmbedsCtx(prompt_embeds=base_prompt_embeds, negative_prompt_embeds=None)
            for hook in self.embedding_hooks:
                try:
                    embeds_ctx = hook(embeds_ctx)
                except Exception as e:
                    logger.error(f"prepare: embedding hook failed: {e}")
                    raise
            self.prompt_embeds = embeds_ctx.prompt_embeds
        else:
            # SD1.5/SD2.1 encode_prompt returns 2 values: (prompt_embeds, negative_prompt_embeds)
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            base_prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

            if self.use_denoising_batch and self.cfg_type == "full":
                uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
            elif self.cfg_type == "initialize":
                uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

            if self.guidance_scale > 1.0 and (
                self.cfg_type == "initialize" or self.cfg_type == "full"
            ):
                base_prompt_embeds = torch.cat(
                    [uncond_prompt_embeds, base_prompt_embeds], dim=0
                )

            # Run embedding hooks (no-op unless modules register)
            embeds_ctx = EmbedsCtx(prompt_embeds=base_prompt_embeds, negative_prompt_embeds=None)
            for hook in self.embedding_hooks:
                try:
                    embeds_ctx = hook(embeds_ctx)
                except Exception as e:
                    logger.error(f"prepare: embedding hook failed: {e}")
                    raise
            self.prompt_embeds = embeds_ctx.prompt_embeds

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        #NOTE: this is a hack. Pipeline needs a major refactor along with stream parameter updater. 
        self.update_prompt(prompt)

        if not self.use_denoising_batch:
            self.sub_timesteps_tensor = self.sub_timesteps_tensor[0]
            self.alpha_prod_t_sqrt = self.alpha_prod_t_sqrt[0]
            self.beta_prod_t_sqrt = self.beta_prod_t_sqrt[0]

        self.sub_timesteps_tensor = self.sub_timesteps_tensor.to(self.device)
        self.c_skip = self.c_skip.to(self.device)
        self.c_out = self.c_out.to(self.device)

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        self._param_updater.update_stream_params(
            prompt_list=[(prompt, 1.0)],
            prompt_interpolation_method="linear"
        )

    @torch.no_grad()
    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        # Prompt blending parameters
        prompt_list: Optional[List[Tuple[str, float]]] = None,
        negative_prompt: Optional[str] = None,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
        normalize_prompt_weights: Optional[bool] = None,
        # Seed blending parameters
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
        normalize_seed_weights: Optional[bool] = None,
        # IPAdapter parameters
        ipadapter_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update streaming parameters efficiently in a single call.

        Parameters
        ----------
        num_inference_steps : Optional[int]
            The number of inference steps to perform.
        guidance_scale : Optional[float]
            The guidance scale to use for CFG.
        delta : Optional[float]
            The delta multiplier of virtual residual noise.
        t_index_list : Optional[List[int]]
            The t_index_list to use for inference.
        seed : Optional[int]
            The random seed to use for noise generation.
        prompt_list : Optional[List[Tuple[str, float]]]
            List of prompts with weights for blending.
        negative_prompt : Optional[str]
            The negative prompt to apply to all blended prompts.
        prompt_interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between prompt embeddings.
        normalize_prompt_weights : Optional[bool]
            Whether to normalize prompt weights in blending to sum to 1, by default None (no change).
            When False, weights > 1 will amplify embeddings.
        seed_list : Optional[List[Tuple[int, float]]]
            List of seeds with weights for blending.
        seed_interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors.
        normalize_seed_weights : Optional[bool]
            Whether to normalize seed weights in blending to sum to 1, by default None (no change).
            When False, weights > 1 will amplify noise.
        ipadapter_config : Optional[Dict[str, Any]]
            IPAdapter configuration dict containing scale, style_image, etc.
        """
        self._param_updater.update_stream_params(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            t_index_list=t_index_list,
            seed=seed,
            prompt_list=prompt_list,
            negative_prompt=negative_prompt,
            prompt_interpolation_method=prompt_interpolation_method,
            seed_list=seed_list,
            seed_interpolation_method=seed_interpolation_method,
            normalize_prompt_weights=normalize_prompt_weights,
            normalize_seed_weights=normalize_seed_weights,
            ipadapter_config=ipadapter_config,
        )



    def get_normalize_prompt_weights(self) -> bool:
        """Get the current prompt weight normalization setting."""
        return self._param_updater.get_normalize_prompt_weights()

    def get_normalize_seed_weights(self) -> bool:
        """Get the current seed weight normalization setting."""
        return self._param_updater.get_normalize_seed_weights()



    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # Prepare UNet call arguments
        unet_kwargs = {
            'sample': x_t_latent_plus_uc,
            'timestep': t_list,
            'encoder_hidden_states': self.prompt_embeds,
            'return_dict': False,
        }
        
        # Add SDXL-specific conditioning if this is an SDXL model
        if self.is_sdxl and hasattr(self, 'add_text_embeds') and hasattr(self, 'add_time_ids'):
            if self.add_text_embeds is not None and self.add_time_ids is not None:
                # Handle batching for CFG - replicate conditioning to match batch size
                batch_size = x_t_latent_plus_uc.shape[0]
                
                # Replicate add_text_embeds and add_time_ids to match the batch size
                if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
                    # For initialize mode: [uncond, cond, cond, ...]
                    add_text_embeds = torch.cat([
                        self.add_text_embeds[0:1],  # uncond
                        self.add_text_embeds[1:2].repeat(batch_size - 1, 1)  # repeat cond
                    ], dim=0)
                    add_time_ids = torch.cat([
                        self.add_time_ids[0:1],  # uncond  
                        self.add_time_ids[1:2].repeat(batch_size - 1, 1)  # repeat cond
                    ], dim=0)
                elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
                    # For full mode: repeat both uncond and cond for each latent
                    repeat_factor = batch_size // 2
                    add_text_embeds = self.add_text_embeds.repeat(repeat_factor, 1)
                    add_time_ids = self.add_time_ids.repeat(repeat_factor, 1)
                else:
                    # No CFG: just repeat the conditioning
                    add_text_embeds = self.add_text_embeds[1:2].repeat(batch_size, 1) if self.add_text_embeds.shape[0] > 1 else self.add_text_embeds.repeat(batch_size, 1)
                    add_time_ids = self.add_time_ids[1:2].repeat(batch_size, 1) if self.add_time_ids.shape[0] > 1 else self.add_time_ids.repeat(batch_size, 1)
                
                unet_kwargs['added_cond_kwargs'] = {
                    'text_embeds': add_text_embeds,
                    'time_ids': add_time_ids
                }
        
        # Allow modules to contribute additional UNet kwargs via hooks
        try:
            step_ctx = StepCtx(
                x_t_latent=x_t_latent_plus_uc,
                t_list=t_list,
                step_index=idx if isinstance(idx, int) else (int(idx) if idx is not None else None),
                guidance_mode=self.cfg_type if self.guidance_scale > 1.0 else "none",
                sdxl_cond=unet_kwargs.get('added_cond_kwargs', None)
            )
            for hook in self.unet_hooks:
                delta: UnetKwargsDelta = hook(step_ctx)
                if delta is None:
                    continue
                if delta.down_block_additional_residuals is not None:
                    unet_kwargs['down_block_additional_residuals'] = delta.down_block_additional_residuals
                if delta.mid_block_additional_residual is not None:
                    unet_kwargs['mid_block_additional_residual'] = delta.mid_block_additional_residual
                if delta.added_cond_kwargs is not None:
                    # Merge SDXL cond if both exist
                    base_added = unet_kwargs.get('added_cond_kwargs', {})
                    base_added.update(delta.added_cond_kwargs)
                    unet_kwargs['added_cond_kwargs'] = base_added
        except Exception as e:
            logger.error(f"unet_step: unet hook failed: {e}")
            raise

        # Call UNet with appropriate conditioning
        if self.is_sdxl:
            try:
                # Add timing to detect hang
                import time
                start_time = time.time()
                
                # Detect UNet type and use appropriate calling convention
                added_cond_kwargs = unet_kwargs.get('added_cond_kwargs', {})
                
                # Check if this is a TensorRT engine or PyTorch UNet
                is_tensorrt_engine = hasattr(self.unet, 'engine') and hasattr(self.unet, 'stream')
                
                if is_tensorrt_engine:
                    # TensorRT engine expects positional args + kwargs
                    # Provide ipadapter_scale vector if engine was built with IP-Adapter
                    extra_kwargs = {}
                    if getattr(self.unet, 'use_ipadapter', False):
                        num_ip_layers = getattr(self.unet, 'num_ip_layers', None)
                        if not isinstance(num_ip_layers, int) or num_ip_layers <= 0:
                            raise RuntimeError("unet_step: Invalid num_ip_layers on TRT engine")
                        base_weight = float(getattr(self, 'ipadapter_scale', 1.0))
                        weight_type = getattr(self, 'ipadapter_weight_type', None)
                        try:
                            from diffusers_ipadapter.ip_adapter.attention_processor import build_layer_weights, build_time_weight_factor
                            weights = build_layer_weights(num_ip_layers, base_weight, weight_type)
                        except Exception:
                            weights = None
                        if weights is None:
                            weights = torch.full((num_ip_layers,), base_weight, dtype=torch.float32, device=self.device)
                        # Apply per-step time factor if applicable
                        try:
                            total_steps = getattr(self, 'denoising_steps_num', None) or (len(self.t_list) if hasattr(self, 't_list') else None)
                            if total_steps is not None and idx is not None:
                                time_factor = build_time_weight_factor(weight_type, int(idx), int(total_steps))
                                weights = weights * float(time_factor)
                        except Exception:
                            pass
                        extra_kwargs['ipadapter_scale'] = weights
                        try:
                            logger.debug(f"pipeline.unet_step: TRT SDXL ipadapter_scale shape={tuple(weights.shape)}, min={float(weights.min().item())}, max={float(weights.max().item())}")
                        except Exception:
                            pass

                    logger.debug(f"pipeline.unet_step: Calling TRT SDXL UNet with extra_kwargs keys={list(extra_kwargs.keys())}")
                    model_pred = self.unet(
                        unet_kwargs['sample'],                    # latent_model_input (positional)
                        unet_kwargs['timestep'],                  # timestep (positional)
                        unet_kwargs['encoder_hidden_states'],     # encoder_hidden_states (positional)
                        **extra_kwargs,
                        **added_cond_kwargs                       # SDXL conditioning as kwargs
                    )[0]
                else:
                    # PyTorch UNet expects diffusers-style named arguments
                    # For PyTorch, optionally apply per-step time scheduling by temporarily scaling processors
                    time_factor = 1.0
                    try:
                        from diffusers_ipadapter.ip_adapter.attention_processor import build_layer_weights, build_time_weight_factor
                        total_steps = getattr(self, 'denoising_steps_num', None) or (len(self.t_list) if hasattr(self, 't_list') else None)
                        if total_steps is not None and idx is not None:
                            time_factor = float(build_time_weight_factor(getattr(self, 'ipadapter_weight_type', None), int(idx), int(total_steps)))
                        # Modulate by time factor using stored _base_scale so user-selected strength is respected
                        if hasattr(self.pipe.unet, 'attn_processors') and time_factor != 1.0:
                            for p in self.pipe.unet.attn_processors.values():
                                if hasattr(p, 'scale') and hasattr(p, '_ip_layer_index'):
                                    base_val = getattr(p, '_base_scale', p.scale)
                                    p.scale = float(base_val) * time_factor
                    except Exception:
                        pass

                    model_pred = self.unet(
                        sample=unet_kwargs['sample'],
                        timestep=unet_kwargs['timestep'],
                        encoder_hidden_states=unet_kwargs['encoder_hidden_states'],
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    # No restoration for per-layer scale; next step will set again via updater/time factor
                
            except Exception as e:
                logger.error(f"[PIPELINE] unet_step: *** ERROR: SDXL UNet call failed: {e} ***")
                import traceback
                traceback.print_exc()
                raise
        else:
            # For SD1.5/SD2.1, use the old calling convention for compatibility
            # If running with TensorRT and IP-Adapter, provide ipadapter_scale as runtime input
            ip_scale_kw = {}
            is_tensorrt_engine = hasattr(self.unet, 'engine') and hasattr(self.unet, 'stream')
            
            if is_tensorrt_engine and getattr(self.unet, 'use_ipadapter', False):
                num_ip_layers = getattr(self.unet, 'num_ip_layers', None)
                if isinstance(num_ip_layers, int) and num_ip_layers > 0:
                    base_weight = float(getattr(self, 'ipadapter_scale', 1.0))
                    weight_type = getattr(self, 'ipadapter_weight_type', None)
                    try:
                        from diffusers_ipadapter.ip_adapter.attention_processor import build_layer_weights, build_time_weight_factor
                        weights = build_layer_weights(num_ip_layers, base_weight, weight_type)
                    except Exception:
                        weights = None
                    if weights is None:
                        weights = torch.full((num_ip_layers,), base_weight, dtype=torch.float32, device=self.device)
                    # Apply per-step time factor if applicable
                    try:
                        total_steps = getattr(self, 'denoising_steps_num', None) or (len(self.t_list) if hasattr(self, 't_list') else None)
                        if total_steps is not None and idx is not None:
                            time_factor = build_time_weight_factor(weight_type, int(idx), int(total_steps))
                            weights = weights * float(time_factor)
                    except Exception:
                        pass
                    ip_scale_kw['ipadapter_scale'] = weights
                    try:
                        logger.debug(f"pipeline.unet_step: TRT SD1.5 ipadapter_scale shape={tuple(weights.shape)}, min={float(weights.min().item())}, max={float(weights.max().item())}")
                    except Exception:
                        pass

            # For PyTorch branch (no TRT), optionally apply per-step time factor
            time_factor = 1.0
            try:
                from diffusers_ipadapter.ip_adapter.attention_processor import build_time_weight_factor
                total_steps = getattr(self, 'denoising_steps_num', None) or (len(self.t_list) if hasattr(self, 't_list') else None)
                if total_steps is not None and idx is not None:
                    time_factor = float(build_time_weight_factor(getattr(self, 'ipadapter_weight_type', None), int(idx), int(total_steps)))
                if hasattr(self.unet, 'attn_processors') and time_factor != 1.0:
                    for p in self.unet.attn_processors.values():
                        if hasattr(p, 'scale') and hasattr(p, '_ip_layer_index'):
                            base_val = getattr(p, '_base_scale', p.scale)
                            p.scale = float(base_val) * time_factor
            except Exception:
                pass

            logger.debug(f"pipeline.unet_step: Calling TRT SD1.5 UNet with ip_scale={ 'ipadapter_scale' in ip_scale_kw }")
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
                **ip_scale_kw,
            )[0]

        # Check for problematic values in model prediction
        if torch.isnan(model_pred).any():
            nan_count = torch.isnan(model_pred).sum().item()
            logger.error(f"[PIPELINE] unet_step: *** ERROR: {nan_count} NaN values in model_pred! ***")
        if torch.isinf(model_pred).any():
            inf_count = torch.isinf(model_pred).sum().item()
            logger.error(f"[PIPELINE] unet_step: *** ERROR: {inf_count} Inf values in model_pred! ***")
        if (model_pred == 0).all():
            logger.error(f"[PIPELINE] unet_step: *** ERROR: All model_pred values are zero! ***")
        

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
            
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
            
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        
        img_latent = img_latent * self.vae.config.scaling_factor
        
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        # Check for problematic values
        if torch.isnan(x_0_pred_out).any():
            nan_count = torch.isnan(x_0_pred_out).sum().item()
            logger.error(f"[PIPELINE] decode_image: *** WARNING: {nan_count} NaN values detected in input! ***")
        if torch.isinf(x_0_pred_out).any():
            inf_count = torch.isinf(x_0_pred_out).sum().item()
            logger.error(f"[PIPELINE] decode_image: *** WARNING: {inf_count} Inf values detected in input! ***")
        if (x_0_pred_out == 0).all():
            logger.error(f"[PIPELINE] decode_image: *** WARNING: All values are zero! ***")
        
        scaled_latent = x_0_pred_out / self.vae.config.scaling_factor
        
        output_latent = self.vae.decode(scaled_latent, return_dict=False)[0]
        
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
            
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(1,).repeat(self.frame_bff_size,)
                
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx)
                
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            
            x_t_latent = self.encode_image(x)
        else:
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        
        # Prepare UNet call arguments
        unet_kwargs = {
            'sample': x_t_latent,
            'timestep': self.sub_timesteps_tensor,
            'encoder_hidden_states': self.prompt_embeds,
            'return_dict': False,
        }
        
        # Add SDXL-specific conditioning if this is an SDXL model
        if self.is_sdxl and hasattr(self, 'add_text_embeds') and hasattr(self, 'add_time_ids'):
            if self.add_text_embeds is not None and self.add_time_ids is not None:
                # For txt2img, replicate conditioning to match batch size
                add_text_embeds = self.add_text_embeds[1:2].repeat(batch_size, 1) if self.add_text_embeds.shape[0] > 1 else self.add_text_embeds.repeat(batch_size, 1)
                add_time_ids = self.add_time_ids[1:2].repeat(batch_size, 1) if self.add_time_ids.shape[0] > 1 else self.add_time_ids.repeat(batch_size, 1)
                
                unet_kwargs['added_cond_kwargs'] = {
                    'text_embeds': add_text_embeds,
                    'time_ids': add_time_ids
                }

        # Call UNet with appropriate conditioning
        if self.is_sdxl:
            model_pred = self.unet(**unet_kwargs)[0]
        else:
            # For SD1.5/SD2.1, use the old calling convention for compatibility
            model_pred = self.unet(
                x_t_latent,
                self.sub_timesteps_tensor,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0]
            
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)
