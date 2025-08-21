import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import (
    LCMScheduler, 
    StableDiffusionPipeline, 
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    TCDScheduler,
)
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
        lora_dict: Optional[Dict[str, float]] = None,
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
        scheduler: Literal["lcm", "tcd", "dpm++ 2m", "uni_pc", "ddim", "euler"] = "lcm",
        sampler: Literal["simple", "sgm uniform", "normal", "ddim", "beta", "karras"] = "normal",
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
        self.scheduler_type = scheduler
        self.sampler_type = sampler

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

        # Initialize scheduler based on configuration
        self.scheduler = self._initialize_scheduler(scheduler, sampler, pipe.scheduler.config)
        
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0

        # Initialize SDXL-specific attributes
        if self.is_sdxl:
            self.add_text_embeds = None
            self.add_time_ids = None
            logger.log(logging.INFO, f"[PIPELINE] SDXL Detected: Using {scheduler} scheduler with {sampler} sampler")

        # Initialize parameter updater
        self._param_updater = StreamParameterUpdater(self, normalize_prompt_weights, normalize_seed_weights)
        # Default IP-Adapter runtime weight mode (None = uniform). Can be set to strings like
        # "ease in", "ease out", "ease in-out", "reverse in-out", "style transfer precise", "composition precise".
        self.ipadapter_weight_type = None

        # Hook containers (step 1: introduced but initially no-op)
        self.embedding_hooks: List[EmbeddingHook] = []
        self.unet_hooks: List[UnetHook] = []
        
        # Cache TensorRT detection to avoid repeated hasattr checks
        self._is_unet_tensorrt = None
        
        # Cache SDXL conditioning tensors to avoid repeated torch.cat/repeat operations
        self._sdxl_conditioning_cache: Dict[str, torch.Tensor] = {}
        self._cached_batch_size: Optional[int] = None
        self._cached_cfg_type: Optional[str] = None
        self._cached_guidance_scale: Optional[float] = None
        

    def _check_unet_tensorrt(self) -> bool:
        """Cache TensorRT detection to avoid repeated hasattr calls"""
        if self._is_unet_tensorrt is None:
            self._is_unet_tensorrt = hasattr(self.unet, 'engine') and hasattr(self.unet, 'stream')
        return self._is_unet_tensorrt

    def _get_cached_sdxl_conditioning(self, batch_size: int, cfg_type: str, guidance_scale: float) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached SDXL conditioning tensors if configuration matches"""
        if (self._cached_batch_size == batch_size and 
            self._cached_cfg_type == cfg_type and 
            self._cached_guidance_scale == guidance_scale and
            len(self._sdxl_conditioning_cache) > 0):
            return {
                'text_embeds': self._sdxl_conditioning_cache.get('text_embeds'),
                'time_ids': self._sdxl_conditioning_cache.get('time_ids')
            }
        return None

    def _cache_sdxl_conditioning(self, batch_size: int, cfg_type: str, guidance_scale: float, 
                                text_embeds: torch.Tensor, time_ids: torch.Tensor) -> None:
        """Cache SDXL conditioning tensors for reuse"""
        self._cached_batch_size = batch_size
        self._cached_cfg_type = cfg_type
        self._cached_guidance_scale = guidance_scale
        self._sdxl_conditioning_cache['text_embeds'] = text_embeds.clone()
        self._sdxl_conditioning_cache['time_ids'] = time_ids.clone()

    def _build_sdxl_conditioning(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Build SDXL conditioning tensors with optimized tensor operations"""
        # Replicate add_text_embeds and add_time_ids to match the batch size
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            # For initialize mode: [uncond, cond, cond, ...]
            # Use more efficient tensor operations
            uncond_text = self.add_text_embeds[0:1]
            cond_text = self.add_text_embeds[1:2]
            uncond_time = self.add_time_ids[0:1]
            cond_time = self.add_time_ids[1:2]
            
            if batch_size > 1:
                cond_text_repeated = cond_text.expand(batch_size - 1, -1).contiguous()
                cond_time_repeated = cond_time.expand(batch_size - 1, -1).contiguous()
                add_text_embeds = torch.cat([uncond_text, cond_text_repeated], dim=0)
                add_time_ids = torch.cat([uncond_time, cond_time_repeated], dim=0)
            else:
                add_text_embeds = uncond_text
                add_time_ids = uncond_time
                
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            # For full mode: repeat both uncond and cond for each latent
            repeat_factor = batch_size // 2
            add_text_embeds = self.add_text_embeds.expand(repeat_factor, -1).contiguous()
            add_time_ids = self.add_time_ids.expand(repeat_factor, -1).contiguous()
        else:
            # No CFG: just repeat the conditioning
            source_text = self.add_text_embeds[1:2] if self.add_text_embeds.shape[0] > 1 else self.add_text_embeds
            source_time = self.add_time_ids[1:2] if self.add_time_ids.shape[0] > 1 else self.add_time_ids
            add_text_embeds = source_text.expand(batch_size, -1).contiguous()
            add_time_ids = source_time.expand(batch_size, -1).contiguous()
        return {
            'text_embeds': add_text_embeds,
            'time_ids': add_time_ids
        }

    def _initialize_scheduler(self, scheduler_type: str, sampler_type: str, config):
        """Initialize scheduler based on type and sampler configuration."""
        # Map sampler types to configuration parameters
        sampler_config = {
            "simple": {"timestep_spacing": "linspace"},
            "sgm uniform": {"timestep_spacing": "trailing"},  # SGM Uniform is typically trailing
            "normal": {},  # Default configuration
            "ddim": {"timestep_spacing": "leading"},  # DDIM default per documentation
            "beta": {"beta_schedule": "scaled_linear"},
            "karras": {},  # Karras sigmas will be enabled in scheduler-specific code
        }
        
        # Get sampler-specific configuration
        sampler_params = sampler_config.get(sampler_type, {})

        print(f"Sampler params: {sampler_params}")
        print(f"Scheduler type: {scheduler_type}")
        
        # Create scheduler based on type
        if scheduler_type == "lcm":
            return LCMScheduler.from_config(config, **sampler_params)
        elif scheduler_type == "tcd":
            return TCDScheduler.from_config(config, **sampler_params)
        elif scheduler_type == "dpm++ 2m":
            # DPM++ 2M typically uses solver_order=2 and algorithm_type="dpmsolver++"
            return DPMSolverMultistepScheduler.from_config(
                config,
                solver_order=2,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=(sampler_type == "karras"),  # Enable Karras sigmas if requested
                **sampler_params
            )
        elif scheduler_type == "uni_pc":
            # UniPC: solver_order=2 for guided sampling, solver_type="bh2" by default
            return UniPCMultistepScheduler.from_config(
                config,
                solver_order=2,  # Good default for guided sampling
                solver_type="bh2",  # Default from documentation
                disable_corrector=[],  # No corrector disabled by default
                use_karras_sigmas=(sampler_type == "karras"),  # Enable Karras sigmas if requested
                **sampler_params
            )
        elif scheduler_type == "ddim":
            # DDIM defaults to leading timestep spacing, but trailing can be better
            return DDIMScheduler.from_config(
                config,
                set_alpha_to_one=True,  # Default per documentation
                steps_offset=0,  # Default per documentation
                prediction_type="epsilon",  # Default per documentation
                **sampler_params
            )
        elif scheduler_type == "euler":
            # Euler can use Karras sigmas for improved quality
            return EulerDiscreteScheduler.from_config(
                config,
                use_karras_sigmas=(sampler_type == "karras"),  # Enable Karras sigmas if requested
                prediction_type="epsilon",  # Default per documentation
                **sampler_params
            )
        else:
            # Default to LCM
            logger.warning(f"Unknown scheduler type '{scheduler_type}', falling back to LCM")
            return LCMScheduler.from_config(config, **sampler_params)

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
            
        self._load_lora_with_offline_fallback(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self._load_lora_with_offline_fallback(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def _load_lora_with_offline_fallback(
        self,
        pretrained: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any],
        **kwargs,
    ) -> None:
        """
        Load LoRA weights, auto-guessing common weight filenames when HF offline mode is enabled.
        """
        try:
            self.pipe.load_lora_weights(pretrained, adapter_name, **kwargs)
            return
        except Exception as e:
            message = str(e)
            is_offline_weight_error = isinstance(e, ValueError) and "must specify a `weight_name`" in message
            if not is_offline_weight_error:
                raise

        candidate_weight_names = (
            "pytorch_lora_weights.safetensors",
            "pytorch_lora_weights.bin",
            "diffusion_pytorch_model.safetensors",
            "adapter_model.safetensors",
            "lora.safetensors",
        )

        last_err: Optional[Exception] = None
        for weight_name in candidate_weight_names:
            try:
                self.pipe.load_lora_weights(
                    pretrained,
                    adapter_name,
                    **{**kwargs, "weight_name": weight_name},
                )
                return
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            raise last_err

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
        max_timestep_index = len(self.timesteps) - 1
        
        for t in self.t_list:
            # Clamp t_index to valid range to prevent index out of bounds
            if t > max_timestep_index:
                logger.warning(f"t_index {t} is out of bounds for scheduler with {len(self.timesteps)} timesteps. Clamping to {max_timestep_index}")
                t = max_timestep_index
            elif t < 0:
                logger.warning(f"t_index {t} is negative. Clamping to 0")
                t = 0
                
            timestep_value = self.timesteps[t]
            # Convert tensor timesteps to scalar values for indexing operations
            if isinstance(timestep_value, torch.Tensor):
                timestep_scalar = timestep_value.cpu().item()
            else:
                timestep_scalar = timestep_value
            self.sub_timesteps.append(timestep_scalar)

        # Create tensor version for UNet calls
        # Handle both integer and floating-point timesteps from different schedulers
        # Some schedulers like Euler may return floating-point timesteps
        if len(self.sub_timesteps) > 0:
            # Always create the tensor from scalar values to avoid device issues
            try:
                # Try integer first for compatibility
                sub_timesteps_tensor = torch.tensor(
                    self.sub_timesteps, dtype=torch.long, device=self.device
                )
            except (TypeError, ValueError):
                # Fallback for floating-point values
                sub_timesteps_tensor = torch.tensor(
                    self.sub_timesteps, dtype=torch.float32, device=self.device
                )
        else:
            sub_timesteps_tensor = torch.tensor([], dtype=torch.long, device=self.device)
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

        # Handle scheduler-specific scaling calculations
        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self._get_scheduler_scalings(timestep)
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
            # Convert floating-point timesteps to integers for tensor indexing
            if isinstance(timestep, float):
                timestep_idx = int(round(timestep))
            else:
                timestep_idx = timestep
            
            # Ensure timestep_idx is within bounds
            max_idx = len(self.scheduler.alphas_cumprod) - 1
            if timestep_idx > max_idx:
                logger.warning(f"Timestep index {timestep_idx} out of bounds for alphas_cumprod (max: {max_idx}). Clamping to {max_idx}")
                timestep_idx = max_idx
            elif timestep_idx < 0:
                logger.warning(f"Timestep index {timestep_idx} is negative. Clamping to 0")
                timestep_idx = 0
            
            # Access scheduler tensors and move to device as needed
            alpha_cumprod = self.scheduler.alphas_cumprod[timestep_idx].to(device=self.device, dtype=self.dtype)
            alpha_prod_t_sqrt = alpha_cumprod.sqrt()
            beta_prod_t_sqrt = (1 - alpha_cumprod).sqrt()
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

        # Only collapse tensors to a single element for non-batched LCM path.
        if (not self.use_denoising_batch) and self._uses_lcm_logic():
            self.sub_timesteps_tensor = self.sub_timesteps_tensor[0]
            self.alpha_prod_t_sqrt = self.alpha_prod_t_sqrt[0]
            self.beta_prod_t_sqrt = self.beta_prod_t_sqrt[0]

        self.sub_timesteps_tensor = self.sub_timesteps_tensor.to(self.device)
        self.c_skip = self.c_skip.to(self.device)
        self.c_out = self.c_out.to(self.device)

    def _get_scheduler_scalings(self, timestep):
        """
        Get LCM-specific scaling factors for boundary conditions.
        Only used for LCMScheduler - other schedulers handle scaling in their step() method.
        """
        if isinstance(self.scheduler, LCMScheduler):
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            # Ensure returned values are tensors on the correct device
            if not isinstance(c_skip, torch.Tensor):
                c_skip = torch.tensor(c_skip, device=self.device, dtype=self.dtype)
            else:
                c_skip = c_skip.to(device=self.device, dtype=self.dtype)
            if not isinstance(c_out, torch.Tensor):
                c_out = torch.tensor(c_out, device=self.device, dtype=self.dtype)
            else:
                c_out = c_out.to(device=self.device, dtype=self.dtype)
            return c_skip, c_out
        else:
            # For non-LCM schedulers, we don't use boundary condition scaling
            # Their step() method handles all the necessary scaling internally
            logger.debug(f"Scheduler {type(self.scheduler)} doesn't use boundary condition scaling")
            c_skip = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            c_out = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            return c_skip, c_out

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        self._param_updater.update_stream_params(
            prompt_list=[(prompt, 1.0)],
            prompt_interpolation_method="linear"
        )

    



    def get_normalize_prompt_weights(self) -> bool:
        """Get the current prompt weight normalization setting."""
        return self._param_updater.get_normalize_prompt_weights()

    def get_normalize_seed_weights(self) -> bool:
        """Get the current seed weight normalization setting."""
        return self._param_updater.get_normalize_seed_weights()

    def set_scheduler(
        self, 
        scheduler: Literal["lcm", "tcd", "dpm++ 2m", "uni_pc", "ddim", "euler"] = None,
        sampler: Literal["simple", "sgm uniform", "normal", "ddim", "beta", "karras"] = None,
    ) -> None:
        """
        Change the scheduler and/or sampler configuration at runtime.
        
        Parameters
        ----------
        scheduler : str, optional
            The scheduler type to use. If None, keeps current scheduler.
        sampler : str, optional
            The sampler type to use. If None, keeps current sampler.
        """
        if scheduler is not None:
            self.scheduler_type = scheduler
        if sampler is not None:
            self.sampler_type = sampler
            
        # Re-initialize scheduler with new configuration
        self.scheduler = self._initialize_scheduler(
            self.scheduler_type, 
            self.sampler_type, 
            self.pipe.scheduler.config
        )
        
        logger.info(f"Scheduler changed to {self.scheduler_type} with {self.sampler_type} sampler")



    def _uses_lcm_logic(self) -> bool:
        """Return True if scheduler uses consistency boundary-condition math (LCM/TCD)."""
        try:
            # Use isinstance checks for more reliable detection
            return isinstance(self.scheduler, LCMScheduler)
        except Exception:
            return False

    def _warned_cfg_mode_fallback(self) -> bool:
        return getattr(self, "_cfg_mode_warning_emitted", False)

    def _emit_cfg_mode_warning_once(self) -> None:
        if not self._warned_cfg_mode_fallback():
            logger.warning(
                "Non-LCM scheduler in use: falling back to standard CFG ('full') semantics. "
                "Custom cfg_type values 'self'/'initialize' are ignored for correctness."
            )
            setattr(self, "_cfg_mode_warning_emitted", True)

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
        """
        Simplified scheduler integration that works with StreamDiffusion's architecture.
        For now, we'll use a hybrid approach until we can properly refactor the pipeline.
        """
        # For LCM, use boundary condition scaling as before
        if self._uses_lcm_logic():
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
        else:
            # For other schedulers, use simple epsilon denoising
            # This is what works reliably with StreamDiffusion's current architecture
            if idx is not None and idx < len(self.alpha_prod_t_sqrt):
                denoised_batch = (
                    x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
                ) / self.alpha_prod_t_sqrt[idx]
            else:
                # Fallback to first timestep if idx is out of bounds
                denoised_batch = (
                    x_t_latent_batch - self.beta_prod_t_sqrt[0] * model_pred_batch
                ) / self.alpha_prod_t_sqrt[0]

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
                
                # Use optimized caching system for SDXL conditioning tensors
                cached_conditioning = self._get_cached_sdxl_conditioning(batch_size, self.cfg_type, self.guidance_scale)
                if cached_conditioning is not None:
                    # Cache hit - reuse existing tensors
                    add_text_embeds = cached_conditioning['text_embeds']
                    add_time_ids = cached_conditioning['time_ids']
                else:
                    # Cache miss - build new tensors using optimized operations
                    conditioning = self._build_sdxl_conditioning(batch_size)
                    add_text_embeds = conditioning['text_embeds']
                    add_time_ids = conditioning['time_ids']
                    # Cache for future use
                    self._cache_sdxl_conditioning(batch_size, self.cfg_type, self.guidance_scale, add_text_embeds, add_time_ids)
                
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
            extra_from_hooks = {}
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
                if getattr(delta, 'extra_unet_kwargs', None):
                    # Merge extra kwargs from hooks (e.g., ipadapter_scale)
                    try:
                        extra_from_hooks.update(delta.extra_unet_kwargs)
                    except Exception:
                        pass
            if extra_from_hooks:
                unet_kwargs['extra_unet_kwargs'] = extra_from_hooks
        except Exception as e:
            logger.error(f"unet_step: unet hook failed: {e}")
            raise

        # Extract potential ControlNet residual kwargs and generic extra kwargs (e.g., ipadapter_scale)
        hook_down_res = unet_kwargs.get('down_block_additional_residuals', None)
        hook_mid_res = unet_kwargs.get('mid_block_additional_residual', None)
        hook_extra_kwargs = unet_kwargs.get('extra_unet_kwargs', None) if 'extra_unet_kwargs' in unet_kwargs else None

        # Call UNet with appropriate conditioning
        if self.is_sdxl:
            try:

                
                # Detect UNet type and use appropriate calling convention
                added_cond_kwargs = unet_kwargs.get('added_cond_kwargs', {})
                
                # Check if this is a TensorRT engine or PyTorch UNet
                is_tensorrt_engine = self._check_unet_tensorrt()
                
                if is_tensorrt_engine:
                    # TensorRT engine expects positional args + kwargs. IP-Adapter scale vector, if any, is provided by hooks via extra_unet_kwargs
                    extra_kwargs = {}
                    if isinstance(hook_extra_kwargs, dict):
                        extra_kwargs.update(hook_extra_kwargs)

                    # Include ControlNet residuals if provided by hooks
                    if hook_down_res is not None:
                        extra_kwargs['down_block_additional_residuals'] = hook_down_res
                    if hook_mid_res is not None:
                        extra_kwargs['mid_block_additional_residual'] = hook_mid_res

                    model_pred = self.unet(
                        unet_kwargs['sample'],                    # latent_model_input (positional)
                        unet_kwargs['timestep'],                  # timestep (positional)
                        unet_kwargs['encoder_hidden_states'],     # encoder_hidden_states (positional)
                        **extra_kwargs,
                        # For TRT engines, ensure SDXL cond shapes match engine builds; if engine expects 81 tokens (77+4), append dummy image tokens when none
                        **added_cond_kwargs                       # SDXL conditioning as kwargs
                    )[0]
                else:
                    # PyTorch UNet expects diffusers-style named arguments. Any processor scaling is handled by IP-Adapter hook

                    call_kwargs = dict(
                        sample=unet_kwargs['sample'],
                        timestep=unet_kwargs['timestep'],
                        encoder_hidden_states=unet_kwargs['encoder_hidden_states'],
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )
                    # Include ControlNet residuals if present
                    if hook_down_res is not None:
                        call_kwargs['down_block_additional_residuals'] = hook_down_res
                    if hook_mid_res is not None:
                        call_kwargs['mid_block_additional_residual'] = hook_mid_res
                    model_pred = self.unet(**call_kwargs)[0]
                    # No restoration for per-layer scale; next step will set again via updater/time factor
                
            except Exception as e:
                logger.error(f"[PIPELINE] unet_step: *** ERROR: SDXL UNet call failed: {e} ***")
                import traceback
                traceback.print_exc()
                raise
        else:
            # For SD1.5/SD2.1, use the old calling convention for compatibility
            # Build kwargs from hooks and include residuals
            ip_scale_kw = {}
            if isinstance(hook_extra_kwargs, dict):
                ip_scale_kw.update(hook_extra_kwargs)

            # PyTorch processor time scaling is handled by the IP-Adapter hook

            # Include ControlNet residuals if present
            if hook_down_res is not None:
                ip_scale_kw['down_block_additional_residuals'] = hook_down_res
            if hook_mid_res is not None:
                ip_scale_kw['mid_block_additional_residual'] = hook_mid_res

            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
                **ip_scale_kw,
            )[0]

        

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

    def _call_unet(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Call the UNet, handling SDXL kwargs and TensorRT engine calling convention."""
        added_cond_kwargs = added_cond_kwargs or {}
        if self.is_sdxl:
            try:
                # Detect TensorRT engine vs PyTorch UNet
                is_tensorrt_engine = hasattr(self.unet, 'engine') and hasattr(self.unet, 'stream')
                if is_tensorrt_engine:
                    out = self.unet(
                        sample,
                        timestep,
                        encoder_hidden_states,
                        **added_cond_kwargs,
                    )[0]
                else:
                    out = self.unet(
                        sample=sample,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
            except Exception as e:
                logger.error(f"[PIPELINE] _call_unet: SDXL UNet call failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        return out

    def _unet_predict_noise_cfg(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        cfg_mode: Literal["none", "full", "self", "initialize"],
    ) -> torch.Tensor:
        """
        Compute noise prediction from UNet with classifier-free guidance applied.
        This function does not apply any scheduler math; it only returns the guided noise.

        For non-LCM schedulers, custom cfg_mode values 'self'/'initialize' are treated
        as 'full' to ensure correctness with scheduler.step().
        """
        effective_cfg = cfg_mode
        if not self._uses_lcm_logic() and cfg_mode in ("self", "initialize"):
            self._emit_cfg_mode_warning_once()
            effective_cfg = "full"

        # Build latent batch for CFG
        if self.guidance_scale > 1.0 and effective_cfg == "full":
            latent_with_uc = torch.cat([latent_model_input, latent_model_input], dim=0)
        elif self.guidance_scale > 1.0 and effective_cfg == "initialize":
            # Keep initialize behavior for LCM only; if we reach here, LCM path
            latent_with_uc = torch.cat([latent_model_input[0:1], latent_model_input], dim=0)
        else:
            latent_with_uc = latent_model_input

        # SDXL added conditioning replication to match batch
        added_cond_kwargs: Dict[str, torch.Tensor] = {}
        if self.is_sdxl and hasattr(self, 'add_text_embeds') and hasattr(self, 'add_time_ids'):
            if self.add_text_embeds is not None and self.add_time_ids is not None:
                batch_size = latent_with_uc.shape[0]
                if self.guidance_scale > 1.0 and effective_cfg == "initialize":
                    add_text_embeds = torch.cat([
                        self.add_text_embeds[0:1],
                        self.add_text_embeds[1:2].repeat(batch_size - 1, 1),
                    ], dim=0)
                    add_time_ids = torch.cat([
                        self.add_time_ids[0:1],
                        self.add_time_ids[1:2].repeat(batch_size - 1, 1),
                    ], dim=0)
                elif self.guidance_scale > 1.0 and effective_cfg == "full":
                    repeat_factor = batch_size // 2
                    add_text_embeds = self.add_text_embeds.repeat(repeat_factor, 1)
                    add_time_ids = self.add_time_ids.repeat(repeat_factor, 1)
                else:
                    add_text_embeds = (
                        self.add_text_embeds[1:2].repeat(batch_size, 1)
                        if self.add_text_embeds.shape[0] > 1
                        else self.add_text_embeds.repeat(batch_size, 1)
                    )
                    add_time_ids = (
                        self.add_time_ids[1:2].repeat(batch_size, 1)
                        if self.add_time_ids.shape[0] > 1
                        else self.add_time_ids.repeat(batch_size, 1)
                    )
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # Call UNet
        model_pred = self._call_unet(
            sample=latent_with_uc,
            timestep=timestep,
            encoder_hidden_states=self.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )

        # Apply CFG
        if self.guidance_scale > 1.0 and effective_cfg == "full":
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            guided = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            return guided
        else:
            return model_pred

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
        
        scaled_latent = x_0_pred_out / self.vae.config.scaling_factor
        
        output_latent = self.vae.decode(scaled_latent, return_dict=False)[0]
        
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer

        # LCM supports our denoising-batch trick. Other schedulers should use step() sequentially
        if self.use_denoising_batch and self._uses_lcm_logic():
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
            # Standard scheduler loop using scale_model_input + scheduler.step()
            sample = x_t_latent
            for idx, timestep in enumerate(self.sub_timesteps_tensor):
                # Ensure timestep tensor on device with correct dtype
                if not isinstance(timestep, torch.Tensor):
                    t = torch.tensor(timestep, device=self.device, dtype=torch.long)
                else:
                    t = timestep.to(self.device)

                # Scale model input per scheduler requirements
                model_input = (
                    self.scheduler.scale_model_input(sample, t)
                    if hasattr(self.scheduler, "scale_model_input")
                    else sample
                )

                # Predict noise with CFG
                noise_pred = self._unet_predict_noise_cfg(
                    latent_model_input=model_input,
                    timestep=t,
                    cfg_mode=self.cfg_type,
                )

                # Advance one step
                step_out = self.scheduler.step(noise_pred, t, sample)
                # diffusers returns a SchedulerOutput; prefer .prev_sample if present
                sample = getattr(step_out, "prev_sample", step_out[0] if isinstance(step_out, (tuple, list)) else step_out)

            # After final step, sample approximates x0 latent
            x_0_pred_out = sample

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
