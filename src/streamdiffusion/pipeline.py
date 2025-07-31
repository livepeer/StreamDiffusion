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
        logger.debug(f"Detected model type: {self.model_type} (confidence: {self.detection_confidence:.2f})")
    
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
            logger.debug(f"WARNING: Skipping LCM LoRA loading for SDXL model. ")
            logger.debug(f"SDXL models are incompatible with SD1.5 LCM LoRAs due to different architectures:")
            logger.debug(f"- Context dimensions: SDXL=2048 vs SD1.5=768")
            logger.debug(f"- Channel configurations: Different U-Net structures")
            logger.debug(f"Use SDXL-specific LCM LoRAs or SDXL-Turbo models instead.")
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
                
                # Set up prompt embeddings for the UNet
                self.prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
                
                # Handle CFG for prompt embeddings
                if self.use_denoising_batch and self.cfg_type == "full":
                    uncond_prompt_embeds = negative_prompt_embeds.repeat(self.batch_size, 1, 1)
                elif self.cfg_type == "initialize":
                    uncond_prompt_embeds = negative_prompt_embeds.repeat(self.frame_bff_size, 1, 1)

                if self.guidance_scale > 1.0 and (
                    self.cfg_type == "initialize" or self.cfg_type == "full"
                ):
                    self.prompt_embeds = torch.cat(
                        [uncond_prompt_embeds, self.prompt_embeds], dim=0
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
                    
                logger.debug(f"SDXL conditioning setup: prompt_embeds {self.prompt_embeds.shape}, "
                      f"add_text_embeds {self.add_text_embeds.shape}, add_time_ids {self.add_time_ids.shape}")
            else:
                raise ValueError(f"SDXL encode_prompt returned {len(encoder_output)} outputs, expected at least 4")
        else:
            # SD1.5/SD2.1 encode_prompt returns 2 values: (prompt_embeds, negative_prompt_embeds)
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

            if self.use_denoising_batch and self.cfg_type == "full":
                uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
            elif self.cfg_type == "initialize":
                uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

            if self.guidance_scale > 1.0 and (
                self.cfg_type == "initialize" or self.cfg_type == "full"
            ):
                self.prompt_embeds = torch.cat(
                    [uncond_prompt_embeds, self.prompt_embeds], dim=0
                )

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
        logger.debug(f"[PIPELINE] unet_step: Starting with latent shape: {x_t_latent.shape}, t_list: {t_list}")
        logger.debug(f"[PIPELINE] unet_step: Input latent range: [{x_t_latent.min().item():.6f}, {x_t_latent.max().item():.6f}]")
        logger.debug(f"[PIPELINE] unet_step: guidance_scale: {self.guidance_scale}, cfg_type: {self.cfg_type}")
        logger.debug(f"[PIPELINE] unet_step: is_sdxl: {self.is_sdxl}")
        
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
            logger.debug(f"[PIPELINE] unet_step: CFG initialize mode - expanded latent shape: {x_t_latent_plus_uc.shape}")
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
            logger.debug(f"[PIPELINE] unet_step: CFG full mode - expanded latent shape: {x_t_latent_plus_uc.shape}")
        else:
            x_t_latent_plus_uc = x_t_latent
            logger.debug(f"[PIPELINE] unet_step: No CFG - using original latent shape: {x_t_latent_plus_uc.shape}")

        # Prepare UNet call arguments
        unet_kwargs = {
            'sample': x_t_latent_plus_uc,
            'timestep': t_list,
            'encoder_hidden_states': self.prompt_embeds,
            'return_dict': False,
        }
        
        logger.debug(f"[PIPELINE] unet_step: Basic UNet kwargs prepared")
        logger.debug(f"[PIPELINE] unet_step: prompt_embeds shape: {self.prompt_embeds.shape}, range: [{self.prompt_embeds.min().item():.6f}, {self.prompt_embeds.max().item():.6f}]")
        
        # Add SDXL-specific conditioning if this is an SDXL model
        if self.is_sdxl and hasattr(self, 'add_text_embeds') and hasattr(self, 'add_time_ids'):
            if self.add_text_embeds is not None and self.add_time_ids is not None:
                logger.debug(f"[PIPELINE] unet_step: Adding SDXL conditioning")
                logger.debug(f"[PIPELINE] unet_step: Original add_text_embeds shape: {self.add_text_embeds.shape}")
                logger.debug(f"[PIPELINE] unet_step: Original add_time_ids shape: {self.add_time_ids.shape}")
                
                # Handle batching for CFG - replicate conditioning to match batch size
                batch_size = x_t_latent_plus_uc.shape[0]
                logger.debug(f"[PIPELINE] unet_step: Target batch size: {batch_size}")
                
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
                    logger.debug(f"[PIPELINE] unet_step: CFG initialize - conditioning shapes: text_embeds={add_text_embeds.shape}, time_ids={add_time_ids.shape}")
                elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
                    # For full mode: repeat both uncond and cond for each latent
                    repeat_factor = batch_size // 2
                    add_text_embeds = self.add_text_embeds.repeat(repeat_factor, 1)
                    add_time_ids = self.add_time_ids.repeat(repeat_factor, 1)
                    logger.debug(f"[PIPELINE] unet_step: CFG full - conditioning shapes: text_embeds={add_text_embeds.shape}, time_ids={add_time_ids.shape}")
                else:
                    # No CFG: just repeat the conditioning
                    add_text_embeds = self.add_text_embeds[1:2].repeat(batch_size, 1) if self.add_text_embeds.shape[0] > 1 else self.add_text_embeds.repeat(batch_size, 1)
                    add_time_ids = self.add_time_ids[1:2].repeat(batch_size, 1) if self.add_time_ids.shape[0] > 1 else self.add_time_ids.repeat(batch_size, 1)
                    logger.debug(f"[PIPELINE] unet_step: No CFG - conditioning shapes: text_embeds={add_text_embeds.shape}, time_ids={add_time_ids.shape}")
                
                unet_kwargs['added_cond_kwargs'] = {
                    'text_embeds': add_text_embeds,
                    'time_ids': add_time_ids
                }
                logger.debug(f"[PIPELINE] unet_step: SDXL added_cond_kwargs added to UNet call")
            else:
                logger.debug(f"[PIPELINE] unet_step: SDXL model but add_text_embeds or add_time_ids is None")
        else:
            logger.debug(f"[PIPELINE] unet_step: Not using SDXL conditioning")

        logger.debug(f"[PIPELINE] unet_step: Calling UNet with kwargs keys: {list(unet_kwargs.keys())}")
        
        # Call UNet with appropriate conditioning
        if self.is_sdxl:
            logger.debug(f"[PIPELINE] unet_step: Using SDXL UNet call")
            logger.debug(f"[PIPELINE] unet_step: About to call self.unet(**unet_kwargs)")
            logger.debug(f"[PIPELINE] unet_step: UNet object type: {type(self.unet)}")
            logger.debug(f"[PIPELINE] unet_step: UNet object class: {self.unet.__class__}")
            logger.debug(f"[PIPELINE] unet_step: UNet object methods: {[method for method in dir(self.unet) if not method.startswith('_')]}")
            logger.debug(f"[PIPELINE] unet_step: Has __call__: {hasattr(self.unet, '__call__')}")
            logger.debug(f"[PIPELINE] unet_step: Has forward: {hasattr(self.unet, 'forward')}")
            
            try:
                logger.debug(f"[PIPELINE] unet_step: 🚀 Starting SDXL UNet call...")
                
                # Add timing to detect hang
                import time
                start_time = time.time()
                
                # Detect UNet type and use appropriate calling convention
                added_cond_kwargs = unet_kwargs.get('added_cond_kwargs', {})
                
                # Check if this is a TensorRT engine or PyTorch UNet
                is_tensorrt_engine = hasattr(self.unet, 'engine') and hasattr(self.unet, 'stream')
                
                if is_tensorrt_engine:
                    logger.debug(f"[PIPELINE] unet_step: Detected TensorRT engine - using TensorRT calling convention")
                    # TensorRT engine expects positional args + kwargs
                    model_pred = self.unet(
                        unet_kwargs['sample'],                    # latent_model_input (positional)
                        unet_kwargs['timestep'],                  # timestep (positional)
                        unet_kwargs['encoder_hidden_states'],     # encoder_hidden_states (positional)
                        **added_cond_kwargs                       # SDXL conditioning as kwargs
                    )[0]
                else:
                    logger.debug(f"[PIPELINE] unet_step: Detected PyTorch UNet - using diffusers calling convention")
                    # PyTorch UNet expects diffusers-style named arguments
                    model_pred = self.unet(
                        sample=unet_kwargs['sample'],
                        timestep=unet_kwargs['timestep'],
                        encoder_hidden_states=unet_kwargs['encoder_hidden_states'],
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                
                elapsed_time = time.time() - start_time
                logger.debug(f"[PIPELINE] unet_step: ✅ SDXL UNet call completed in {elapsed_time:.3f}s!")
                
            except Exception as e:
                logger.error(f"[PIPELINE] unet_step: *** ERROR: SDXL UNet call failed: {e} ***")
                import traceback
                traceback.print_exc()
                raise
        else:
            logger.debug(f"[PIPELINE] unet_step: Using legacy UNet call")
            # For SD1.5/SD2.1, use the old calling convention for compatibility
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0]

        logger.debug(f"[PIPELINE] unet_step: UNet inference completed")
        logger.debug(f"[PIPELINE] unet_step: model_pred shape: {model_pred.shape}, range: [{model_pred.min().item():.6f}, {model_pred.max().item():.6f}]")
        
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
            logger.debug(f"[PIPELINE] unet_step: CFG initialize - noise_pred_text shape: {noise_pred_text.shape}")
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            logger.debug(f"[PIPELINE] unet_step: CFG full - noise_pred_uncond: {noise_pred_uncond.shape}, noise_pred_text: {noise_pred_text.shape}")
        else:
            noise_pred_text = model_pred
            logger.debug(f"[PIPELINE] unet_step: No CFG - using model_pred directly: {noise_pred_text.shape}")
            
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
            logger.debug(f"[PIPELINE] unet_step: Using stock_noise for uncond guidance: {noise_pred_uncond.shape}")
            
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            logger.debug(f"[PIPELINE] unet_step: Applied CFG with guidance_scale={self.guidance_scale}")
            logger.debug(f"[PIPELINE] unet_step: Final model_pred range: [{model_pred.min().item():.6f}, {model_pred.max().item():.6f}]")
        else:
            model_pred = noise_pred_text
            logger.debug(f"[PIPELINE] unet_step: No CFG applied - using noise_pred_text directly")

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            logger.debug(f"[PIPELINE] unet_step: Using denoising batch scheduler step")
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            logger.debug(f"[PIPELINE] unet_step: denoised_batch shape: {denoised_batch.shape}, range: [{denoised_batch.min().item():.6f}, {denoised_batch.max().item():.6f}]")
            
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                logger.debug(f"[PIPELINE] unet_step: Applying additional self/initialize CFG processing")
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
            logger.debug(f"[PIPELINE] unet_step: Using single scheduler step")
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            logger.debug(f"[PIPELINE] unet_step: denoised_batch shape: {denoised_batch.shape}, range: [{denoised_batch.min().item():.6f}, {denoised_batch.max().item():.6f}]")

        logger.debug(f"[PIPELINE] unet_step: Returning denoised_batch and model_pred")
        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        logger.debug(f"[PIPELINE] encode_image: Input shape: {image_tensors.shape}, dtype: {image_tensors.dtype}, device: {image_tensors.device}")
        logger.debug(f"[PIPELINE] encode_image: Input range: [{image_tensors.min().item():.6f}, {image_tensors.max().item():.6f}]")
        
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        logger.debug(f"[PIPELINE] encode_image: After device/dtype conversion: {image_tensors.shape}, {image_tensors.dtype}")
        
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        logger.debug(f"[PIPELINE] encode_image: After VAE encode: {img_latent.shape}, range: [{img_latent.min().item():.6f}, {img_latent.max().item():.6f}]")
        
        img_latent = img_latent * self.vae.config.scaling_factor
        logger.debug(f"[PIPELINE] encode_image: After scaling (factor={self.vae.config.scaling_factor}): range: [{img_latent.min().item():.6f}, {img_latent.max().item():.6f}]")
        
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        logger.debug(f"[PIPELINE] encode_image: After add_noise: {x_t_latent.shape}, range: [{x_t_latent.min().item():.6f}, {x_t_latent.max().item():.6f}]")
        
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        logger.debug(f"[PIPELINE] decode_image: Input shape: {x_0_pred_out.shape}, dtype: {x_0_pred_out.dtype}")
        logger.debug(f"[PIPELINE] decode_image: Input range: [{x_0_pred_out.min().item():.6f}, {x_0_pred_out.max().item():.6f}]")
        
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
        logger.debug(f"[PIPELINE] decode_image: After scaling (factor={self.vae.config.scaling_factor}): range: [{scaled_latent.min().item():.6f}, {scaled_latent.max().item():.6f}]")
        
        output_latent = self.vae.decode(scaled_latent, return_dict=False)[0]
        logger.debug(f"[PIPELINE] decode_image: After VAE decode: {output_latent.shape}, range: [{output_latent.min().item():.6f}, {output_latent.max().item():.6f}]")
        
        return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        logger.debug(f"[PIPELINE] predict_x0_batch: Input shape: {x_t_latent.shape}, dtype: {x_t_latent.dtype}")
        logger.debug(f"[PIPELINE] predict_x0_batch: Input range: [{x_t_latent.min().item():.6f}, {x_t_latent.max().item():.6f}]")
        logger.debug(f"[PIPELINE] predict_x0_batch: use_denoising_batch: {self.use_denoising_batch}")
        logger.debug(f"[PIPELINE] predict_x0_batch: denoising_steps_num: {self.denoising_steps_num}")
        
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            logger.debug(f"[PIPELINE] predict_x0_batch: t_list: {t_list}")
            
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                logger.debug(f"[PIPELINE] predict_x0_batch: After cat with prev_latent_batch: {x_t_latent.shape}")
                
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
            
            logger.debug(f"[PIPELINE] predict_x0_batch: Calling unet_step with latent shape: {x_t_latent.shape}")
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list)
            logger.debug(f"[PIPELINE] predict_x0_batch: After unet_step - x_0_pred_batch: {x_0_pred_batch.shape}, model_pred: {model_pred.shape}")
            logger.debug(f"[PIPELINE] predict_x0_batch: x_0_pred_batch range: [{x_0_pred_batch.min().item():.6f}, {x_0_pred_batch.max().item():.6f}]")
            logger.debug(f"[PIPELINE] predict_x0_batch: model_pred range: [{model_pred.min().item():.6f}, {model_pred.max().item():.6f}]")

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                logger.debug(f"[PIPELINE] predict_x0_batch: Multi-step - using last batch element: {x_0_pred_out.shape}")
                
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
                logger.debug(f"[PIPELINE] predict_x0_batch: Single-step - using full batch: {x_0_pred_out.shape}")
                self.x_t_latent_buffer = None
        else:
            logger.debug(f"[PIPELINE] predict_x0_batch: Not using denoising batch - iterative mode")
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                logger.debug(f"[PIPELINE] predict_x0_batch: Step {idx}, timestep: {t}")
                t = t.view(1,).repeat(self.frame_bff_size,)
                
                logger.debug(f"[PIPELINE] predict_x0_batch: Calling unet_step for step {idx}")
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx)
                logger.debug(f"[PIPELINE] predict_x0_batch: Step {idx} result - x_0_pred: {x_0_pred.shape}, range: [{x_0_pred.min().item():.6f}, {x_0_pred.max().item():.6f}]")
                
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
                    logger.debug(f"[PIPELINE] predict_x0_batch: Step {idx} - updated x_t_latent for next step: range: [{x_t_latent.min().item():.6f}, {x_t_latent.max().item():.6f}]")
            x_0_pred_out = x_0_pred

        logger.debug(f"[PIPELINE] predict_x0_batch: Final result shape: {x_0_pred_out.shape}")
        logger.debug(f"[PIPELINE] predict_x0_batch: Final result range: [{x_0_pred_out.min().item():.6f}, {x_0_pred_out.max().item():.6f}]")
        
        # Check for problematic values in final result
        if torch.isnan(x_0_pred_out).any():
            nan_count = torch.isnan(x_0_pred_out).sum().item()
            logger.debug(f"[PIPELINE] predict_x0_batch: *** ERROR: {nan_count} NaN values in final result! ***")
        if torch.isinf(x_0_pred_out).any():
            inf_count = torch.isinf(x_0_pred_out).sum().item()
            logger.debug(f"[PIPELINE] predict_x0_batch: *** ERROR: {inf_count} Inf values in final result! ***")
        if (x_0_pred_out == 0).all():
            logger.debug(f"[PIPELINE] predict_x0_batch: *** ERROR: All output values are zero! ***")

        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        logger.debug(f"[PIPELINE] __call__: Starting inference with input type: {type(x)}")
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if x is not None:
            logger.debug(f"[PIPELINE] __call__: Processing input image")
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            logger.debug(f"[PIPELINE] __call__: After preprocessing: {x.shape}, range: [{x.min().item():.6f}, {x.max().item():.6f}]")
            
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    logger.debug(f"[PIPELINE] __call__: Similar filter rejected image, returning previous result")
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            
            logger.debug(f"[PIPELINE] __call__: Calling encode_image...")
            x_t_latent = self.encode_image(x)
            logger.debug(f"[PIPELINE] __call__: encode_image result: {x_t_latent.shape}, range: [{x_t_latent.min().item():.6f}, {x_t_latent.max().item():.6f}]")
        else:
            logger.debug(f"[PIPELINE] __call__: Generating random latent")
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
            logger.debug(f"[PIPELINE] __call__: Random latent: {x_t_latent.shape}, range: [{x_t_latent.min().item():.6f}, {x_t_latent.max().item():.6f}]")
        
        logger.debug(f"[PIPELINE] __call__: Calling predict_x0_batch...")
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        logger.debug(f"[PIPELINE] __call__: predict_x0_batch result: {x_0_pred_out.shape}, range: [{x_0_pred_out.min().item():.6f}, {x_0_pred_out.max().item():.6f}]")
        
        logger.debug(f"[PIPELINE] __call__: Calling decode_image...")
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        logger.debug(f"[PIPELINE] __call__: decode_image result: {x_output.shape}, range: [{x_output.min().item():.6f}, {x_output.max().item():.6f}]")

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        logger.debug(f"[PIPELINE] __call__: Inference completed in {inference_time:.3f}s, returning result")
        
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
