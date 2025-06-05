import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionControlNetImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from diffusers.utils import PIL_INTERPOLATION

from streamdiffusion.image_filter import SimilarImageFilter
from diffusers import ControlNetModel, MultiControlNetModel # Assuming these imports are available or can be added


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnet) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        
    def forward(self, sample, timestep, encoder_hidden_states, image):
        # hard-coded since it is not clear how to integrate this argument into tensorrt
        conditioning_scale = 1.0

        down_samples, mid_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=image,
            guess_mode=False,
            return_dict=False,
        )

        down_block_res_samples = [
            down_sample * conditioning_scale
            for down_sample in down_samples
        ]
        mid_block_res_sample = conditioning_scale * mid_sample
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )
        return noise_pred


class StreamUNetControlDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionControlNetImg2ImgPipeline,
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        strength: float = 0.8,
        num_inference_steps: int = 4,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        
        self.strength = strength
        self.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps, self.num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, self.device
        )
        self.t_list = list(range(self.num_inference_steps))
        self.denoising_steps_num = self.num_inference_steps

        self.cfg_type = cfg_type

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

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor, do_convert_rgb=True)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler
        self.text_encoder = pipe.text_encoder
        self.unet = UNet2DConditionControlNetModel(
            pipe.unet, pipe.controlnet
        )
        self.vae = pipe.config.vae

        self.inference_time_ema = 0
        self.prompt_update_idx = 1

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: int = 2,
    ) -> None:
        if generator is None:
            generator = torch.Generator(device=self.device)
        self.generator = generator
        self.generator.manual_seed(seed)
        
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
            self.control_x_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    3,
                    self.height,
                    self.width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None
            self.control_x_buffer = None

        # print('x_t_latent_buffer: ', self.x_t_latent_buffer.shape, 'control_x_buffer: ', self.control_x_buffer.shape)

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True
        self.do_classifier_free_guidance = do_classifier_free_guidance

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        # print('denoise_steps_num: ', self.denoising_steps_num)
        # print('encoder_output: ', encoder_output[0].shape, encoder_output[1].shape)
        self.prompt_embeds = encoder_output[0].repeat(self.denoising_steps_num, 1, 1)
        # print('prompt_embeds: ', self.prompt_embeds.shape)

        if self.use_denoising_batch and self.cfg_type == "full":
            self.uncond_prompt_embeds = encoder_output[1].repeat(self.denoising_steps_num, 1, 1)
        elif self.use_denoising_batch and self.cfg_type == "initialize":
            self.uncond_prompt_embeds = encoder_output[1]

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [self.uncond_prompt_embeds, self.prompt_embeds], dim=0
            )
            # print('uncond_prompt_embeds: ', self.uncond_prompt_embeds.shape)
            # print('prompt_embeds: ', self.prompt_embeds.shape)

        #self.scheduler.set_timesteps(num_inference_steps, self.device)
        #self.timesteps = self.scheduler.timesteps.to(self.device)

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
        # print('sub_timesteps_tensor: ', self.sub_timesteps_tensor)

        # self.init_noise = torch.randn(
        #     (self.batch_size, 4, self.latent_height, self.latent_width),
        #     generator=generator,
        # ).to(device=self.device, dtype=self.dtype)
        self.init_noise = torch.randn(
            (self.denoising_steps_num, 4, self.latent_height, self.latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        self.init_noise = torch.repeat_interleave(
            self.init_noise,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

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
        self.c_skip = torch.repeat_interleave(
            self.c_skip,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.c_out = torch.repeat_interleave(
            self.c_out,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
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

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        # TODO: what if we have negative embeds
        if not self.do_classifier_free_guidance:
            self.prompt_embeds = torch.cat((encoder_output[0], self.prompt_embeds[:-self.frame_bff_size]), dim=0)
        elif self.cfg_type == "initialize":
            positive_prompt_embeds = self.prompt_embeds[self.frame_bff_size:]
            positive_prompt_embeds = torch.cat((encoder_output[0], positive_prompt_embeds[:-self.frame_bff_size]), dim=0)
            self.prompt_embeds = torch.cat(
                [self.uncond_prompt_embeds, positive_prompt_embeds], dim=0
            )
        elif self.cfg_type == "full":
            positive_prompt_embeds = self.prompt_embeds[self.batch_size:]
            positive_prompt_embeds = torch.cat((encoder_output[0], positive_prompt_embeds[:-self.frame_bff_size]), dim=0)
            self.prompt_embeds = torch.cat(
                [self.uncond_prompt_embeds, positive_prompt_embeds], dim=0
            )
        else:
            raise ValueError("Invalid cfg_type")

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
        image, 
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[:self.frame_bff_size], x_t_latent], dim=0)
            t_list = torch.concat([t_list[:self.frame_bff_size], t_list], dim=0)
            image = torch.concat([image[:self.frame_bff_size], image], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
            image = torch.concat([image, image], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # print('=' * 50)
        # print(
        #     'sample: ', x_t_latent_plus_uc.shape,
        #     't_list: ', t_list.shape,
        #     'prompt_embeds: ', self.prompt_embeds.shape, 
        #     'image: ', image.shape
        # )
        # print('=' * 50)

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            self.prompt_embeds,
            image,
        )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[self.frame_bff_size:]
            self.stock_noise = torch.concat(
                [model_pred[:self.frame_bff_size], self.stock_noise[self.frame_bff_size:]], dim=0
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
                        self.alpha_prod_t_sqrt[self.frame_bff_size:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:self.frame_bff_size]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[self.frame_bff_size:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:self.frame_bff_size]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[self.frame_bff_size:], self.init_noise[0:self.frame_bff_size]], dim=0
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
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent

    def predict_x0_batch(
        self, 
        x_t_latent: torch.Tensor,
        image,
    ) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer
        prev_image = self.control_x_buffer
        # print('prev_image: ', prev_image.shape, 'image: ', image.shape)

        if self.use_denoising_batch:
            # construct input batch with current input and buffer
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:self.frame_bff_size], self.stock_noise[:-self.frame_bff_size]), dim=0
                )
                image = torch.cat((image, prev_image), dim=0)
            
            t_list = self.sub_timesteps_tensor
            x_0_pred_batch, model_pred = self.unet_step(
                x_t_latent, image, t_list
            )

            # update buffer
            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-self.frame_bff_size:]
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[self.frame_bff_size:] * x_0_pred_batch[:-self.frame_bff_size]
                        + self.beta_prod_t_sqrt[self.frame_bff_size:] * self.init_noise[self.frame_bff_size:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[self.frame_bff_size:] * x_0_pred_batch[:-self.frame_bff_size]
                    )
                self.control_x_buffer = image[:-self.frame_bff_size]
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
                self.control_x_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(
                    x_t_latent, image, t_list, idx
                )
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
        self, 
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],control_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert image is not None
        if isinstance(image, torch.Tensor) or isinstance(image, list):
            assert len(image) == self.frame_bff_size
        elif isinstance(image, PIL.Image.Image):
            assert self.frame_bff_size == 1
        else:
            raise ValueError("Invalid input type")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = self.pipe.image_processor.preprocess(image, self.height, self.width).to(
            dtype=self.dtype, device=self.device
        )

        control_x = self.pipe.prepare_control_image(
            image=control_images,
            width=self.width,
            height=self.height,
            batch_size=1,
            num_images_per_prompt=1,
            device=self.device,
            dtype=self.dtype,
            do_classifier_free_guidance=False, # self.do_classifier_free_guidance,
            guess_mode=False,
        )
        # print('control_x: ', control_x.shape)

        if self.similar_image_filter:
            x = self.similar_filter(x)
            if x is None:
                time.sleep(self.inference_time_ema)
                return self.prev_image_result

        x_t_latent = self.encode_image(x)
        x_0_pred_out = self.predict_x0_batch(x_t_latent, control_x)
        x_output = self.decode_image(x_0_pred_out)

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output