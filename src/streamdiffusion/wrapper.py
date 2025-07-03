import gc
import os
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any, Tuple

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from .pipeline import StreamDiffusion
from .image_utils import postprocess_image

from .acceleration.tensorrt.model_detection import detect_model_from_diffusers_unet

from .pipeline import StreamDiffusion
from .image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamDiffusionWrapper:
    """
    StreamDiffusionWrapper for real-time image generation.
    
    This wrapper provides a unified interface for both single prompts and prompt blending:
    
    ## Unified Interface:
    ```python
    # Single prompt
    wrapper.prepare("a beautiful cat")
    
    # Prompt blending
    wrapper.prepare([("cat", 0.7), ("dog", 0.3)])
    
    # Prompt + seed blending  
    wrapper.prepare(
        prompt=[("style1", 0.6), ("style2", 0.4)],
        seed_list=[(123, 0.8), (456, 0.2)]
    )
    ```
    
    ## Runtime Updates:
    ```python
    # Update single prompt
    wrapper.update_prompt("new prompt")
    
    # Update prompt blending
    wrapper.update_prompt([("new1", 0.5), ("new2", 0.5)])
    
    # Update combined parameters
    wrapper.update_stream_params(
        prompt_list=[("bird", 0.6), ("fish", 0.4)],
        seed_list=[(789, 0.3), (101, 0.7)]
    )
    ```
    
    ## Weight Management:
    - All weights are normalized by default (sum to 1.0) unless normalize_weights=False
    - Use update_prompt_weights([0.8, 0.2]) to change weights without re-encoding prompts
    - Use update_seed_weights([0.3, 0.7]) to change weights without regenerating noise
    
    ## Cache Management:
    - Prompt embeddings and seed noise tensors are automatically cached for performance
    - Use get_cache_info() to inspect cache statistics
    - Use clear_caches() to free memory
    """
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        build_engines_if_missing: bool = True,
        normalize_weights: bool = True,
        # ControlNet options
        use_controlnet: bool = False,
        controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        normalize_weights : bool, optional
            Whether to normalize weights in prompt and seed blending to sum to 1,
            by default True. When False, weights > 1 will amplify embeddings/noise.
        use_controlnet : bool, optional
            Whether to enable ControlNet support, by default False.
        controlnet_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            ControlNet configuration(s), by default None.
            Can be a single config dict or list of config dicts for multiple ControlNets.
            Each config should contain: model_id, preprocessor (optional), conditioning_scale, etc.
        """
        self.sd_turbo = "turbo" in model_id_or_path
        self.use_controlnet = use_controlnet

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            width=width,
            height=height,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            build_engines_if_missing=build_engines_if_missing,
            normalize_weights=normalize_weights,
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
        )

        # Store acceleration settings for ControlNet integration
        self._acceleration = acceleration
        self._engine_dir = engine_dir

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        prompt: Union[str, List[Tuple[str, float]]],
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        # Blending-specific parameters (only used when prompt is a list)
        interpolation_method: Literal["linear", "slerp"] = "slerp",
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
    ) -> None:
        """
        Prepares the model for inference.

        Supports both single prompts and prompt blending based on the prompt parameter type.

        Parameters
        ----------
        prompt : Union[str, List[Tuple[str, float]]]
            Either a single prompt string or a list of (prompt, weight) tuples for blending.
            Examples:
            - Single: "a beautiful cat"
            - Blending: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : str, optional
            The negative prompt, by default "".
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise, by default 1.0.
        interpolation_method : Literal["linear", "slerp"], optional
            Method for interpolating between prompt embeddings (only used for prompt blending), 
            by default "slerp".
        seed_list : Optional[List[Tuple[int, float]]], optional
            List of seeds with weights for blending, by default None.
        seed_interpolation_method : Literal["linear", "slerp"], optional
            Method for interpolating between seed noise tensors, by default "linear".
        """
        

        # Handle both single prompt and prompt blending
        if isinstance(prompt, str):
            # Single prompt mode (legacy interface)
            self.stream.prepare(
                prompt,
                negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta,
            )
            
            # Apply seed blending if provided
            if seed_list is not None:
                self.stream.update_stream_params(
                    seed_list=seed_list,
                    seed_interpolation_method=seed_interpolation_method,
                )
        
        elif isinstance(prompt, list):
            # Prompt blending mode
            if not prompt:
                raise ValueError("prepare: prompt list cannot be empty")
            
            # Prepare with first prompt to initialize the pipeline
            first_prompt = prompt[0][0]
            self.stream.prepare(
                first_prompt,
                negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta,
            )
            
            # Then apply prompt blending (and seed blending if provided)
            self.stream.update_stream_params(
                prompt_list=prompt,
                negative_prompt=negative_prompt,
                interpolation_method=interpolation_method,
                seed_list=seed_list,
                seed_interpolation_method=seed_interpolation_method,
            )
        
        else:
            raise TypeError(f"prepare: prompt must be str or List[Tuple[str, float]], got {type(prompt)}")

    def update_prompt(
        self, 
        prompt: Union[str, List[Tuple[str, float]]], 
        negative_prompt: str = "",
        interpolation_method: Literal["linear", "slerp"] = "slerp",
        clear_blending: bool = True,
        warn_about_conflicts: bool = True
    ) -> None:
        """
        Update to a new prompt or prompt blending configuration.
        
        Supports both single prompts and prompt blending based on the prompt parameter type.
        
        Parameters
        ----------
        prompt : Union[str, List[Tuple[str, float]]]
            Either a single prompt string or a list of (prompt, weight) tuples for blending.
            Examples:
            - Single: "a beautiful cat"
            - Blending: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : str, optional
            The negative prompt (used with blending), by default "".
        interpolation_method : Literal["linear", "slerp"], optional
            Method for interpolating between prompt embeddings (used with blending), by default "slerp".
        clear_blending : bool, optional
            Whether to clear existing blending when switching to single prompt, by default True.
        warn_about_conflicts : bool, optional
            Whether to warn about conflicts when switching between modes, by default True.
        """
        # Handle both single prompt and prompt blending
        if isinstance(prompt, str):
            # Single prompt mode
            current_prompts = self.stream._param_updater.get_current_prompts()
            if current_prompts and len(current_prompts) > 1 and warn_about_conflicts:
                print("update_prompt: WARNING: Active prompt blending detected!")
                print(f"  Current blended prompts: {len(current_prompts)} prompts")
                print("  Switching to single prompt mode.")
                if clear_blending:
                    print("  Clearing prompt blending cache...")
            
            if clear_blending:
                # Clear the blending caches to avoid conflicts
                self.stream._param_updater.clear_caches()
            
            # Use the legacy single prompt update
            self.stream.update_prompt(prompt)
        
        elif isinstance(prompt, list):
            # Prompt blending mode
            if not prompt:
                raise ValueError("update_prompt: prompt list cannot be empty")
                
            current_prompts = self.stream._param_updater.get_current_prompts()
            if len(current_prompts) <= 1 and warn_about_conflicts:
                print("update_prompt: Switching from single prompt to prompt blending mode.")
            
            # Apply prompt blending
            self.stream.update_stream_params(
                prompt_list=prompt,
                negative_prompt=negative_prompt,
                interpolation_method=interpolation_method,
            )
        
        else:
            raise TypeError(f"update_prompt: prompt must be str or List[Tuple[str, float]], got {type(prompt)}")

    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        # New prompt blending parameters
        prompt_list: Optional[List[Tuple[str, float]]] = None,
        negative_prompt: Optional[str] = None,
        interpolation_method: Literal["linear", "slerp"] = "slerp",
        # New seed blending parameters  
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
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
        width : Optional[int]
            The image width (must be multiple of 64, between 512-1024).
        height : Optional[int]
            The image height (must be multiple of 64, between 512-1024).
        prompt_list : Optional[List[Tuple[str, float]]]
            List of prompts with weights for blending. Each tuple contains (prompt_text, weight).
            Example: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : Optional[str]
            The negative prompt to apply to all blended prompts.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between prompt embeddings, by default "slerp".
        seed_list : Optional[List[Tuple[int, float]]]
            List of seeds with weights for blending. Each tuple contains (seed_value, weight).
            Example: [(123, 0.6), (456, 0.4)]
        seed_interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        """
        new_stream = self.stream._param_updater.update_stream_params(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            t_index_list=t_index_list,
            seed=seed,
            width=width,
            height=height,
            prompt_list=prompt_list,
            negative_prompt=negative_prompt,
            interpolation_method=interpolation_method,
            seed_list=seed_list,
            seed_interpolation_method=seed_interpolation_method,
        )
        # If a new stream was created (pipeline restart), update self.stream
        if new_stream is not None:
            self.stream = new_stream
        # Update wrapper's width/height properties if resolution changed
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

    def set_normalize_weights(self, normalize: bool) -> None:
        """Set whether to normalize weights in prompt and seed blending operations."""
        self.stream.set_normalize_weights(normalize)
        
    def get_normalize_weights(self) -> bool:
        """Get the current weight normalization setting."""
        return self.stream.get_normalize_weights()

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from. If provided, will update to single prompt mode
            and may conflict with active prompt blending.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.update_prompt(prompt, warn_about_conflicts=True)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers.models.clip import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(device=self.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            # Use stream's current resolution for fallback image
            self.nsfw_fallback_img = Image.new("RGB", (self.stream.height, self.stream.width), (0, 0, 0))

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from. If provided, will update to single prompt mode
            and may conflict with active prompt blending.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.update_prompt(prompt, warn_about_conflicts=True)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers.models.clip import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(device=self.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            # Use stream's current resolution for fallback image
            self.nsfw_fallback_img = Image.new("RGB", (self.stream.height, self.stream.width), (0, 0, 0))

        return image

    def preprocess_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        # Use stream's current resolution instead of wrapper's cached values
        current_width = self.stream.width
        current_height = self.stream.height
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((current_width, current_height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((current_width, current_height))

        return self.stream.image_processor.preprocess(
            image, current_height, current_width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image (OPTIMIZED VERSION)

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        # Fast paths for non-PIL outputs (avoid unnecessary conversions)
        if output_type == "latent":
            return image_tensor
        elif output_type == "pt":
            # Denormalize on GPU, return tensor
            return self._denormalize_on_gpu(image_tensor)
        elif output_type == "np":
            # Denormalize on GPU, then single efficient CPU transfer
            denormalized = self._denormalize_on_gpu(image_tensor)
            return denormalized.cpu().permute(0, 2, 3, 1).float().numpy()


        # PIL output path (optimized)
        if output_type == "pil":
            if self.frame_buffer_size > 1:
                return self._tensor_to_pil_optimized(image_tensor)
            else:
                return self._tensor_to_pil_optimized(image_tensor)[0]


        # Fallback to original method for any unexpected output types
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]


    def _denormalize_on_gpu(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize image tensor on GPU for efficiency


        Args:
            image_tensor: Input tensor on GPU


        Returns:
            Denormalized tensor on GPU, clamped to [0,1]
        """
        return (image_tensor / 2 + 0.5).clamp(0, 1)


    def _tensor_to_pil_optimized(self, image_tensor: torch.Tensor) -> List[Image.Image]:
        """
        Optimized tensor to PIL conversion with minimal CPU transfers


        Args:
            image_tensor: Input tensor on GPU


        Returns:
            List of PIL Images
        """
        # Denormalize on GPU first
        denormalized = self._denormalize_on_gpu(image_tensor)


        # Convert to uint8 on GPU to reduce transfer size
        # Scale to [0, 255] and convert to uint8
        # Scale to [0, 255] and convert to uint8
        uint8_tensor = (denormalized * 255).clamp(0, 255).to(torch.uint8)


        # Single efficient CPU transfer
        cpu_tensor = uint8_tensor.cpu()


        # Convert to HWC format for PIL
        # From BCHW to BHWC
        cpu_tensor = cpu_tensor.permute(0, 2, 3, 1)


        # Convert to PIL images efficiently
        pil_images = []
        for i in range(cpu_tensor.shape[0]):
            img_array = cpu_tensor[i].numpy()


            if img_array.shape[-1] == 1:
                # Grayscale
                pil_images.append(Image.fromarray(img_array.squeeze(-1), mode="L"))
            else:
                # RGB
                pil_images.append(Image.fromarray(img_array))


        return pil_images

    def _load_model(
        self,
        model_id_or_path: str,
        width: int,
        height: int,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        build_engines_if_missing: bool = True,
        normalize_weights: bool = True,
        use_controlnet: bool = False,
        controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.
        7. Apply ControlNet patch if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_controlnet : bool, optional
            Whether to apply ControlNet patch, by default False.
        controlnet_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            ControlNet configuration(s), by default None.

        Returns
        -------
        StreamDiffusion
            The loaded model (potentially wrapped with ControlNet pipeline).
        """

        # Try different loading methods in order - prioritize AutoPipeline for safety
        from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image
        
        loading_methods = [
            (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
            (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
            (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file")
        ]

        pipe = None
        for method, method_name in loading_methods:
            try:
                pipe = method(model_id_or_path).to(device=self.device, dtype=self.dtype)
                print(f"_load_model: Successfully loaded using {method_name}")
                break
            except Exception as e:
                continue
        
        if pipe is None:
            traceback.print_exc()
            print("_load_model: All loading methods failed. Model doesn't exist or is incompatible.")
            exit()

        # Use existing model detection instead of guessing from config
        model_type = detect_model_from_diffusers_unet(pipe.unet)
        is_sdxl = model_type == "SDXL"
        
        # Store model info for later use (after TensorRT conversion)
        self._detected_model_type = model_type

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
            normalize_weights=normalize_weights,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                # Use TAESD XL for SDXL models, regular TAESD for SD 1.5
                taesd_model = "madebyollin/taesdxl" if is_sdxl else "madebyollin/taesd"
                stream.vae = AutoencoderTiny.from_pretrained(taesd_model).to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )
                # Add ControlNet detection and support
                from streamdiffusion.acceleration.tensorrt.model_detection import (
                    extract_unet_architecture,
                    validate_architecture
                )
                from streamdiffusion.acceleration.tensorrt.controlnet_wrapper import create_controlnet_wrapper

                def create_prefix(
                    model_id_or_path: str,
                    max_batch: int,
                    min_batch_size: int,
                    width: int,
                    height: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    # Use dynamic engine naming to distinguish from static engines
                    dynamic_suffix = "dyn-512-1024"
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch}--min_batch-{min_batch_size}--mode-{self.mode}--{dynamic_suffix}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch}--min_batch-{min_batch_size}--mode-{self.mode}--{dynamic_suffix}"

                # Always enable ControlNet TensorRT support to create universal engines
                use_controlnet_trt = False
                unet_arch = {}
                
                if acceleration == "tensorrt":
                    try:
                        model_type = detect_model_from_diffusers_unet(stream.unet)
                        unet_arch = extract_unet_architecture(stream.unet)
                        unet_arch = validate_architecture(unet_arch, model_type)
                        use_controlnet_trt = True
                        print(f"Building universal TensorRT engines with ControlNet support for {model_type}")
                    except Exception as e:
                        print(f"ControlNet architecture detection failed: {e}, building engines without ControlNet support")
                        use_controlnet_trt = False

                # Use the engine_dir parameter passed to this function, with fallback to instance variable
                engine_dir = Path(engine_dir if engine_dir else getattr(self, '_engine_dir', 'engines'))
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        width=self.width,
                        height=self.height,
                    ),
                    "vae_decoder.engine",
                )

                # Check if all required engines exist
                missing_engines = []
                if not os.path.exists(unet_path):
                    missing_engines.append(f"UNet engine: {unet_path}")
                if not os.path.exists(vae_decoder_path):
                    missing_engines.append(f"VAE decoder engine: {vae_decoder_path}")
                if not os.path.exists(vae_encoder_path):
                    missing_engines.append(f"VAE encoder engine: {vae_encoder_path}")

                if missing_engines:
                    if build_engines_if_missing:
                        print(f"Missing TensorRT engines, building them...")
                        for engine in missing_engines:
                            print(f"  - {engine}")
                    else:
                        error_msg = f"Required TensorRT engines are missing and build_engines_if_missing=False:\n"
                        for engine in missing_engines:
                            error_msg += f"  - {engine}\n"
                        error_msg += f"\nTo build engines, set build_engines_if_missing=True or run the build script manually."
                        raise RuntimeError(error_msg)


                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)

                    print(f"Creating UNet model for image size: {self.width}x{self.height}")


                    print(f"Creating UNet model for image size: {self.width}x{self.height}")

                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                        use_control=use_controlnet_trt,
                        unet_arch=unet_arch if use_controlnet_trt else None,
                        image_height=self.height,
                        image_width=self.width,
                    )


                    # Use ControlNet wrapper if ControlNet support is enabled
                    if use_controlnet_trt:
                        control_input_names = unet_model.get_input_names()
                        wrapped_unet = create_controlnet_wrapper(stream.unet, control_input_names)
                        compile_unet(
                            wrapped_unet,
                            unet_model,
                            unet_path + ".onnx",
                            unet_path + ".opt.onnx",
                            unet_path,
                            opt_batch_size=stream.trt_unet_batch_size,
                            engine_build_options={
                                'opt_image_height': self.height,
                                'opt_image_width': self.width,
                                'build_dynamic_shape': True,  # Force dynamic shapes
                                'min_image_resolution': 512,
                                'max_image_resolution': 1024,
                            },
                        )
                    else:
                        compile_unet(
                            stream.unet,
                            unet_model,
                            unet_path + ".onnx",
                            unet_path + ".opt.onnx",
                            unet_path,
                            opt_batch_size=stream.trt_unet_batch_size,
                            engine_build_options={
                                'opt_image_height': self.height,
                                'opt_image_width': self.width,
                                'build_dynamic_shape': True,  # Force dynamic shapes
                                'min_image_resolution': 512,
                                'max_image_resolution': 1024,
                            },
                        )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        engine_build_options={
                            'opt_image_height': self.height,
                            'opt_image_width': self.width,
                            'build_dynamic_shape': True,  # Force dynamic shapes
                            'min_image_resolution': 512,
                            'max_image_resolution': 1024,
                        },
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        engine_build_options={
                            'opt_image_height': self.height,
                            'opt_image_width': self.width,
                            'build_dynamic_shape': True,  # Force dynamic shapes
                            'min_image_resolution': 512,
                            'max_image_resolution': 1024,
                        },
                    )

                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_stream, use_cuda_graph=False
                )

                # Always set ControlNet support to True for universal TensorRT engines
                # This allows the engine to accept dummy inputs when no ControlNets are used
                stream.unet.use_control = True
                if use_controlnet_trt and unet_arch:
                    stream.unet.unet_arch = unet_arch
                    stream.unet.unet_arch = unet_arch

                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                stream.vae.config = vae_config
                stream.vae.dtype = vae_dtype
                stream.vae.config = vae_config
                stream.vae.dtype = vae_dtype

                gc.collect()
                torch.cuda.empty_cache()

            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")
            # TODO: Make pytorch fallback configurable
            raise NotImplementedError("Acceleration has failed. Automatic pytorch inference fallback temporarily disabled.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers.models.clip import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(device=pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            # Use stream's current resolution for fallback image
            self.nsfw_fallback_img = Image.new("RGB", (stream.height, stream.width), (0, 0, 0))

        # Apply ControlNet patch if needed
        if use_controlnet and controlnet_config:
            stream = self._apply_controlnet_patch(stream, controlnet_config, acceleration, engine_dir, self._detected_model_type)

        return stream

    def _apply_controlnet_patch(self, stream: StreamDiffusion, controlnet_config: Union[Dict[str, Any], List[Dict[str, Any]]], acceleration: str = "none", engine_dir: str = "engines", model_type: str = "SD15") -> Any:
        """
        Apply ControlNet patch to StreamDiffusion using detected model type

        Args:
            stream: Base StreamDiffusion instance
            controlnet_config: ControlNet configuration(s)
            model_type: Detected model type from original UNet

        Returns:
            ControlNet-enabled pipeline (ControlNetPipeline or SDXLTurboControlNetPipeline)
        """
        # Use provided model type (detected before TensorRT conversion)
        if model_type == "SDXL":
            from streamdiffusion.controlnet.controlnet_sdxlturbo_pipeline import SDXLTurboControlNetPipeline
            controlnet_pipeline = SDXLTurboControlNetPipeline(stream, self.device, self.dtype)
        else:  # SD15, SD21, etc. all use same ControlNet pipeline
            from streamdiffusion.controlnet.controlnet_pipeline import ControlNetPipeline
            controlnet_pipeline = ControlNetPipeline(stream, self.device, self.dtype)

        # Check if we should use TensorRT ControlNet acceleration
        use_controlnet_tensorrt = (acceleration == "tensorrt")
        
        # Set the detected model type to avoid re-detection from TensorRT engine
        controlnet_pipeline._detected_model_type = model_type


        # Initialize ControlNet engine pool if using TensorRT acceleration
        if use_controlnet_tensorrt:
            from streamdiffusion.acceleration.tensorrt.engine_pool import ControlNetEnginePool
            from polygraphy import cuda


            # Create engine pool with same engine directory structure as UNet
            stream_cuda = cuda.Stream()
            controlnet_pool = ControlNetEnginePool(engine_dir, stream_cuda, self.width, self.height)

            # Store pool on the pipeline for later use
            controlnet_pipeline._controlnet_pool = controlnet_pool
            controlnet_pipeline._use_tensorrt = True
            # Also set on stream where ControlNet pipeline expects to find it
            stream.controlnet_engine_pool = controlnet_pool
            print("Initialized ControlNet TensorRT engine pool")
        else:
            controlnet_pipeline._use_tensorrt = False
            print("Loading ControlNet in PyTorch mode (no TensorRT acceleration)")


        # Setup ControlNets from config
        if not isinstance(controlnet_config, list):
            controlnet_config = [controlnet_config]


        for config in controlnet_config:
            model_id = config.get('model_id')
            if not model_id:
                continue

            preprocessor = config.get('preprocessor', None)
            conditioning_scale = config.get('conditioning_scale', 1.0)
            enabled = config.get('enabled', True)
            preprocessor_params = config.get('preprocessor_params', None)
            control_image = config.get('control_image', None)  # Extract control image from config

            try:
                # Pass config dictionary directly
                cn_config = {
                    'model_id': model_id,
                    'preprocessor': preprocessor,
                    'conditioning_scale': conditioning_scale,
                    'enabled': enabled,
                    'preprocessor_params': preprocessor_params or {}
                }

                # Add ControlNet with control image if provided
                controlnet_pipeline.add_controlnet(cn_config, control_image)
                print(f"_apply_controlnet_patch: Successfully added ControlNet: {model_id}")
            except Exception as e:
                print(f"_apply_controlnet_patch: Failed to add ControlNet {model_id}: {e}")
                import traceback
                traceback.print_exc()

        return controlnet_pipeline

    # ControlNet convenience methods
    def add_controlnet(self,
                      model_id: str,
                      preprocessor: Optional[str] = None,
                      conditioning_scale: float = 1.0,
                      control_image: Optional[Union[str, Image.Image, np.ndarray, torch.Tensor]] = None,
                      enabled: bool = True,
                      preprocessor_params: Optional[Dict[str, Any]] = None) -> int:
        """Forward add_controlnet call to the underlying ControlNet pipeline"""
        if not self.use_controlnet:
            raise RuntimeError("add_controlnet: ControlNet support not enabled. Set use_controlnet=True in constructor.")

        cn_config = {
            'model_id': model_id,
            'preprocessor': preprocessor,
            'conditioning_scale': conditioning_scale,
            'enabled': enabled,
            'preprocessor_params': preprocessor_params or {}
        }
        return self.stream.add_controlnet(cn_config, control_image)



    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor], index: Optional[int] = None) -> None:
        """Forward update_control_image_efficient call to the underlying ControlNet pipeline"""
        if not self.use_controlnet:
            raise RuntimeError("update_control_image_efficient: ControlNet support not enabled. Set use_controlnet=True in constructor.")
        
        self.stream.update_control_image_efficient(control_image, index=index)

    def update_controlnet_scale(self, index: int, scale: float) -> None:
        """Forward update_controlnet_scale call to the underlying ControlNet pipeline"""
        if not self.use_controlnet:
            raise RuntimeError("update_controlnet_scale: ControlNet support not enabled. Set use_controlnet=True in constructor.")
        
        self.stream.update_controlnet_scale(index, scale)

    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """Forward get_last_processed_image call to the underlying ControlNet pipeline"""
        if not self.use_controlnet:
            raise RuntimeError("get_last_processed_image: ControlNet support not enabled. Set use_controlnet=True in constructor.")
        
        return self.stream.get_last_processed_image(index)


    
    def update_seed_blending(
        self, 
        seed_list: List[Tuple[int, float]], 
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """
        Update seed blending with multiple weighted seeds.
        
        Parameters
        ----------
        seed_list : List[Tuple[int, float]]
            List of seeds with weights. Each tuple contains (seed_value, weight).
            Example: [(123, 0.6), (456, 0.4)]
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        """
        self.stream._param_updater.update_stream_params(
            seed_list=seed_list,
            seed_interpolation_method=interpolation_method
        )
    
    def update_prompt_weights(
        self, 
        prompt_weights: List[float],
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """
        Update weights for current prompt list without re-encoding prompts.
        
        Parameters
        ----------
        prompt_weights : List[float]
            New weights for the current prompt list.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between prompt embeddings, by default "slerp".
        """
        self.stream._param_updater.update_prompt_weights(prompt_weights, interpolation_method)
    
    def update_seed_weights(
        self, 
        seed_weights: List[float],
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """
        Update weights for current seed list without regenerating noise.
        
        Parameters
        ----------
        seed_weights : List[float]
            New weights for the current seed list.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        """
        self.stream._param_updater.update_seed_weights(seed_weights, interpolation_method)

    def get_current_prompts(self) -> List[Tuple[str, float]]:
        """
        Get the current prompt list with weights.
        
        Returns
        -------
        List[Tuple[str, float]]
            Current prompt list with weights.
        """
        return self.stream._param_updater.get_current_prompts()
    
    def get_current_seeds(self) -> List[Tuple[int, float]]:
        """
        Get the current seed list with weights.
        
        Returns
        -------
        List[Tuple[int, float]]
            Current seed list with weights.
        """
        return self.stream._param_updater.get_current_seeds()
    
    def get_cache_info(self) -> Dict:
        """
        Get cache statistics for prompt and seed blending.
        
        Returns
        -------
        Dict
            Cache information including hits, misses, and cache sizes.
        """
        return self.stream._param_updater.get_cache_info()
    
    def clear_caches(self) -> None:
        """Clear all cached prompt embeddings and seed noise tensors."""
        self.stream._param_updater.clear_caches()

    def update_prompt_at_index(
        self, 
        index: int, 
        new_prompt: str,
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """
        Update a specific prompt by index without changing other prompts.
        
        Parameters
        ----------
        index : int
            Index of the prompt to update.
        new_prompt : str
            New prompt text.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between prompt embeddings, by default "slerp".
        """
        self.stream._param_updater.update_prompt_at_index(index, new_prompt, interpolation_method)
    
    def add_prompt(
        self, 
        prompt: str, 
        weight: float = 1.0,
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """
        Add a new prompt to the current blending configuration.
        
        Parameters
        ----------
        prompt : str
            Prompt text to add.
        weight : float
            Weight for the new prompt, by default 1.0.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between prompt embeddings, by default "slerp".
        """
        self.stream._param_updater.add_prompt(prompt, weight, interpolation_method)
    
    def remove_prompt_at_index(
        self, 
        index: int,
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """
        Remove a prompt from the current blending configuration by index.
        
        Parameters
        ----------
        index : int
            Index of the prompt to remove.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between remaining prompt embeddings, by default "slerp".
        """
        self.stream._param_updater.remove_prompt_at_index(index, interpolation_method)
    
    def update_seed_at_index(
        self, 
        index: int, 
        new_seed: int,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """
        Update a specific seed by index without changing other seeds.
        
        Parameters
        ----------
        index : int
            Index of the seed to update.
        new_seed : int
            New seed value.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        """
        self.stream._param_updater.update_seed_at_index(index, new_seed, interpolation_method)
    
    def add_seed(
        self, 
        seed: int, 
        weight: float = 1.0,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """
        Add a new seed to the current blending configuration.
        
        Parameters
        ----------
        seed : int
            Seed value to add.
        weight : float
            Weight for the new seed, by default 1.0.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        """
        self.stream._param_updater.add_seed(seed, weight, interpolation_method)
    
    def remove_seed_at_index(
        self, 
        index: int,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """
        Remove a seed from the current blending configuration by index.
        
        Parameters
        ----------
        index : int
            Index of the seed to remove.
        interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between remaining seed noise tensors, by default "linear".
        """
        self.stream._param_updater.remove_seed_at_index(index, interpolation_method)




