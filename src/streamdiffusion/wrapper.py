import gc
import os
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any, Tuple

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from PIL import Image

import logging
logger = logging.getLogger(__name__)

from .pipeline import StreamDiffusion
from .image_utils import postprocess_image

from .model_detection import detect_model

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
    - Prompt weights are normalized by default (sum to 1.0) unless normalize_prompt_weights=False
    - Seed weights are normalized by default (sum to 1.0) unless normalize_seed_weights=False
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
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
        # ControlNet options
        use_controlnet: bool = False,
        controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        enable_pytorch_fallback: bool = False,
        # IPAdapter options
        use_ipadapter: bool = False,
        ipadapter_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        # Postprocessing options
        use_postprocessing: bool = False,
        postprocessing_config: Optional[List[Dict[str, Any]]] = None,
        skip_diffusion: bool = False,
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
        normalize_prompt_weights : bool, optional
            Whether to normalize prompt weights in blending to sum to 1,
            by default True. When False, weights > 1 will amplify embeddings.
        normalize_seed_weights : bool, optional
            Whether to normalize seed weights in blending to sum to 1,
            by default True. When False, weights > 1 will amplify noise.
        use_controlnet : bool, optional
            Whether to enable ControlNet support, by default False.
        controlnet_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            ControlNet configuration(s), by default None.
            Can be a single config dict or list of config dicts for multiple ControlNets.
            Each config should contain: model_id, preprocessor (optional), conditioning_scale, etc.
        enable_pytorch_fallback : bool, optional
            Whether to enable PyTorch fallback when acceleration fails, by default False.
            When True, falls back to PyTorch inference if TensorRT/xformers acceleration fails.
            When False, raises an exception when acceleration fails.
        """
        self.sd_turbo = "turbo" in model_id_or_path
        self.use_controlnet = use_controlnet
        self.enable_pytorch_fallback = enable_pytorch_fallback
        self.use_ipadapter = use_ipadapter
        self.ipadapter_config = ipadapter_config
        self.use_postprocessing = use_postprocessing
        self.postprocessing_config = postprocessing_config
        self.skip_diffusion = skip_diffusion

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
            normalize_prompt_weights=normalize_prompt_weights,
            normalize_seed_weights=normalize_seed_weights,
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
            enable_pytorch_fallback=enable_pytorch_fallback,
            use_ipadapter=use_ipadapter,
            ipadapter_config=ipadapter_config,
            use_postprocessing=use_postprocessing,
            postprocessing_config=postprocessing_config,
        )

        # Set wrapper reference on parameter updater so it can access pipeline structure
        self.stream._param_updater.wrapper = self

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
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
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
        prompt_interpolation_method : Literal["linear", "slerp"], optional
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
                self.update_stream_params(
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
            self.update_stream_params(
                prompt_list=prompt,
                negative_prompt=negative_prompt,
                prompt_interpolation_method=prompt_interpolation_method,
                seed_list=seed_list,
                seed_interpolation_method=seed_interpolation_method,
            )

        else:
            raise TypeError(f"prepare: prompt must be str or List[Tuple[str, float]], got {type(prompt)}")

    def update_prompt(
        self,
        prompt: Union[str, List[Tuple[str, float]]],
        negative_prompt: str = "",
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
        clear_blending: bool = True,
        warn_about_conflicts: bool = True
    ) -> None:
        """
        Update to a new prompt or prompt blending configuration.

        Supports both single prompts and prompt blending based on the prompt parameter type.

        This is for legacy compatibility, use update_stream_params instead

        Parameters
        ----------
        prompt : Union[str, List[Tuple[str, float]]]
            Either a single prompt string or a list of (prompt, weight) tuples for blending.
            Examples:
            - Single: "a beautiful cat"
            - Blending: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : str, optional
            The negative prompt (used with blending), by default "".
        prompt_interpolation_method : Literal["linear", "slerp"], optional
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
                logger.warning("update_prompt: WARNING: Active prompt blending detected!")
                logger.warning(f"  Current blended prompts: {len(current_prompts)} prompts")
                logger.warning("  Switching to single prompt mode.")
                if clear_blending:
                    logger.warning("  Clearing prompt blending cache...")

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
                logger.warning("update_prompt: Switching from single prompt to prompt blending mode.")

            # Apply prompt blending
            self.update_stream_params(
                prompt_list=prompt,
                negative_prompt=negative_prompt,
                prompt_interpolation_method=prompt_interpolation_method,
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
        # Prompt blending parameters
        prompt_list: Optional[List[Tuple[str, float]]] = None,
        negative_prompt: Optional[str] = None,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
        normalize_prompt_weights: Optional[bool] = None,
        # Seed blending parameters
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
        normalize_seed_weights: Optional[bool] = None,
        # ControlNet configuration
        controlnet_config: Optional[List[Dict[str, Any]]] = None,
        # IPAdapter configuration
        ipadapter_config: Optional[Dict[str, Any]] = None,
        # Postprocessing configuration
        postprocessing_config: Optional[List[Dict[str, Any]]] = None,
        skip_diffusion: Optional[bool] = None,
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
            List of prompts with weights for blending. Each tuple contains (prompt_text, weight).
            Example: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : Optional[str]
            The negative prompt to apply to all blended prompts.
        prompt_interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between prompt embeddings, by default "slerp".
        normalize_prompt_weights : Optional[bool]
            Whether to normalize prompt weights in blending to sum to 1, by default None (no change).
            When False, weights > 1 will amplify embeddings.
        seed_list : Optional[List[Tuple[int, float]]]
            List of seeds with weights for blending. Each tuple contains (seed_value, weight).
            Example: [(123, 0.6), (456, 0.4)]
        seed_interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        normalize_seed_weights : Optional[bool]
            Whether to normalize seed weights in blending to sum to 1, by default None (no change).
            When False, weights > 1 will amplify noise.
        controlnet_config : Optional[List[Dict[str, Any]]]
            Complete ControlNet configuration list defining the desired state.
            Each dict contains: model_id, preprocessor, conditioning_scale, enabled, 
            preprocessor_params, etc. System will diff current vs desired state and 
            perform minimal add/remove/update operations.
        ipadapter_config : Optional[Dict[str, Any]]
            IPAdapter configuration dict containing scale, style_image, etc.
        postprocessing_config : Optional[List[Dict[str, Any]]]
            List of postprocessor configurations defining the desired state.
            Each dict contains: name, enabled, scale, preprocessor_params, etc.
        skip_diffusion : Optional[bool]
            Whether to skip diffusion entirely and process input directly.
        """
        # Handle wrapper-level state
        if skip_diffusion is not None:
            self.skip_diffusion = skip_diffusion
        
        # Handle all stream parameters via parameter updater (including postprocessing)
        self.stream._param_updater.update_stream_params(
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
            controlnet_config=controlnet_config,
            ipadapter_config=ipadapter_config,
            postprocessing_config=postprocessing_config,
        )

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
        if self.skip_diffusion:
            return self._process_skip_diffusion(image, prompt)
        
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def _process_skip_diffusion(
        self, 
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None, 
        prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Process input directly without diffusion.
        
        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to process directly.
        prompt : Optional[str]
            Prompt (ignored in skip mode, but kept for API consistency).
            
        Returns
        -------
        Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]
            The processed image.
        """
        if self.mode == "txt2img":
            logger.warning("_process_skip_diffusion: skip_diffusion mode not applicable for txt2img - no input image")
            return self.txt2img(prompt)
        
        if image is None:
            raise ValueError("_process_skip_diffusion: image required for skip diffusion mode")
        
        # Handle input tensor normalization
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)
        elif isinstance(image, torch.Tensor):
            # Check if tensor is in [0,1] range and needs normalization to [-1,1]
            if image.min().item() >= 0.0 and image.max().item() <= 1.0:
                image = image * 2.0 - 1.0
        
        # Process with postprocessing if enabled, otherwise return input directly
        return self.postprocess_image(image, output_type=self.output_type)

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
        elif isinstance(image, torch.Tensor):
            # Check if tensor is in [0,1] range (from bytes_to_pt) and needs normalization to [-1,1]
            if image.min().item() >= 0.0 and image.max().item() <= 1.0:
                image = image * 2.0 - 1.0  # Convert [0,1] to [-1,1]

        # Normal diffusion pipeline
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
        # Apply postprocessing first if enabled
        if self.use_postprocessing:
            # Convert from [-1,1] to [0,1] range for postprocessors
            normalized_tensor = self._denormalize_on_gpu(image_tensor)
            
            # Use synchronous processing for skip diffusion, pipelined for normal diffusion
            if self.skip_diffusion:
                processed_tensor = self._postprocessing_orchestrator.process_postprocessors_sync(
                    normalized_tensor, self._postprocessor_instances
                )
            else:
                processed_tensor = self._postprocessing_orchestrator.process_postprocessors_pipelined(
                    normalized_tensor, self._postprocessor_instances
                )
            
            # Convert back to [-1,1] range for output format handling
            image_tensor = self._denormalize_from_processors(processed_tensor)
        
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

    def _denormalize_from_processors(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor from [0,1] (processor range) back to [-1,1] (diffusion range)"""
        return (image_tensor * 2 - 1).clamp(-1, 1)

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
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
        use_controlnet: bool = False,
        controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        enable_pytorch_fallback: bool = False,
        use_ipadapter: bool = False,
        ipadapter_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        use_postprocessing: bool = False,
        postprocessing_config: Optional[List[Dict[str, Any]]] = None,
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
        use_ipadapter : bool, optional
            Whether to apply IPAdapter patch, by default False.
        ipadapter_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            IPAdapter configuration(s), by default None.

        Returns
        -------
        StreamDiffusion
            The loaded model (potentially wrapped with ControlNet pipeline).
        """

        # Clean up GPU memory before loading new model to prevent OOM errors
        try:
            self.cleanup_gpu_memory()
        except Exception as e:
            logger.warning(f"GPU cleanup warning: {e}")

        # First, try to detect if this is an SDXL model before loading
        # TODO: CAN we do this step with model_detection.py?
        is_sdxl_model = False
        model_path_lower = model_id_or_path.lower()
        
        # Check path for SDXL indicators
        if any(indicator in model_path_lower for indicator in ['sdxl', 'xl', '1024']):
            is_sdxl_model = True
            logger.info(f"_load_model: Path suggests SDXL model: {model_id_or_path}")
        
        # For .safetensor files, we need to be more careful about pipeline selection
        if model_id_or_path.endswith('.safetensors'):
            # For .safetensor files, try SDXL pipeline first if path suggests SDXL
            if is_sdxl_model:
                loading_methods = [
                    (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file"),
                    (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
                    (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
                ]
            else:
                loading_methods = [
                    (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
                    (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
                    (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file")
                ]
        else:
            # For regular model directories or checkpoints, use the original order
            loading_methods = [
                (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
                (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
                (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file")
            ]

        pipe = None
        last_error = None
        for method, method_name in loading_methods:
            try:
                logger.info(f"_load_model: Attempting to load with {method_name}...")
                pipe = method(model_id_or_path).to(device=self.device, dtype=self.dtype)
                logger.info(f"_load_model: Successfully loaded using {method_name}")
                
                # Verify that we have the right pipeline type for SDXL models
                if is_sdxl_model and not isinstance(pipe, StableDiffusionXLPipeline):
                    logger.warning(f"_load_model: SDXL model detected but loaded with non-SDXL pipeline: {type(pipe)}")
                    # Try to explicitly load with SDXL pipeline instead
                    try:
                        logger.info(f"_load_model: Retrying with StableDiffusionXLPipeline...")
                        pipe = StableDiffusionXLPipeline.from_single_file(model_id_or_path).to(device=self.device, dtype=self.dtype)
                        logger.info(f"_load_model: Successfully loaded using SDXL pipeline on retry")
                    except Exception as retry_error:
                        logger.warning(f"_load_model: SDXL pipeline retry failed: {retry_error}")
                        # Continue with the originally loaded pipeline
                
                break
            except Exception as e:
                logger.warning(f"_load_model: {method_name} failed: {e}")
                last_error = e
                continue

        if pipe is None:
            error_msg = f"_load_model: All loading methods failed for model '{model_id_or_path}'. Last error: {last_error}"
            logger.error(error_msg)
            if last_error:
                logger.warning("Full traceback of last error:")
                import traceback
                traceback.print_exc()
            raise RuntimeError(error_msg)

        # If we get here, the model loaded successfully - break out of retry loop
        logger.info(f"Model loading succeeded")

        # Use comprehensive model detection instead of basic detection
        detection_result = detect_model(pipe.unet, pipe)
        model_type = detection_result['model_type']
        is_sdxl = detection_result['is_sdxl']
        is_turbo = detection_result['is_turbo']
        confidence = detection_result['confidence']
        
        # Store comprehensive model info for later use (after TensorRT conversion)
        self._detected_model_type = model_type
        self._detection_confidence = confidence
        self._is_turbo = is_turbo
        self._is_sdxl = is_sdxl
        
        logger.info(f"_load_model: Detected model type: {model_type} (confidence: {confidence:.2f})")

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
            normalize_prompt_weights=normalize_prompt_weights,
            normalize_seed_weights=normalize_seed_weights,
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
                from streamdiffusion.acceleration.tensorrt import TorchVAEEncoder
                from streamdiffusion.acceleration.tensorrt.runtime_engines.unet_engine import AutoencoderKLEngine
                from streamdiffusion.acceleration.tensorrt.models.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType
                # Add ControlNet detection and support
                from streamdiffusion.model_detection import (
                    extract_unet_architecture,
                    validate_architecture
                )
                from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_controlnet_export import create_controlnet_wrapper
                from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_ipadapter_export import create_ipadapter_wrapper

                # Legacy TensorRT implementation (fallback)
                # Initialize engine manager
                engine_manager = EngineManager(engine_dir)

                # Enhanced SDXL and ControlNet TensorRT support
                use_controlnet_trt = False
                use_ipadapter_trt = False
                unet_arch = {}
                is_sdxl_model = False
                
                # Use the explicit use_ipadapter parameter
                has_ipadapter = use_ipadapter
                
                # Determine IP-Adapter presence and token count directly from config (no legacy pipeline)
                if has_ipadapter and not ipadapter_config:
                    has_ipadapter = False
                
                try:
                    # Use model detection results already computed during model loading
                    model_type = getattr(self, '_detected_model_type', 'SD15')
                    is_sdxl = getattr(self, '_is_sdxl', False)
                    is_turbo = getattr(self, '_is_turbo', False)
                    confidence = getattr(self, '_detection_confidence', 0.0)
                    
                    if is_sdxl:
                        logger.info(f"Building TensorRT engines for SDXL model: {model_type}")
                        logger.info(f"   Turbo variant: {is_turbo}")
                        logger.info(f"   Detection confidence: {confidence:.2f}")
                    else:
                        logger.info(f"Building TensorRT engines for {model_type}")
                    
                    # Enable IPAdapter TensorRT if configured and available
                    if has_ipadapter:
                        use_ipadapter_trt = True
                        cross_attention_dim = stream.unet.config.cross_attention_dim
                    
                    # Only enable ControlNet for legacy TensorRT if ControlNet is actually being used
                    if self.use_controlnet:
                        try:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                            use_controlnet_trt = True
                            logger.info(f"   Including ControlNet support for {model_type}")
                        except Exception as e:
                            logger.warning(f"   ControlNet architecture detection failed: {e}")
                            use_controlnet_trt = False
                    
                    # Set up architecture info for enabled modes
                    if use_controlnet_trt and not use_ipadapter_trt:
                        # ControlNet only: Full architecture needed
                        if not unet_arch:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                    elif use_ipadapter_trt and not use_controlnet_trt:
                        # IPAdapter only: Cross-attention dim needed
                        unet_arch = {"context_dim": stream.unet.config.cross_attention_dim}
                    elif use_controlnet_trt and use_ipadapter_trt:
                        # Combined mode: Full architecture + cross-attention dim
                        if not unet_arch:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                        unet_arch["context_dim"] = stream.unet.config.cross_attention_dim
                    else:
                        # Neither enabled: Standard UNet
                        unet_arch = {}
                        
                except Exception as e:
                    logger.error(f"Advanced model detection failed: {e}")
                    logger.error("   Falling back to basic TensorRT")
                    
                    # Fallback to basic detection
                    try:
                        detection_result = detect_model(stream.unet, None)
                        model_type = detection_result['model_type']
                        is_sdxl = detection_result['is_sdxl']
                        if self.use_controlnet:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                            use_controlnet_trt = True
                    except Exception:
                        pass
                
                if not use_controlnet_trt and not self.use_controlnet:
                    logger.info("ControlNet not enabled, building engines without ControlNet support")

                # Use the engine_dir parameter passed to this function, with fallback to instance variable
                engine_dir = engine_dir if engine_dir else getattr(self, '_engine_dir', 'engines')
                
                # Resolve IP-Adapter runtime params from config
                # Strength is now a runtime input, so we do NOT bake scale into engine identity
                ipadapter_scale = None
                ipadapter_tokens = None
                if use_ipadapter_trt and has_ipadapter and ipadapter_config:
                    cfg0 = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config
                    # scale omitted from engine naming; runtime will pass ipadapter_scale vector
                    ipadapter_tokens = cfg0.get('num_image_tokens', 4)
                    # Determine FaceID type from config for engine naming
                    is_faceid = (cfg0.get('type') == 'faceid' or bool(cfg0.get('is_faceid', False)))
                # Generate engine paths using EngineManager
                unet_path = engine_manager.get_engine_path(
                    EngineType.UNET,
                    model_id_or_path=model_id_or_path,
                    max_batch=stream.trt_unet_batch_size,
                    min_batch_size=stream.trt_unet_batch_size,
                    mode=self.mode,
                    use_lcm_lora=use_lcm_lora,
                    use_tiny_vae=use_tiny_vae,
                    ipadapter_scale=ipadapter_scale,
                    ipadapter_tokens=ipadapter_tokens,
                    is_faceid=is_faceid if use_ipadapter_trt else None
                )
                vae_encoder_path = engine_manager.get_engine_path(
                    EngineType.VAE_ENCODER,
                    model_id_or_path=model_id_or_path,
                    max_batch=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    mode=self.mode,
                    use_lcm_lora=use_lcm_lora,
                    use_tiny_vae=use_tiny_vae,
                    ipadapter_scale=ipadapter_scale,
                    ipadapter_tokens=ipadapter_tokens,
                    is_faceid=is_faceid if use_ipadapter_trt else None
                )
                vae_decoder_path = engine_manager.get_engine_path(
                    EngineType.VAE_DECODER,
                    model_id_or_path=model_id_or_path,
                    max_batch=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    mode=self.mode,
                    use_lcm_lora=use_lcm_lora,
                    use_tiny_vae=use_tiny_vae,
                    ipadapter_scale=ipadapter_scale,
                    ipadapter_tokens=ipadapter_tokens,
                    is_faceid=is_faceid if use_ipadapter_trt else None
                )

                # Check if all required engines exist
                missing_engines = []
                if not unet_path.exists():
                    missing_engines.append(f"UNet engine: {unet_path}")
                if not vae_decoder_path.exists():
                    missing_engines.append(f"VAE decoder engine: {vae_decoder_path}")
                if not vae_encoder_path.exists():
                    missing_engines.append(f"VAE encoder engine: {vae_encoder_path}")

                if missing_engines:
                    if build_engines_if_missing:
                        logger.info(f"Missing TensorRT engines, building them...")
                        for engine in missing_engines:
                            logger.info(f"  - {engine}")
                    else:
                        error_msg = f"Required TensorRT engines are missing and build_engines_if_missing=False:\n"
                        for engine in missing_engines:
                            error_msg += f"  - {engine}\n"
                        error_msg += f"\nTo build engines, set build_engines_if_missing=True or run the build script manually."
                        raise RuntimeError(error_msg)

                # Determine correct embedding dimension based on model type
                if is_sdxl:
                    # SDXL uses concatenated embeddings from dual text encoders (768 + 1280 = 2048)
                    embedding_dim = 2048
                    logger.info(f"SDXL model detected! Using embedding_dim = {embedding_dim}")
                else:
                    # SD1.5, SD2.1, etc. use single text encoder
                    embedding_dim = stream.text_encoder.config.hidden_size
                    logger.info(f"Non-SDXL model ({model_type}) detected! Using embedding_dim = {embedding_dim}")

                # Gather parameters for unified wrapper - validate IPAdapter first for consistent token count
                control_input_names = None
                num_tokens = 4  # Default for non-IPAdapter mode
                
                if use_ipadapter_trt:
                    # Use token count resolved from configuration (default to 4)
                    num_tokens = ipadapter_tokens if isinstance(ipadapter_tokens, int) else 4

                # Compile UNet engine using EngineManager
                logger.info(f"compile_and_load_engine: Compiling UNet engine for image size: {self.width}x{self.height}")
                try:
                    logger.debug(f"compile_and_load_engine: use_ipadapter_trt={use_ipadapter_trt}, num_ip_layers={num_ip_layers}, tokens={num_tokens}")
                except Exception:
                    pass
                
                # If using TensorRT with IP-Adapter, ensure processors and weights are installed BEFORE export
                if use_ipadapter_trt and has_ipadapter and ipadapter_config and not hasattr(stream, '_ipadapter_module'):
                    try:
                        from streamdiffusion.modules.ipadapter_module import IPAdapterModule, IPAdapterConfig
                        cfg = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config
                        ip_cfg = IPAdapterConfig(
                            style_image_key=cfg.get('style_image_key') or 'ipadapter_main',
                            num_image_tokens=cfg.get('num_image_tokens', 4),
                            ipadapter_model_path=cfg['ipadapter_model_path'],
                            image_encoder_path=cfg['image_encoder_path'],
                            style_image=cfg.get('style_image'),
                            scale=cfg.get('scale', 1.0),
                            is_faceid=(cfg.get('type') == 'faceid' or bool(cfg.get('is_faceid', False))),
                            insightface_model_name=cfg.get('insightface_model_name'),
                        )
                        ip_module_for_export = IPAdapterModule(ip_cfg)
                        ip_module_for_export.install(stream)
                        setattr(stream, '_ipadapter_module', ip_module_for_export)
                        try:
                            logger.info("Installed IP-Adapter processors prior to TensorRT export")
                        except Exception:
                            pass
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        logger.error("Failed to pre-install IP-Adapter prior to TensorRT export")

                # NOTE: When IPAdapter is enabled, we must pass num_ip_layers. We cannot know it until after
                # installing processors in the export wrapper. We construct the wrapper first to discover it,
                # then construct UNet model with that value.

                # Build a temporary unified wrapper to install processors and discover num_ip_layers
                from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export import UnifiedExportWrapper
                temp_wrapped_unet = UnifiedExportWrapper(
                    stream.unet,
                    use_controlnet=use_controlnet_trt,
                    use_ipadapter=use_ipadapter_trt,
                    control_input_names=None,
                    num_tokens=num_tokens
                )

                num_ip_layers = None
                if use_ipadapter_trt:
                    # Access underlying IPAdapter wrapper
                    if hasattr(temp_wrapped_unet, 'ipadapter_wrapper') and temp_wrapped_unet.ipadapter_wrapper:
                        num_ip_layers = getattr(temp_wrapped_unet.ipadapter_wrapper, 'num_ip_layers', None)
                        if not isinstance(num_ip_layers, int) or num_ip_layers <= 0:
                            raise RuntimeError("Failed to determine num_ip_layers for IP-Adapter")
                        try:
                            logger.info(f"compile_and_load_engine: discovered num_ip_layers={num_ip_layers}")
                        except Exception:
                            pass

                unet_model = UNet(
                    fp16=True,
                    device=stream.device,
                    max_batch=stream.trt_unet_batch_size,
                    min_batch_size=stream.trt_unet_batch_size,
                    embedding_dim=embedding_dim,
                    unet_dim=stream.unet.config.in_channels,
                    use_control=use_controlnet_trt,
                    unet_arch=unet_arch if use_controlnet_trt else None,
                    use_ipadapter=use_ipadapter_trt,
                    num_image_tokens=num_tokens,
                    num_ip_layers=num_ip_layers if use_ipadapter_trt else None,
                    image_height=self.height,
                    image_width=self.width,
                )

                # Use ControlNet wrapper if ControlNet support is enabled
                if use_controlnet_trt:
                    # Build control_input_names excluding ipadapter_scale so indices align to 3-base offset
                    all_input_names = unet_model.get_input_names()
                    control_input_names = [name for name in all_input_names if name != 'ipadapter_scale']

                # Unified compilation path 
                # Recreate wrapped_unet with control input names if needed (after unet_model is ready)
                wrapped_unet = UnifiedExportWrapper(
                    stream.unet,
                    use_controlnet=use_controlnet_trt,
                    use_ipadapter=use_ipadapter_trt,
                    control_input_names=control_input_names,
                    num_tokens=num_tokens
                )

                # Compile VAE decoder engine using EngineManager
                vae_decoder_model = VAE(
                    device=stream.device,
                    max_batch=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                )

                engine_manager.compile_and_load_engine(
                    EngineType.VAE_DECODER,
                    vae_decoder_path,
                    model=stream.vae,
                    model_config=vae_decoder_model,
                    batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    cuda_stream=None,
                    stream_vae=stream.vae,
                    engine_build_options={
                        'opt_image_height': self.height,
                        'opt_image_width': self.width,
                        'build_dynamic_shape': True,
                        'min_image_resolution': 384,
                        'max_image_resolution': 1024,
                    }
                )

                # Compile VAE encoder engine using EngineManager
                vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                vae_encoder_model = VAEEncoder(
                    device=stream.device,
                    max_batch=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                )

                vae_encoder_path = engine_manager.compile_and_load_engine(
                    EngineType.VAE_ENCODER,
                    vae_encoder_path,
                    model=vae_encoder,
                    model_config=vae_encoder_model,
                    batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    cuda_stream=None,
                    engine_build_options={
                        'opt_image_height': self.height,
                        'opt_image_width': self.width,
                        'build_dynamic_shape': True,
                        'min_image_resolution': 384,
                        'max_image_resolution': 1024,
                    }
                )

                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                # Try to load TensorRT UNet engine with OOM recovery
                tensorrt_unet_loaded = False
                try:
                    logger.info("Loading TensorRT UNet engine...")
                    # Compile and load UNet engine using EngineManager
                    stream.unet = engine_manager.compile_and_load_engine(
                        EngineType.UNET,
                        unet_path,
                        model=wrapped_unet,
                        model_config=unet_model,
                        batch_size=stream.trt_unet_batch_size,
                        cuda_stream=cuda_stream,
                        use_controlnet_trt=use_controlnet_trt,
                        use_ipadapter_trt=use_ipadapter_trt,
                        unet_arch=unet_arch,
                        num_ip_layers=num_ip_layers if use_ipadapter_trt else None,
                        engine_build_options={
                            'opt_image_height': self.height,
                            'opt_image_width': self.width,
                        }
                    )
                    
                    tensorrt_unet_loaded = True
                    logger.info("TensorRT UNet engine loaded successfully")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    is_oom_error = ('out of memory' in error_msg or 'outofmemory' in error_msg or 
                                   'oom' in error_msg or 'cuda error' in error_msg)
                    
                    if is_oom_error:
                        logger.error(f"TensorRT UNet engine OOM: {e}")
                        logger.info("Falling back to PyTorch UNet (no TensorRT acceleration)")
                        logger.info("This will be slower but should work with less memory")
                        
                        # Clean up any partial TensorRT state
                        if hasattr(stream, 'unet'):
                            try:
                                del stream.unet
                            except:
                                pass
                        
                        self.cleanup_gpu_memory()
                        
                        # Fall back to original PyTorch UNet
                        try:
                            logger.info("Loading PyTorch UNet as fallback...")
                            # Keep the original UNet from the pipe
                            if hasattr(stream, 'pipe') and hasattr(stream.pipe, 'unet'):
                                stream.unet = stream.pipe.unet
                                logger.info("PyTorch UNet fallback successful")
                            else:
                                raise RuntimeError("No PyTorch UNet available for fallback")
                        except Exception as fallback_error:
                            logger.error(f"PyTorch UNet fallback also failed: {fallback_error}")
                            raise RuntimeError(f"Both TensorRT and PyTorch UNet loading failed. TensorRT error: {e}, Fallback error: {fallback_error}")
                    else:
                        # Non-OOM error, re-raise
                        logger.error(f"TensorRT UNet engine loading failed (non-OOM): {e}")
                        raise e

                # Load VAE engines using paths returned by EngineManager
                stream.vae = AutoencoderKLEngine(
                    str(vae_encoder_path),
                    str(vae_decoder_path),
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=True,
                )
                stream.vae.config = vae_config
                stream.vae.dtype = vae_dtype

                gc.collect()
                torch.cuda.empty_cache()

                # Try to load TensorRT VAE engines with OOM recovery
                tensorrt_vae_loaded = False
                try:
                    logger.info("Loading TensorRT VAE engines...")
                    stream.vae = AutoencoderKLEngine(
                        vae_encoder_path,
                        vae_decoder_path,
                        cuda_stream,
                        stream.pipe.vae_scale_factor,
                        use_cuda_graph=True,
                    )
                    stream.vae.config = vae_config
                    stream.vae.dtype = vae_dtype
                    
                    tensorrt_vae_loaded = True
                    logger.info("TensorRT VAE engines loaded successfully")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    is_oom_error = ('out of memory' in error_msg or 'outofmemory' in error_msg or 
                                   'oom' in error_msg or 'cuda error' in error_msg)
                    
                    if is_oom_error:
                        logger.error(f"TensorRT VAE engine OOM: {e}")
                        logger.info("Falling back to PyTorch VAE (no TensorRT acceleration)")
                        logger.info("This will be slower but should work with less memory")
                        
                        # Clean up any partial TensorRT state
                        if hasattr(stream, 'vae'):
                            try:
                                del stream.vae
                            except:
                                pass
                        
                        self.cleanup_gpu_memory()
                        
                        # Fall back to original PyTorch VAE
                        try:
                            logger.info("Loading PyTorch VAE as fallback...")
                            # Keep the original VAE from the pipe
                            if hasattr(stream, 'pipe') and hasattr(stream.pipe, 'vae'):
                                stream.vae = stream.pipe.vae
                                logger.info("PyTorch VAE fallback successful")
                            else:
                                raise RuntimeError("No PyTorch VAE available for fallback")
                        except Exception as fallback_error:
                            logger.error(f"PyTorch VAE fallback also failed: {fallback_error}")
                            raise RuntimeError(f"Both TensorRT and PyTorch VAE loading failed. TensorRT error: {e}, Fallback error: {fallback_error}")
                    else:
                        # Non-OOM error, re-raise
                        logger.error(f"TensorRT VAE engine loading failed (non-OOM): {e}")
                        raise e
                    
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.error("Acceleration has failed. Falling back to normal mode.")
            if not self.enable_pytorch_fallback:
                raise NotImplementedError("Acceleration has failed. Automatic pytorch inference fallback disabled.")
            else:
                logger.error("Acceleration has failed. Falling back to PyTorch inference.")

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

        # Install modules via hooks instead of patching (wrapper keeps forwarding updates only)
        if use_controlnet and controlnet_config:
            try:
                from streamdiffusion.modules.controlnet_module import ControlNetModule, ControlNetConfig
                cn_module = ControlNetModule(device=self.device, dtype=self.dtype)
                cn_module.install(stream)
                # Normalize to list of configs
                configs = controlnet_config if isinstance(controlnet_config, list) else [controlnet_config]
                for cfg in configs:
                    if not cfg.get('model_id'):
                        continue
                    cn_cfg = ControlNetConfig(
                        model_id=cfg['model_id'],
                        preprocessor=cfg.get('preprocessor'),
                        conditioning_scale=cfg.get('conditioning_scale', 1.0),
                        enabled=cfg.get('enabled', True),
                        preprocessor_params=cfg.get('preprocessor_params'),
                    )
                    cn_module.add_controlnet(cn_cfg, control_image=cfg.get('control_image'))
                # Expose for later updates if needed by caller code
                stream._controlnet_module = cn_module

                # If TensorRT UNet is active, proactively compile/load ControlNet TRT engines for each model
                #TODO: make unet cnet trt acceleration independent and configurable
                try:
                    use_trt_unet = hasattr(stream, 'unet') and hasattr(stream.unet, 'engine')
                except Exception:
                    use_trt_unet = False
                if use_trt_unet:
                    try:
                        compiled_cn_engines = []
                        for cfg, cn_model in zip(configs, cn_module.controlnets):
                            if not cfg or not cfg.get('model_id') or cn_model is None:
                                continue
                            try:
                                engine = engine_manager.get_or_load_controlnet_engine(
                                    model_id=cfg['model_id'],
                                    pytorch_model=cn_model,
                                    model_type=model_type,
                                    batch_size=stream.trt_unet_batch_size,
                                    cuda_stream=cuda_stream,
                                    use_cuda_graph=False,
                                    unet=None,
                                    model_path=cfg['model_id']
                                )
                                try:
                                    setattr(engine, 'model_id', cfg['model_id'])
                                except Exception:
                                    pass
                                compiled_cn_engines.append(engine)
                            except Exception as e:
                                logger.warning(f"Failed to compile/load ControlNet engine for {cfg.get('model_id')}: {e}")
                        if compiled_cn_engines:
                            setattr(stream, 'controlnet_engines', compiled_cn_engines)
                            try:
                                logger.info(f"Compiled/loaded {len(compiled_cn_engines)} ControlNet TensorRT engine(s)")
                            except Exception:
                                pass
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        logger.warning("ControlNet TensorRT engine build step encountered an issue; continuing with PyTorch ControlNet")
            except Exception:
                import traceback
                traceback.print_exc()
                logger.error("Failed to install ControlNetModule")
                raise

        if use_ipadapter and ipadapter_config and not hasattr(stream, '_ipadapter_module'):
            try:
                from streamdiffusion.modules.ipadapter_module import IPAdapterModule, IPAdapterConfig
                # Use first config if list provided
                cfg = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config
                ip_cfg = IPAdapterConfig(
                    style_image_key=cfg.get('style_image_key') or 'ipadapter_main',
                    num_image_tokens=cfg.get('num_image_tokens', 4),
                    ipadapter_model_path=cfg['ipadapter_model_path'],
                    image_encoder_path=cfg['image_encoder_path'],
                    style_image=cfg.get('style_image'),
                    scale=cfg.get('scale', 1.0),
                    is_faceid=(cfg.get('type') == 'faceid' or bool(cfg.get('is_faceid', False))),
                    insightface_model_name=cfg.get('insightface_model_name'),
                )
                ip_module = IPAdapterModule(ip_cfg)
                ip_module.install(stream)
                # Expose for later updates
                stream._ipadapter_module = ip_module
            except Exception:
                import traceback
                traceback.print_exc()
                logger.error("Failed to install IPAdapterModule")
                raise

        # Initialize postprocessing
        if use_postprocessing and postprocessing_config:
            try:
                from .preprocessing.postprocessing_orchestrator import PostprocessingOrchestrator
                from .preprocessing.processors import get_preprocessor
                
                self._postprocessing_orchestrator = PostprocessingOrchestrator(
                    device=self.device, 
                    dtype=self.dtype, 
                    max_workers=4
                )
                
                # Build postprocessor instances using existing registry
                self._postprocessor_instances = []
                for proc_config in postprocessing_config:
                    if proc_config.get('enabled', True):
                        processor = get_preprocessor(proc_config['name'])
                        
                        # Set system parameters (same pattern as ControlNet)
                        try:
                            if hasattr(processor, 'params') and isinstance(getattr(processor, 'params'), dict):
                                processor.params['image_width'] = int(width)
                                processor.params['image_height'] = int(height)
                            if hasattr(processor, 'image_width'):
                                setattr(processor, 'image_width', int(width))
                            if hasattr(processor, 'image_height'):
                                setattr(processor, 'image_height', int(height))
                        except Exception:
                            pass
                        
                        # Configure processor with preprocessor_params if provided (same pattern as ControlNet)
                        if proc_config.get('preprocessor_params'):
                            params = proc_config['preprocessor_params']
                            # If the processor exposes a 'params' dict, update it
                            if hasattr(processor, 'params') and isinstance(getattr(processor, 'params'), dict):
                                processor.params.update(params)
                            # Also set individual attributes
                            for param_name, param_value in params.items():
                                if hasattr(processor, param_name):
                                    setattr(processor, param_name, param_value)
                        self._postprocessor_instances.append(processor)
                
                logger.info(f"_load_model: Postprocessing initialized with {len(self._postprocessor_instances)} processors")
            except Exception as e:
                logger.error(f"_load_model: Postprocessing initialization failed: {e}")
                raise RuntimeError(f"Postprocessing setup failed: {e}")
        else:
            self._postprocessing_orchestrator = None
            self._postprocessor_instances = []

        return stream

    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """Forward get_last_processed_image call to the underlying ControlNet pipeline"""
        if not self.use_controlnet:
            raise RuntimeError("get_last_processed_image: ControlNet support not enabled. Set use_controlnet=True in constructor.")

        return self.stream.get_last_processed_image(index)
        
    def cleanup_controlnets(self) -> None:
        """Cleanup ControlNet resources including background threads and VRAM"""
        if not self.use_controlnet:
            return
            
        if hasattr(self, 'stream') and self.stream and hasattr(self.stream, 'cleanup'):
            self.stream.cleanup_controlnets()



    def clear_caches(self) -> None:
        """Clear all cached prompt embeddings and seed noise tensors."""
        self.stream._param_updater.clear_caches()
        if self._postprocessing_orchestrator:
            self._postprocessing_orchestrator.clear_cache()

    def get_stream_state(self, include_caches: bool = False) -> Dict[str, Any]:
        """Get a unified snapshot of the current stream state.

        Args:
            include_caches: When True, include cache statistics in the response

        Returns:
            Dict[str, Any]: Consolidated state including prompts/seeds, runtime settings,
                            module configs, and basic pipeline info.
        """
        stream = self.stream
        updater = stream._param_updater

        # Prompts / Seeds
        prompts = updater.get_current_prompts()
        seeds = updater.get_current_seeds()

        # Normalization flags
        normalize_prompt_weights = updater.get_normalize_prompt_weights()
        normalize_seed_weights = updater.get_normalize_seed_weights()

        # Core runtime params
        guidance_scale = getattr(stream, 'guidance_scale', None)
        delta = getattr(stream, 'delta', None)
        t_index_list = list(getattr(stream, 't_list', []))
        current_seed = getattr(stream, 'current_seed', None)
        num_inference_steps = None
        try:
            if hasattr(stream, 'timesteps') and stream.timesteps is not None:
                num_inference_steps = int(len(stream.timesteps))
        except Exception:
            pass

        # Resolution and model/pipeline info
        state: Dict[str, Any] = {
            'width': getattr(stream, 'width', None),
            'height': getattr(stream, 'height', None),
            'latent_width': getattr(stream, 'latent_width', None),
            'latent_height': getattr(stream, 'latent_height', None),
            'device': getattr(stream, 'device', None).type if hasattr(getattr(stream, 'device', None), 'type') else getattr(stream, 'device', None),
            'dtype': str(getattr(stream, 'dtype', None)),
            'model_type': getattr(stream, 'model_type', None),
            'is_sdxl': getattr(stream, 'is_sdxl', None),
            'is_turbo': getattr(stream, 'is_turbo', None),
            'cfg_type': getattr(stream, 'cfg_type', None),
            'use_denoising_batch': getattr(stream, 'use_denoising_batch', None),
            'batch_size': getattr(stream, 'batch_size', None),
        }

        # Blending state
        state.update({
            'prompt_list': prompts,
            'seed_list': seeds,
            'normalize_prompt_weights': normalize_prompt_weights,
            'normalize_seed_weights': normalize_seed_weights,
            'negative_prompt': getattr(updater, '_current_negative_prompt', ""),
        })

        # Core runtime knobs
        state.update({
            'guidance_scale': guidance_scale,
            'delta': delta,
            't_index_list': t_index_list,
            'current_seed': current_seed,
            'num_inference_steps': num_inference_steps,
        })

        # Module configs (ControlNet, IP-Adapter)
        try:
            controlnet_config = updater._get_current_controlnet_config()
        except Exception:
            controlnet_config = []
        try:
            ipadapter_config = updater._get_current_ipadapter_config()
        except Exception:
            ipadapter_config = None
        state.update({
            'controlnet_config': controlnet_config,
            'ipadapter_config': ipadapter_config,
        })

        # Postprocessing state
        postprocessing_state = {
            'enabled': self.use_postprocessing,
            'skip_diffusion': self.skip_diffusion,
            'processors': []
        }
        if self.use_postprocessing and self._postprocessor_instances:
            for i, processor in enumerate(self._postprocessor_instances):
                proc_state = {
                    'name': processor.__class__.__name__.replace('Preprocessor', '').lower(),
                    'enabled': True,
                    'type': processor.__class__.__name__
                }
                postprocessing_state['processors'].append(proc_state)
        
        state.update({
            'postprocessing': postprocessing_state,
        })

        # Optional caches
        if include_caches:
            try:
                state['caches'] = updater.get_cache_info()
            except Exception:
                state['caches'] = None

        return state
    
    def cleanup_gpu_memory(self) -> None:
        """Comprehensive GPU memory cleanup for model switching."""
        import gc
        import torch
        
        logger.info("Cleaning up GPU memory...")
        
        # Clear prompt caches
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream._param_updater.clear_caches()
                logger.info("   Cleared prompt caches")
            except:
                pass
        
        # Clear postprocessing caches
        if hasattr(self, '_postprocessing_orchestrator') and self._postprocessing_orchestrator:
            try:
                self._postprocessing_orchestrator.clear_cache()
                logger.info("   Cleared postprocessing caches")
            except:
                pass
        
        # Enhanced TensorRT engine cleanup
        if hasattr(self, 'stream') and self.stream:
            try:
                # Cleanup UNet TensorRT engine
                if hasattr(self.stream, 'unet'):
                    unet_engine = self.stream.unet
                    logger.info("   Cleaning up TensorRT UNet engine...")
                    
                    # Check if it's a TensorRT engine and cleanup properly
                    if hasattr(unet_engine, 'engine') and hasattr(unet_engine.engine, '__del__'):
                        try:
                            # Call the engine's destructor explicitly
                            unet_engine.engine.__del__()
                        except:
                            pass
                    
                    # Clear all engine-related attributes
                    if hasattr(unet_engine, 'context'):
                        try:
                            del unet_engine.context
                        except:
                            pass
                    if hasattr(unet_engine, 'engine'):
                        try:
                            del unet_engine.engine.engine  # TensorRT runtime engine
                            del unet_engine.engine
                        except:
                            pass
                    
                    del self.stream.unet
                    logger.info("   UNet engine cleanup completed")
                    
                # Cleanup VAE TensorRT engines
                if hasattr(self.stream, 'vae'):
                    vae_engine = self.stream.vae
                    logger.info("   Cleaning up TensorRT VAE engines...")
                    
                    # VAE has encoder and decoder engines
                    for engine_name in ['vae_encoder', 'vae_decoder']:
                        if hasattr(vae_engine, engine_name):
                            engine = getattr(vae_engine, engine_name)
                            if hasattr(engine, 'engine') and hasattr(engine.engine, '__del__'):
                                try:
                                    engine.engine.__del__()
                                except:
                                    pass
                            try:
                                delattr(vae_engine, engine_name)
                            except:
                                pass
                    
                    del self.stream.vae
                    logger.info("   VAE engines cleanup completed")
                
                # Cleanup ControlNet engine pool if it exists
                if hasattr(self.stream, 'controlnet_engine_pool'):
                    logger.info("   Cleaning up ControlNet engine pool...")
                    try:
                        self.stream.controlnet_engine_pool.cleanup()
                        del self.stream.controlnet_engine_pool
                        logger.info("   ControlNet engine pool cleanup completed")
                    except:
                        pass
                    
            except Exception as e:
                logger.error(f"   TensorRT cleanup warning: {e}")
        
        # Clear the entire stream object to free all models
        if hasattr(self, 'stream'):
            try:
                del self.stream
                logger.info("   Cleared stream object")
            except:
                pass
            self.stream = None
        
        # Force multiple garbage collection cycles for thorough cleanup
        for i in range(3):
            gc.collect()
        
        # Clear CUDA cache multiple times
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force additional memory cleanup
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            # Get memory info
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            cached = torch.cuda.memory_reserved() / (1024**3)     # GB
            logger.info(f"   GPU Memory after cleanup: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        logger.info("   Enhanced GPU memory cleanup complete")

    def check_gpu_memory_for_engine(self, engine_size_gb: float) -> bool:
        """
        Check if there's enough GPU memory to load a TensorRT engine.
        
        Args:
            engine_size_gb: Expected engine size in GB
            
        Returns:
            True if enough memory is available, False otherwise
        """
        if not torch.cuda.is_available():
            return True  # Assume OK if CUDA not available
        
        try:
            # Get current memory status
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory = total_memory - allocated
            
            # Add 20% overhead for safety
            required_memory = engine_size_gb * 1.2
            
            logger.info(f"GPU Memory Check:")
            logger.info(f"   Total: {total_memory:.2f}GB")
            logger.info(f"   Allocated: {allocated:.2f}GB") 
            logger.info(f"   Cached: {cached:.2f}GB")
            logger.info(f"   Free: {free_memory:.2f}GB")
            logger.info(f"   Required: {required_memory:.2f}GB (engine: {engine_size_gb:.2f}GB + 20% overhead)")
            
            if free_memory >= required_memory:
                logger.info(f"   Sufficient memory available")
                return True
            else:
                logger.error(f"   Insufficient memory! Need {required_memory:.2f}GB but only {free_memory:.2f}GB available")
                return False
                
        except Exception as e:
            logger.error(f"   Memory check failed: {e}")
            return True  # Assume OK if check fails

    def cleanup_engines_and_rebuild(self, reduce_batch_size: bool = True, reduce_resolution: bool = False) -> None:
        """
        Clean up TensorRT engines and rebuild with smaller settings to fix OOM issues.
        
        Parameters:
        -----------
        reduce_batch_size : bool
            If True, reduce batch size to 1
        reduce_resolution : bool  
            If True, reduce resolution by half
        """
        import shutil
        import os
        
        logger.info("Cleaning up engines and rebuilding with smaller settings...")
        
        # Clean up GPU memory first
        self.cleanup_gpu_memory()
        
        # Remove engines directory
        engines_dir = "engines"
        if os.path.exists(engines_dir):
            try:
                shutil.rmtree(engines_dir)
                logger.info(f"   Removed engines directory: {engines_dir}")
            except Exception as e:
                logger.error(f"   Failed to remove engines: {e}")
        
        # Reduce settings
        if reduce_batch_size:
            if hasattr(self, 'batch_size') and self.batch_size > 1:
                old_batch = self.batch_size
                self.batch_size = 1
                logger.info(f"   Reduced batch size: {old_batch} -> {self.batch_size}")
            
            # Also reduce frame buffer size if needed
            if hasattr(self, 'frame_buffer_size') and self.frame_buffer_size > 1:
                old_buffer = self.frame_buffer_size
                self.frame_buffer_size = 1  
                logger.info(f"   Reduced frame buffer size: {old_buffer} -> {self.frame_buffer_size}")
        
        if reduce_resolution:
            if hasattr(self, 'width') and hasattr(self, 'height'):
                old_width, old_height = self.width, self.height
                self.width = max(512, self.width // 2)
                self.height = max(512, self.height // 2)
                # Round to multiples of 64 for compatibility
                self.width = (self.width // 64) * 64
                self.height = (self.height // 64) * 64
                logger.info(f"   Reduced resolution: {old_width}x{old_height} -> {self.width}x{self.height}")
        
        logger.info("   Next model load will rebuild engines with these smaller settings")
