import gc
import os
import traceback
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Any

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from .pipeline import StreamDiffusion
from .image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamDiffusionWrapper:
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
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """

        # Apply ControlNet if configured
        if self.use_controlnet and hasattr(self, 'controlnet_configs') and self.controlnet_configs:
            # Apply ControlNet integration
            self._apply_controlnet_integration()

        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
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
        """
        self.stream.update_stream_params(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            t_index_list=t_index_list,
            seed=seed,
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
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

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

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

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
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
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

        # Determine if this should be an SDXL pipeline from controlnet config
        pipeline_type = self._get_pipeline_type_from_config(controlnet_config)
        is_sdxl = pipeline_type == "sdxlturbo"

        try:  # Load from local directory
            if is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)
            else:
                pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)

        except ValueError:  # Load from huggingface
            if is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)
            else:
                pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

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
                    detect_model_from_diffusers_unet,
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
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch}--min_batch-{min_batch_size}--mode-{self.mode}--width-{width}--height-{height}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch}--min_batch-{min_batch_size}--mode-{self.mode}--width-{width}--height-{height}"

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
                setattr(stream.unet, 'use_control', True)
                if use_controlnet_trt and unet_arch:
                    setattr(stream.unet, 'unet_arch', unet_arch)

                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

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
            self.nsfw_fallback_img = Image.new("RGB", (self.height, self.width), (0, 0, 0))

        # Apply ControlNet patch if needed
        if use_controlnet and controlnet_config:
            stream = self._apply_controlnet_patch(stream, controlnet_config, acceleration, engine_dir)

        return stream

    def _apply_controlnet_patch(self, stream: StreamDiffusion, controlnet_config: Union[Dict[str, Any], List[Dict[str, Any]]], acceleration: str = "none", engine_dir: str = "engines") -> Any:
        """
        Apply ControlNet patch to StreamDiffusion based on pipeline_type

        Args:
            stream: Base StreamDiffusion instance
            controlnet_config: ControlNet configuration(s)

        Returns:
            ControlNet-enabled pipeline (ControlNetPipeline or SDXLTurboControlNetPipeline)
        """
        # Extract pipeline_type from controlnet_config if it's in there
        if isinstance(controlnet_config, list):
            config_dict = controlnet_config[0] if controlnet_config else {}
        else:
            config_dict = controlnet_config

        pipeline_type = config_dict.get('pipeline_type', 'sd1.5')

        # Check if we should use TensorRT ControlNet acceleration
        use_controlnet_tensorrt = (acceleration == "tensorrt")

        if pipeline_type == "sdxlturbo":
            from streamdiffusion.controlnet.controlnet_sdxlturbo_pipeline import SDXLTurboControlNetPipeline
            controlnet_pipeline = SDXLTurboControlNetPipeline(stream, self.device, self.dtype)
        elif pipeline_type in ["sd1.5", "sdturbo"]:
            from streamdiffusion.controlnet.controlnet_pipeline import ControlNetPipeline
            controlnet_pipeline = ControlNetPipeline(stream, self.device, self.dtype)
        else:
            raise ValueError(f"Unsupported pipeline_type: {pipeline_type}")

        # Initialize ControlNet engine pool if using TensorRT acceleration
        if use_controlnet_tensorrt:
            from streamdiffusion.acceleration.tensorrt.engine_pool import ControlNetEnginePool
            from polygraphy import cuda

            # Create engine pool with same engine directory structure as UNet
            stream_cuda = cuda.Stream()
            controlnet_pool = ControlNetEnginePool(engine_dir, stream_cuda, self.width, self.height)

            # Store pool on the pipeline for later use
            setattr(controlnet_pipeline, '_controlnet_pool', controlnet_pool)
            setattr(controlnet_pipeline, '_use_tensorrt', True)
            # Also set on stream where ControlNet pipeline expects to find it
            setattr(stream, 'controlnet_engine_pool', controlnet_pool)
            print("Initialized ControlNet TensorRT engine pool")
        else:
            setattr(controlnet_pipeline, '_use_tensorrt', False)
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

            try:
                # Pass config dictionary directly
                cn_config = {
                    'model_id': model_id,
                    'preprocessor': preprocessor,
                    'conditioning_scale': conditioning_scale,
                    'enabled': enabled,
                    'preprocessor_params': preprocessor_params or {}
                }

                controlnet_pipeline.add_controlnet(cn_config)
            except Exception as e:
                pass

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

        if hasattr(self.stream, 'add_controlnet'):
            cn_config = {
                'model_id': model_id,
                'preprocessor': preprocessor,
                'conditioning_scale': conditioning_scale,
                'enabled': enabled,
                'preprocessor_params': preprocessor_params or {}
            }
            return self.stream.add_controlnet(cn_config, control_image)
        else:
            raise RuntimeError("add_controlnet: ControlNet functionality not available on this pipeline")



    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor], index: Optional[int] = None) -> None:
        """Forward update_control_image_efficient call to the underlying ControlNet pipeline"""
        if self.use_controlnet and hasattr(self.stream, 'update_control_image_efficient'):
            self.stream.update_control_image_efficient(control_image, index=index)

    def update_controlnet_scale(self, index: int, scale: float) -> None:
        """Forward update_controlnet_scale call to the underlying ControlNet pipeline"""
        if self.use_controlnet and hasattr(self.stream, 'update_controlnet_scale'):
            self.stream.update_controlnet_scale(index, scale)

    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """Forward get_last_processed_image call to the underlying ControlNet pipeline"""
        if self.use_controlnet and hasattr(self.stream, 'get_last_processed_image'):
            return self.stream.get_last_processed_image(index)
        return None

    def _get_pipeline_type_from_config(self, controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> str:
        """
        Extracts the pipeline_type from controlnet_config if it exists.

        Args:
            controlnet_config: ControlNet configuration(s)

        Returns:
            pipeline_type: Extracted pipeline_type or 'sd1.5' as default
        """
        if controlnet_config is None:
            return 'sd1.5'  # Default to SD 1.5

        if isinstance(controlnet_config, list):
            config_dict = controlnet_config[0] if controlnet_config else {}
        else:
            config_dict = controlnet_config

        return config_dict.get('pipeline_type', 'sd1.5')  # Default to SD 1.5 if not specified
