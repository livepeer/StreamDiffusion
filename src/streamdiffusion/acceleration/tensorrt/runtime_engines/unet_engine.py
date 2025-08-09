from typing import *

import torch
import logging
import os

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.autoencoders.autoencoder_kl import DecoderOutput
from polygraphy import cuda

from ..utilities import Engine

# Set up logger for this module
logger = logging.getLogger(__name__)


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: 'cuda.Stream', use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.use_control = False  # Will be set to True by wrapper if engine has ControlNet support
        self._cached_dummy_controlnet_inputs = None
        
        # Enable VRAM monitoring only if explicitly requested (defaults to False for performance)
        self.debug_vram = os.getenv('STREAMDIFFUSION_DEBUG_VRAM', '').lower() in ('1', 'true')

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        controlnet_conditioning: Optional[Dict[str, List[torch.Tensor]]] = None,
        **kwargs,
    ) -> Any:
        logger.debug("UNet2DConditionModelEngine.__call__: enter")
        logger.debug(f"UNet2DConditionModelEngine.__call__: use_ipadapter={getattr(self, 'use_ipadapter', False)}, use_control={self.use_control}")
        logger.debug(f"UNet2DConditionModelEngine.__call__: latent_model_input shape={tuple(latent_model_input.shape)}, dtype={latent_model_input.dtype}, device={latent_model_input.device}")
        logger.debug(f"UNet2DConditionModelEngine.__call__: timestep shape={tuple(timestep.shape)}, dtype={timestep.dtype}")
        logger.debug(f"UNet2DConditionModelEngine.__call__: encoder_hidden_states shape={tuple(encoder_hidden_states.shape)}, dtype={encoder_hidden_states.dtype}")
        

        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        # Prepare base shape and input dictionaries
        shape_dict = {
            "sample": latent_model_input.shape,
            "timestep": timestep.shape,
            "encoder_hidden_states": encoder_hidden_states.shape,
            "latent": latent_model_input.shape,
        }
        
        input_dict = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        
        # Handle IP-Adapter runtime scale vector if engine was built with it
        if getattr(self, 'use_ipadapter', False):
            if 'ipadapter_scale' not in kwargs:
                logger.error("UNet2DConditionModelEngine: ipadapter_scale missing but required (use_ipadapter=True)")
                raise RuntimeError("UNet2DConditionModelEngine: ipadapter_scale is required for IP-Adapter engines")
            ip_scale = kwargs['ipadapter_scale']
            if not isinstance(ip_scale, torch.Tensor):
                logger.error(f"UNet2DConditionModelEngine: ipadapter_scale has wrong type: {type(ip_scale)}")
                raise TypeError("ipadapter_scale must be a torch.Tensor")
            logger.debug(f"UNet2DConditionModelEngine.__call__: ipadapter_scale shape={tuple(ip_scale.shape)}, dtype={ip_scale.dtype}, device={ip_scale.device}, min={float(ip_scale.min().item()) if ip_scale.numel()>0 else 'n/a'}, max={float(ip_scale.max().item()) if ip_scale.numel()>0 else 'n/a'}")
            shape_dict["ipadapter_scale"] = ip_scale.shape
            input_dict["ipadapter_scale"] = ip_scale
            


        # Handle ControlNet inputs if provided
        if controlnet_conditioning is not None:
            # Option 1: Direct ControlNet conditioning dict (organized by type)
            self._add_controlnet_conditioning_dict(controlnet_conditioning, shape_dict, input_dict)
        elif down_block_additional_residuals is not None or mid_block_additional_residual is not None:
            # Option 2: Diffusers-style ControlNet residuals
            self._add_controlnet_residuals(
                down_block_additional_residuals, 
                mid_block_additional_residual, 
                shape_dict, 
                input_dict
            )
        else:
            # Check if this engine was compiled with ControlNet support but no conditioning is provided
            if self.use_control:
                unet_arch = getattr(self, 'unet_arch', {})
                if unet_arch:
                    current_latent_height = latent_model_input.shape[2]
                    current_latent_width = latent_model_input.shape[3]
                    
                    # Check if cached dummy inputs exist and have correct dimensions
                    if (self._cached_dummy_controlnet_inputs is None or 
                        not hasattr(self, '_cached_latent_dims') or
                        self._cached_latent_dims != (current_latent_height, current_latent_width)):
                        
                        try:
                            self._cached_dummy_controlnet_inputs = self._generate_dummy_controlnet_specs(latent_model_input)
                            self._cached_latent_dims = (current_latent_height, current_latent_width)
                        except RuntimeError:
                            self._cached_dummy_controlnet_inputs = None
                    
                    if self._cached_dummy_controlnet_inputs is not None:
                        self._add_cached_dummy_inputs(self._cached_dummy_controlnet_inputs, latent_model_input, shape_dict, input_dict)

        # Allocate buffers and run inference
        if self.debug_vram:
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"VRAM before allocation: {allocated_before:.2f}GB")
        
        logger.debug(f"UNet2DConditionModelEngine.__call__: shape_dict={ {k: tuple(v) if hasattr(v,'__iter__') else v for k,v in shape_dict.items()} }")
        self.engine.allocate_buffers(shape_dict=shape_dict, device=latent_model_input.device)
        
        if self.debug_vram:
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"VRAM after allocation: {allocated_after:.2f}GB")

        logger.debug(f"UNet2DConditionModelEngine.__call__: input_dict keys={list(input_dict.keys())}")
        try:
            outputs = self.engine.infer(
                input_dict,
                self.stream,
                use_cuda_graph=self.use_cuda_graph,
            )
        except Exception as e:
            logger.exception(f"UNet2DConditionModelEngine.__call__: Engine.infer failed: {e}")
            raise
        
        
        if self.debug_vram:
            allocated_final = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"VRAM after inference: {allocated_final:.2f}GB")
        
        if "latent" not in outputs:
            logger.error(f"*** ERROR: 'latent' output not found in TensorRT outputs! Available keys: {list(outputs.keys())} ***")
            raise ValueError("TensorRT engine did not produce expected 'latent' output")
        
        noise_pred = outputs["latent"]
        logger.debug(f"UNet2DConditionModelEngine.__call__: output shape={tuple(noise_pred.shape)}, dtype={noise_pred.dtype}, device={noise_pred.device}")
        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            logger.error("UNet2DConditionModelEngine.__call__: output contains NaN/Inf")
      
        

        
        return UNet2DConditionOutput(sample=noise_pred)

    def _add_controlnet_conditioning_dict(self, 
                                        controlnet_conditioning: Dict[str, List[torch.Tensor]], 
                                        shape_dict: Dict, 
                                        input_dict: Dict):
        """
        Add ControlNet conditioning from organized dictionary
        
        Args:
            controlnet_conditioning: Dict with 'input', 'output', 'middle' keys
            shape_dict: Shape dictionary to update
            input_dict: Input dictionary to update
        """
        # Add input controls (down blocks)
        if 'input' in controlnet_conditioning:
            for i, tensor in enumerate(controlnet_conditioning['input']):
                input_name = f"input_control_{i:02d}"  # Use zero-padded names
                shape_dict[input_name] = tensor.shape
                input_dict[input_name] = tensor
        
        # Add output controls (up blocks) 
        if 'output' in controlnet_conditioning:
            for i, tensor in enumerate(controlnet_conditioning['output']):
                input_name = f"output_control_{i:02d}"  # Use zero-padded names
                shape_dict[input_name] = tensor.shape
                input_dict[input_name] = tensor
        
        # Add middle controls
        if 'middle' in controlnet_conditioning:
            for i, tensor in enumerate(controlnet_conditioning['middle']):
                input_name = f"input_control_middle"  # Use consistent middle naming
                shape_dict[input_name] = tensor.shape
                input_dict[input_name] = tensor

    def _add_controlnet_residuals(self, 
                                down_block_additional_residuals: Optional[List[torch.Tensor]], 
                                mid_block_additional_residual: Optional[torch.Tensor],
                                shape_dict: Dict, 
                                input_dict: Dict):
        """
        Add ControlNet residuals in diffusers format
        
        Args:
            down_block_additional_residuals: List of down block residuals
            mid_block_additional_residual: Middle block residual
            shape_dict: Shape dictionary to update
            input_dict: Input dictionary to update
        """
        
        # Add down block residuals as input controls
        if down_block_additional_residuals is not None:
            # Map directly to engine input names (no reversal needed for our approach)
            for i, tensor in enumerate(down_block_additional_residuals):
                input_name = f"input_control_{i:02d}"  # Use zero-padded names to match engine
                shape_dict[input_name] = tensor.shape
                input_dict[input_name] = tensor
        
        # Add middle block residual
        if mid_block_additional_residual is not None:
            input_name = "input_control_middle"  # Match engine middle control name
            shape_dict[input_name] = mid_block_additional_residual.shape
            input_dict[input_name] = mid_block_additional_residual

    def _add_cached_dummy_inputs(self, 
                               dummy_inputs: Dict, 
                               latent_model_input: torch.Tensor,
                               shape_dict: Dict, 
                               input_dict: Dict):
        """
        Add cached dummy inputs to the shape dictionary and input dictionary
        
        Args:
            dummy_inputs: Dictionary containing dummy input specifications
            latent_model_input: The main latent input tensor (used for device/dtype reference)
            shape_dict: Shape dictionary to update
            input_dict: Input dictionary to update
        """
        for input_name, shape_spec in dummy_inputs.items():
            channels = shape_spec["channels"]
            height = shape_spec["height"] 
            width = shape_spec["width"]
            
            # Create zero tensor with appropriate shape
            zero_tensor = torch.zeros(
                latent_model_input.shape[0], channels, height, width,
                dtype=latent_model_input.dtype, device=latent_model_input.device
            )
            
            shape_dict[input_name] = zero_tensor.shape
            input_dict[input_name] = zero_tensor

    def _generate_dummy_controlnet_specs(self, latent_model_input: torch.Tensor) -> Dict:
        """
        Generate dummy ControlNet input specifications once and cache them.
        
        Args:
            latent_model_input: The main latent input tensor (used for dimensions)
            
        Returns:
            Dictionary containing dummy input specifications
        """
        # Get latent dimensions
        latent_height = latent_model_input.shape[2]
        latent_width = latent_model_input.shape[3]
        
        # Calculate image dimensions (assuming 8x upsampling from latent)
        image_height = latent_height * 8
        image_width = latent_width * 8
        
        # Get stored architecture info from engine (set during building)
        unet_arch = getattr(self, 'unet_arch', {})
        
        if not unet_arch:
            raise RuntimeError("No ControlNet architecture info available on engine. Cannot generate dummy inputs.")
        
        # Use the same logic as UNet.get_control() to generate control input specs
        from ..models.models import UNet
        
        # Create a temporary UNet model instance just to use its get_control method
        temp_unet = UNet(
            use_control=True,
            unet_arch=unet_arch,
            image_height=image_height,
            image_width=image_width,
            min_batch_size=1  # Minimal params needed for get_control
        )
        
        return temp_unet.get_control(image_height, image_width)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: 'cuda.Stream',
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def encode(self, images: torch.Tensor, **kwargs):
        self.encoder.allocate_buffers(
            shape_dict={
                "images": images.shape,
                "latent": (
                    images.shape[0],
                    4,
                    images.shape[2] // self.vae_scale_factor,
                    images.shape[3] // self.vae_scale_factor,
                ),
            },
            device=images.device,
        )
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        self.decoder.allocate_buffers(
            shape_dict={
                "latent": latent.shape,
                "images": (
                    latent.shape[0],
                    3,
                    latent.shape[2] * self.vae_scale_factor,
                    latent.shape[3] * self.vae_scale_factor,
                ),
            },
            device=latent.device,
        )
        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
