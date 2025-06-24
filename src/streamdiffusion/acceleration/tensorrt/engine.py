from typing import *

import torch

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.autoencoders.autoencoder_kl import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

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

        # Allocate buffers and run inference
        self.engine.allocate_buffers(shape_dict=shape_dict, device=latent_model_input.device)

        noise_pred = self.engine.infer(
            input_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        
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

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
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
