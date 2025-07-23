from typing import *

import torch
import logging

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.autoencoders.autoencoder_kl import DecoderOutput
from polygraphy import cuda

from .utilities import Engine

# Set up logger for this module
logger = logging.getLogger(__name__)


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: 'cuda.Stream', use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.use_control = False  # Will be set to True by wrapper if engine has ControlNet support
        self._cached_dummy_controlnet_inputs = None

        logger.debug(f"UNet2DConditionModelEngine.__init__: Loading TensorRT engine from {filepath}")
        self.engine.load()
        self.engine.activate()
        logger.debug(f"UNet2DConditionModelEngine.__init__: TensorRT engine loaded and activated successfully")

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
        
        logger.debug(f"[UNET_ENGINE] __call__: *** UNet2DConditionModelEngine called! ***")
        logger.debug(f"[UNET_ENGINE] __call__: latent_model_input shape: {latent_model_input.shape}")
        logger.debug(f"[UNET_ENGINE] __call__: timestep shape: {timestep.shape}")
        logger.debug(f"[UNET_ENGINE] __call__: encoder_hidden_states shape: {encoder_hidden_states.shape}")
        logger.debug(f"[UNET_ENGINE] __call__: kwargs keys: {list(kwargs.keys())}")
        logger.debug(f"[UNET_ENGINE] __call__: About to start detailed processing...")
        
        # Check for NaN/Inf in inputs
        if torch.isnan(latent_model_input).any():
            logger.warning(f"*** WARNING: NaN detected in latent_model_input! ***")
        if torch.isinf(latent_model_input).any():
            logger.warning(f"*** WARNING: Inf detected in latent_model_input! ***")
        if torch.isnan(timestep).any():
            logger.warning(f"*** WARNING: NaN detected in timestep! ***")
        if torch.isnan(encoder_hidden_states).any():
            logger.warning(f"*** WARNING: NaN detected in encoder_hidden_states! ***")
        
        # Print value ranges
        logger.debug(f"Input ranges - latent: [{latent_model_input.min().item():.6f}, {latent_model_input.max().item():.6f}]")
        logger.debug(f"Input ranges - timestep: [{timestep.min().item():.6f}, {timestep.max().item():.6f}]") 
        logger.debug(f"Input ranges - encoder: [{encoder_hidden_states.min().item():.6f}, {encoder_hidden_states.max().item():.6f}]")

        if timestep.dtype != torch.float32:
            logger.debug(f"Converting timestep from {timestep.dtype} to float32")
            timestep = timestep.float()
        logger.debug(f"UNetEngine: Main input shapes - latent: {latent_model_input.shape}, timestep: {timestep.shape}, encoder: {encoder_hidden_states.shape}")

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
            logger.debug(f"Adding ControlNet conditioning dict")
            # Option 1: Direct ControlNet conditioning dict (organized by type)
            self._add_controlnet_conditioning_dict(controlnet_conditioning, shape_dict, input_dict)
        elif down_block_additional_residuals is not None or mid_block_additional_residual is not None:
            logger.debug(f"Adding ControlNet residuals")
            # Option 2: Diffusers-style ControlNet residuals
            self._add_controlnet_residuals(
                down_block_additional_residuals, 
                mid_block_additional_residual, 
                shape_dict, 
                input_dict
            )
        else:
            # Check if this engine was compiled with ControlNet support but no conditioning is provided
            # In that case, we need to provide dummy zero tensors for the expected ControlNet inputs
            if self.use_control:
                logger.debug(f"Engine has ControlNet support but no conditioning provided - checking for dummy inputs")
                
                # Check if we have the required architecture info for dummy input generation
                unet_arch = getattr(self, 'unet_arch', {})
                
                if not unet_arch:
                    logger.warning(f"Engine was built with ControlNet support but no architecture info available.")
                    logger.warning(f"Proceeding without ControlNet inputs - this may work if the engine can handle missing ControlNet inputs.")
                    # Don't try to generate dummy inputs, just proceed
                else:
                    logger.debug(f"Architecture info available, generating dummy ControlNet inputs")
                    
                    # Check if we need to regenerate dummy inputs due to dimension change
                    current_latent_height = latent_model_input.shape[2]
                    current_latent_width = latent_model_input.shape[3]
                    
                    # Check if cached dummy inputs exist and have correct dimensions
                    if (self._cached_dummy_controlnet_inputs is None or 
                        not hasattr(self, '_cached_latent_dims') or
                        self._cached_latent_dims != (current_latent_height, current_latent_width)):
                        
                        logger.debug(f"Regenerating dummy inputs for latent dimensions {current_latent_height}x{current_latent_width}")
                        try:
                            self._cached_dummy_controlnet_inputs = self._generate_dummy_controlnet_specs(latent_model_input)
                            self._cached_latent_dims = (current_latent_height, current_latent_width)
                        except RuntimeError as e:
                            logger.warning(f"Failed to generate dummy ControlNet inputs: {e}")
                            logger.warning(f"Proceeding without ControlNet inputs")
                            self._cached_dummy_controlnet_inputs = None
                    
                    # Use cached dummy inputs if available
                    if self._cached_dummy_controlnet_inputs is not None:
                        self._add_cached_dummy_inputs(self._cached_dummy_controlnet_inputs, latent_model_input, shape_dict, input_dict)

        logger.debug(f"Final shape_dict keys: {list(shape_dict.keys())}")
        logger.debug(f"Final input_dict keys: {list(input_dict.keys())}")
        for key, shape in shape_dict.items():
            if key.startswith('input_control'):
                logger.debug(f"UNetEngine: Control input {key}: {shape}")

        # Allocate buffers and run inference
        logger.debug(f"UNetEngine: Allocating TensorRT buffers...")
        logger.debug(f"[UNET_ENGINE] About to allocate TensorRT buffers with shape_dict: {[(k, v) for k, v in shape_dict.items()]}")
        
        # Check VRAM before allocation
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"[UNET_ENGINE] VRAM before allocation - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        try:
            self.engine.allocate_buffers(shape_dict=shape_dict, device=latent_model_input.device)
            logger.debug(f"[UNET_ENGINE] ✅ Buffer allocation completed successfully")
            
            # Check VRAM after allocation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"[UNET_ENGINE] VRAM after allocation - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        except Exception as e:
            logger.debug(f"[UNET_ENGINE] *** ERROR: Buffer allocation failed: {e} ***")
            raise

        logger.debug(f"UNetEngine: Running TensorRT inference...")
        logger.debug(f"[UNET_ENGINE] About to call TensorRT engine.infer()...")
        logger.debug(f"[UNET_ENGINE] Input dict keys: {list(input_dict.keys())}")
        logger.debug(f"[UNET_ENGINE] Input dict shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in input_dict.items()]}")
        logger.debug(f"[UNET_ENGINE] use_cuda_graph: {self.use_cuda_graph}")
        
        # Check VRAM before inference
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"[UNET_ENGINE] VRAM before inference - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        try:
            logger.debug(f"[UNET_ENGINE] 🚀 Starting TensorRT inference...")
            logger.debug(f"[UNET_ENGINE] This call may hang if there are VRAM issues...")
            
            # Set a timeout hint for debugging
            import time
            start_time = time.time()
            
            outputs = self.engine.infer(
                input_dict,
                self.stream,
                use_cuda_graph=self.use_cuda_graph,
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"[UNET_ENGINE] ✅ TensorRT inference completed successfully in {elapsed_time:.3f}s!")
            logger.debug(f"[UNET_ENGINE] Output keys: {list(outputs.keys())}")
            
            # Check VRAM after inference
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"[UNET_ENGINE] VRAM after inference - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
                
        except Exception as e:
            logger.debug(f"[UNET_ENGINE] *** ERROR: TensorRT inference failed: {e} ***")
            import traceback
            traceback.print_exc()
            raise
        
        if "latent" not in outputs:
            logger.error(f"*** ERROR: 'latent' output not found in TensorRT outputs! Available keys: {list(outputs.keys())} ***")
            logger.debug(f"[UNET_ENGINE] *** ERROR: Expected 'latent' output not found! ***")
            raise ValueError("TensorRT engine did not produce expected 'latent' output")
        
        noise_pred = outputs["latent"]
        logger.debug(f"TensorRT inference completed")
        logger.debug(f"Output shape: {noise_pred.shape}, dtype: {noise_pred.dtype}")
        logger.debug(f"Output range: [{noise_pred.min().item():.6f}, {noise_pred.max().item():.6f}]")
        logger.debug(f"[UNET_ENGINE] Output tensor - shape: {noise_pred.shape}, range: [{noise_pred.min().item():.6f}, {noise_pred.max().item():.6f}]")
        
        # Check for NaN/Inf in outputs  
        if torch.isnan(noise_pred).any():
            logger.error(f"*** ERROR: NaN detected in TensorRT output! ***")
            nan_count = torch.isnan(noise_pred).sum().item()
            total_elements = noise_pred.numel()
            logger.error(f"*** NaN count: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%) ***")
            logger.error(f"[UNET_ENGINE] *** ERROR: NaN values detected in output! ***")
        if torch.isinf(noise_pred).any():
            logger.error(f"*** ERROR: Inf detected in TensorRT output! ***")
            inf_count = torch.isinf(noise_pred).sum().item()
            total_elements = noise_pred.numel()
            logger.error(f"*** Inf count: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%) ***")
            logger.error(f"[UNET_ENGINE] *** ERROR: Inf values detected in output! ***")
        
        logger.debug(f"[UNET_ENGINE] Returning UNet2DConditionOutput...")
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
        logger.debug(f"UNetEngine: Adding ControlNet residuals - down_blocks: {len(down_block_additional_residuals) if down_block_additional_residuals else 0}, mid_block: {mid_block_additional_residual is not None}")
        
        # Add down block residuals as input controls
        if down_block_additional_residuals is not None:
            # Map directly to engine input names (no reversal needed for our approach)
            for i, tensor in enumerate(down_block_additional_residuals):
                input_name = f"input_control_{i:02d}"  # Use zero-padded names to match engine
                shape_dict[input_name] = tensor.shape
                input_dict[input_name] = tensor
                logger.debug(f"UNetEngine: Added control input {input_name}: {tensor.shape}")
        
        # Add middle block residual
        if mid_block_additional_residual is not None:
            input_name = "input_control_middle"  # Match engine middle control name
            shape_dict[input_name] = mid_block_additional_residual.shape
            input_dict[input_name] = mid_block_additional_residual
            logger.debug(f"UNetEngine: Added middle control input {input_name}: {mid_block_additional_residual.shape}")

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
        from .models import UNet
        
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
