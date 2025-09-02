import torch
from diffusers import UNet2DConditionModel
from typing import Optional, List
from .unet_controlnet_export import create_controlnet_wrapper
from .unet_ipadapter_export import create_ipadapter_wrapper

class UnifiedExportWrapper(torch.nn.Module):
    """
    Unified wrapper that composes wrappers for conditioning modules. 
    """
    
    def __init__(self, 
                 unet: UNet2DConditionModel, 
                 use_controlnet: bool = False,
                 use_ipadapter: bool = False,
                 control_input_names: Optional[List[str]] = None,
                 num_tokens: int = 4,
                 **kwargs):
        super().__init__()
        self.use_controlnet = use_controlnet
        self.use_ipadapter = use_ipadapter
        self.controlnet_wrapper = None
        self.ipadapter_wrapper = None
        self.unet = unet
        
        # Apply IPAdapter first (installs processors into UNet)
        if use_ipadapter:
            ipadapter_kwargs = {k: v for k, v in kwargs.items() if k in ['install_processors']}
            if 'install_processors' not in ipadapter_kwargs:
                ipadapter_kwargs['install_processors'] = True
            

            self.ipadapter_wrapper = create_ipadapter_wrapper(unet, num_tokens=num_tokens, **ipadapter_kwargs)
            self.unet = self.ipadapter_wrapper.unet
        
        # Apply ControlNet second (wraps whatever UNet we have)
        # Skip ControlNet when IPAdapter is present to avoid compatibility issues
        if use_controlnet and control_input_names and not use_ipadapter:
            controlnet_kwargs = {k: v for k, v in kwargs.items() if k in ['num_controlnets', 'conditioning_scales']}

            self.controlnet_wrapper = create_controlnet_wrapper(self.unet, control_input_names, **controlnet_kwargs)
        else:
            self.controlnet_wrapper = None
        
        # Set up forward strategy based on what we created
        if self.controlnet_wrapper:
            self._forward_impl = self.controlnet_wrapper
        else:
            self._forward_impl = self._basic_unet_forward
        
    def _basic_unet_forward(self, sample, timestep, encoder_hidden_states, *control_args, **kwargs):
        """Basic UNet forward that passes through all parameters to handle any model type"""
        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
            **kwargs  # Pass through all additional parameters (SDXL, future model types, etc.)
        }
        return self.unet(**unet_kwargs)
        
    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                ipadapter_scale=None,
                *control_args,
                **kwargs) -> torch.Tensor:
        """Forward pass that handles any UNet parameters via **kwargs passthrough"""
        # Handle IP-Adapter runtime scale vector - can come as positional arg (ONNX) or in control_args (inference)
        if self.use_ipadapter and self.ipadapter_wrapper is not None:
            # Check if ipadapter_scale was passed as a direct positional argument (ONNX export case)
            if ipadapter_scale is not None and isinstance(ipadapter_scale, torch.Tensor):
                pass  # Use the ipadapter_scale that was passed directly
            # Otherwise, try to get it from control_args (normal inference case)
            elif len(control_args) > 0 and isinstance(control_args[0], torch.Tensor):
                ipadapter_scale = control_args[0]
                control_args = control_args[1:]  # Remove it from control_args
            else:
                import logging
                logging.getLogger(__name__).error("UnifiedExportWrapper: ipadapter_scale missing; required when use_ipadapter=True")
                raise RuntimeError("UnifiedExportWrapper: ipadapter_scale tensor is required when use_ipadapter=True")

            if not isinstance(ipadapter_scale, torch.Tensor):
                import logging
                logging.getLogger(__name__).error(f"UnifiedExportWrapper: ipadapter_scale wrong type: {type(ipadapter_scale)}")
                raise TypeError("ipadapter_scale must be a torch.Tensor")
            try:
                import logging
                logging.getLogger(__name__).debug(f"UnifiedExportWrapper: ipadapter_scale shape={tuple(ipadapter_scale.shape)}, dtype={ipadapter_scale.dtype}")
            except Exception:
                pass
            # assign per-layer scale tensors into processors
            self.ipadapter_wrapper.set_ipadapter_scale(ipadapter_scale)

        # Auto-generate SDXL conditioning if needed
        if ('added_cond_kwargs' not in kwargs and
            hasattr(self.unet, 'config') and hasattr(self.unet.config, 'addition_embed_type') and
            self.unet.config.addition_embed_type == 'text_time'):

            device = sample.device
            batch_size = sample.shape[0]

            import logging
            logging.getLogger(__name__).info("UnifiedExportWrapper: Auto-generating required SDXL conditioning...")
            kwargs['added_cond_kwargs'] = {
                'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
            }

        # Skip ControlNet when IPAdapter is present to avoid compatibility issues
        if self.controlnet_wrapper and not self.use_ipadapter:
            # ControlNet wrapper handles the UNet call with all parameters
            return self.controlnet_wrapper(sample, timestep, encoder_hidden_states, *control_args, **kwargs)
        else:
            # Basic UNet call with all parameters passed through
            return self._basic_unet_forward(sample, timestep, encoder_hidden_states, *control_args, **kwargs)

def create_conditioning_wrapper(unet: UNet2DConditionModel, 
                              use_controlnet: bool = False, 
                              use_ipadapter: bool = False,
                              control_input_names: Optional[List[str]] = None,
                              num_tokens: int = 4,
                              **kwargs) -> UnifiedExportWrapper:
    return UnifiedExportWrapper(
        unet, use_controlnet, use_ipadapter, control_input_names, num_tokens, **kwargs
    ) 