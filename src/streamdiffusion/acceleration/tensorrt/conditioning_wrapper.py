import torch
from diffusers import UNet2DConditionModel
from typing import Optional, List
from .controlnet_wrapper import create_controlnet_wrapper
from .ipadapter_wrapper import create_ipadapter_wrapper

class ConditioningWrapper(torch.nn.Module):
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
        if use_controlnet and control_input_names:
            controlnet_kwargs = {k: v for k, v in kwargs.items() if k in ['num_controlnets', 'conditioning_scales']}

            self.controlnet_wrapper = create_controlnet_wrapper(self.unet, control_input_names, **controlnet_kwargs)
        
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
                *control_args,
                **kwargs) -> torch.Tensor:
        """Forward pass that handles any UNet parameters via **kwargs passthrough"""
        if self.controlnet_wrapper:
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
                              **kwargs) -> ConditioningWrapper:
    return ConditioningWrapper(
        unet, use_controlnet, use_ipadapter, control_input_names, num_tokens, **kwargs
    ) 