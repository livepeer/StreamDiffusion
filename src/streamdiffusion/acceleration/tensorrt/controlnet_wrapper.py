"""ControlNet-aware UNet wrapper for ONNX export"""

import torch
from typing import List, Optional, Dict, Any
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class ControlNetUNetWrapper(torch.nn.Module):
    """Wrapper that combines UNet with ControlNet inputs for ONNX export"""
    
    def __init__(self, unet: UNet2DConditionModel, control_input_names: List[str]):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        
        self.input_control_indices = []
        self.output_control_indices = []
        self.middle_control_indices = []
        
        for i, name in enumerate(control_input_names):
            if name in ["sample", "timestep", "encoder_hidden_states"]:
                continue
                
            if "input_control" in name:
                self.input_control_indices.append(i)
            elif "output_control" in name:
                self.output_control_indices.append(i)
            elif "middle_control" in name:
                self.middle_control_indices.append(i)
    
    def forward(self, sample, timestep, encoder_hidden_states, *control_args):
        """Forward pass that organizes control inputs and calls UNet"""
        
        down_block_controls = []
        mid_block_control = None
        
        input_control_count = len(self.input_control_indices)
        
        if input_control_count > 0:
            all_control_tensors = []
            middle_tensor = None
            
            for i, idx in enumerate(self.input_control_indices):
                control_arg_idx = idx - 3
                if control_arg_idx < len(control_args):
                    tensor = control_args[control_arg_idx]
                    
                    if i == input_control_count - 1:
                        middle_tensor = tensor
                    else:
                        all_control_tensors.append(tensor)
            
            if len(all_control_tensors) == 12:
                down_block_controls = all_control_tensors
                mid_block_control = middle_tensor
            else:
                print(f"ERROR: Expected 12 down block tensors, got {len(all_control_tensors)}")
                down_block_controls = None
                mid_block_control = None
        
        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        
        if down_block_controls:
            unet_kwargs['down_block_additional_residuals'] = down_block_controls
        
        if mid_block_control is not None:
            unet_kwargs['mid_block_additional_residual'] = mid_block_control
        
        return self.unet(**unet_kwargs)


class MultiControlNetUNetWrapper(torch.nn.Module):
    """Advanced wrapper for multiple ControlNets with different scales"""
    
    def __init__(self, 
                 unet: UNet2DConditionModel, 
                 control_input_names: List[str],
                 num_controlnets: int = 1,
                 conditioning_scales: Optional[List[float]] = None):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        self.num_controlnets = num_controlnets
        self.conditioning_scales = conditioning_scales or [1.0] * num_controlnets
        
        self.controlnet_indices = []
        controls_per_net = (len(control_input_names) - 3) // num_controlnets
        
        for cn_idx in range(num_controlnets):
            start_idx = 3 + cn_idx * controls_per_net
            end_idx = start_idx + controls_per_net
            self.controlnet_indices.append(list(range(start_idx, end_idx)))
    
    def forward(self, sample, timestep, encoder_hidden_states, *control_args):
        """Forward pass for multiple ControlNets"""
        combined_down_controls = None
        combined_mid_control = None
        
        for cn_idx, indices in enumerate(self.controlnet_indices):
            scale = self.conditioning_scales[cn_idx]
            if scale == 0:
                continue
            
            cn_controls = [control_args[i - 3] for i in indices if i - 3 < len(control_args)]
            
            if not cn_controls:
                continue
            
            num_down = len(cn_controls) - 1
            down_controls = cn_controls[:num_down]
            mid_control = cn_controls[num_down] if num_down < len(cn_controls) else None
            
            scaled_down = [ctrl * scale for ctrl in down_controls]
            scaled_mid = mid_control * scale if mid_control is not None else None
            
            if combined_down_controls is None:
                combined_down_controls = scaled_down
                combined_mid_control = scaled_mid
            else:
                for i in range(min(len(combined_down_controls), len(scaled_down))):
                    combined_down_controls[i] += scaled_down[i]
                if scaled_mid is not None and combined_mid_control is not None:
                    combined_mid_control += scaled_mid
        
        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        
        if combined_down_controls:
            unet_kwargs['down_block_additional_residuals'] = list(reversed(combined_down_controls))
        if combined_mid_control is not None:
            unet_kwargs['mid_block_additional_residual'] = combined_mid_control
        
        return self.unet(**unet_kwargs)


def create_controlnet_wrapper(unet: UNet2DConditionModel, 
                            control_input_names: List[str],
                            num_controlnets: int = 1,
                            conditioning_scales: Optional[List[float]] = None) -> torch.nn.Module:
    """Factory function to create appropriate ControlNet wrapper"""
    if num_controlnets == 1:
        return ControlNetUNetWrapper(unet, control_input_names)
    else:
        return MultiControlNetUNetWrapper(
            unet, control_input_names, num_controlnets, conditioning_scales
        )


def organize_control_tensors(control_tensors: List[torch.Tensor], 
                           control_input_names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Organize control tensors by type (input, output, middle)"""
    organized = {'input': [], 'output': [], 'middle': []}
    
    for tensor, name in zip(control_tensors, control_input_names):
        if "input_control" in name:
            organized['input'].append(tensor)
        elif "output_control" in name:
            organized['output'].append(tensor)
        elif "middle_control" in name:
            organized['middle'].append(tensor)
    
    return organized 