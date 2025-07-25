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
        
        # Detect if this is SDXL based on UNet config
        self.is_sdxl = self._detect_sdxl_architecture(unet)
        
        # SDXL ControlNet has different structure than SD1.5
        if self.is_sdxl:
            # SDXL typically has 10 down blocks instead of 12
            self.expected_down_blocks = 10
            print(f"🎯 Detected SDXL ControlNet - expecting {self.expected_down_blocks} down blocks")
        else:
            # SD1.5 has 12 down blocks
            self.expected_down_blocks = 12
            print(f"🎯 Detected SD1.5 ControlNet - expecting {self.expected_down_blocks} down blocks")
        
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
    
    def _detect_sdxl_architecture(self, unet):
        """Detect if this is an SDXL UNet based on its configuration"""
        try:
            # SDXL has cross_attention_dim of 2048, SD1.5 has 768
            if hasattr(unet.config, 'cross_attention_dim') and unet.config.cross_attention_dim == 2048:
                return True
            # SDXL has addition_embed_type
            if hasattr(unet.config, 'addition_embed_type') and unet.config.addition_embed_type:
                return True
            # SDXL typically has sample_size of 128 (for 1024x1024 images), SD1.5 has 64
            if hasattr(unet.config, 'sample_size') and unet.config.sample_size >= 128:
                return True
        except:
            pass
        return False
    
    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        """Forward pass that organizes control inputs and calls UNet"""
        
        down_block_controls = []
        mid_block_control = None
        
        # Extract control arguments from *args 
        control_args = args
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
            
            if len(all_control_tensors) == self.expected_down_blocks:
                down_block_controls = all_control_tensors
                mid_block_control = middle_tensor
                print(f"✅ ControlNet: Got expected {len(all_control_tensors)} down block tensors")
            else:
                print(f"⚠️ ControlNet: Expected {self.expected_down_blocks} down block tensors, got {len(all_control_tensors)}")
                
                # Try to adapt the available tensors
                if len(all_control_tensors) > 0:
                    if len(all_control_tensors) > self.expected_down_blocks:
                        # Too many tensors - take the first expected_down_blocks
                        down_block_controls = all_control_tensors[:self.expected_down_blocks]
                        print(f"🔧 Truncated to {len(down_block_controls)} tensors")
                    else:
                        # Too few tensors - use what we have
                        down_block_controls = all_control_tensors
                        print(f"🔧 Using available {len(down_block_controls)} tensors")
                    mid_block_control = middle_tensor
                else:
                    # No control tensors available - skip ControlNet
                    print("🔧 No control tensors available - proceeding without ControlNet")
                    down_block_controls = None
                    mid_block_control = None
        
        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        
        # Pass through all additional kwargs (for SDXL models)
        unet_kwargs.update(kwargs)
        
        if down_block_controls:
            # Adapt control tensor shapes for SDXL if needed
            adapted_controls = self._adapt_control_tensors(down_block_controls, sample)
            unet_kwargs['down_block_additional_residuals'] = adapted_controls
        
        if mid_block_control is not None:
            # Adapt middle control tensor shape if needed
            adapted_mid_control = self._adapt_middle_control_tensor(mid_block_control, sample)
            unet_kwargs['mid_block_additional_residual'] = adapted_mid_control
        
        return self.unet(**unet_kwargs)
    
    def _adapt_control_tensors(self, control_tensors, sample):
        """Adapt control tensor shapes to match UNet expectations"""
        if not control_tensors:
            return control_tensors
            
        adapted_tensors = []
        sample_height, sample_width = sample.shape[-2:]
        
        for i, control_tensor in enumerate(control_tensors):
            if control_tensor is None:
                adapted_tensors.append(control_tensor)
                continue
                
            # Check if tensor needs spatial adaptation
            if len(control_tensor.shape) >= 4:
                control_height, control_width = control_tensor.shape[-2:]
                
                # Calculate expected size for this layer (typically powers of 2 downsampling)
                layer_downsample = 2 ** (i // 3)  # Rough estimate
                expected_height = sample_height // layer_downsample
                expected_width = sample_width // layer_downsample
                
                if control_height != expected_height or control_width != expected_width:
                    print(f"🔧 Adapting control tensor {i}: {control_tensor.shape} -> expected size ~{expected_height}x{expected_width}")
                    # Use interpolation to adapt size
                    import torch.nn.functional as F
                    adapted_tensor = F.interpolate(
                        control_tensor, 
                        size=(expected_height, expected_width),
                        mode='bilinear', 
                        align_corners=False
                    )
                    adapted_tensors.append(adapted_tensor)
                else:
                    adapted_tensors.append(control_tensor)
            else:
                adapted_tensors.append(control_tensor)
                
        return adapted_tensors
    
    def _adapt_middle_control_tensor(self, mid_control, sample):
        """Adapt middle control tensor shape to match UNet expectations"""
        if mid_control is None:
            return mid_control
            
        # Middle control is typically at the bottleneck, so heavily downsampled
        if len(mid_control.shape) >= 4 and len(sample.shape) >= 4:
            sample_height, sample_width = sample.shape[-2:]
            control_height, control_width = mid_control.shape[-2:]
            
            # Middle block is typically downsampled by factor of 8 or 16
            expected_height = sample_height // 8
            expected_width = sample_width // 8
            
            if control_height != expected_height or control_width != expected_width:
                print(f"🔧 Adapting middle control tensor: {mid_control.shape} -> expected size ~{expected_height}x{expected_width}")
                import torch.nn.functional as F
                adapted_tensor = F.interpolate(
                    mid_control,
                    size=(expected_height, expected_width),
                    mode='bilinear',
                    align_corners=False
                )
                return adapted_tensor
                
        return mid_control


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