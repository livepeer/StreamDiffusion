"""ControlNet-aware UNet wrapper for ONNX export"""

import torch
from typing import List, Optional, Dict, Any
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ....model_detection import detect_model

class ControlNetUNetExportWrapper(torch.nn.Module):
    """Wrapper that combines UNet with ControlNet inputs for ONNX export"""
    
    def __init__(self, unet: UNet2DConditionModel, control_input_names: List[str]):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        
        # Detect if this is SDXL based on UNet config
        detection_result = detect_model(self.unet)
        self.is_sdxl = detection_result.get('is_sdxl', False)
        
        # Detect if UNet has IPAdapter modifications
        self.has_ipadapter = self._detect_ipadapter_modifications(unet)
        
        # SDXL ControlNet has different structure than SD1.5
        if self.is_sdxl:
            # SDXL has 1 initial + 3 down blocks producing 9 control tensors total
            self.expected_down_blocks = 9
        else:
            # SD1.5 has 12 down blocks
            self.expected_down_blocks = 12
        
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
    
    def _detect_ipadapter_modifications(self, unet):
        """Detect if UNet has been modified by IPAdapter"""
        # Check for IPAdapter attention processors
        if hasattr(unet, 'attn_processors'):
            for processor in unet.attn_processors.values():
                if hasattr(processor, '__class__') and 'TRTIPAttn' in processor.__class__.__name__:
                    return True
        return False
    
    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        """Forward pass that organizes control inputs and calls UNet"""
        
        down_block_controls = []
        mid_block_control = None
        
        # Extract control args (skip sample, timestep, encoder_hidden_states)
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
            else:
                # Try to adapt the available tensors
                if len(all_control_tensors) > 0:
                    if len(all_control_tensors) > self.expected_down_blocks:
                        # Too many tensors - take the first expected_down_blocks
                        down_block_controls = all_control_tensors[:self.expected_down_blocks]
                    else:
                        # Too few tensors - use what we have
                        down_block_controls = all_control_tensors
                    mid_block_control = middle_tensor
                else:
                    # No control tensors available - skip ControlNet
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
        
        # Handle SDXL conditioning - ensure added_cond_kwargs is never None for SDXL models
        if self.is_sdxl and ('added_cond_kwargs' not in unet_kwargs or unet_kwargs['added_cond_kwargs'] is None):
            # Auto-generate minimal SDXL conditioning if missing
            batch_size = sample.shape[0]
            device = sample.device
            dtype = sample.dtype
            unet_kwargs['added_cond_kwargs'] = {
                'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=dtype),
                'time_ids': torch.zeros(batch_size, 6, device=device, dtype=dtype)
            }
        
        if down_block_controls:
            # Adapt control tensor shapes for SDXL if needed
            adapted_controls = self._adapt_control_tensors(down_block_controls, sample)
            
            # Control tensors are now generated in the correct order to match UNet's down_block_res_samples
            # For SDXL: [88x88, 88x88, 88x88, 44x44, 44x44, 44x44, 22x22, 22x22, 22x22]
            # This directly aligns with UNet's: [initial_sample] + [block0_residuals] + [block1_residuals] + [block2_residuals]
            unet_kwargs['down_block_additional_residuals'] = adapted_controls
        
        if mid_block_control is not None:
            # Adapt middle control tensor shape if needed
            adapted_mid_control = self._adapt_middle_control_tensor(mid_block_control, sample)
            unet_kwargs['mid_block_additional_residual'] = adapted_mid_control
        
        try:
            result = self.unet(**unet_kwargs)
            return result
        except Exception as e:
            print(f"❌ DEBUG: UNet forward failed: {e}")
            raise
    
    def _adapt_control_tensors(self, control_tensors, sample):
        """Adapt control tensor shapes to match UNet expectations"""
        if not control_tensors:
            return control_tensors
            
        adapted_tensors = []
        sample_height, sample_width = sample.shape[-2:]
        
        # Updated factors to match the corrected control tensor generation
        # SDXL: 9 tensors [88x88, 88x88, 88x88, 44x44, 44x44, 44x44, 22x22, 22x22, 22x22]
        # Factors: [1, 1, 1, 2, 2, 2, 4, 4, 4] to match UNet down_block_res_samples structure
        if self.is_sdxl:
            expected_downsample_factors = [1, 1, 1, 2, 2, 2, 4, 4, 4]  # 9 tensors for SDXL
            # SDXL expected channel dimensions: [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
            expected_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        else:
            expected_downsample_factors = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]  # 12 tensors for SD1.5
            expected_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
        
        for i, control_tensor in enumerate(control_tensors):
            if control_tensor is None:
                adapted_tensors.append(control_tensor)
                continue
                
            # Check if tensor needs adaptation
            if len(control_tensor.shape) >= 4:
                batch_size, current_channels, control_height, control_width = control_tensor.shape
                
                # Check if we need channel adaptation for SDXL
                expected_channel_count = expected_channels[i] if i < len(expected_channels) else current_channels
                
                if self.is_sdxl and current_channels != expected_channel_count:
                    # Adapt channel dimensions for SDXL
                    device = control_tensor.device
                    dtype = control_tensor.dtype
                    
                    if current_channels < expected_channel_count:
                        # Pad channels by repeating the tensor
                        repeat_factor = expected_channel_count // current_channels
                        remainder = expected_channel_count % current_channels
                        
                        repeated_tensor = control_tensor.repeat(1, repeat_factor, 1, 1)
                        if remainder > 0:
                            # Add partial repetition for remainder
                            partial_tensor = control_tensor[:, :remainder, :, :]
                            control_tensor = torch.cat([repeated_tensor, partial_tensor], dim=1)
                        else:
                            control_tensor = repeated_tensor
                            
                        print(f"🔧 Adapted ControlNet tensor {i}: {current_channels} -> {expected_channel_count} channels")
                    
                    elif current_channels > expected_channel_count:
                        # Truncate channels
                        control_tensor = control_tensor[:, :expected_channel_count, :, :]
                        print(f"🔧 Truncated ControlNet tensor {i}: {current_channels} -> {expected_channel_count} channels")
                
                # Use the correct downsampling factor for this tensor index
                if i < len(expected_downsample_factors):
                    downsample_factor = expected_downsample_factors[i]
                    expected_height = sample_height // downsample_factor
                    expected_width = sample_width // downsample_factor
                    
                    if control_height != expected_height or control_width != expected_width:
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
                    # Fallback for unexpected tensor count
                    adapted_tensors.append(control_tensor)
            else:
                adapted_tensors.append(control_tensor)
                
        return adapted_tensors
    
    def _adapt_middle_control_tensor(self, mid_control, sample):
        """Adapt middle control tensor shape to match UNet expectations"""
        if mid_control is None:
            return mid_control
            
        # Check if channel adaptation is needed for SDXL
        if len(mid_control.shape) >= 4:
            batch_size, current_channels, control_height, control_width = mid_control.shape
            
            if self.is_sdxl:
                expected_channels = 1280  # SDXL middle block expects 1280 channels
                
                if current_channels != expected_channels:
                    device = mid_control.device
                    dtype = mid_control.dtype
                    
                    if current_channels < expected_channels:
                        # Pad channels by repeating the tensor
                        repeat_factor = expected_channels // current_channels
                        remainder = expected_channels % current_channels
                        
                        repeated_tensor = mid_control.repeat(1, repeat_factor, 1, 1)
                        if remainder > 0:
                            # Add partial repetition for remainder
                            partial_tensor = mid_control[:, :remainder, :, :]
                            mid_control = torch.cat([repeated_tensor, partial_tensor], dim=1)
                        else:
                            mid_control = repeated_tensor
                            
                        print(f"🔧 Adapted ControlNet middle tensor: {current_channels} -> {expected_channels} channels")
                    
                    elif current_channels > expected_channels:
                        # Truncate channels
                        mid_control = mid_control[:, :expected_channels, :, :]
                        print(f"🔧 Truncated ControlNet middle tensor: {current_channels} -> {expected_channels} channels")
            
        # Middle control is typically at the bottleneck, so heavily downsampled
        if len(mid_control.shape) >= 4 and len(sample.shape) >= 4:
            sample_height, sample_width = sample.shape[-2:]
            control_height, control_width = mid_control.shape[-2:]
            
            # For SDXL: middle block is at 4x downsampling (22x22 from 88x88)
            # For SD1.5: middle block is at 8x downsampling
            expected_factor = 4 if self.is_sdxl else 8
            expected_height = sample_height // expected_factor
            expected_width = sample_width // expected_factor
            
            if control_height != expected_height or control_width != expected_width:
                import torch.nn.functional as F
                adapted_tensor = F.interpolate(
                    mid_control,
                    size=(expected_height, expected_width),
                    mode='bilinear',
                    align_corners=False
                )
                return adapted_tensor
                
        return mid_control


class MultiControlNetUNetExportWrapper(torch.nn.Module):
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
        return ControlNetUNetExportWrapper(unet, control_input_names)
    else:
        return MultiControlNetUNetExportWrapper(
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