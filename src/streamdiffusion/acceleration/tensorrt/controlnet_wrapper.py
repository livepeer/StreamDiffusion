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
            # SDXL has 1 initial + 3 down blocks producing 9 control tensors total
            # Initial: 1 tensor, Block 0: 3 tensors, Block 1: 3 tensors, Block 2: 2 tensors
            self.expected_down_blocks = 9
            print(f"🎯 Detected SDXL ControlNet - expecting {self.expected_down_blocks} control tensors")
        else:
            # SD1.5 has 12 down blocks
            self.expected_down_blocks = 12
            print(f"🎯 Detected SD1.5 ControlNet - expecting {self.expected_down_blocks} down blocks")
        
        self.input_control_indices = []
        self.output_control_indices = []
        self.middle_control_indices = []
        
        print(f"🔍 ControlNet input names: {control_input_names}")
        for i, name in enumerate(control_input_names):
            if name in ["sample", "timestep", "encoder_hidden_states"]:
                continue
                
            if "input_control" in name:
                self.input_control_indices.append(i)
                print(f"🔍 Found input_control at index {i}: {name}")
            elif "output_control" in name:
                self.output_control_indices.append(i)
                print(f"🔍 Found output_control at index {i}: {name}")
            elif "middle_control" in name:
                self.middle_control_indices.append(i)
                print(f"🔍 Found middle_control at index {i}: {name}")
        
        print(f"🔍 Final indices - input_control: {self.input_control_indices}, middle: {self.middle_control_indices}")
    
    def _detect_sdxl_architecture(self, unet):
        """Detect if this is an SDXL UNet based on its configuration"""
        try:
            config = unet.config
            print(f"🔍 UNet detection: cross_attention_dim={getattr(config, 'cross_attention_dim', None)}")
            print(f"🔍 UNet detection: block_out_channels={getattr(config, 'block_out_channels', None)}")
            print(f"🔍 UNet detection: down_block_types={getattr(config, 'down_block_types', None)}")
            print(f"🔍 UNet detection: addition_embed_type={getattr(config, 'addition_embed_type', None)}")
            
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
        
        print(f"🔍 Forward pass - args count: {len(args)}, control_args count: {len(control_args)}")
        print(f"🔍 Input control count: {input_control_count}, indices: {self.input_control_indices}")
        
        if input_control_count > 0:
            all_control_tensors = []
            middle_tensor = None
            
            for i, idx in enumerate(self.input_control_indices):
                control_arg_idx = idx - 3
                print(f"🔍 Processing control index {i}: input_name_idx={idx}, control_arg_idx={control_arg_idx}")
                if control_arg_idx < len(control_args):
                    tensor = control_args[control_arg_idx]
                    print(f"🔍 Control tensor {i}: shape={tensor.shape if tensor is not None else None}")
                    
                    if i == input_control_count - 1:
                        middle_tensor = tensor
                        print(f"🔍 Assigned as middle tensor: {tensor.shape if tensor is not None else None}")
                    else:
                        all_control_tensors.append(tensor)
                        print(f"🔍 Added to control tensors: {tensor.shape if tensor is not None else None}")
            
            if len(all_control_tensors) == self.expected_down_blocks:
                down_block_controls = all_control_tensors
                mid_block_control = middle_tensor
                print(f"✅ ControlNet: Got expected {len(all_control_tensors)} down block tensors")
                for i, tensor in enumerate(all_control_tensors):
                    print(f"🔧 ControlNet tensor {i}: shape={tensor.shape} (channels={tensor.shape[1]})")
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
            print(f"🔧 ControlNet: Feeding {len(adapted_controls)} adapted control tensors to UNet:")
            for i, tensor in enumerate(adapted_controls):
                print(f"   adapted_control[{i}]: shape={tensor.shape} (channels={tensor.shape[1]})")
            
            print(f"🔧 Current order (fine→coarse): channels = {[t.shape[1] for t in adapted_controls]}")
            print(f"🔧 Current order (fine→coarse): spatial = {[f'{t.shape[-2]}x{t.shape[-1]}' for t in adapted_controls]}")
            
            reversed_controls = list(reversed(adapted_controls))
            print(f"🔧 Reversed order (coarse→fine): channels = {[t.shape[1] for t in reversed_controls]}")
            print(f"🔧 Reversed order (coarse→fine): spatial = {[f'{t.shape[-2]}x{t.shape[-1]}' for t in reversed_controls]}")
            
            # Control tensors are now generated in the correct order to match UNet's down_block_res_samples
            # For SDXL: [88x88, 88x88, 88x88, 44x44, 44x44, 44x44, 22x22, 22x22, 22x22]
            # This directly aligns with UNet's: [initial_sample] + [block0_residuals] + [block1_residuals] + [block2_residuals]
            if self.is_sdxl and len(adapted_controls) == 9:
                print(f"🔧 SDXL: Using 9 control tensors in correct order for UNet down_block_res_samples")
                for i, tensor in enumerate(adapted_controls):
                    print(f"   control[{i}]: shape={tensor.shape} (channels={tensor.shape[1]}, spatial={tensor.shape[-2]}x{tensor.shape[-1]})")
                
                unet_kwargs['down_block_additional_residuals'] = adapted_controls
            else:
                print(f"🔧 Using {len(adapted_controls)} control tensors as-is")
                unet_kwargs['down_block_additional_residuals'] = adapted_controls
        
        if mid_block_control is not None:
            # Adapt middle control tensor shape if needed
            adapted_mid_control = self._adapt_middle_control_tensor(mid_block_control, sample)
            unet_kwargs['mid_block_additional_residual'] = adapted_mid_control
        
        print(f"🔍 DEBUG: Calling UNet with kwargs keys: {list(unet_kwargs.keys())}")
        if 'down_block_additional_residuals' in unet_kwargs:
            residuals = unet_kwargs['down_block_additional_residuals']
            print(f"🔍 DEBUG: down_block_additional_residuals count={len(residuals)}")
            for i, r in enumerate(residuals):
                print(f"   residual[{i}]: {r.shape}")
        
        # Create a debug wrapper to catch the exact failure point
        class UNetDebugWrapper:
            def __init__(self, unet):
                self.unet = unet
                
            def __call__(self, **kwargs):
                try:
                    print(f"🔍 DEBUG: UNet.__call__ starting...")
                    
                    # Monkey patch the down blocks to add debug info
                    original_forward_methods = []
                    if hasattr(self.unet, 'down_blocks'):
                        for i, block in enumerate(self.unet.down_blocks):
                            original_forward = block.forward
                            original_forward_methods.append(original_forward)
                            
                            def make_debug_forward(block_idx, orig_forward):
                                def debug_forward(hidden_states, temb=None, encoder_hidden_states=None, 
                                                attention_mask=None, cross_attention_kwargs=None, 
                                                encoder_attention_mask=None, additional_residuals=None, **forward_kwargs):
                                    print(f"🔍 DEBUG: down_block[{block_idx}] input shape: {hidden_states.shape}")
                                    if additional_residuals is not None:
                                        if isinstance(additional_residuals, (list, tuple)):
                                            print(f"🔍 DEBUG: down_block[{block_idx}] additional_residuals: {len(additional_residuals)} tensors")
                                            for j, res in enumerate(additional_residuals):
                                                if res is not None:
                                                    print(f"   additional_residual[{j}]: {res.shape}")
                                        else:
                                            print(f"🔍 DEBUG: down_block[{block_idx}] additional_residual: {additional_residuals.shape}")
                                    
                                    try:
                                        result = orig_forward(hidden_states, temb, encoder_hidden_states, 
                                                            attention_mask, cross_attention_kwargs, 
                                                            encoder_attention_mask, additional_residuals, **forward_kwargs)
                                        if isinstance(result, tuple):
                                            output_states, down_block_res_samples = result
                                            print(f"🔍 DEBUG: down_block[{block_idx}] output shape: {output_states.shape}")
                                            print(f"🔍 DEBUG: down_block[{block_idx}] res_samples: {len(down_block_res_samples)} tensors")
                                            for j, res in enumerate(down_block_res_samples):
                                                print(f"   res_sample[{j}]: {res.shape}")
                                        else:
                                            print(f"🔍 DEBUG: down_block[{block_idx}] output: {result.shape}")
                                        return result
                                    except Exception as e:
                                        print(f"❌ DEBUG: down_block[{block_idx}] failed: {e}")
                                        print(f"❌ DEBUG: Input hidden_states: {hidden_states.shape}")
                                        if additional_residuals is not None:
                                            if isinstance(additional_residuals, (list, tuple)):
                                                for j, res in enumerate(additional_residuals):
                                                    if res is not None:
                                                        print(f"❌ DEBUG: additional_residual[{j}]: {res.shape}")
                                            else:
                                                print(f"❌ DEBUG: additional_residual: {additional_residuals.shape}")
                                        raise
                                return debug_forward
                            
                            block.forward = make_debug_forward(i, original_forward)
                    
                    # Call the actual UNet
                    result = self.unet(**kwargs)
                    
                    # Restore original forward methods
                    if hasattr(self.unet, 'down_blocks'):
                        for i, block in enumerate(self.unet.down_blocks):
                            block.forward = original_forward_methods[i]
                    
                    print(f"✅ DEBUG: UNet.__call__ successful")
                    return result
                    
                except Exception as e:
                    # Restore original forward methods on error
                    if hasattr(self.unet, 'down_blocks'):
                        for i, block in enumerate(self.unet.down_blocks):
                            if i < len(original_forward_methods):
                                block.forward = original_forward_methods[i]
                    
                    print(f"❌ DEBUG: UNet.__call__ failed: {e}")
                    raise
        
        debug_wrapper = UNetDebugWrapper(self.unet)
        
        try:
            result = debug_wrapper(**unet_kwargs)
            print(f"✅ DEBUG: UNet forward successful")
            return result
        except Exception as e:
            print(f"❌ DEBUG: UNet forward failed: {e}")
            print(f"❌ DEBUG: UNet input shapes - sample: {sample.shape}")
            if 'down_block_additional_residuals' in unet_kwargs:
                for i, r in enumerate(unet_kwargs['down_block_additional_residuals']):
                    print(f"❌ DEBUG: down_block_additional_residuals[{i}]: {r.shape}")
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
        else:
            expected_downsample_factors = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]  # 12 tensors for SD1.5
        
        print(f"🔍 DEBUG: _adapt_control_tensors - SDXL={self.is_sdxl}, tensor_count={len(control_tensors)}")
        print(f"🔍 DEBUG: expected_downsample_factors = {expected_downsample_factors}")
        print(f"🔍 DEBUG: sample spatial = {sample_height}x{sample_width}")
        
        for i, control_tensor in enumerate(control_tensors):
            if control_tensor is None:
                adapted_tensors.append(control_tensor)
                continue
                
            # Check if tensor needs spatial adaptation
            if len(control_tensor.shape) >= 4:
                control_height, control_width = control_tensor.shape[-2:]
                
                # Use the correct downsampling factor for this tensor index
                if i < len(expected_downsample_factors):
                    downsample_factor = expected_downsample_factors[i]
                    expected_height = sample_height // downsample_factor
                    expected_width = sample_width // downsample_factor
                    
                    if control_height != expected_height or control_width != expected_width:
                        print(f"🔧 Adapting control tensor {i}: {control_tensor.shape} -> expected size {expected_height}x{expected_width} (factor={downsample_factor})")
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
                        print(f"🔧 Control tensor {i}: {control_tensor.shape} already correct size (factor={downsample_factor})")
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
                print(f"🔧 Adapting middle control tensor: {mid_control.shape} -> expected size {expected_height}x{expected_width} (factor={expected_factor})")
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