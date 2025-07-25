"""
SDXL Support for TensorRT Acceleration
Handles the complexities of SDXL models with TensorRT including dual encoders,
conditioning parameters, and Turbo variants
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from diffusers import UNet2DConditionModel
from .model_detection import (
    detect_model_comprehensive, 
    detect_unet_characteristics,
    is_sdxl_model_path,
    is_turbo_model_path
)

# Handle different diffusers versions for CLIPTextModel import
try:
    from diffusers.models.transformers.clip_text_model import CLIPTextModel
except ImportError:
    try:
        from diffusers.models.clip_text_model import CLIPTextModel  
    except ImportError:
        try:
            from transformers import CLIPTextModel
        except ImportError:
            # If CLIPTextModel isn't available, we'll work without it
            CLIPTextModel = None



class SDXLConditioningHandler:
    """Handles SDXL conditioning parameters and dual text encoders"""
    
    def __init__(self, unet_info: Dict[str, Any]):
        self.unet_info = unet_info
        self.is_sdxl = unet_info['is_sdxl']
        self.has_time_cond = unet_info['has_time_cond']
        self.has_addition_embed = unet_info['has_addition_embed']
    
    def get_conditioning_spec(self) -> Dict[str, Any]:
        """Get conditioning specification for ONNX export and TensorRT"""
        spec = {
            'text_encoder_dim': 768,  # CLIP ViT-L
            'context_dim': 768,       # Default SD1.5
            'pooled_embeds': False,
            'time_ids': False,
            'dual_encoders': False
        }
        
        if self.is_sdxl:
            spec.update({
                'text_encoder_dim': 768,      # CLIP ViT-L  
                'text_encoder_2_dim': 1280,   # OpenCLIP ViT-bigG
                'context_dim': 2048,          # Concatenated 768 + 1280
                'pooled_embeds': True,        # Pooled text embeddings
                'time_ids': self.has_time_cond,  # Size/crop conditioning
                'dual_encoders': True
            })
        
        return spec
    
    def create_sample_conditioning(self, batch_size: int = 1, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Create sample conditioning tensors for testing/export"""
        spec = self.get_conditioning_spec()
        dtype = torch.float16
        
        conditioning = {
            'encoder_hidden_states': torch.randn(
                batch_size, 77, spec['context_dim'], 
                device=device, dtype=dtype
            )
        }
        
        if spec['pooled_embeds']:
            conditioning['text_embeds'] = torch.randn(
                batch_size, spec['text_encoder_2_dim'],
                device=device, dtype=dtype
            )
        
        if spec['time_ids']:
            conditioning['time_ids'] = torch.randn(
                batch_size, 6,  # [height, width, crop_h, crop_w, target_height, target_width]
                device=device, dtype=dtype
            )
        
        return conditioning
    
    def test_unet_conditioning(self, unet: UNet2DConditionModel) -> Dict[str, bool]:
        """Test what conditioning the UNet actually supports"""
        results = {
            'basic': False,
            'added_cond_kwargs': False,
            'separate_args': False
        }
        
        try:
            # Ensure model is on CUDA and in eval mode for testing
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            unet_test = unet.to(device).eval()
            
            # Create test inputs on the same device
            sample = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float16)
            timestep = torch.tensor([0.5], device=device, dtype=torch.float32)
            conditioning = self.create_sample_conditioning(1, device=device)
            
            # Test basic call
            try:
                with torch.no_grad():
                    _ = unet_test(sample, timestep, conditioning['encoder_hidden_states'])
                results['basic'] = True
            except Exception:
                pass
            
            # Test added_cond_kwargs (standard SDXL)
            if self.is_sdxl:
                try:
                    added_cond = {}
                    if 'text_embeds' in conditioning:
                        added_cond['text_embeds'] = conditioning['text_embeds']
                    if 'time_ids' in conditioning:
                        added_cond['time_ids'] = conditioning['time_ids']
                    
                    with torch.no_grad():
                        _ = unet_test(sample, timestep, conditioning['encoder_hidden_states'], 
                               added_cond_kwargs=added_cond)
                    results['added_cond_kwargs'] = True
                except Exception:
                    pass
                
                # Test separate arguments (some implementations)
                try:
                    args = [sample, timestep, conditioning['encoder_hidden_states']]
                    if 'text_embeds' in conditioning:
                        args.append(conditioning['text_embeds'])
                    if 'time_ids' in conditioning:
                        args.append(conditioning['time_ids'])
                    
                    with torch.no_grad():
                        _ = unet_test(*args)
                    results['separate_args'] = True
                except Exception:
                    pass
                    
        except Exception as e:
            # If testing fails completely, provide safe defaults
            print(f"⚠️ UNet conditioning test setup failed: {e}")
            results = {
                'basic': True,  # Assume basic call works
                'added_cond_kwargs': self.is_sdxl,  # Assume SDXL models support this
                'separate_args': False
            }
        
        return results


class SDXLTensorRTWrapper(torch.nn.Module):
    """Advanced SDXL wrapper for TensorRT with intelligent conditioning detection"""
    
    def __init__(self, unet: UNet2DConditionModel, model_path: str = ""):
        super().__init__()
        self.unet = unet
        self.model_path = model_path
        
        # Use comprehensive model detection
        model_info = detect_model_comprehensive(unet, model_path)
        
        self.unet_info = model_info
        self.is_sdxl = model_info['is_sdxl']
        self.is_turbo = model_info['is_turbo']
        
        self.conditioning_handler = SDXLConditioningHandler(self.unet_info)
        
        # Test what the UNet actually supports (with error handling)
        try:
            self.supported_calls = self.conditioning_handler.test_unet_conditioning(unet)
        except Exception as e:
            print(f"⚠️ UNet conditioning test failed: {e}")
            # Provide safe defaults
            self.supported_calls = {
                'basic': True,
                'added_cond_kwargs': self.is_sdxl,
                'separate_args': False
            }
        
        print(f"🔍 SDXL Model Analysis:")
        print(f"  Model Path: {model_path}")
        print(f"  Is SDXL: {self.is_sdxl}")
        print(f"  Is Turbo: {self.is_turbo}")
        print(f"  UNet Info: {self.unet_info}")
        print(f"  Supported Calls: {self.supported_calls}")
    
    def forward(self, *args, **kwargs):
        """Intelligent forward pass based on detected capabilities"""
        
        # CRITICAL: Always ensure added_cond_kwargs is never None to prevent TypeError
        if 'added_cond_kwargs' in kwargs and kwargs['added_cond_kwargs'] is None:
            kwargs['added_cond_kwargs'] = {}
        
        # For any model that might be SDXL or have SDXL-like requirements
        if self.is_sdxl or self.unet_info.get('has_addition_embed', False):
            
            if len(args) >= 3:  # sample, timestep, encoder_hidden_states
                sample, timestep, encoder_hidden_states = args[:3]
                
                # For SDXL models that require text_time conditioning, ensure it's provided
                if self.unet_info.get('addition_embed_type') == 'text_time' and 'added_cond_kwargs' not in kwargs:
                    print("🔧 Auto-generating required SDXL conditioning for ONNX export...")
                    device = sample.device
                    batch_size = sample.shape[0]
                    
                    # Generate the required conditioning
                    kwargs['added_cond_kwargs'] = {
                        'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                        'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
                    }
                
                # Try added_cond_kwargs first (standard SDXL)
                if self.supported_calls.get('added_cond_kwargs', False):
                    # Ensure we always have the dict, even if empty
                    if 'added_cond_kwargs' not in kwargs:
                        kwargs['added_cond_kwargs'] = {}
                    return self.unet(sample, timestep, encoder_hidden_states, **kwargs)
                
                # Try separate arguments (some SDXL implementations)
                elif self.supported_calls.get('separate_args', False) and len(args) > 3:
                    return self.unet(*args)
                
                # Fall back to basic call 
                elif self.supported_calls.get('basic', True):
                    return self.unet(sample, timestep, encoder_hidden_states)
                
                # Last resort - create minimal valid SDXL conditioning
                else:
                    try:
                        # For SDXL models that require conditioning, provide minimal valid data
                        if self.unet_info.get('addition_embed_type') == 'text_time':
                            # Create minimal conditioning that SDXL expects
                            device = sample.device
                            batch_size = sample.shape[0]
                            
                            minimal_conditioning = {
                                'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                                'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
                            }
                            return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=minimal_conditioning)
                        else:
                            # Non-text_time SDXL, try empty dict
                            return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs={})
                    except Exception as e:
                        print(f"⚠️ SDXL fallback with conditioning failed: {e}")
                        # Absolute last resort: try without added_cond_kwargs but ensure no None issues
                        try:
                            return self.unet(sample, timestep, encoder_hidden_states)
                        except TypeError as te:
                            if "NoneType" in str(te):
                                # The UNet is expecting added_cond_kwargs but gets None internally
                                # This means we need to provide it, even if minimal
                                minimal_conditioning = {
                                    'text_embeds': torch.zeros(sample.shape[0], 1280, device=sample.device, dtype=sample.dtype),
                                    'time_ids': torch.zeros(sample.shape[0], 6, device=sample.device, dtype=sample.dtype)
                                }
                                return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=minimal_conditioning)
                            else:
                                raise te
        
        # For non-SDXL models, pass through directly but still protect against None
        else:
            # Remove problematic added_cond_kwargs for non-SDXL models
            if 'added_cond_kwargs' in kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k != 'added_cond_kwargs'}
            return self.unet(*args, **kwargs)
    
    def get_onnx_export_spec(self) -> Dict[str, Any]:
        """Get specification for ONNX export"""
        spec = self.conditioning_handler.get_conditioning_spec()
        
        # Add export-specific details
        spec.update({
            'input_names': ['sample', 'timestep', 'encoder_hidden_states'],
            'output_names': ['noise_pred'],
            'dynamic_axes': {
                'sample': {0: 'batch_size'},
                'timestep': {0: 'batch_size'},
                'encoder_hidden_states': {0: 'batch_size'},
                'noise_pred': {0: 'batch_size'}
            }
        })
        
        # Add SDXL-specific inputs if supported
        if self.is_sdxl and self.supported_calls['added_cond_kwargs']:
            if spec['pooled_embeds']:
                spec['input_names'].append('text_embeds')
                spec['dynamic_axes']['text_embeds'] = {0: 'batch_size'}
            
            if spec['time_ids']:
                spec['input_names'].append('time_ids')
                spec['dynamic_axes']['time_ids'] = {0: 'batch_size'}
        
        return spec


def create_sdxl_tensorrt_wrapper(unet: UNet2DConditionModel, model_path: str = "") -> torch.nn.Module:
    """Factory function to create appropriate SDXL TensorRT wrapper"""
    return SDXLTensorRTWrapper(unet, model_path)


def get_sdxl_tensorrt_config(model_path: str, unet: UNet2DConditionModel) -> Dict[str, Any]:
    """Get complete TensorRT configuration for SDXL model"""
    config = detect_model_comprehensive(unet, model_path)
    
    # Add conditioning specification
    conditioning_handler = SDXLConditioningHandler(config)
    config['conditioning_spec'] = conditioning_handler.get_conditioning_spec()
    
    return config 