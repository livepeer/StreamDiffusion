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



def get_sdxl_tensorrt_config(model_path: str, unet: UNet2DConditionModel) -> Dict[str, Any]:
    """Get complete TensorRT configuration for SDXL model"""
    config = detect_model_comprehensive(unet, model_path)
    
    # Add conditioning specification
    conditioning_handler = SDXLConditioningHandler(config)
    config['conditioning_spec'] = conditioning_handler.get_conditioning_spec()
    
    return config 