"""
SDXL Support for TensorRT Acceleration
Handles the complexities of SDXL models with TensorRT including dual encoders,
conditioning parameters, and Turbo variants
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from diffusers import UNet2DConditionModel
from ..utilities.model_detection import (
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

# Set up logger for this module
logger = logging.getLogger(__name__)


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



class SDXLExportWrapper(torch.nn.Module):
    """Wrapper for SDXL UNet to handle optional conditioning in legacy TensorRT"""
    
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.base_unet = self._get_base_unet(unet)
        self.supports_added_cond = self._test_added_cond_support()
        
    def _get_base_unet(self, unet):
        """Extract the base UNet from wrappers like ControlNetUNetExportWrapper"""
        # Handle ControlNet wrapper
        if hasattr(unet, 'unet_model') and hasattr(unet.unet_model, 'config'):
            return unet.unet_model
        elif hasattr(unet, 'unet') and hasattr(unet.unet, 'config'):
            return unet.unet
        elif hasattr(unet, 'config'):
            return unet
        else:
            # Fallback: try to find any attribute that has config
            for attr_name in dir(unet):
                if not attr_name.startswith('_'):
                    attr = getattr(unet, attr_name, None)
                    if hasattr(attr, 'config') and hasattr(attr.config, 'addition_embed_type'):
                        return attr
            return unet
        
    def _test_added_cond_support(self):
        """Test if this SDXL model supports added_cond_kwargs"""
        try:
            # Create minimal test inputs
            sample = torch.randn(1, 4, 8, 8, device='cuda', dtype=torch.float16)
            timestep = torch.tensor([0.5], device='cuda', dtype=torch.float32)
            encoder_hidden_states = torch.randn(1, 77, 2048, device='cuda', dtype=torch.float16)
            
            # Test with added_cond_kwargs
            test_added_cond = {
                'text_embeds': torch.randn(1, 1280, device='cuda', dtype=torch.float16),
                'time_ids': torch.randn(1, 6, device='cuda', dtype=torch.float16)
            }
            
            with torch.no_grad():
                _ = self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=test_added_cond)
            
            logger.info("SDXL model supports added_cond_kwargs")
            return True
            
        except Exception as e:
            logger.error(f"SDXL model does not support added_cond_kwargs: {e}")
            return False
        
    def forward(self, *args, **kwargs):
        """Forward pass that handles SDXL conditioning gracefully"""
        logger.debug(f"[SDXL_WRAPPER] forward: Called with {len(args)} args and {len(kwargs)} kwargs")
        logger.debug(f"[SDXL_WRAPPER] forward: Args shapes: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}")
        logger.debug(f"[SDXL_WRAPPER] forward: Kwargs keys: {list(kwargs.keys())}")
        logger.debug(f"[SDXL_WRAPPER] forward: self.supports_added_cond: {self.supports_added_cond}")
        logger.debug(f"[SDXL_WRAPPER] forward: Underlying UNet type: {type(self.unet)}")
        
        try:
            # Ensure added_cond_kwargs is never None to prevent TypeError
            if 'added_cond_kwargs' in kwargs and kwargs['added_cond_kwargs'] is None:
                logger.debug(f"[SDXL_WRAPPER] forward: Setting added_cond_kwargs from None to empty dict")
                kwargs['added_cond_kwargs'] = {}
            
            # Auto-generate SDXL conditioning if missing and model needs it
            if (len(args) >= 3 and 'added_cond_kwargs' not in kwargs and 
                hasattr(self.base_unet.config, 'addition_embed_type') and 
                self.base_unet.config.addition_embed_type == 'text_time'):
                
                sample = args[0]
                device = sample.device
                batch_size = sample.shape[0]
                
                logger.info("Auto-generating required SDXL conditioning...")
                kwargs['added_cond_kwargs'] = {
                    'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                    'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
                }
                
            # If model supports added conditioning and we have the kwargs, use them
            if self.supports_added_cond and 'added_cond_kwargs' in kwargs:
                logger.debug(f"[SDXL_WRAPPER] forward: Using full SDXL call with added_cond_kwargs")
                logger.debug(f"[SDXL_WRAPPER] forward: About to call self.unet(*args, **kwargs)")
                logger.debug(f"[SDXL_WRAPPER] forward: Starting underlying UNet call...")
                
                import time
                start_time = time.time()
                result = self.unet(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                logger.debug(f"[SDXL_WRAPPER] forward: Underlying UNet call completed in {elapsed_time:.3f}s")
                return result
            elif len(args) >= 3:
                logger.debug(f"[SDXL_WRAPPER] forward: Using basic SDXL call (no added_cond_kwargs)")
                logger.debug(f"[SDXL_WRAPPER] forward: About to call self.unet(args[0], args[1], args[2])")
                
                import time
                start_time = time.time()
                result = self.unet(args[0], args[1], args[2])
                elapsed_time = time.time() - start_time
                
                logger.debug(f"[SDXL_WRAPPER] forward: Basic UNet call completed in {elapsed_time:.3f}s")
                return result
            else:
                logger.debug(f"[SDXL_WRAPPER] forward: Using fallback call")
                # Fallback
                return self.unet(*args, **kwargs)
                
        except (TypeError, AttributeError) as e:
            logger.error(f"[SDXL_WRAPPER] forward: Exception caught: {e}")
            if "NoneType" in str(e) or "iterable" in str(e) or "text_embeds" in str(e):
                # Handle SDXL-Turbo models that need proper conditioning
                logger.info(f"Providing minimal SDXL conditioning due to: {e}")
                if len(args) >= 3:
                    sample, timestep, encoder_hidden_states = args[0], args[1], args[2]
                    device = sample.device
                    batch_size = sample.shape[0]
                    
                    # Create minimal valid SDXL conditioning
                    minimal_conditioning = {
                        'text_embeds': torch.zeros(batch_size, 1280, device=device, dtype=sample.dtype),
                        'time_ids': torch.zeros(batch_size, 6, device=device, dtype=sample.dtype)
                    }
                    
                    try:
                        logger.debug(f"[SDXL_WRAPPER] forward: Trying with minimal conditioning...")
                        return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=minimal_conditioning)
                    except Exception as final_e:
                        logger.info(f"Final fallback to basic call: {final_e}")
                        return self.unet(sample, timestep, encoder_hidden_states)
                else:
                    return self.unet(*args)
            else:
                raise e


        
class SDXLControlNetWrapper(torch.nn.Module):
    """Wrapper for SDXL ControlNet models to handle added_cond_kwargs properly during ONNX export"""
    
    def __init__(self, controlnet_model):
        super().__init__()
        self.controlnet = controlnet_model
        
        # Get device and dtype from model
        if hasattr(controlnet_model, 'device'):
            self.device = controlnet_model.device
        else:
            # Try to infer from first parameter
            try:
                self.device = next(controlnet_model.parameters()).device
            except:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hasattr(controlnet_model, 'dtype'):
            self.dtype = controlnet_model.dtype
        else:
            # Try to infer from first parameter
            try:
                self.dtype = next(controlnet_model.parameters()).dtype
            except:
                self.dtype = torch.float16
    
    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond, conditioning_scale):
        """Forward pass that handles SDXL ControlNet requirements and produces 9 down blocks"""
        batch_size = sample.shape[0]
        
        # Create proper added_cond_kwargs for SDXL
        added_cond_kwargs = {
            'text_embeds': torch.randn(batch_size, 1280, dtype=self.dtype, device=self.device),
            'time_ids': torch.randn(batch_size, 6, dtype=self.dtype, device=self.device)
        }
        
        # Call the ControlNet with proper arguments including conditioning_scale
        result = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )
        
        # Extract down blocks and mid block from result
        if isinstance(result, tuple) and len(result) >= 2:
            down_block_res_samples, mid_block_res_sample = result[0], result[1]
        elif hasattr(result, 'down_block_res_samples') and hasattr(result, 'mid_block_res_sample'):
            down_block_res_samples = result.down_block_res_samples
            mid_block_res_sample = result.mid_block_res_sample
        else:
            raise ValueError(f"Unexpected ControlNet output format: {type(result)}")
        
        # SDXL ControlNet should have exactly 9 down blocks
        if len(down_block_res_samples) != 9:
            raise ValueError(f"SDXL ControlNet expected 9 down blocks, got {len(down_block_res_samples)}")
        
        # Return 9 down blocks + 1 mid block with explicit names matching UNet pattern
        # Following the pattern from controlnet_wrapper.py and models.py:
        # down_block_00: Initial sample (320 channels)
        # down_block_01-03: Block 0 residuals (320 channels) 
        # down_block_04-06: Block 1 residuals (640 channels)
        # down_block_07-08: Block 2 residuals (1280 channels)
        down_block_00 = down_block_res_samples[0]  # Initial: 320 channels, 88x88
        down_block_01 = down_block_res_samples[1]  # Block0: 320 channels, 88x88
        down_block_02 = down_block_res_samples[2]  # Block0: 320 channels, 88x88  
        down_block_03 = down_block_res_samples[3]  # Block0: 320 channels, 44x44
        down_block_04 = down_block_res_samples[4]  # Block1: 640 channels, 44x44
        down_block_05 = down_block_res_samples[5]  # Block1: 640 channels, 44x44
        down_block_06 = down_block_res_samples[6]  # Block1: 640 channels, 22x22
        down_block_07 = down_block_res_samples[7]  # Block2: 1280 channels, 22x22
        down_block_08 = down_block_res_samples[8]  # Block2: 1280 channels, 22x22
        mid_block = mid_block_res_sample            # Mid: 1280 channels, 22x22
        
        # Return as individual tensors to preserve names in ONNX
        return (down_block_00, down_block_01, down_block_02, down_block_03, 
                down_block_04, down_block_05, down_block_06, down_block_07, 
                down_block_08, mid_block)


def create_sdxl_export_wrapper(unet: UNet2DConditionModel) -> SDXLExportWrapper:
    """Factory function to create SDXL export wrapper"""
    return SDXLExportWrapper(unet)


def get_sdxl_tensorrt_config(model_path: str, unet: UNet2DConditionModel) -> Dict[str, Any]:
    """Get complete TensorRT configuration for SDXL model"""
    config = detect_model_comprehensive(unet, model_path)
    
    # Add conditioning specification
    conditioning_handler = SDXLConditioningHandler(config)
    config['conditioning_spec'] = conditioning_handler.get_conditioning_spec()
    
    return config 