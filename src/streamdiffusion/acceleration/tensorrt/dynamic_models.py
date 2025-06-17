from typing import Dict, List, Optional, Union
import tensorrt as trt
import torch

from .models import BaseModel, UNet, VAE, VAEEncoder


class DynamicUNet(UNet):
    """
    UNet model with dynamic batch size support for StreamDiffusion batched denoising
    """
    
    def __init__(
        self,
        fp16: bool = True,
        device: Union[str, torch.device] = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
        opt_batch_size: Optional[int] = None,
        embedding_dim: int = 768,
        unet_dim: int = 4,
        use_control: bool = False,
        unet_arch: Optional[Dict] = None,
        enable_dynamic_batch: bool = True,
    ):
        # Set optimal batch size to frame_buffer_size if not specified
        self.opt_batch_size = opt_batch_size or max_batch_size
        self.enable_dynamic_batch = enable_dynamic_batch
        
        print(f"\n--- DynamicUNet Initialization ---")
        print(f"  min_batch_size: {min_batch_size}")
        print(f"  opt_batch_size: {self.opt_batch_size}")  
        print(f"  max_batch_size: {max_batch_size}")
        print(f"  enable_dynamic_batch: {enable_dynamic_batch}")
        print(f"  use_control: {use_control}")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  unet_dim: {unet_dim}")
        
        # Initialize parent UNet class
        super().__init__(
            fp16=fp16,
            device=device,
            max_batch=max_batch_size,  # Note: base class uses max_batch
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=77,
            unet_dim=unet_dim,
            use_control=use_control,
            unet_arch=unet_arch,
        )
        
        # Store dynamic batch parameters for later use
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        
        print(f"  Initialized with base class max_batch: {self.max_batch}")
        print(f"  Initialized with base class min_batch: {self.min_batch}")
        print("----------------------------------\n")
        
    def get_input_names(self) -> List[str]:
        """Get input names for dynamic batch ONNX export"""
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        
        if self.use_control:
            # Add ControlNet inputs - support multiple ControlNets
            for i in range(4):  # Support up to 4 ControlNets
                input_names.extend([
                    f"controlnet_cond_{i}",
                    f"conditioning_scale_{i}",
                ])
        
        return input_names
    
    def get_output_names(self) -> List[str]:
        """Get output names for ONNX export"""
        return ["latent"]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Define dynamic axes for batch dimension"""
        if not self.enable_dynamic_batch:
            return {}
            
        dynamic_axes = {
            "sample": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"}, 
            "latent": {0: "batch_size"},
        }
        
        if self.use_control:
            # Add dynamic batch dimension for ControlNet inputs
            for i in range(4):
                dynamic_axes[f"controlnet_cond_{i}"] = {0: "batch_size"}
                
        return dynamic_axes
    
    def get_shape_dict(self, opt_batch_size: int, opt_image_height: int, opt_image_width: int) -> Dict[str, List]:
        """Get shape dictionary for TensorRT optimization profile"""
        latent_height = opt_image_height // 8
        latent_width = opt_image_width // 8
        
        if self.enable_dynamic_batch:
            # Dynamic batch size shapes: min, opt, max
            shapes = {
                "sample": [
                    [self.min_batch_size, self.unet_dim, latent_height, latent_width],  # min
                    [self.opt_batch_size, self.unet_dim, latent_height, latent_width],   # opt  
                    [self.max_batch_size, self.unet_dim, latent_height, latent_width],  # max
                ],
                "encoder_hidden_states": [
                    [self.min_batch_size, 77, self.embedding_dim],  # min
                    [self.opt_batch_size, 77, self.embedding_dim],   # opt
                    [self.max_batch_size, 77, self.embedding_dim],  # max
                ],
                "timestep": [
                    [self.min_batch_size],  # min
                    [self.opt_batch_size],   # opt  
                    [self.max_batch_size],  # max
                ],
            }
            
            if self.use_control:
                # Add ControlNet input shapes
                for i in range(4):
                    shapes[f"controlnet_cond_{i}"] = [
                        [self.min_batch_size, 3, opt_image_height, opt_image_width],  # min
                        [self.opt_batch_size, 3, opt_image_height, opt_image_width],   # opt
                        [self.max_batch_size, 3, opt_image_height, opt_image_width],  # max
                    ]
                    shapes[f"conditioning_scale_{i}"] = [
                        [1],  # min (scalar)
                        [1],  # opt  
                        [1],  # max
                    ]
        else:
            # Fixed batch size shapes (current behavior)
            shapes = super().get_shape_dict(opt_batch_size, opt_image_height, opt_image_width)
            
        return shapes
    
    def get_dynamic_input_profile(self, opt_batch_size: int, opt_image_height: int, opt_image_width: int, min_batch_size: int = 1, max_batch_size: int = 4):
        """Get input profile for TensorRT with dynamic batch support"""
        if not self.enable_dynamic_batch:
            return super().get_input_profile(opt_batch_size, opt_image_height, opt_image_width)
        
        print(f"\n--- DynamicUNet Profile Generation ---")
        print(f"  Parameters: min_batch={min_batch_size}, opt_batch={opt_batch_size}, max_batch={max_batch_size}")
        print(f"  Image dimensions: {opt_image_height}x{opt_image_width}")
        
        latent_height = opt_image_height // 8
        latent_width = opt_image_width // 8
        
        # Use the batch sizes directly as passed in (they're already calculated correctly)
        # No need to double them here - that's handled by the caller
        
        # Return dictionary format compatible with TensorRT engine builder
        profile = {
            "sample": [
                (min_batch_size, self.unet_dim, latent_height, latent_width),  # min
                (opt_batch_size, self.unet_dim, latent_height, latent_width),  # opt
                (max_batch_size, self.unet_dim, latent_height, latent_width),  # max
            ],
            "timestep": [
                (min_batch_size,),  # min
                (opt_batch_size,),  # opt
                (max_batch_size,),  # max
            ],
            "encoder_hidden_states": [
                (min_batch_size, 77, self.embedding_dim),  # min
                (opt_batch_size, 77, self.embedding_dim),  # opt
                (max_batch_size, 77, self.embedding_dim),  # max
            ],
        }
        
        print(f"  Generated TensorRT profile:")
        for input_name, shapes in profile.items():
            print(f"    {input_name}:")
            print(f"      min: {shapes[0]}")
            print(f"      opt: {shapes[1]}")
            print(f"      max: {shapes[2]}")
        
        # Add ControlNet inputs if needed
        if self.use_control:
            profile["controlnet_cond"] = [
                (min_batch_size, 3, opt_image_height, opt_image_width),  # min
                (opt_batch_size, 3, opt_image_height, opt_image_width),  # opt
                (max_batch_size, 3, opt_image_height, opt_image_width),  # max
            ]
            print(f"    controlnet_cond:")
            print(f"      min: {profile['controlnet_cond'][0]}")
            print(f"      opt: {profile['controlnet_cond'][1]}")
            print(f"      max: {profile['controlnet_cond'][2]}")
            
        print("--------------------------------------\n")
        return profile

    def get_sample_input(self, batch_size: int, image_height: int, image_width: int):
        """Get sample input for ONNX export with dynamic batch support"""
        if not self.enable_dynamic_batch:
            return super().get_sample_input(batch_size, image_height, image_width)
        
        # For dynamic batch export, use the optimal batch size (batch_size parameter)
        # TensorRT expects ONNX model dimensions to match kOPT dimensions in profile
        effective_batch_size = batch_size  # Use optimal batch size for ONNX export
        latent_height, latent_width = self.check_dims(effective_batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        
        print(f"\n--- DynamicUNet ONNX Sample Input ---")
        print(f"  Called with batch_size: {batch_size}")
        print(f"  Using effective_batch_size: {effective_batch_size} (optimal batch size)")
        print(f"  Image dimensions: {image_height}x{image_width}")
        print(f"  Latent dimensions: {latent_height}x{latent_width}")
        print(f"  dtype: {dtype}")
        
        base_inputs = [
            torch.randn(
                effective_batch_size, self.unet_dim, latent_height, latent_width,
                dtype=torch.float32, device=self.device  # Use float32 for ONNX export
            ),
            torch.ones((effective_batch_size,), dtype=torch.float32, device=self.device),  # timestep
            torch.randn(effective_batch_size, 77, self.embedding_dim, dtype=dtype, device=self.device),  # encoder_hidden_states
        ]
        
        print(f"  Generated input shapes:")
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        for i, (name, inp) in enumerate(zip(input_names, base_inputs)):
            print(f"    {name}: {list(inp.shape)} ({inp.dtype})")
        
        # Add ControlNet input if needed
        if self.use_control:
            control_input = torch.randn(
                effective_batch_size, 3, image_height, image_width,
                dtype=dtype, device=self.device
            )
            base_inputs.append(control_input)
            print(f"    controlnet_cond: {list(control_input.shape)} ({control_input.dtype})")
        
        print("-------------------------------------\n")
        return tuple(base_inputs)  # Return as tuple, not list


class DynamicVAE(VAE):
    """
    VAE model with dynamic batch size support
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
        opt_batch_size: Optional[int] = None,
        enable_dynamic_batch: bool = True,
    ):
        super().__init__(
            device=device,
            max_batch=max_batch_size,  # Note: base class uses max_batch
            min_batch_size=min_batch_size,
        )
        
        self.opt_batch_size = opt_batch_size or max_batch_size
        self.enable_dynamic_batch = enable_dynamic_batch
        # Store dynamic batch parameters for later use
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        
        print(f"\n--- DynamicVAE Initialization ---")
        print(f"  min_batch_size: {min_batch_size}")
        print(f"  opt_batch_size: {self.opt_batch_size}")  
        print(f"  max_batch_size: {max_batch_size}")
        print(f"  enable_dynamic_batch: {enable_dynamic_batch}")
        print("---------------------------------\n")
        
    def get_input_names(self) -> List[str]:
        return ["latent"]
    
    def get_output_names(self) -> List[str]:
        return ["images"]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Define dynamic axes for batch dimension"""
        if not self.enable_dynamic_batch:
            return {}
            
        return {
            "latent": {0: "batch_size"},
            "images": {0: "batch_size"},
        }
    
    def get_shape_dict(self, opt_batch_size: int, opt_image_height: int, opt_image_width: int) -> Dict[str, List]:
        """Get shape dictionary for TensorRT optimization profile"""
        latent_height = opt_image_height // 8
        latent_width = opt_image_width // 8
        
        if self.enable_dynamic_batch:
            return {
                "latent": [
                    [self.min_batch_size, 4, latent_height, latent_width],  # min
                    [self.opt_batch_size, 4, latent_height, latent_width],   # opt
                    [self.max_batch_size, 4, latent_height, latent_width],  # max
                ],
            }
        else:
            return super().get_shape_dict(opt_batch_size, opt_image_height, opt_image_width)
    
    def get_dynamic_input_profile(self, opt_batch_size: int, opt_image_height: int, opt_image_width: int, min_batch_size: int = 1, max_batch_size: int = 4):
        """Get input profile for TensorRT with dynamic batch support"""
        if not self.enable_dynamic_batch:
            return super().get_input_profile(opt_batch_size, opt_image_height, opt_image_width)
        
        latent_height = opt_image_height // 8
        latent_width = opt_image_width // 8
        
        # Use batch sizes directly as passed in (no CFG doubling for VAE)
        # Return dictionary format compatible with TensorRT engine builder
        profile = {
            "latent": [
                (min_batch_size, 4, latent_height, latent_width),  # min
                (opt_batch_size, 4, latent_height, latent_width),  # opt
                (max_batch_size, 4, latent_height, latent_width),  # max
            ],
        }
                
        return profile

    def get_sample_input(self, batch_size: int, image_height: int, image_width: int):
        """Get sample input for ONNX export with dynamic batch support"""
        if not self.enable_dynamic_batch:
            return super().get_sample_input(batch_size, image_height, image_width)
        
        # For dynamic batch export, use the optimal batch size (batch_size parameter)
        # TensorRT expects ONNX model dimensions to match kOPT dimensions in profile
        effective_batch_size = batch_size  # Use optimal batch size for ONNX export
        latent_height, latent_width = self.check_dims(effective_batch_size, image_height, image_width)
        
        print(f"\n--- DynamicVAE ONNX Sample Input ---")
        print(f"  Called with batch_size: {batch_size}")
        print(f"  Using effective_batch_size: {effective_batch_size} (optimal batch size)")
        print(f"  Image dimensions: {image_height}x{image_width}")
        print(f"  Latent dimensions: {latent_height}x{latent_width}")
        
        result = torch.randn(
            effective_batch_size, 4, latent_height, latent_width,
            dtype=torch.float32, device=self.device
        )
        
        print(f"  Generated VAE input: {list(result.shape)} ({result.dtype})")
        print("------------------------------------\n")
        
        return result


class DynamicVAEEncoder(VAEEncoder):
    """
    VAE Encoder model with dynamic batch size support
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        max_batch_size: int = 4,
        min_batch_size: int = 1,
        opt_batch_size: Optional[int] = None,
        enable_dynamic_batch: bool = True,
    ):
        super().__init__(
            device=device,
            max_batch=max_batch_size,  # Note: base class uses max_batch
            min_batch_size=min_batch_size,
        )
        
        self.opt_batch_size = opt_batch_size or max_batch_size
        self.enable_dynamic_batch = enable_dynamic_batch
        # Store dynamic batch parameters for later use  
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        
    def get_input_names(self) -> List[str]:
        return ["images"]
    
    def get_output_names(self) -> List[str]:
        return ["latent"]
    
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Define dynamic axes for batch dimension"""
        if not self.enable_dynamic_batch:
            return {}
            
        return {
            "images": {0: "batch_size"},
            "latent": {0: "batch_size"},
        }
    
    def get_shape_dict(self, opt_batch_size: int, opt_image_height: int, opt_image_width: int) -> Dict[str, List]:
        """Get shape dictionary for TensorRT optimization profile"""
        if self.enable_dynamic_batch:
            return {
                "images": [
                    [self.min_batch_size, 3, opt_image_height, opt_image_width],  # min
                    [self.opt_batch_size, 3, opt_image_height, opt_image_width],   # opt
                    [self.max_batch_size, 3, opt_image_height, opt_image_width],  # max
                ],
            }
        else:
            return super().get_shape_dict(opt_batch_size, opt_image_height, opt_image_width)
    
    def get_dynamic_input_profile(self, opt_batch_size: int, opt_image_height: int, opt_image_width: int, min_batch_size: int = 1, max_batch_size: int = 4):
        """Get input profile for TensorRT with dynamic batch support"""
        if not self.enable_dynamic_batch:
            return super().get_input_profile(opt_batch_size, opt_image_height, opt_image_width)
        
        # Use batch sizes directly as passed in (no CFG doubling for VAE Encoder)
        # Return dictionary format compatible with TensorRT engine builder
        profile = {
            "images": [
                (min_batch_size, 3, opt_image_height, opt_image_width),  # min
                (opt_batch_size, 3, opt_image_height, opt_image_width),   # opt
                (max_batch_size, 3, opt_image_height, opt_image_width),   # max
            ],
        }
                
        return profile

    def get_sample_input(self, batch_size: int, image_height: int, image_width: int):
        """Get sample input for ONNX export with dynamic batch support"""
        if not self.enable_dynamic_batch:
            return super().get_sample_input(batch_size, image_height, image_width)
        
        # For dynamic batch export, use the optimal batch size (batch_size parameter)
        # TensorRT expects ONNX model dimensions to match kOPT dimensions in profile
        effective_batch_size = batch_size  # Use optimal batch size for ONNX export
        self.check_dims(effective_batch_size, image_height, image_width)
        
        return torch.randn(
            effective_batch_size, 3, image_height, image_width,
            dtype=torch.float32, device=self.device
        )