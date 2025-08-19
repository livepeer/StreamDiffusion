import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Any, Tuple
from .base import PipelineAwareProcessor

try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from torchvision.utils import flow_to_image
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# torchvision.transforms not needed - we use tensor operations directly


class TemporalNetPreprocessor(PipelineAwareProcessor):
    """
    TemporalNet v2 preprocessor for temporal consistency using optical flow.
    
    This preprocessor computes optical flow between the current input frame and the previous
    pipeline output, then warps the previous frame to create temporal guidance for the
    TemporalNet ControlNet. This ensures temporal consistency in video generation.
    
    The preprocessor follows the FeedbackPreprocessor pattern to access previous pipeline
    results through pipeline_ref.
    
    Key features:
    - RAFT-based optical flow computation for high quality motion estimation
    - Frame warping with hole detection for temporal alignment
    - Fallback handling for first frames and missing dependencies
    - GPU-optimized processing paths
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "TemporalNet v2",
            "description": "Computes optical flow between frames and warps previous output for temporal consistency in video generation.",
            "parameters": {
                "flow_strength": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 2.0],
                    "step": 0.1,
                    "description": "Strength of optical flow warping (1.0 = normal, higher = more warping)"
                },
                "detect_resolution": {
                    "type": "int",
                    "default": 512,
                    "range": [256, 1024],
                    "step": 64,
                    "description": "Resolution for optical flow computation (affects quality vs speed)"
                },

                "output_format": {
                    "type": "str", 
                    "default": "concat",
                    "options": ["concat", "warped_only"],
                    "description": "Output format: 'concat' for 6-channel (current+warped), 'warped_only' for 3-channel warped frame"
                },
                "fast_mode": {
                    "type": "bool",
                    "default": False,
                    "description": "Fast mode: duplicate current frame instead of optical flow (much faster, still provides temporal consistency)"
                }
            },
            "use_cases": ["Video generation", "Temporal consistency", "Animation", "Motion control"]
        }
    
    def __init__(self, 
                 pipeline_ref: Any,
                 image_resolution: int = 512,
                 flow_strength: float = 1.0,
                 detect_resolution: int = 512,
                 output_format: str = "concat",
                 fast_mode: bool = False,
                 **kwargs):
        """
        Initialize TemporalNet preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            image_resolution: Output image resolution
            flow_strength: Strength of optical flow warping
            detect_resolution: Resolution for optical flow computation
            output_format: "concat" for 6-channel TemporalNetV2, "warped_only" for 3-channel
            fast_mode: Fast mode - duplicate current frame instead of optical flow
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for TemporalNet preprocessing. "
                "Install it with: pip install torchvision"
            )
        
        super().__init__(
            pipeline_ref=pipeline_ref,
            image_resolution=image_resolution,
            flow_strength=flow_strength,
            detect_resolution=detect_resolution,
            output_format=output_format,
            fast_mode=fast_mode,
            **kwargs
        )
        
        self.flow_strength = max(0.0, min(2.0, flow_strength))
        self.detect_resolution = detect_resolution
        self._first_frame = True
        self._raft_model = None
    
    @property
    def raft_model(self):
        """Lazy loading of the RAFT optical flow model"""
        if self._raft_model is None:
            print("temporal_net._process_core: Loading RAFT optical flow model")
            self._raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
            self._raft_model = self._raft_model.to(device=self.device)
            self._raft_model.eval()
        return self._raft_model
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Process using optical flow warping of previous frame output
        
        Args:
            image: Current input image
            
        Returns:
            Warped previous frame for temporal guidance, or fallback for first frame
        """
        # Check fast_mode first
        fast_mode = self.params.get('fast_mode', False)
        
        # Check if we have a pipeline reference and previous output
        if (not fast_mode and
            self.pipeline_ref is not None and 
            hasattr(self.pipeline_ref, 'prev_image_result') and 
            self.pipeline_ref.prev_image_result is not None and
            not self._first_frame):
            
            # Get previous output from pipeline and convert to GPU tensors
            prev_output_tensor = self.pipeline_ref.prev_image_result
            if prev_output_tensor.dim() == 4:
                prev_output_tensor = prev_output_tensor[0]  # Remove batch dimension
            
            # Convert from VAE output format [-1, 1] to [0, 1] and ensure on GPU
            prev_tensor = ((prev_output_tensor / 2.0 + 0.5).clamp(0, 1)).to(device=self.device, dtype=self.dtype)
            current_tensor = self.pil_to_tensor(image).squeeze(0).to(device=self.device, dtype=self.dtype)
            
            try:
                
                # Compute optical flow and warp on GPU
                warped_tensor = self._compute_and_warp_tensor(current_tensor, prev_tensor)
                
                # Check output format and return tensor result
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Concatenate current frame + warped frame for TemporalNet2 (6 channels)
                    result_tensor = self._concatenate_frames_tensor(current_tensor, warped_tensor)
                    return self.tensor_to_pil(result_tensor)
                else:
                    # Return only warped frame (3 channels)
                    return self.tensor_to_pil(warped_tensor)
            except Exception as e:
                print(f"temporal_net._process_core: Optical flow failed, using fallback: {e}")
                # Create 6-channel fallback by concatenating current frame with itself
                return self._concatenate_frames(image, image)
        else:
            # First frame or no previous output available
            self._first_frame = False
            
            # For first frame, duplicate current frame to create 6-channel output
            return self._concatenate_frames(image, image)
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process using optical flow warping (GPU-optimized path)
        
        Args:
            tensor: Current input tensor
            
        Returns:
            Warped previous frame tensor for temporal guidance
        """
        # Check fast_mode first
        fast_mode = self.params.get('fast_mode', False)
        
        # Check if we have a pipeline reference and previous output
        if (not fast_mode and
            self.pipeline_ref is not None and 
            hasattr(self.pipeline_ref, 'prev_image_result') and 
            self.pipeline_ref.prev_image_result is not None and
            not self._first_frame):
            
            prev_output = self.pipeline_ref.prev_image_result
            
            # Convert from VAE output format [-1, 1] to [0, 1]
            prev_output = (prev_output / 2.0 + 0.5).clamp(0, 1)
            
            # Normalize input tensor
            input_tensor = tensor
            if input_tensor.max() > 1.0:
                input_tensor = input_tensor / 255.0
            
            # Ensure consistent format
            if prev_output.dim() == 4 and prev_output.shape[0] == 1:
                prev_output = prev_output[0]
            if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
                input_tensor = input_tensor[0]
            
            try:
                # Compute optical flow and warp on GPU
                warped_tensor = self._compute_and_warp_tensor(input_tensor, prev_output)
                
                # Check output format
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Concatenate current frame + warped frame for TemporalNet2 (6 channels)
                    result_tensor = self._concatenate_frames_tensor(input_tensor, warped_tensor)
                else:
                    # Return only warped frame (3 channels)
                    result_tensor = warped_tensor
                
                # Ensure correct output format
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                
                return result_tensor.to(device=self.device, dtype=self.dtype)
            except Exception as e:
                print(f"temporal_net._process_tensor_core: Optical flow failed, using fallback: {e}")
                output_format = self.params.get('output_format', 'concat')
                if output_format == "concat":
                    # Create 6-channel fallback by concatenating current frame with itself
                    result_tensor = self._concatenate_frames_tensor(input_tensor, input_tensor)
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.unsqueeze(0)
                    return result_tensor.to(device=self.device, dtype=self.dtype)
                else:
                    # Create 6-channel fallback by concatenating current frame with itself
                    result_tensor = self._concatenate_frames_tensor(input_tensor, input_tensor)
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.unsqueeze(0)
                    return result_tensor.to(device=self.device, dtype=self.dtype)
        else:
            # First frame or no previous output available
            self._first_frame = False
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            # Handle 6-channel output for first frame
            output_format = self.params.get('output_format', 'concat')
            if output_format == "concat":
                # For first frame, duplicate current frame to create 6-channel output
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                result_tensor = self._concatenate_frames_tensor(current_tensor, current_tensor)
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                return result_tensor.to(device=self.device, dtype=self.dtype)
            else:
                # Create 6-channel fallback by concatenating current frame with itself
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                result_tensor = self._concatenate_frames_tensor(current_tensor, current_tensor)
                if result_tensor.dim() == 3:
                    result_tensor = result_tensor.unsqueeze(0)
                return result_tensor.to(device=self.device, dtype=self.dtype)
    

    
    def _compute_and_warp_tensor(self, current_tensor: torch.Tensor, prev_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow and warp previous tensor using GPU processing
        
        Args:
            current_tensor: Current input frame tensor (CHW format, [0,1]) on GPU
            prev_tensor: Previous pipeline output tensor (CHW format, [0,1]) on GPU
            
        Returns:
            Warped previous frame tensor on GPU
        """
        detect_resolution = self.params.get('detect_resolution', 512)
        target_width, target_height = self.get_target_dimensions()
        
        # Convert to float32 once at the beginning for entire pipeline
        current_tensor = current_tensor.to(device=self.device, dtype=torch.float32)
        prev_tensor = prev_tensor.to(device=self.device, dtype=torch.float32)
        
        # Resize for flow computation if needed (keep on GPU)
        if current_tensor.shape[-1] != detect_resolution or current_tensor.shape[-2] != detect_resolution:
            current_resized = F.interpolate(
                current_tensor.unsqueeze(0), 
                size=(detect_resolution, detect_resolution),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            prev_resized = F.interpolate(
                prev_tensor.unsqueeze(0),
                size=(detect_resolution, detect_resolution), 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            current_resized = current_tensor
            prev_resized = prev_tensor
        
        # Compute optical flow using RAFT (stays on GPU)
        flow = self._compute_optical_flow(current_resized, prev_resized)
        
        # Apply flow strength scaling (GPU operation)
        flow_strength = self.params.get('flow_strength', 1.0)
        if flow_strength != 1.0:
            flow = flow * flow_strength
        
        # Warp previous frame using flow (GPU operation)
        warped_frame = self._warp_frame_tensor(prev_resized, flow)
        
        # Resize back to target resolution if needed (keep on GPU)
        if warped_frame.shape[-1] != target_width or warped_frame.shape[-2] != target_height:
            warped_frame = F.interpolate(
                warped_frame.unsqueeze(0),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Convert to processor's dtype only at the very end
        return warped_frame.to(dtype=self.dtype)
    
    def _compute_optical_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between two frames using RAFT
        
        Args:
            frame1: First frame tensor (CHW format, [0,1])
            frame2: Second frame tensor (CHW format, [0,1])
            
        Returns:
            Optical flow tensor (2HW format)
        """
        # Frames already in float32, just add batch dimension
        frame1_batch = frame1.unsqueeze(0)
        frame2_batch = frame2.unsqueeze(0)
        
        # Apply RAFT preprocessing if available
        weights = Raft_Large_Weights.DEFAULT
        if hasattr(weights, 'transforms') and weights.transforms is not None:
            transforms = weights.transforms()
            frame1_batch, frame2_batch = transforms(frame1_batch, frame2_batch)
        
        # Compute flow
        with torch.no_grad():
            flow_predictions = self.raft_model(frame1_batch, frame2_batch)
            flow = flow_predictions[-1][0]  # Take final prediction, remove batch dim
        
        # Keep flow in float32 for downstream use
        return flow
    
    def _warp_frame_tensor(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp frame using optical flow
        
        Args:
            frame: Frame to warp (CHW format)
            flow: Optical flow (2HW format)
            
        Returns:
            Warped frame tensor
        """
        # Frame already in float32 from pipeline
        H, W = frame.shape[-2:]
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Apply flow to coordinates
        new_x = grid_x + flow[0]
        new_y = grid_y + flow[1]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0
        
        # Create sampling grid (HW2 format for grid_sample)
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        
        # Warp frame
        frame_batch = frame.unsqueeze(0)
        warped_batch = F.grid_sample(
            frame_batch, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Keep in float32 until final conversion
        return warped_batch.squeeze(0)
    

    
    def _concatenate_frames(self, current_image: Image.Image, warped_image: Image.Image) -> Image.Image:
        """
        Concatenate current frame and warped previous frame for TemporalNet2 (6-channel input)
        
        Args:
            current_image: Current input frame
            warped_image: Warped previous frame
            
        Returns:
            PIL Image with concatenated frames (will be converted to 6-channel tensor by ControlNet)
        """
        import numpy as np
        
        # Convert to numpy arrays
        current_np = np.array(current_image)
        warped_np = np.array(warped_image)
        
        # Ensure same size
        if current_np.shape != warped_np.shape:
            target_width, target_height = self.get_target_dimensions()
            current_image = current_image.resize((target_width, target_height), Image.LANCZOS)
            warped_image = warped_image.resize((target_width, target_height), Image.LANCZOS)
            current_np = np.array(current_image)
            warped_np = np.array(warped_image)
        
        # Concatenate along channel dimension: [current_R, current_G, current_B, warped_R, warped_G, warped_B]
        concatenated = np.concatenate([current_np, warped_np], axis=-1)
        
        # Convert back to PIL Image (will be 6 channels)
        return Image.fromarray(concatenated.astype(np.uint8))
    
    def _concatenate_frames_tensor(self, current_tensor: torch.Tensor, warped_tensor: torch.Tensor) -> torch.Tensor:
        """
        Concatenate current frame and warped previous frame tensors for TemporalNet2 (6-channel input)
        
        Args:
            current_tensor: Current input frame tensor (CHW format)
            warped_tensor: Warped previous frame tensor (CHW format)
            
        Returns:
            Concatenated tensor (6CHW format)
        """
        # Ensure same size
        if current_tensor.shape != warped_tensor.shape:
            target_width, target_height = self.get_target_dimensions()
            
            if current_tensor.shape[-2:] != (target_height, target_width):
                current_tensor = F.interpolate(
                    current_tensor.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            if warped_tensor.shape[-2:] != (target_height, target_width):
                warped_tensor = F.interpolate(
                    warped_tensor.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
        
        # Concatenate along channel dimension: [current_R, current_G, current_B, warped_R, warped_G, warped_B]
        concatenated = torch.cat([current_tensor, warped_tensor], dim=0)
        
        return concatenated
    
    def reset(self):
        """
        Reset the preprocessor state (useful for new sequences)
        """
        self._first_frame = True
