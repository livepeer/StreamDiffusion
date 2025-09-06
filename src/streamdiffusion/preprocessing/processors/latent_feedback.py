import torch
import math
from typing import Optional, Any, List, Literal
from .base import PipelineAwareProcessor


class LatentFeedbackPreprocessor(PipelineAwareProcessor):
    """
    Enhanced latent domain feedback preprocessor with temporal and nonlinear effects
    
    Creates configurable feedback loops in latent space with multiple modes:
    - Linear blending (original): Simple weighted blend
    - Temporal decay/accumulation: Accumulates feedback history with decay
    - Nonlinear functions: tanh, sin, sigmoid for trippy effects
    
    Temporal Modes:
    - "none": No temporal accumulation (original behavior)
    - "decay": Exponential decay of feedback history
    - "accumulate": Simple accumulation of recent frames
    
    Nonlinear Modes:
    - "linear": Standard linear blending (original)
    - "tanh": Hyperbolic tangent for smooth saturation
    - "sin": Sinusoidal feedback for oscillating effects
    - "sigmoid": Sigmoid function for smooth transitions
    
    The preprocessor accesses the pipeline's prev_latent_result to get the previous latent output.
    For the first frame (when no previous output exists), it falls back to the input latent.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Enhanced Latent Feedback Loop",
            "description": "Creates configurable feedback loops in latent space with temporal accumulation and nonlinear effects for trippy visual effects.",
            "parameters": {
                "feedback_strength": {
                    "type": "float",
                    "default": 0.5,
                    "range": [0.0, 2.0],
                    "step": 0.01,
                    "description": "Strength of latent feedback blend (0.0 = pure input, 1.0 = equal blend, >1.0 = amplified feedback)"
                },
                "temporal_mode": {
                    "type": "string",
                    "default": "none",
                    "options": ["none", "decay", "accumulate"],
                    "description": "Temporal feedback mode: none (original), decay (exponential), accumulate (simple sum)"
                },
                "history_length": {
                    "type": "int",
                    "default": 5,
                    "range": [1, 20],
                    "description": "Number of previous frames to keep in temporal history"
                },
                "decay_factor": {
                    "type": "float",
                    "default": 0.8,
                    "range": [0.1, 0.99],
                    "step": 0.01,
                    "description": "Decay factor for temporal mode (higher = longer memory)"
                },
                "nonlinear_mode": {
                    "type": "string",
                    "default": "linear",
                    "options": ["linear", "tanh", "sin", "sigmoid"],
                    "description": "Nonlinear feedback function for trippy effects"
                },
                "frequency": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.1, 10.0],
                    "step": 0.1,
                    "description": "Frequency parameter for sin mode (higher = faster oscillation)"
                },
                "phase_speed": {
                    "type": "float",
                    "default": 0.1,
                    "range": [0.0, 1.0],
                    "step": 0.01,
                    "description": "Phase evolution speed for sin mode (creates moving patterns)"
                }
            },
            "use_cases": ["Trippy feedback effects", "Temporal consistency", "Psychedelic visuals", "Oscillating patterns", "Memory effects", "Latent space exploration"]
        }
    
    def __init__(self, 
                 pipeline_ref: Any,
                 feedback_strength: float = 0.5,
                 temporal_mode: Literal["none", "decay", "accumulate"] = "none",
                 history_length: int = 5,
                 decay_factor: float = 0.8,
                 nonlinear_mode: Literal["linear", "tanh", "sin", "sigmoid"] = "linear",
                 frequency: float = 1.0,
                 phase_speed: float = 0.1,
                 **kwargs):
        """
        Initialize enhanced latent feedback preprocessor
        
        Args:
            pipeline_ref: Reference to the StreamDiffusion pipeline instance (required)
            feedback_strength: Strength of feedback blend (0.0 = pure input, 1.0 = equal blend, >1.0 = amplified)
            temporal_mode: Temporal feedback mode ("none", "decay", "accumulate")
            history_length: Number of previous frames to keep in temporal history
            decay_factor: Decay factor for temporal mode (0.1-0.99, higher = longer memory)
            nonlinear_mode: Nonlinear feedback function ("linear", "tanh", "sin", "sigmoid")
            frequency: Frequency parameter for sin mode (0.1-10.0, higher = faster oscillation)
            phase_speed: Phase evolution speed for sin mode (0.0-1.0, creates moving patterns)
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            pipeline_ref=pipeline_ref,
            feedback_strength=feedback_strength,
            temporal_mode=temporal_mode,
            history_length=history_length,
            decay_factor=decay_factor,
            nonlinear_mode=nonlinear_mode,
            frequency=frequency,
            phase_speed=phase_speed,
            **kwargs
        )
        
        # Clamp parameters to safe ranges
        self.feedback_strength = max(0.0, min(2.0, feedback_strength))  # Allow amplified feedback
        self.temporal_mode = temporal_mode
        self.history_length = max(1, min(20, history_length))
        self.decay_factor = max(0.1, min(0.99, decay_factor))
        self.nonlinear_mode = nonlinear_mode
        self.frequency = max(0.1, min(10.0, frequency))
        self.phase_speed = max(0.0, min(1.0, phase_speed))
        
        # State tracking
        self._first_frame = True
        self._frame_count = 0
        self._phase = 0.0
        
        # Temporal history for decay/accumulate modes
        self._feedback_history: List[torch.Tensor] = []
    
    def _get_previous_data(self):
        """Get previous frame latent data from pipeline"""
        if self.pipeline_ref is not None:
            # Get previous OUTPUT latent (after diffusion), not input latent
            # Check for prev_latent_result (the actual attribute name used by the pipeline)
            if hasattr(self.pipeline_ref, 'prev_latent_result'):
                if self.pipeline_ref.prev_latent_result is not None and not self._first_frame:
                    return self.pipeline_ref.prev_latent_result
        return None
    
    def _update_temporal_history(self, latent: torch.Tensor) -> None:
        """Update temporal feedback history with new latent"""
        if self.temporal_mode == "none":
            return
            
        # Add current latent to history
        self._feedback_history.append(latent.detach().clone())
        
        # Limit history length
        if len(self._feedback_history) > self.history_length:
            self._feedback_history.pop(0)
    
    def _compute_temporal_feedback(self) -> Optional[torch.Tensor]:
        """Compute temporal feedback based on history"""
        if self.temporal_mode == "none" or not self._feedback_history:
            return None
            
        if self.temporal_mode == "decay":
            # Exponential decay of feedback history
            accumulated = torch.zeros_like(self._feedback_history[-1])
            for i, hist_latent in enumerate(reversed(self._feedback_history)):
                decay_weight = self.decay_factor ** i
                accumulated += decay_weight * hist_latent
            return accumulated
            
        elif self.temporal_mode == "accumulate":
            # Simple accumulation of recent frames
            accumulated = torch.zeros_like(self._feedback_history[-1])
            for hist_latent in self._feedback_history:
                accumulated += hist_latent
            # Normalize by history length to prevent explosion
            return accumulated / len(self._feedback_history)
            
        return None
    
    def _apply_nonlinear_function(self, feedback_tensor: torch.Tensor) -> torch.Tensor:
        """Apply nonlinear function to feedback tensor"""
        if self.nonlinear_mode == "linear":
            return feedback_tensor
            
        elif self.nonlinear_mode == "tanh":
            # Hyperbolic tangent for smooth saturation
            return torch.tanh(feedback_tensor * self.feedback_strength)
            
        elif self.nonlinear_mode == "sin":
            # Sinusoidal feedback for oscillating effects
            # Update phase for moving patterns
            self._phase += self.phase_speed
            if self._phase > 2 * math.pi:
                self._phase -= 2 * math.pi
                
            return torch.sin(feedback_tensor * self.frequency + self._phase) * self.feedback_strength
            
        elif self.nonlinear_mode == "sigmoid":
            # Sigmoid function for smooth transitions
            return torch.sigmoid(feedback_tensor * self.feedback_strength * 2 - 1) * 2 - 1  # Scale to [-1, 1]
            
        return feedback_tensor
    
    #TODO: eventually, these processors should be divided by input and output domain rather than overriding image-first basec class
    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate latent tensor input - preserve batch dimensions for latent processing
        
        Args:
            latent_tensor: Input latent tensor in format [B, C, H/8, W/8]
            
        Returns:
            Validated latent tensor with preserved batch dimension
        """
        # For latent processing, we want to preserve the batch dimension
        # Only ensure correct device and dtype
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)
        return latent_tensor
        
    #TODO: eventually, these processors should be divided by input and output domain rather than overriding image-first basec class
    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Override base class resize logic - latent tensors should NOT be resized to image dimensions
        
        For latent domain processing, we want to preserve the latent space dimensions,
        not resize to image target dimensions like image-domain processors.
        """
        # For latent feedback, just return the tensor as-is without any resizing
        return tensor
    
    def _process_core(self, image):
        """
        For latent feedback, we don't process PIL images directly.
        This method should not be called in normal latent preprocessing workflows.
        """
        raise NotImplementedError(
            "LatentFeedbackPreprocessor is designed for latent domain processing. "
            "Use _process_tensor_core or process_tensor for latent tensors."
        )
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process latent tensor with enhanced feedback blending including temporal and nonlinear effects
        
        Args:
            tensor: Current input latent tensor in format [B, C, H/8, W/8]
            
        Returns:
            Enhanced blended latent tensor with temporal and nonlinear effects applied
        """
        input_latent = tensor.to(device=self.device, dtype=self.dtype)
        self._frame_count += 1
        
        # Handle first frame
        if self._first_frame:
            self._first_frame = False
            # Initialize temporal history with input for temporal modes
            if self.temporal_mode != "none":
                self._update_temporal_history(input_latent)
            return input_latent
        
        # Get feedback source based on temporal mode
        if self.temporal_mode == "none":
            # Original behavior: use previous frame
            feedback_source = self._get_previous_data()
            if feedback_source is None:
                return input_latent
        else:
            # Temporal modes: use accumulated history
            feedback_source = self._compute_temporal_feedback()
            if feedback_source is None:
                # Update history and return input if no history yet
                self._update_temporal_history(input_latent)
                return input_latent
        
        # Ensure tensors have compatible shapes
        if feedback_source.shape[0] != input_latent.shape[0]:
            if feedback_source.shape[0] == 1:
                feedback_source = feedback_source.expand(input_latent.shape[0], -1, -1, -1)
            elif input_latent.shape[0] == 1:
                input_latent = input_latent.expand(feedback_source.shape[0], -1, -1, -1)
            else:
                min_batch = min(feedback_source.shape[0], input_latent.shape[0])
                feedback_source = feedback_source[:min_batch]
                input_latent = input_latent[:min_batch]
        
        # Resize spatial dimensions if they don't match
        if feedback_source.shape[2:] != input_latent.shape[2:]:
            target_size = input_latent.shape[2:]
            feedback_source = torch.nn.functional.interpolate(
                feedback_source, size=target_size, mode='bilinear', align_corners=False
            )
        
        # Apply nonlinear function to feedback
        if self.nonlinear_mode == "linear":
            # Original linear blending
            blended_latent = (1 - self.feedback_strength) * input_latent + self.feedback_strength * feedback_source
        else:
            # Nonlinear feedback modes
            nonlinear_feedback = self._apply_nonlinear_function(feedback_source)
            blended_latent = input_latent + nonlinear_feedback
        
        # Safety clamping with wider range for trippy effects
        blended_latent = torch.clamp(blended_latent, min=-15.0, max=15.0)
        
        # Update temporal history for next frame
        if self.temporal_mode != "none":
            self._update_temporal_history(blended_latent.detach())
        
        return blended_latent.to(device=self.device, dtype=self.dtype)
