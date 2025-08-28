from typing import List, Optional, Any
import torch

from ..preprocessing.orchestrator_user import OrchestratorUser
from ..hooks import LatentCtx, LatentHook


class LatentProcessingModule(OrchestratorUser):
    """
    Shared base class for latent domain processing modules.
    
    Handles sequential chain execution for both preprocessing and postprocessing
    timing variants. Processing domain is always latent tensors.
    """
    
    def __init__(self, processors: List[Any], order: Optional[List[int]] = None):
        """
        Initialize latent processing module.
        
        Args:
            processors: List of processor instances for sequential execution
            order: Optional list of indices for custom processor ordering
        """
        self.processors = processors
        self.order = order or list(range(len(processors)))
        
    def _process_latent_chain(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Execute sequential chain of processors in latent domain.
        
        Uses the shared orchestrator's sequential chain processing.
        """
        if not self.processors:
            return input_latent
            
        ordered_processors = self._get_ordered_processors()
        return self._preprocessing_orchestrator.execute_pipeline_chain(
            input_latent, ordered_processors, processing_domain="latent"
        )
    
    def _get_ordered_processors(self) -> List[Any]:
        """Return processors in execution order based on configured ordering."""
        if self.order and len(self.order) == len(self.processors):
            return [self.processors[i] for i in self.order]
        return self.processors


class LatentPreprocessingModule(LatentProcessingModule):
    """
    Latent domain preprocessing module - executes after VAE encoding, before diffusion.
    
    Timing: After encode_image(), before predict_x0_batch()
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrator."""
        self.attach_orchestrator(stream)
        stream.latent_preprocessing_hooks.append(self.build_latent_hook())
    
    def build_latent_hook(self) -> LatentHook:
        """Build hook function that processes latent context."""
        def hook(ctx: LatentCtx) -> LatentCtx:
            ctx.latent = self._process_latent_chain(ctx.latent)
            return ctx
        return hook


class LatentPostprocessingModule(LatentProcessingModule):
    """
    Latent domain postprocessing module - executes after diffusion, before VAE decoding.
    
    Timing: After predict_x0_batch(), before decode_image()
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrator."""
        self.attach_orchestrator(stream)
        stream.latent_postprocessing_hooks.append(self.build_latent_hook())
    
    def build_latent_hook(self) -> LatentHook:
        """Build hook function that processes latent context."""
        def hook(ctx: LatentCtx) -> LatentCtx:
            ctx.latent = self._process_latent_chain(ctx.latent)
            return ctx
        return hook
