from typing import List, Optional, Any
import torch

from ..preprocessing.orchestrator_user import OrchestratorUser
from ..hooks import ImageCtx, ImageHook


class ImageProcessingModule(OrchestratorUser):
    """
    Shared base class for image domain processing modules.
    
    Handles sequential chain execution for both preprocessing and postprocessing
    timing variants. Processing domain is always image tensors.
    """
    
    def __init__(self, processors: List[Any], order: Optional[List[int]] = None):
        """
        Initialize image processing module.
        
        Args:
            processors: List of processor instances for sequential execution
            order: Optional list of indices for custom processor ordering
        """
        self.processors = processors
        self.order = order or list(range(len(processors)))
        
    def _process_image_chain(self, input_image: torch.Tensor) -> torch.Tensor:
        """Execute sequential chain of processors in image domain.
        
        Uses the shared orchestrator's sequential chain processing.
        """
        if not self.processors:
            return input_image
            
        ordered_processors = self._get_ordered_processors()
        return self._preprocessing_orchestrator.execute_pipeline_chain(
            input_image, ordered_processors, processing_domain="image"
        )
    
    def _get_ordered_processors(self) -> List[Any]:
        """Return processors in execution order based on configured ordering."""
        if self.order and len(self.order) == len(self.processors):
            return [self.processors[i] for i in self.order]
        return self.processors


class ImagePreprocessingModule(ImageProcessingModule):
    """
    Image domain preprocessing module - executes before VAE encoding.
    
    Timing: After image_processor.preprocess(), before similar_image_filter
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrator."""
        self.attach_orchestrator(stream)
        stream.image_preprocessing_hooks.append(self.build_image_hook())
    
    def build_image_hook(self) -> ImageHook:
        """Build hook function that processes image context."""
        def hook(ctx: ImageCtx) -> ImageCtx:
            ctx.image = self._process_image_chain(ctx.image)
            return ctx
        return hook


class ImagePostprocessingModule(ImageProcessingModule):
    """
    Image domain postprocessing module - executes after VAE decoding.
    
    Timing: After decode_image(), before returning final output
    """
    
    def install(self, stream) -> None:
        """Install module by registering hook with stream and attaching orchestrator."""
        self.attach_orchestrator(stream)
        stream.image_postprocessing_hooks.append(self.build_image_hook())
    
    def build_image_hook(self) -> ImageHook:
        """Build hook function that processes image context."""
        def hook(ctx: ImageCtx) -> ImageCtx:
            ctx.image = self._process_image_chain(ctx.image)
            return ctx
        return hook
