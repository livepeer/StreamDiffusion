import torch
from typing import List, Optional, Union, Dict, Any
import logging
from .base_orchestrator import BaseOrchestrator

logger = logging.getLogger(__name__)


class PostprocessingOrchestrator(BaseOrchestrator[torch.Tensor, torch.Tensor]):
    """
    Orchestrates postprocessing with parallelization and pipelining.
    
    Handles super-resolution, enhancement, style transfer, and other postprocessing operations
    that are applied to generated images after diffusion.
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4):
        super().__init__(device, dtype, max_workers)
        
        # Postprocessing-specific state
        self._last_input_tensor = None
        self._postprocessor_cache: Dict[str, torch.Tensor] = {}
    
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.
        
        For postprocessing, we typically don't need sync processing since most
        postprocessors are stateless and don't have temporal feedback requirements.
        
        Returns:
            False - postprocessing can typically always use pipelined processing
        """
        # Future: Could check for specific postprocessor types that need sync processing
        return False
    
    def process_sync(self, 
                   input_tensor: torch.Tensor,
                   postprocessors: List[Any],
                   *args, **kwargs) -> torch.Tensor:
        """
        Process tensor through postprocessors synchronously.
        
        Args:
            input_tensor: Input tensor to postprocess (typically from diffusion output)
            postprocessors: List of postprocessor instances
            *args, **kwargs: Additional arguments for postprocessors
            
        Returns:
            Postprocessed tensor
        """
        if not postprocessors:
            return input_tensor
        
        # Sequential application of postprocessors
        current_tensor = input_tensor
        for postprocessor in postprocessors:
            if postprocessor is not None:
                current_tensor = self._apply_single_postprocessor(current_tensor, postprocessor)
        
        return current_tensor
    
    def _process_frame_background(self, 
                                input_tensor: torch.Tensor,
                                postprocessors: List[Any],
                                *args, **kwargs) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Implementation of BaseOrchestrator._process_frame_background for postprocessing.
        
        Returns:
            Dictionary containing processing results and status
        """
        try:
            if not postprocessors:
                return {
                    'result': input_tensor,
                    'status': 'success'
                }
            
            # Check for cache hit (same input tensor)
            if (self._last_input_tensor is not None and 
                torch.equal(input_tensor, self._last_input_tensor)):
                return {
                    'result': input_tensor,  # Return original if same input
                    'status': 'success',
                    'cache_hit': True
                }
            
            self._last_input_tensor = input_tensor.clone()
            
            # Process postprocessors in parallel if multiple, sequential if single
            if len(postprocessors) > 1:
                result = self._process_postprocessors_parallel(input_tensor, postprocessors)
            else:
                result = self._apply_single_postprocessor(input_tensor, postprocessors[0])
            
            return {
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"PostprocessingOrchestrator: Background processing failed: {e}")
            return {
                'result': input_tensor,  # Return original on error
                'error': str(e),
                'status': 'error'
            }
    
    def _apply_current_frame_processing(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply processing results from previous iteration.
        
        Implementation of BaseOrchestrator._apply_current_frame_processing for postprocessing.
        
        Returns:
            Processed tensor, or original tensor if no results available
        """
        if not hasattr(self, '_next_frame_result') or self._next_frame_result is None:
            # First frame or no background results - return the stored current input
            if hasattr(self, '_current_input_tensor') and self._current_input_tensor is not None:
                logger.debug("PostprocessingOrchestrator: No background results available, returning original input")
                return self._current_input_tensor
            
            # If we don't have the current input stored, we have an issue
            logger.error("PostprocessingOrchestrator: No background results and no current input tensor available")
            raise RuntimeError("PostprocessingOrchestrator: No processing results available")
        
        result = self._next_frame_result
        if result['status'] != 'success':
            logger.warning(f"PostprocessingOrchestrator: Background processing failed: {result.get('error', 'Unknown error')}")
            # Return original input on error
            if hasattr(self, '_current_input_tensor') and self._current_input_tensor is not None:
                return self._current_input_tensor
            raise RuntimeError("PostprocessingOrchestrator: Background processing failed and no fallback available")
        
        return result['result']
    
    def _process_postprocessors_parallel(self, 
                                       input_tensor: torch.Tensor, 
                                       postprocessors: List[Any]) -> torch.Tensor:
        """
        Process multiple postprocessors in parallel.
        
        Note: This applies postprocessors sequentially for now, but could be extended
        to support parallel processing for independent postprocessors in the future.
        
        Args:
            input_tensor: Input tensor to process
            postprocessors: List of postprocessor instances
            
        Returns:
            Processed tensor
        """
        # For now, apply sequentially since most postprocessors are dependent
        # Future enhancement: Detect independent postprocessors and run in parallel
        current_tensor = input_tensor
        for postprocessor in postprocessors:
            if postprocessor is not None:
                current_tensor = self._apply_single_postprocessor(current_tensor, postprocessor)
        
        return current_tensor
    
    def _apply_single_postprocessor(self, 
                                  input_tensor: torch.Tensor, 
                                  postprocessor: Any) -> torch.Tensor:
        """
        Apply a single postprocessor to the input tensor.
        
        Args:
            input_tensor: Input tensor to process
            postprocessor: Postprocessor instance
            
        Returns:
            Processed tensor
        """
        try:
            # Ensure tensor is on correct device and dtype
            processed_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            
            # Apply postprocessor
            if hasattr(postprocessor, 'process_tensor'):
                # Prefer tensor processing if available
                result = postprocessor.process_tensor(processed_tensor)
            elif hasattr(postprocessor, 'process'):
                # Fallback to general process method
                result = postprocessor.process(processed_tensor)
            elif callable(postprocessor):
                # Treat as callable
                result = postprocessor(processed_tensor)
            else:
                logger.warning(f"PostprocessingOrchestrator: Unknown postprocessor type: {type(postprocessor)}")
                return processed_tensor
            
            # Ensure result is a tensor
            if isinstance(result, torch.Tensor):
                return result.to(device=self.device, dtype=self.dtype)
            else:
                logger.warning(f"PostprocessingOrchestrator: Postprocessor returned non-tensor: {type(result)}")
                return processed_tensor
                
        except Exception as e:
            logger.error(f"PostprocessingOrchestrator: Postprocessor failed: {e}")
            return input_tensor  # Return original on error
    
    def clear_cache(self) -> None:
        """Clear postprocessing cache"""
        self._postprocessor_cache.clear()
        self._last_input_tensor = None
    
    def process_postprocessors_pipelined(self,
                                       input_tensor: torch.Tensor,
                                       postprocessors: List[Any]) -> torch.Tensor:
        """
        Process postprocessors with inter-frame pipelining.
        
        Frame N+1 postprocessing runs during frame N display for maximum performance.
        
        Args:
            input_tensor: Input tensor to postprocess
            postprocessors: List of postprocessor instances
            
        Returns:
            Postprocessed tensor
        """
        # Store current input tensor for fallback use
        self._current_input_tensor = input_tensor.clone()
        
        return self.process_pipelined(input_tensor, postprocessors)
    
    def process_postprocessors_sync(self,
                                  input_tensor: torch.Tensor,
                                  postprocessors: List[Any]) -> torch.Tensor:
        """
        Process postprocessors synchronously.
        
        Args:
            input_tensor: Input tensor to postprocess
            postprocessors: List of postprocessor instances
            
        Returns:
            Postprocessed tensor
        """
        return self.process_sync(input_tensor, postprocessors)
