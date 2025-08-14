import torch
from typing import List, Optional, Union, Dict, Any, Tuple, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
import numpy as np
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

# Type variables for generic orchestrator
T = TypeVar('T')  # Input type (e.g., ControlImage for preprocessing)
R = TypeVar('R')  # Result type (e.g., List[torch.Tensor] for preprocessing)


class BaseOrchestrator(Generic[T, R], ABC):
    """
    Generic base orchestrator for parallelized and pipelined processing.
    
    Handles thread pool management, pipeline state, and inter-frame pipelining
    while leaving domain-specific processing logic to subclasses.
    
    Type Parameters:
        T: Input type for processing operations
        R: Result type returned from processing operations
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4):
        self.device = device
        self.dtype = dtype
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Pipeline state for pipelined processing
        self._next_frame_future = None
        self._next_frame_result = None
        
        # Pipeline state for embedding preprocessing (used by preprocessing orchestrator)
        self._next_embedding_future = None
        self._next_embedding_result = None
    
    def cleanup(self) -> None:
        """Cleanup thread pool resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass
    
    @abstractmethod
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Determine if synchronous processing should be used instead of pipelined.
        
        Subclasses implement domain-specific logic (e.g., feedback preprocessor detection).
        
        Returns:
            True if sync processing should be used, False for pipelined processing
        """
        pass
    
    @abstractmethod
    def _process_frame_background(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Subclasses implement their specific processing logic here.
        
        Returns:
            Dictionary containing processing results and status
        """
        pass
    
    def process_pipelined(self, input_data: T, *args, **kwargs) -> R:
        """
        Process input with intelligent pipelining.
        
        Automatically falls back to sync processing when required by domain logic,
        otherwise uses pipelined processing for performance.
        
        Args:
            input_data: Input data to process
            *args, **kwargs: Additional arguments passed to processing methods
        
        Returns:
            Processing results
        """
        # Check if sync processing is required (domain-specific logic)
        if self._should_use_sync_processing(*args, **kwargs):
            return self.process_sync(input_data, *args, **kwargs)
        
        # Use pipelined processing
        # Wait for previous frame processing; non-blocking with short timeout
        self._wait_for_previous_processing()
        
        # Start next frame processing in background
        self._start_next_frame_processing(input_data, *args, **kwargs)
        
        # Apply current frame processing results if available; otherwise signal no update
        return self._apply_current_frame_processing(*args, **kwargs)
    
    @abstractmethod
    def process_sync(self, input_data: T, *args, **kwargs) -> R:
        """
        Process input synchronously.
        
        Subclasses implement their specific synchronous processing logic.
        
        Args:
            input_data: Input data to process
            *args, **kwargs: Additional arguments passed to processing methods
        
        Returns:
            Processing results
        """
        pass
    
    def _start_next_frame_processing(self, input_data: T, *args, **kwargs) -> None:
        """Start processing for next frame in background thread"""
        # Submit background processing
        self._next_frame_future = self._executor.submit(
            self._process_frame_background, input_data, *args, **kwargs
        )
    
    def _wait_for_previous_processing(self) -> None:
        """Wait for previous frame processing with optimized timeout"""
        if hasattr(self, '_next_frame_future') and self._next_frame_future is not None:
            try:
                # Reduced timeout: 10ms for lower latency in real-time
                self._next_frame_result = self._next_frame_future.result(timeout=0.01)
            except concurrent.futures.TimeoutError:
                # Non-blocking: skip applying results this frame
                self._next_frame_result = None
            except Exception as e:
                logger.error(f"BaseOrchestrator: Processing error: {e}")
                self._next_frame_result = None
        else:
            self._next_frame_result = None
    
    @abstractmethod
    def _apply_current_frame_processing(self, *args, **kwargs) -> R:
        """
        Apply processing results from previous iteration.
        
        Subclasses implement how to handle and return the processed results.
        
        Returns:
            Processing results, or appropriate "no update" signal
        """
        pass
