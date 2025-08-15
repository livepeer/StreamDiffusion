import torch
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from PIL import Image
import numpy as np
import concurrent.futures
import logging
from diffusers.utils import load_image
import torchvision.transforms as transforms
from .base_orchestrator import BaseOrchestrator

logger = logging.getLogger(__name__)

# Type alias for control image input
ControlImage = Union[str, Image.Image, np.ndarray, torch.Tensor]


class PreprocessingOrchestrator(BaseOrchestrator[ControlImage, List[Optional[torch.Tensor]]]):
    """
    Orchestrates module preprocessing with typical orchestrator pipelining, but with additional intraframe parallelization, caching, and optimization.
    Modules (IPAdapter, Controlnet) share intraframe parallelism. 
    Handles image format conversion (while most are GPU native,some preprocessors are CPU only), preprocessor execution, and result caching.
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4):
        # Preprocessing: 10ms timeout for fast frame-skipping behavior
        super().__init__(device, dtype, max_workers, timeout_ms=10.0)
        
        # Caching
        self._preprocessed_cache: Dict[str, torch.Tensor] = {}
        self._last_input_frame = None
        
        # Optimized transforms
        self._cached_transform = transforms.ToTensor()
        
        # Cache pipelining decision to avoid hot path checks
        self._preprocessors_cache_key = None
        self._has_feedback_cache = False
        
    
    #Abstract method implementations
    def process_sync(self, 
                   control_image: ControlImage,
                   preprocessors: List[Optional[Any]],
                   scales: List[float] = None,
                   stream_width: int = None,
                   stream_height: int = None,
                   index: Optional[int] = None,
                   processing_type: str = "controlnet") -> Union[List[Optional[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process images synchronously for ControlNet or IPAdapter preprocessing.
        
        Args:
            control_image: Input image to process
            preprocessors: List of preprocessor instances
            scales: List of conditioning scales (ControlNet only, optional for IPAdapter)
            stream_width: Target width for processing
            stream_height: Target height for processing
            index: If specified, only process this single ControlNet index (ControlNet only)
            processing_type: "controlnet" or "ipadapter" to specify processing mode
            
        Returns:
            ControlNet: List of processed tensors for each ControlNet
            IPAdapter: List of (positive_embeds, negative_embeds) tuples
        """
        if processing_type == "ipadapter":
            # IPAdapter processing - no scales needed
            return self._process_multiple_ipadapters_sync(control_image, preprocessors, stream_width, stream_height)
        elif processing_type == "controlnet":
            # ControlNet processing - requires scales
            if scales is None:
                raise ValueError("ControlNet processing requires 'scales' parameter")
            if index is not None:
                return self._process_single_controlnet(
                    control_image, preprocessors, scales, stream_width, stream_height, index
                )
            return self._process_multiple_controlnets_sync(
                control_image, preprocessors, scales, stream_width, stream_height
            )
        else:
            raise ValueError(f"Invalid processing_type: {processing_type}. Must be 'controlnet' or 'ipadapter'")
    
    def _should_use_sync_processing(self, *args, **kwargs) -> bool:
        """
        Check for feedback preprocessors that require sync processing.
        
        Feedback preprocessors need synchronous processing to avoid temporal artifacts.
        
        Args:
            *args: Arguments from process_pipelined call (preprocessors, scales, stream_width, stream_height)
            **kwargs: Keyword arguments
            
        Returns:
            True if feedback preprocessors detected, False otherwise
        """
        # Extract preprocessors from args - they're the first argument after control_image
        if len(args) < 1:
            return False
        
        preprocessors = args[0]  # preprocessors is first arg after control_image
        return self._check_feedback_cached(preprocessors)

    def _process_frame_background(self, 
                                control_image: ControlImage,
                                *args, **kwargs) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Implementation of BaseOrchestrator._process_frame_background for ControlNet preprocessing.
        Automatically detects processing mode based on current state.
        
        Returns:
            Dictionary containing processing results and status
        """
        # Check if we're in embedding processing mode
        if hasattr(self, '_current_processing_mode') and self._current_processing_mode == "embedding":
            # Handle embedding preprocessing
            embedding_preprocessors = args[0]
            stream_width = args[1]  
            stream_height = args[2]
            
            # Prepare processing data
            control_variants = self._prepare_input_variants(control_image, thread_safe=True)
            
            # Process using existing IPAdapter logic
            try:
                results = self._process_ipadapter_preprocessors_parallel(
                    embedding_preprocessors, control_variants, stream_width, stream_height
                )
                return {
                    'results': results,
                    'status': 'success'
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    'error': str(e),
                    'status': 'error'
                }
        else:
            # Handle ControlNet preprocessing (default mode)
            preprocessors = args[0]
            scales = args[1]
            stream_width = args[2]
            stream_height = args[3]
            
            # Check if any processing is needed
            if not any(scale > 0 for scale in scales):
                return {'status': 'success', 'results': [None] * len(preprocessors)}
            #TODO: can we reuse similarity filter here?
            if (self._last_input_frame is not None and 
                isinstance(control_image, (torch.Tensor, np.ndarray, Image.Image)) and 
                control_image is self._last_input_frame):
                return {'status': 'success', 'results': []}  # Signal no update needed
            
            self._last_input_frame = control_image
            
            # Prepare processing data
            preprocessor_groups = self._group_preprocessors(preprocessors, scales)
            active_indices = [i for i, scale in enumerate(scales) if scale > 0]
            
            if not active_indices:
                return {'status': 'success', 'results': [None] * len(preprocessors)}
            
            # Optimize input preparation
            control_variants = self._prepare_input_variants(control_image, thread_safe=True)
            
            # Process using unified parallel logic
            processed_images = self._process_controlnet_preprocessors_parallel(
                preprocessor_groups, control_variants, stream_width, stream_height, preprocessors
            )
            
            return {
                'results': processed_images,
                'status': 'success'
            }
    
    def _apply_current_frame_processing(self, 
                                      preprocessors: List[Optional[Any]] = None,
                                      scales: List[float] = None, 
                                      *args, **kwargs) -> List[Optional[torch.Tensor]]:
        """
        Apply processing results from previous iteration.
        
        Overrides BaseOrchestrator._apply_current_frame_processing for module preprocessing.
        
        Returns:
            List of processed tensors, or empty list to signal no update needed
        """
        if not hasattr(self, '_next_frame_result') or self._next_frame_result is None:
            # Return empty list to signal no update needed
            return []
        
        # Handle case where preprocessors is None
        if preprocessors is None:
            return []
        
        processed_images = [None] * len(preprocessors)
        
        result = self._next_frame_result
        if result['status'] != 'success':
            # Return empty list to signal no update needed on error
            return []
        
        # Handle case where no update is needed (cached input)
        if 'results' in result and len(result['results']) == 0:
            return []
        
        # Get the processed results directly
        processed_images = result.get('results', [])
        if not processed_images:
            return []
        
        return processed_images
    
    #Controlnet methods
    def prepare_control_image(self,
                            control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                            preprocessor: Optional[Any],
                            target_width: int,
                            target_height: int) -> torch.Tensor:
        """
        Prepare a single control image for ControlNet input with format conversion and preprocessing.
        
        Args:
            control_image: Input image in various formats
            preprocessor: Optional preprocessor to apply
            target_width: Target width for the output tensor
            target_height: Target height for the output tensor
            
        Returns:
            Processed tensor ready for ControlNet
        """
        # Load image if path
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        
        # Fast tensor processing path
        if isinstance(control_image, torch.Tensor):
            return self._process_tensor_input(control_image, preprocessor, target_width, target_height)
        
        # Apply preprocessor to non-tensor inputs
        if preprocessor is not None:
            control_image = preprocessor.process(control_image)
        
        # Convert to tensor
        return self._convert_to_tensor(control_image, target_width, target_height)
     
    def _process_multiple_controlnets_sync(self,
                                         control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                                         preprocessors: List[Optional[Any]],
                                         scales: List[float],
                                         stream_width: int,
                                         stream_height: int) -> List[Optional[torch.Tensor]]:
        """Process multiple ControlNets synchronously with parallel execution"""
        # Check if any processing is needed
        if not any(scale > 0 for scale in scales):
            return [None] * len(preprocessors)
        
        #TODO: can we reuse similarity filter here?
        # Check cache for same input - return early without changing anything
        if (self._last_input_frame is not None and 
            isinstance(control_image, (torch.Tensor, np.ndarray, Image.Image)) and 
            control_image is self._last_input_frame):
            # Return empty list to signal no update needed
            return []
        
        self._last_input_frame = control_image
        self.clear_cache()
        
        # Prepare input variants for optimal processing
        control_variants = self._prepare_input_variants(control_image, stream_width, stream_height)
        
        # Group preprocessors to avoid duplicate work
        preprocessor_groups = self._group_preprocessors(preprocessors, scales)
        
        if not preprocessor_groups:
            return [None] * len(preprocessors)
        
        # Process groups using parallel logic (efficient for 1 or many items)
        return self._process_controlnet_preprocessors_parallel(
            preprocessor_groups, control_variants, stream_width, stream_height, preprocessors
        )
   
    def _process_single_controlnet(self,
                                 control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                                 preprocessors: List[Optional[Any]],
                                 scales: List[float],
                                 stream_width: int,
                                 stream_height: int,
                                 index: int) -> List[Optional[torch.Tensor]]:
        """Process a single ControlNet by index"""
        if not (0 <= index < len(preprocessors)):
            raise IndexError(f"ControlNet index {index} out of range")
        
        if scales[index] == 0:
            return [None] * len(preprocessors)
        
        processed_images = [None] * len(preprocessors)
        processed_image = self.prepare_control_image(
            control_image, preprocessors[index], stream_width, stream_height
        )
        processed_images[index] = processed_image
        
        return processed_images
     
    def _process_controlnet_preprocessors_parallel(self,
                               preprocessor_groups: Dict[str, Dict[str, Any]],
                               control_variants: Dict[str, Any],
                               stream_width: int,
                               stream_height: int,
                               preprocessors: List[Optional[Any]]) -> List[Optional[torch.Tensor]]:
        """Process ControlNet preprocessor groups in parallel"""
        futures = [
            self._executor.submit(
                self._process_single_preprocessor_group,
                prep_key, group, control_variants, stream_width, stream_height
            )
            for prep_key, group in preprocessor_groups.items()
        ]
        
        processed_images = [None] * len(preprocessors)
        
        for future in futures:
            result = future.result()
            if result and result['processed_image'] is not None:
                prep_key = result['prep_key']
                processed_image = result['processed_image']
                indices = result['indices']
                
                # Cache and assign
                cache_key = f"prep_{prep_key}"
                self._preprocessed_cache[cache_key] = processed_image
                for index in indices:
                    processed_images[index] = processed_image
        
        return processed_images
    
    #IPAdapter methods
    def _process_multiple_ipadapters_sync(self, 
                              control_image: ControlImage,
                              preprocessors: List[Optional[Any]],
                              stream_width: int,
                              stream_height: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process IPAdapter preprocessors synchronously.
        
        This is the implementation that was previously in process_ipadapter_preprocessors().
        """
        if not preprocessors:
            return []
        
        # For IPAdapter preprocessing, we don't skip on cache hits - we need the actual embeddings
        # (Unlike spatial preprocessing where empty list means "no update needed")
        
        # Prepare input variants for processing
        control_variants = self._prepare_input_variants(control_image, stream_width, stream_height)
        
        # Process using parallel logic (efficient for 1 or many items)
        results = self._process_ipadapter_preprocessors_parallel(
            preprocessors, control_variants, stream_width, stream_height
        )
        
        return results
   
    def _process_ipadapter_preprocessors_parallel(self,
                                                ipadapter_preprocessors: List[Any],
                                                control_variants: Dict[str, Any],
                                                stream_width: int,
                                                stream_height: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Process multiple IPAdapter preprocessors in parallel"""
        futures = [
            self._executor.submit(
                self._process_single_ipadapter,
                i, preprocessor, control_variants, stream_width, stream_height
            )
            for i, preprocessor in enumerate(ipadapter_preprocessors)
        ]
        
        results = [None] * len(ipadapter_preprocessors)
        
        for future in futures:
            result = future.result()
            if result and result['embeddings'] is not None:
                index = result['index']
                embeddings = result['embeddings']
                results[index] = embeddings
        
        return results
    
    def _process_single_ipadapter(self,
                                             index: int,
                                             preprocessor: Any,
                                             control_variants: Dict[str, Any],
                                             stream_width: int,
                                             stream_height: int) -> Optional[Dict[str, Any]]:
        """Process a single IPAdapter preprocessor"""
        try:
            # Use tensor processing if available and input is tensor
            if (hasattr(preprocessor, 'process_tensor') and 
                control_variants['tensor'] is not None):
                embeddings = preprocessor.process_tensor(control_variants['tensor'])
                return {
                    'index': index,
                    'embeddings': embeddings
                }
            
            # Use PIL processing for non-tensor inputs
            if control_variants['image'] is not None:
                embeddings = preprocessor.process(control_variants['image'])
                return {
                    'index': index,
                    'embeddings': embeddings
                }
            
            return None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    #Helper methods
    def _check_feedback_cached(self, preprocessors: List[Optional[Any]]) -> bool:
        """
        Efficiently check for feedback preprocessors using caching
        
        Only performs expensive isinstance checks when preprocessor list actually changes.
        """
        # Create cache key from preprocessor identities
        cache_key = tuple(id(p) for p in preprocessors)
        
        # Return cached result if preprocessors haven't changed
        if cache_key == self._preprocessors_cache_key:
            return self._has_feedback_cache
        
        # Preprocessors changed - recompute and cache
        self._preprocessors_cache_key = cache_key
        self._has_feedback_cache = False
        
        try:
            from .processors.feedback import FeedbackPreprocessor
            for prep in preprocessors:
                if isinstance(prep, FeedbackPreprocessor):
                    self._has_feedback_cache = True
                    break
        except Exception:
            # Fallback on class name check without importing
            for prep in preprocessors:
                if prep is not None and prep.__class__.__name__.lower().startswith('feedback'):
                    self._has_feedback_cache = True
                    break
        
        return self._has_feedback_cache

    def clear_cache(self) -> None:
        """Clear preprocessing cache"""
        self._preprocessed_cache.clear()
        self._last_input_frame = None
    
    def _process_tensor_input(self,
                            control_tensor: torch.Tensor,
                            preprocessor: Optional[Any],
                            target_width: int,
                            target_height: int) -> torch.Tensor:
        """Process tensor input with GPU acceleration when possible"""
        # Fast path for tensor input with GPU preprocessor
        if preprocessor is not None and hasattr(preprocessor, 'process_tensor'):
            try:
                processed_tensor = preprocessor.process_tensor(control_tensor)
                # Ensure NCHW shape
                if processed_tensor.dim() == 3:
                    processed_tensor = processed_tensor.unsqueeze(0)
                return processed_tensor.to(device=self.device, dtype=self.dtype)
            except Exception:
                pass  # Fall through to standard processing
        
        # Direct tensor passthrough (no preprocessor) - preprocessors handle their own sizing
        if preprocessor is None:
            # For passthrough, we still need basic format handling
            if control_tensor.dim() == 3:
                control_tensor = control_tensor.unsqueeze(0)
            return control_tensor.to(device=self.device, dtype=self.dtype)
        
        # Convert to PIL for preprocessor, then back to tensor
        if control_tensor.dim() == 4:
            control_tensor = control_tensor[0]
        if control_tensor.dim() == 3 and control_tensor.shape[0] in [1, 3]:
            control_tensor = control_tensor.permute(1, 2, 0)
        
        if control_tensor.is_cuda:
            control_tensor = control_tensor.cpu()
        
        control_array = control_tensor.numpy()
        if control_array.max() <= 1.0:
            control_array = (control_array * 255).astype(np.uint8)
        
        control_image = Image.fromarray(control_array.astype(np.uint8))
        return self.prepare_control_image(control_image, preprocessor, target_width, target_height)
     
    def _convert_to_tensor(self,
                         control_image: Union[Image.Image, np.ndarray],
                         target_width: int,
                         target_height: int) -> torch.Tensor:
        """Convert PIL Image or numpy array to tensor - preprocessors handle their own sizing"""
        # Handle PIL Images - no resizing here, preprocessors handle their target size
        if isinstance(control_image, Image.Image):
            control_tensor = self._cached_transform(control_image).unsqueeze(0)
            return control_tensor.to(device=self.device, dtype=self.dtype)
        
        # Handle numpy arrays
        if isinstance(control_image, np.ndarray):
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image)
            return self._convert_to_tensor(control_image, target_width, target_height)
        
        raise ValueError(f"Unsupported control image type: {type(control_image)}")
    
    def _to_tensor_safe(self, image: Image.Image) -> torch.Tensor:
        """Thread-safe tensor conversion from PIL Image"""
        return self._cached_transform(image).unsqueeze(0).to(device=self.device, dtype=self.dtype)
    
    def _prepare_input_variants(self,
                              control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                              stream_width: int = None,
                              stream_height: int = None,
                              thread_safe: bool = False) -> Dict[str, Any]:
        """Prepare optimized input variants for different processing paths
        
        Args:
            control_image: Input image in various formats
            stream_width: Target width (unused, kept for backward compatibility)
            stream_height: Target height (unused, kept for backward compatibility)
            thread_safe: If True, use thread-safe key naming for background processing
            
        Returns:
            Dictionary with 'tensor' and 'image'/'image_safe' keys
        """
        image_key = 'image_safe' if thread_safe else 'image'
        
        if isinstance(control_image, torch.Tensor):
            return {
                'tensor': control_image,
                image_key: None  # Will create if needed
            }
        elif isinstance(control_image, Image.Image):
            image_copy = control_image.copy()
            return {
                image_key: image_copy,
                'tensor': self._to_tensor_safe(image_copy)
            }
        elif isinstance(control_image, str):
            image_loaded = load_image(control_image)
            return {
                image_key: image_loaded,
                'tensor': self._to_tensor_safe(image_loaded)
            }
        else:
            return {
                image_key: control_image,
                'tensor': None
            }
    
    def _group_preprocessors(self,
                           preprocessors: List[Optional[Any]],
                           scales: List[float]) -> Dict[str, Dict[str, Any]]:
        """Group preprocessors by type to avoid duplicate processing"""
        preprocessor_groups = {}
        
        for i, scale in enumerate(scales):
            if scale > 0:
                preprocessor = preprocessors[i]
                preprocessor_key = id(preprocessor) if preprocessor is not None else 'passthrough'
                
                if preprocessor_key not in preprocessor_groups:
                    preprocessor_groups[preprocessor_key] = {
                        'preprocessor': preprocessor,
                        'indices': []
                    }
                preprocessor_groups[preprocessor_key]['indices'].append(i)
        
        return preprocessor_groups

    def _process_single_preprocessor_group(self,
                                         prep_key: str,
                                         group: Dict[str, Any],
                                         control_variants: Dict[str, Any],
                                         stream_width: int,
                                         stream_height: int) -> Optional[Dict[str, Any]]:
        """Process a single preprocessor group with optimal input selection"""
        try:
            preprocessor = group['preprocessor']
            indices = group['indices']
            
            # Try tensor processing first (fastest path)
            if (preprocessor is not None and 
                hasattr(preprocessor, 'process_tensor') and 
                control_variants['tensor'] is not None):
                try:
                    processed_image = self.prepare_control_image(
                        control_variants['tensor'], preprocessor, stream_width, stream_height
                    )
                    return {
                        'prep_key': prep_key,
                        'indices': indices,
                        'processed_image': processed_image
                    }
                except Exception:
                    pass  # Fall through to PIL processing
            
            # PIL processing fallback
            if control_variants['image'] is not None:
                processed_image = self.prepare_control_image(
                    control_variants['image'], preprocessor, stream_width, stream_height
                )
                return {
                    'prep_key': prep_key,
                    'indices': indices,
                    'processed_image': processed_image
                }
            
            return None
            
        except Exception as e:
            logger.error(f"PreprocessingOrchestrator: Preprocessor {prep_key} failed: {e}")
            return None

