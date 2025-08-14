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
    Orchestrates ControlNet preprocessing with parallelization, caching, and optimization.
    Handles image format conversion, preprocessor execution, and result caching.
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, max_workers: int = 4):
        super().__init__(device, dtype, max_workers)
        
        # Caching
        self._preprocessed_cache: Dict[str, torch.Tensor] = {}
        self._last_input_frame = None
        
        # Optimized transforms
        self._cached_transform = transforms.ToTensor()
        
        # Cache pipelining decision to avoid hot path checks
        self._preprocessors_cache_key = None
        self._has_feedback_cache = False
    
    def cleanup(self) -> None:
        """Cleanup thread pool resources"""
        super().cleanup()
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass
    
    def _should_use_sync_processing(self, 
                                  preprocessors: List[Optional[Any]], 
                                  *args, **kwargs) -> bool:
        """
        _check_feedback_cached: Efficiently check for feedback preprocessors using caching
        
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
    
    def process_sync(self, 
                   control_image: ControlImage,
                   preprocessors: List[Optional[Any]],
                   scales: List[float],
                   stream_width: int,
                   stream_height: int,
                   index: Optional[int] = None) -> List[Optional[torch.Tensor]]:
        """
        Process control images synchronously for all ControlNets.
        
        Implementation of BaseOrchestrator.process_sync for ControlNet preprocessing.
        
        Args:
            control_image: Input image to process
            preprocessors: List of preprocessor instances
            scales: List of conditioning scales
            stream_width: Target width for processing
            stream_height: Target height for processing
            index: If specified, only process this single ControlNet index
            
        Returns:
            List of processed tensors for each ControlNet
        """
        if index is not None:
            return self._process_single_controlnet(
                control_image, preprocessors, scales, stream_width, stream_height, index
            )
        
        return self._process_multiple_controlnets_sync(
            control_image, preprocessors, scales, stream_width, stream_height
        )
    
    def _process_frame_background(self, 
                                control_image: ControlImage,
                                preprocessors: List[Optional[Any]],
                                scales: List[float],
                                stream_width: int,
                                stream_height: int) -> Dict[str, Any]:
        """
        Process a frame in the background thread.
        
        Implementation of BaseOrchestrator._process_frame_background for ControlNet preprocessing.
        
        Returns:
            Dictionary containing processing results and status
        """
        # Check if any processing is needed
        if not any(scale > 0 for scale in scales):
            return {'status': 'success', 'results': [None] * len(preprocessors)}
        
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
        
        # Process using the existing optimized logic
        return self._execute_background_processing(
            preprocessor_groups,
            control_variants,
            active_indices,
            stream_width,
            stream_height
        )
    
    def _apply_current_frame_processing(self, 
                                      preprocessors: List[Optional[Any]],
                                      scales: List[float]) -> List[Optional[torch.Tensor]]:
        """
        Apply processing results from previous iteration.
        
        Implementation of BaseOrchestrator._apply_current_frame_processing for ControlNet preprocessing.
        
        Returns:
            List of processed tensors, or empty list to signal no update needed
        """
        if not hasattr(self, '_next_frame_result') or self._next_frame_result is None:
            # Return empty list to signal no update needed
            return []
        
        processed_images = [None] * len(preprocessors)
        
        result = self._next_frame_result
        if result['status'] != 'success':
            # Return empty list to signal no update needed on error
            return []
        
        # Handle case where no update is needed (cached input)
        if 'results' in result and len(result['results']) == 0:
            return []
        
        # Apply results to pipeline state
        processed_cache = result.get('processed_cache', {})
        preprocessor_groups = result.get('preprocessor_groups', {})
        
        # Update processed_images with processed results
        for prep_key, group in preprocessor_groups.items():
            cache_key = f"prep_{prep_key}"
            if cache_key in processed_cache:
                processed_image = processed_cache[cache_key]
                for index in group['indices']:
                    processed_images[index] = processed_image
        
        # Update internal cache
        self._preprocessed_cache.clear()
        self._preprocessed_cache.update(processed_cache)
        
        return processed_images
    
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
    
    def clear_cache(self) -> None:
        """Clear preprocessing cache"""
        self._preprocessed_cache.clear()
        self._last_input_frame = None
    
    def process_embedding_preprocessors(self, 
                                      input_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                                      embedding_preprocessors: List[Any],
                                      stream_width: int,
                                      stream_height: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process IPAdapter embedding preprocessors with same parallelism as ControlNet.
        
        Args:
            input_image: Input image to process
            embedding_preprocessors: List of IPAdapterEmbeddingPreprocessor instances
            stream_width: Target width for processing
            stream_height: Target height for processing
            
        Returns:
            List of (positive_embeds, negative_embeds) tuples
        """
        if not embedding_preprocessors:
            return []
        
        # For embedding preprocessing, we don't skip on cache hits - we need the actual embeddings
        # (Unlike spatial preprocessing where empty list means "no update needed")
        
        # Prepare input variants for processing
        control_variants = self._prepare_input_variants(input_image, stream_width, stream_height)
        
        # Process in parallel if multiple preprocessors, otherwise process directly
        if len(embedding_preprocessors) > 1:
            results = self._process_embedding_preprocessors_parallel(
                embedding_preprocessors, control_variants, stream_width, stream_height
            )
        else:
            results = self._process_embedding_preprocessors_sequential(
                embedding_preprocessors, control_variants, stream_width, stream_height
            )
        
        return results
    
    def process_embedding_preprocessors_pipelined(self, 
                                                input_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                                                embedding_preprocessors: List[Any],
                                                stream_width: int,
                                                stream_height: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process IPAdapter embedding preprocessors with inter-frame pipelining.
        
        Frame N+1 embeddings are generated during frame N diffusion for maximum performance.
        
        Returns:
            List of (positive_embeds, negative_embeds) tuples
        """
        if not embedding_preprocessors:
            return []
        
        # Wait for previous frame embedding preprocessing
        self._wait_for_previous_embedding_preprocessing()
        
        # Start next frame embedding preprocessing in background
        self._start_next_frame_embedding_preprocessing(
            input_image, embedding_preprocessors, stream_width, stream_height
        )
        
        # Apply current frame embedding preprocessing results
        return self._apply_current_frame_embedding_preprocessing(embedding_preprocessors)
    
    def _process_embedding_preprocessors_parallel(self,
                                                embedding_preprocessors: List[Any],
                                                control_variants: Dict[str, Any],
                                                stream_width: int,
                                                stream_height: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Process multiple embedding preprocessors in parallel"""
        futures = [
            self._executor.submit(
                self._process_single_embedding_preprocessor,
                i, preprocessor, control_variants, stream_width, stream_height
            )
            for i, preprocessor in enumerate(embedding_preprocessors)
        ]
        
        results = [None] * len(embedding_preprocessors)
        
        for future in futures:
            result = future.result()
            if result and result['embeddings'] is not None:
                index = result['index']
                embeddings = result['embeddings']
                results[index] = embeddings
        
        return results
    
    def _process_embedding_preprocessors_sequential(self,
                                                  embedding_preprocessors: List[Any],
                                                  control_variants: Dict[str, Any],
                                                  stream_width: int,
                                                  stream_height: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Process single embedding preprocessor directly"""
        result = self._process_single_embedding_preprocessor(
            0, embedding_preprocessors[0], control_variants, stream_width, stream_height
        )
        
        if result and result['embeddings'] is not None:
            return [result['embeddings']]
        else:
            return [None]
    
    def _process_single_embedding_preprocessor(self,
                                             index: int,
                                             preprocessor: Any,
                                             control_variants: Dict[str, Any],
                                             stream_width: int,
                                             stream_height: int) -> Optional[Dict[str, Any]]:
        """Process a single embedding preprocessor"""
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
    
    # Private methods for implementation
    
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
        
        # Process groups in parallel if multiple, otherwise process directly
        if len(preprocessor_groups) > 1:
            return self._process_groups_parallel(
                preprocessor_groups, control_variants, stream_width, stream_height, preprocessors
            )
        else:
            return self._process_groups_sequential(
                preprocessor_groups, control_variants, stream_width, stream_height, preprocessors
            )
    
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
    
    def _process_groups_parallel(self,
                               preprocessor_groups: Dict[str, Dict[str, Any]],
                               control_variants: Dict[str, Any],
                               stream_width: int,
                               stream_height: int,
                               preprocessors: List[Optional[Any]]) -> List[Optional[torch.Tensor]]:
        """Process preprocessor groups in parallel"""
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
    
    def _process_groups_sequential(self,
                                 preprocessor_groups: Dict[str, Dict[str, Any]],
                                 control_variants: Dict[str, Any],
                                 stream_width: int,
                                 stream_height: int,
                                 preprocessors: List[Optional[Any]]) -> List[Optional[torch.Tensor]]:
        """Process single preprocessor group directly"""
        prep_key, group = next(iter(preprocessor_groups.items()))
        result = self._process_single_preprocessor_group(
            prep_key, group, control_variants, stream_width, stream_height
        )
        
        processed_images = [None] * len(preprocessors)
        
        if result and result['processed_image'] is not None:
            processed_image = result['processed_image']
            indices = result['indices']
            
            # Cache and assign
            cache_key = f"prep_{prep_key}"
            self._preprocessed_cache[cache_key] = processed_image
            for index in indices:
                processed_images[index] = processed_image
        
        return processed_images
    
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
                # Resize to target spatial resolution if needed to match stream dimensions
                processed_tensor = self._resize_tensor_if_needed(
                    processed_tensor, target_width, target_height
                )
                return processed_tensor.to(device=self.device, dtype=self.dtype)
            except Exception:
                pass  # Fall through to standard processing
        
        # Direct tensor passthrough (no preprocessor)
        if preprocessor is None:
            return self._resize_tensor_if_needed(control_tensor, target_width, target_height)
        
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
    
    def _resize_tensor_if_needed(self,
                               control_tensor: torch.Tensor,
                               target_width: int,
                               target_height: int) -> torch.Tensor:
        """Resize tensor to target size if needed"""
        target_size = (target_height, target_width)
        
        # Handle dimensions efficiently
        if control_tensor.dim() == 4:
            control_tensor = control_tensor[0]
        if control_tensor.dim() == 3 and control_tensor.shape[0] not in [1, 3]:
            control_tensor = control_tensor.permute(2, 0, 1)
        
        # Resize if needed
        if control_tensor.shape[-2:] != target_size:
            if control_tensor.dim() == 3:
                control_tensor = control_tensor.unsqueeze(0)
            control_tensor = torch.nn.functional.interpolate(
                control_tensor, size=target_size, mode='bilinear', align_corners=False
            )
            if control_tensor.shape[0] == 1:
                control_tensor = control_tensor.squeeze(0)
        
        if control_tensor.dim() == 3:
            control_tensor = control_tensor.unsqueeze(0)
        
        return control_tensor.to(device=self.device, dtype=self.dtype)
    
    def _convert_to_tensor(self,
                         control_image: Union[Image.Image, np.ndarray],
                         target_width: int,
                         target_height: int) -> torch.Tensor:
        """Convert PIL Image or numpy array to tensor"""
        # Handle PIL Images
        if isinstance(control_image, Image.Image):
            target_size = (target_width, target_height)
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.LANCZOS)
            
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
    

    
    # Pipelined processing methods (now handled by BaseOrchestrator)
    

    
    def _execute_background_processing(self,
                                     preprocessor_groups: Dict[str, Dict[str, Any]],
                                     control_variants: Dict[str, Any],
                                     active_indices: List[int],
                                     stream_width: int,
                                     stream_height: int) -> Dict[str, Any]:
        """Execute preprocessing in background thread with parallel processing"""
        try:
            processed_cache = {}
            
            if len(preprocessor_groups) > 1:
                # Parallel processing for multiple preprocessors
                futures = []
                for prep_key, group in preprocessor_groups.items():
                    future = self._executor.submit(
                        self._process_single_preprocessor_threadsafe,
                        prep_key, group, control_variants, stream_width, stream_height
                    )
                    futures.append((future, prep_key, group))
                
                # Collect results
                for future, prep_key, group in futures:
                    result = future.result()
                    if result and result[2] is not None:
                        cache_key = f"prep_{prep_key}"
                        processed_cache[cache_key] = result[2]
            else:
                # Single preprocessor - direct processing
                prep_key, group = next(iter(preprocessor_groups.items()))
                result = self._process_single_preprocessor_threadsafe(
                    prep_key, group, control_variants, stream_width, stream_height
                )
                if result and result[2] is not None:
                    cache_key = f"prep_{prep_key}"
                    processed_cache[cache_key] = result[2]
            
            return {
                'processed_cache': processed_cache,
                'preprocessor_groups': preprocessor_groups,
                'active_indices': active_indices,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def _process_single_preprocessor_threadsafe(self,
                                               preprocessor_key: str,
                                               group: Dict[str, Any],
                                               control_variants: Dict[str, Any],
                                               stream_width: int,
                                               stream_height: int) -> Optional[Tuple[str, List[int], torch.Tensor]]:
        """Thread-safe preprocessor logic for background processing"""
        try:
            preprocessor = group['preprocessor']
            
            # Prioritize tensor path for speed
            if (preprocessor is not None and 
                hasattr(preprocessor, 'process_tensor') and 
                control_variants['tensor'] is not None):
                try:
                    processed_image = self.prepare_control_image(
                        control_variants['tensor'], preprocessor, stream_width, stream_height
                    )
                    return preprocessor_key, group['indices'], processed_image
                except Exception:
                    pass  # Fall through to PIL processing
            
            # PIL processing fallback
            image_key = 'image_safe' if 'image_safe' in control_variants else 'image'
            if control_variants[image_key] is not None:
                processed_image = self.prepare_control_image(
                    control_variants[image_key], preprocessor, stream_width, stream_height
                )
                return preprocessor_key, group['indices'], processed_image
            
            # Last resort: create tensor from scratch
            if control_variants['tensor'] is not None:
                processed_image = self.prepare_control_image(
                    control_variants['tensor'], preprocessor, stream_width, stream_height
                )
                return preprocessor_key, group['indices'], processed_image
            
            return None
            
        except Exception as e:
            logger.error(f"PreprocessingOrchestrator: Preprocessor {preprocessor_key} failed: {e}")
            return None
    
        


    
    # Embedding pipelining methods
    
    def _start_next_frame_embedding_preprocessing(self,
                                                input_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                                                embedding_preprocessors: List[Any],
                                                stream_width: int,
                                                stream_height: int) -> None:
        """Start embedding preprocessing for next frame in background thread"""
        if not embedding_preprocessors:
            self._next_embedding_future = None
            return
        
        # Prepare processing data
        control_variants = self._prepare_input_variants(input_image, thread_safe=True)
        
        # Submit optimized background processing
        self._next_embedding_future = self._executor.submit(
            self._process_embedding_preprocessors_background,
            embedding_preprocessors,
            control_variants,
            stream_width,
            stream_height
        )
    
    def _process_embedding_preprocessors_background(self,
                                                  embedding_preprocessors: List[Any],
                                                  control_variants: Dict[str, Any],
                                                  stream_width: int,
                                                  stream_height: int) -> Dict[str, Any]:
        """Background embedding preprocessing in separate thread"""
        try:
            
            if len(embedding_preprocessors) > 1:
                # Parallel processing for multiple preprocessors
                futures = []
                for i, preprocessor in enumerate(embedding_preprocessors):
                    future = self._executor.submit(
                        self._process_single_embedding_preprocessor,
                        i, preprocessor, control_variants, stream_width, stream_height
                    )
                    futures.append((future, i))
                
                # Collect results
                results = [None] * len(embedding_preprocessors)
                for future, index in futures:
                    result = future.result()
                    if result and result['embeddings'] is not None:
                        results[index] = result['embeddings']
            else:
                # Single preprocessor - direct processing
                result = self._process_single_embedding_preprocessor(
                    0, embedding_preprocessors[0], control_variants, stream_width, stream_height
                )
                results = [result['embeddings'] if result and result['embeddings'] is not None else None]
            
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
    
    def _wait_for_previous_embedding_preprocessing(self) -> None:
        """Wait for previous frame embedding preprocessing with optimized timeout"""
        if hasattr(self, '_next_embedding_future') and self._next_embedding_future is not None:
            try:
                # Reduced timeout: 50ms for real-time performance
                self._next_embedding_result = self._next_embedding_future.result(timeout=0.05)
            except concurrent.futures.TimeoutError:
                raise RuntimeError("_wait_for_previous_embedding_preprocessing: Background embedding preprocessing timed out")
            except Exception as e:
                raise RuntimeError(f"_wait_for_previous_embedding_preprocessing: Background embedding preprocessing failed: {e}")
        else:
            self._next_embedding_result = None
    
    def _apply_current_frame_embedding_preprocessing(self,
                                                   embedding_preprocessors: List[Any]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply embedding preprocessing results from previous iteration"""
        if not hasattr(self, '_next_embedding_result') or self._next_embedding_result is None:
            # First frame - no background results available yet
            return [None] * len(embedding_preprocessors)
        
        result = self._next_embedding_result
        if result['status'] != 'success':
            raise RuntimeError(f"_apply_current_frame_embedding_preprocessing: Background processing failed: {result.get('error', 'Unknown error')}")
        
        return result['results']