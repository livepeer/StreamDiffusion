"""
Input Source Management for StreamDiffusion Realtime Img2Img Demo

This module provides a flexible input routing system for different components
(ControlNet, IPAdapter, and base pipeline) to use different input sources
(webcam, uploaded images, uploaded videos).
"""

import logging
from enum import Enum
from typing import Dict, Optional, Union, Any
from pathlib import Path
import torch
from PIL import Image
import cv2
import numpy as np

from util import bytes_to_pt
from utils.video_utils import VideoFrameExtractor


class InputSourceType(Enum):
    """Types of input sources available."""
    WEBCAM = "webcam"
    UPLOADED_IMAGE = "uploaded_image"
    UPLOADED_VIDEO = "uploaded_video"


class InputSource:
    """
    Represents an input source for a component.
    
    Handles different types of inputs (webcam, image, video) and provides
    a unified interface to get the current frame as a tensor.
    """
    
    def __init__(self, source_type: InputSourceType, source_data: Any = None):
        """
        Initialize an input source.
        
        Args:
            source_type: Type of input source
            source_data: Data for the source (PIL Image, video path, or None for webcam)
        """
        self.source_type = source_type
        self.source_data = source_data
        self.is_stream = source_type in [InputSourceType.WEBCAM, InputSourceType.UPLOADED_VIDEO]
        self._current_frame = None
        self._video_extractor = None
        self._logger = logging.getLogger(f"InputSource.{source_type.value}")
        
        # Initialize video extractor if needed
        if source_type == InputSourceType.UPLOADED_VIDEO and source_data:
            self._init_video_extractor()
    
    def _init_video_extractor(self):
        """Initialize video extractor for video input sources."""
        if self.source_data and Path(self.source_data).exists():
            try:
                self._video_extractor = VideoFrameExtractor(str(self.source_data))
                self._logger.info(f"Initialized video extractor for: {self.source_data}")
            except Exception as e:
                self._logger.error(f"Failed to initialize video extractor: {e}")
                self._video_extractor = None
        else:
            self._logger.error(f"Video file not found: {self.source_data}")
    
    def get_frame(self) -> Optional[torch.Tensor]:
        """
        Get the current frame as a PyTorch tensor.
        
        Returns:
            torch.Tensor: Current frame with shape (C, H, W), values in [0, 1], dtype float32
            None: If no frame is available
        """
        try:
            if self.source_type == InputSourceType.WEBCAM:
                # For webcam, return cached frame (will be updated externally)
                return self._current_frame
            
            elif self.source_type == InputSourceType.UPLOADED_IMAGE:
                # For static image, convert to tensor if not already done
                if self._current_frame is None and self.source_data:
                    if isinstance(self.source_data, Image.Image):
                        # Convert PIL Image to tensor
                        img_array = np.array(self.source_data)
                        if img_array.ndim == 3:
                            # Convert HWC to CHW and normalize
                            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                            self._current_frame = img_tensor
                    elif isinstance(self.source_data, bytes):
                        # Convert bytes to tensor using existing utility
                        self._current_frame = bytes_to_pt(self.source_data)
                
                return self._current_frame
            
            elif self.source_type == InputSourceType.UPLOADED_VIDEO:
                # For video, get next frame
                return self._get_video_frame()
            
        except Exception as e:
            self._logger.error(f"Error getting frame from {self.source_type.value}: {e}")
            
        return None
    
    def _get_video_frame(self) -> Optional[torch.Tensor]:
        """Get the next frame from video source."""
        if not self._video_extractor:
            return None
        
        return self._video_extractor.get_frame()
    
    def update_webcam_frame(self, frame_data: Union[bytes, torch.Tensor]):
        """
        Update the current frame for webcam sources.
        
        Args:
            frame_data: Frame data as bytes or tensor
        """
        if self.source_type != InputSourceType.WEBCAM:
            return
        
        try:
            if isinstance(frame_data, bytes):
                self._current_frame = bytes_to_pt(frame_data)
            elif isinstance(frame_data, torch.Tensor):
                self._current_frame = frame_data
        except Exception as e:
            self._logger.error(f"Error updating webcam frame: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self._video_extractor:
            self._video_extractor.cleanup()
            self._video_extractor = None


class InputSourceManager:
    """
    Manages input sources for different components in the pipeline.
    
    Provides a centralized way to set and get input sources for:
    - ControlNet instances (indexed)
    - IPAdapter 
    - Base pipeline
    """
    
    def __init__(self):
        """Initialize the input source manager."""
        self.sources = {
            'controlnet': {},  # {index: InputSource}
            'ipadapter': None,  # Single InputSource
            'base': None       # Single InputSource for main pipeline
        }
        self._logger = logging.getLogger("InputSourceManager")
        
        # Default to webcam for base pipeline
        self.sources['base'] = InputSource(InputSourceType.WEBCAM)
    
    def set_source(self, component: str, source: InputSource, index: Optional[int] = None):
        """
        Set input source for a component.
        
        Args:
            component: Component name ('controlnet', 'ipadapter', 'base')
            source: InputSource instance
            index: Index for ControlNet instances (required for 'controlnet')
        """
        try:
            if component == 'controlnet':
                if index is None:
                    raise ValueError("Index is required for ControlNet components")
                
                # Clean up existing source if any
                if index in self.sources['controlnet']:
                    self.sources['controlnet'][index].cleanup()
                
                self.sources['controlnet'][index] = source
                self._logger.info(f"Set ControlNet {index} input source to {source.source_type.value}")
                
            elif component in ['ipadapter', 'base']:
                # Clean up existing source if any
                if self.sources[component]:
                    self.sources[component].cleanup()
                
                self.sources[component] = source
                self._logger.info(f"Set {component} input source to {source.source_type.value}")
                
            else:
                raise ValueError(f"Unknown component: {component}")
                
        except Exception as e:
            self._logger.error(f"Error setting source for {component}: {e}")
    
    def get_frame(self, component: str, index: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get current frame for a component.
        
        Args:
            component: Component name ('controlnet', 'ipadapter', 'base')
            index: Index for ControlNet instances (required for 'controlnet')
            
        Returns:
            torch.Tensor: Current frame or None if not available
        """
        try:
            if component == 'controlnet':
                if index is None:
                    raise ValueError("Index is required for ControlNet components")
                
                source = self.sources['controlnet'].get(index)
                if source:
                    frame = source.get_frame()
                    if frame is not None:
                        return frame
                
                # Fallback to base pipeline input
                self._logger.debug(f"ControlNet {index} has no input, falling back to base")
                return self._get_fallback_frame()
                
            elif component in ['ipadapter', 'base']:
                source = self.sources[component]
                if source:
                    frame = source.get_frame()
                    if frame is not None:
                        return frame
                
                # Fallback to base pipeline input if not base itself
                if component != 'base':
                    self._logger.debug(f"{component} has no input, falling back to base")
                    return self._get_fallback_frame()
                
        except Exception as e:
            self._logger.error(f"Error getting frame for {component}: {e}")
        
        return None
    
    def _get_fallback_frame(self) -> Optional[torch.Tensor]:
        """Get frame from base pipeline as fallback."""
        base_source = self.sources['base']
        if base_source:
            return base_source.get_frame()
        return None
    
    def update_webcam_frame(self, frame_data: Union[bytes, torch.Tensor]):
        """
        Update webcam frame for all webcam sources.
        
        Args:
            frame_data: Frame data as bytes or tensor
        """
        # Update base pipeline if it's webcam
        if (self.sources['base'] and 
            self.sources['base'].source_type == InputSourceType.WEBCAM):
            self.sources['base'].update_webcam_frame(frame_data)
        
        # Update ControlNet webcam sources
        for source in self.sources['controlnet'].values():
            if source.source_type == InputSourceType.WEBCAM:
                source.update_webcam_frame(frame_data)
        
        # Update IPAdapter if it's webcam
        if (self.sources['ipadapter'] and 
            self.sources['ipadapter'].source_type == InputSourceType.WEBCAM):
            self.sources['ipadapter'].update_webcam_frame(frame_data)
    
    def get_source_info(self, component: str, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a component's input source.
        
        Returns:
            Dictionary with source type and metadata
        """
        try:
            if component == 'controlnet':
                if index is None or index not in self.sources['controlnet']:
                    return {'type': 'fallback', 'source_type': 'base_pipeline'}
                source = self.sources['controlnet'][index]
            elif component in ['ipadapter', 'base']:
                source = self.sources[component]
                if not source:
                    return {'type': 'none'}
            else:
                return {'type': 'unknown'}
            
            return {
                'type': source.source_type.value,
                'is_stream': source.is_stream,
                'has_data': source.source_data is not None
            }
            
        except Exception as e:
            self._logger.error(f"Error getting source info for {component}: {e}")
            return {'type': 'error', 'error': str(e)}
    
    def cleanup(self):
        """Clean up all sources."""
        for source in self.sources['controlnet'].values():
            source.cleanup()
        
        if self.sources['ipadapter']:
            self.sources['ipadapter'].cleanup()
        
        if self.sources['base']:
            self.sources['base'].cleanup()
