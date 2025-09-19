"""
Video utilities for handling video input sources in StreamDiffusion.

This module provides classes and functions for extracting frames from video files
and managing video playback for input sources.
"""

import logging
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple


class VideoFrameExtractor:
    """
    Extracts frames from video files for use as input sources.
    
    Handles video playback, looping, and frame extraction with automatic
    conversion to PyTorch tensors.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize the video frame extractor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.current_frame_idx = 0
        self._logger = logging.getLogger(f"VideoFrameExtractor.{self.video_path.name}")
        
        self._initialize_capture()
    
    def _initialize_capture(self):
        """Initialize the video capture object."""
        if not self.video_path.exists():
            self._logger.error(f"Video file not found: {self.video_path}")
            return
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            self._logger.error(f"Failed to open video file: {self.video_path}")
            return
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self._logger.info(f"Initialized video: {self.video_path.name}, "
                         f"FPS: {self.fps:.2f}, Frames: {self.frame_count}")
    
    def get_frame(self) -> Optional[torch.Tensor]:
        """
        Extract the current frame and advance to the next frame.
        
        Automatically loops back to the beginning when reaching the end.
        
        Returns:
            torch.Tensor: Frame as tensor with shape (C, H, W), values in [0, 1], dtype float32
            None: If frame extraction fails
        """
        if not self.cap or not self.cap.isOpened():
            self._logger.error("Video capture not initialized or closed")
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video, loop back to beginning
            self._logger.debug("End of video reached, looping back to start")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            ret, frame = self.cap.read()
            
            if not ret:
                self._logger.error("Failed to read frame even after reset")
                return None
        
        self.current_frame_idx += 1
        
        # Convert frame to tensor
        return self._frame_to_tensor(frame)
    
    def get_frame_at_time(self, timestamp: float) -> Optional[torch.Tensor]:
        """
        Get frame at a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            torch.Tensor: Frame at the specified time or None if failed
        """
        if not self.cap or not self.cap.isOpened():
            return None
        
        # Convert timestamp to frame number
        frame_number = int(timestamp * self.fps)
        frame_number = max(0, min(frame_number, self.frame_count - 1))
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame_idx = frame_number
        
        ret, frame = self.cap.read()
        if ret:
            return self._frame_to_tensor(frame)
        
        return None
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert OpenCV frame to PyTorch tensor.
        
        Args:
            frame: OpenCV frame in BGR format
            
        Returns:
            torch.Tensor: Frame tensor in RGB format with shape (C, H, W)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        
        return frame_tensor
    
    def get_video_info(self) -> dict:
        """
        Get information about the video.
        
        Returns:
            Dictionary with video metadata
        """
        if not self.cap:
            return {}
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        return {
            'path': str(self.video_path),
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'current_frame': self.current_frame_idx
        }
    
    def reset(self):
        """Reset video to beginning."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self._logger.debug("Video reset to beginning")
    
    def cleanup(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self._logger.debug("Video capture released")


def get_video_thumbnail(video_path: str, timestamp: float = 0.0) -> Optional[torch.Tensor]:
    """
    Get a thumbnail frame from a video file.
    
    Args:
        video_path: Path to the video file
        timestamp: Time in seconds to extract thumbnail from
        
    Returns:
        torch.Tensor: Thumbnail frame or None if failed
    """
    try:
        extractor = VideoFrameExtractor(video_path)
        
        if timestamp > 0:
            thumbnail = extractor.get_frame_at_time(timestamp)
        else:
            thumbnail = extractor.get_frame()
        
        extractor.cleanup()
        return thumbnail
        
    except Exception as e:
        logging.getLogger("video_utils").error(f"Failed to get thumbnail: {e}")
        return None


def validate_video_file(video_path: str) -> Tuple[bool, str]:
    """
    Validate if a file is a readable video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(video_path)
        
        if not path.exists():
            return False, "Video file does not exist"
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "Cannot read frames from video"
        
        return True, "Video file is valid"
        
    except Exception as e:
        return False, f"Video validation error: {str(e)}"


# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', 
    '.m4v', '.3gp', '.ogv', '.ts', '.m2ts', '.mts'
}


def is_supported_video_format(filename: str) -> bool:
    """
    Check if a file has a supported video format.
    
    Args:
        filename: Name or path of the file
        
    Returns:
        bool: True if format is supported
    """
    return Path(filename).suffix.lower() in SUPPORTED_VIDEO_FORMATS
