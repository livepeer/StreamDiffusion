from .base import BasePreprocessor
from .canny import CannyPreprocessor
from .depth import DepthPreprocessor
from .openpose import OpenPosePreprocessor
from .lineart import LineartPreprocessor
from .passthrough import PassthroughPreprocessor
from .browser import BrowserPreprocessor

# Try to import TensorRT preprocessors - might not be available on all systems
try:
    from .depth_tensorrt import DepthAnythingTensorrtPreprocessor
    DEPTH_TENSORRT_AVAILABLE = True
except ImportError:
    DepthAnythingTensorrtPreprocessor = None
    DEPTH_TENSORRT_AVAILABLE = False

try:
    from .pose_tensorrt import YoloNasPoseTensorrtPreprocessor
    POSE_TENSORRT_AVAILABLE = True
except ImportError:
    YoloNasPoseTensorrtPreprocessor = None
    POSE_TENSORRT_AVAILABLE = False

try:
    from .mediapipe_pose import MediaPipePosePreprocessor
    MEDIAPIPE_POSE_AVAILABLE = True
except ImportError:
    MediaPipePosePreprocessor = None
    MEDIAPIPE_POSE_AVAILABLE = False

# Registry for easy lookup
_preprocessor_registry = {
    "canny": CannyPreprocessor,
    "depth": DepthPreprocessor,
    "openpose": OpenPosePreprocessor,
    "lineart": LineartPreprocessor,
    "passthrough": PassthroughPreprocessor,
    "browser": BrowserPreprocessor,
}

# Add TensorRT preprocessors if available
if DEPTH_TENSORRT_AVAILABLE:
    _preprocessor_registry["depth_tensorrt"] = DepthAnythingTensorrtPreprocessor

if POSE_TENSORRT_AVAILABLE:
    _preprocessor_registry["pose_tensorrt"] = YoloNasPoseTensorrtPreprocessor

# Add MediaPipe preprocessor if available
if MEDIAPIPE_POSE_AVAILABLE:
    _preprocessor_registry["mediapipe_pose"] = MediaPipePosePreprocessor


def get_preprocessor(name: str) -> BasePreprocessor:
    """
    Get a preprocessor by name
    
    Args:
        name: Name of the preprocessor
        
    Returns:
        Preprocessor instance
        
    Raises:
        ValueError: If preprocessor name is not found
    """
    if name not in _preprocessor_registry:
        available = ", ".join(_preprocessor_registry.keys())
        raise ValueError(f"Unknown preprocessor '{name}'. Available: {available}")
    
    return _preprocessor_registry[name]()


def register_preprocessor(name: str, preprocessor_class):
    """
    Register a new preprocessor
    
    Args:
        name: Name to register under
        preprocessor_class: Preprocessor class
    """
    _preprocessor_registry[name] = preprocessor_class


def list_preprocessors():
    """List all available preprocessors"""
    return list(_preprocessor_registry.keys())


__all__ = [
    "BasePreprocessor",
    "CannyPreprocessor",
    "DepthPreprocessor", 
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "PassthroughPreprocessor",
    "BrowserPreprocessor",
    "get_preprocessor",
    "register_preprocessor",
    "list_preprocessors",
]

if DEPTH_TENSORRT_AVAILABLE:
    __all__.append("DepthAnythingTensorrtPreprocessor")

if POSE_TENSORRT_AVAILABLE:
    __all__.append("YoloNasPoseTensorrtPreprocessor")

if MEDIAPIPE_POSE_AVAILABLE:
    __all__.append("MediaPipePosePreprocessor") 