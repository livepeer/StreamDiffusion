from .pipeline import StreamDiffusion
from utils.wrapper import StreamDiffusionWrapper

# ControlNet support
try:
    from .controlnet import (
        ControlNetPipeline,
        create_controlnet_pipeline,
        ControlNetConfig,
        StreamDiffusionControlNetConfig,
        load_config,
        save_config,
        create_wrapper_from_config,
        load_config,
        BasePreprocessor,
        CannyPreprocessor,
        DepthPreprocessor,
        OpenPosePreprocessor,
        LineartPreprocessor,
        get_preprocessor,
    )

    __all__ = [
        "StreamDiffusion",
        "StreamDiffusionWrapper",
        "ControlNetPipeline",
        "create_controlnet_pipeline",
        "ControlNetConfig",
        "StreamDiffusionControlNetConfig",
        "load_config",
        "save_config",
        "create_wrapper_from_config",
        "load_config",
        "BasePreprocessor",
        "CannyPreprocessor",
        "DepthPreprocessor",
        "OpenPosePreprocessor",
        "LineartPreprocessor",
        "get_preprocessor",
    ]
except ImportError as e:
    print(f"Warning: ControlNet support not available: {e}")
    __all__ = ["StreamDiffusion", "StreamDiffusionWrapper"]
