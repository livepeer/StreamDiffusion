from typing import Literal, Optional

import fire
from packaging.version import Version

from ..pip_utils import is_installed, run_pip, version
import platform


def get_cuda_version_from_torch() -> Optional[Literal["11", "12"]]:
    try:
        import torch
    except ImportError:
        return None

    return torch.version.cuda.split(".")[0]


def install(cu: Optional[Literal["11", "12"]] = get_cuda_version_from_torch()):
    if cu is None or cu not in ["11", "12"]:
        print("Could not detect CUDA version. Please specify manually.")
        return
    print("Installing TensorRT requirements...")

    # Install nvidia-pyindex first (required for NVIDIA packages)
    if not is_installed("nvidia_pyindex"):
        print("Installing nvidia-pyindex...")
        run_pip("install nvidia-pyindex")

    # Install cuda-python (required for TensorRT utilities)
    if not is_installed("cuda"):
        print("Installing cuda-python...")
        run_pip("install cuda-python")

    # Install onnxruntime with GPU support (required for ONNX operations)
    if not is_installed("onnxruntime"):
        print("Installing onnxruntime-gpu...")
        # Install GPU version for better performance and newer IR version support
        run_pip("install onnxruntime-gpu")

    # Install colored for better logging output
    if not is_installed("colored"):
        print("Installing colored...")
        run_pip("install colored")

    if is_installed("tensorrt"):
        current_version = version("tensorrt")
        if current_version and current_version < Version("10.0.0"):
            print("Uninstalling old TensorRT version...")
            run_pip("uninstall -y tensorrt")

    # Use CUDA-specific TensorRT packages
    if cu == "12":
        tensorrt_package = "tensorrt-cu12"
        cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.4.25"
    else:
        tensorrt_package = "tensorrt-cu11" 
        cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.4.25"

    if not is_installed("tensorrt"):
        print(f"Installing {cudnn_name}...")
        run_pip(f"install {cudnn_name} --no-cache-dir")
        print(f"Installing {tensorrt_package}...")
        run_pip(f"install {tensorrt_package} --no-cache-dir")

    if not is_installed("polygraphy"):
        print("Installing polygraphy...")
        run_pip(
            "install polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if not is_installed("onnx_graphsurgeon"):
        print("Installing onnx-graphsurgeon...")
        run_pip(
            "install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        print("Installing pywin32...")
        run_pip(
            "install pywin32"
        )

    print("TensorRT installation completed!")

if __name__ == "__main__":
    fire.Fire(install)