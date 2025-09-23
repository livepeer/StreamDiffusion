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

    if is_installed("tensorrt"):
        try:
            if version("tensorrt") and version("tensorrt") < Version("9.0.0"):
                run_pip("uninstall -y tensorrt")
        except Exception:
            # best-effort cleanup; proceed with install
            pass

    cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.4.25"

    if not is_installed("tensorrt"):
        # Ensure CuDNN for the correct CUDA major is present
        run_pip(f"install {cudnn_name} --no-cache-dir")
        # Install a stable, known-good TensorRT build from NVIDIA PyPI for CUDA {cu}
        # Post11 builds are for CUDA 12.x; for CUDA 11 we fallback to the matching 8.x series
        if cu == "12":
            trt_spec = "tensorrt==9.1.0.post12"  # stable CUDA 12 build
        else:
            trt_spec = "tensorrt==8.6.1"  # last stable for CUDA 11 runtime
        run_pip(
            f"install --extra-index-url https://pypi.nvidia.com {trt_spec} --no-cache-dir"
        )

    if not is_installed("polygraphy"):
        run_pip(
            "install polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if not is_installed("onnx_graphsurgeon"):
        run_pip(
            "install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        run_pip(
            "install pywin32"
        )

    pass


if __name__ == "__main__":
    fire.Fire(install)
