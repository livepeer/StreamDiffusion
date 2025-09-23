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

    cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.7.29"

    if not is_installed("tensorrt"):
        run_pip(f"install {cudnn_name} --no-cache-dir")
        if cu == "12":
            run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt==10.12.0.36 --no-cache-dir")
            run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt-cu12-bindings==10.12.0.36 --no-cache-dir")
            run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt-cu12-libs==10.12.0.36 --no-cache-dir")
        else:
            # CUDA 11 fallback to last supported TRT 8.x
            run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt==8.6.1 --no-cache-dir")

    if not is_installed("polygraphy"):
        run_pip(
            "install polygraphy==0.49.24 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if not is_installed("onnx_graphsurgeon"):
        run_pip(
            "install onnx-graphsurgeon==0.5.8 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        run_pip(
            "install pywin32"
        )

    pass


if __name__ == "__main__":
    fire.Fire(install)
