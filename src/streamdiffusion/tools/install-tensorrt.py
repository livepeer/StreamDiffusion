from typing import Literal, Optional

import fire
from ..pip_utils import is_installed, run_pip
import platform


def _detect_cuda_major() -> Optional[Literal["11", "12"]]:
    try:
        import torch
        return torch.version.cuda.split(".")[0]  # type: ignore
    except Exception:
        return None


def install(cu: Optional[Literal["11", "12"]] = _detect_cuda_major()):
    print("Installing TensorRT requirements...")
    if cu not in ("11", "12"):
        raise RuntimeError("CUDA major version not detected. Pass --cu 11 or --cu 12 explicitly.")

    cudnn_name = (
        f"nvidia-cudnn-cu12==9.7.1.26" if cu == "12" else f"nvidia-cudnn-cu11==8.9.7.29"
    )

    run_pip(f"install {cudnn_name} --no-cache-dir")

    if cu == "12":
        run_pip("install --extra-index-url https://pypi.nvidia.com --no-cache-dir "
                "tensorrt==10.12.0.36 "
                "tensorrt-cu12-bindings==10.12.0.36 "
                "tensorrt-cu12-libs==10.12.0.36")
    else:
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
            "install pywin32==306"
        )


if __name__ == "__main__":
    fire.Fire(install)
