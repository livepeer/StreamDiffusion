from typing import Literal

import fire
from ..pip_utils import is_installed, run_pip
import platform


def install(cu: Literal["11", "12"]):
    print("Installing TensorRT requirements...")

    cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.7.29"

    run_pip(f"install {cudnn_name} --no-cache-dir")

    if cu == "12":
        run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt==10.12.0.36 --no-cache-dir")
        run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt-cu12-bindings==10.12.0.36 --no-cache-dir")
        run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt-cu12-libs==10.12.0.36 --no-cache-dir")
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
