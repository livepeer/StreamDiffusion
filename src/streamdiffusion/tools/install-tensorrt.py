from typing import Literal, Optional

import fire
from packaging.version import Version

from ..pip_utils import is_installed, run_pip, version, get_cuda_major
import platform


def install(cu: Optional[Literal["11", "12"]] = get_cuda_major()):
    if cu not in ("11", "12"):
        raise RuntimeError("CUDA major version not detected. Pass --cu 11 or --cu 12 explicitly.")

    print("Installing TensorRT requirements...")

    trt_version = version("tensorrt")


    if cu == "12":
        if trt_version and trt_version < Version("12.0.0"):
            run_pip("uninstall -y tensorrt")

        run_pip(f"install nvidia-cudnn-cu12==9.7.1.26 --no-cache-dir")
        run_pip("install --extra-index-url https://pypi.nvidia.com --no-cache-dir "
                "tensorrt==10.12.0.36 "
                "tensorrt-cu12-bindings==10.12.0.36 "
                "tensorrt-cu12-libs==10.12.0.36")
    else:
        if trt_version and trt_version < Version("9.0.0"):
            run_pip("uninstall -y tensorrt")

        run_pip(f"install nvidia-cudnn-cu11==8.9.7.29 --no-cache-dir")
        run_pip("install --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir")

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
