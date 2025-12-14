import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional
import fire
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt.git"
REPO_COMMIT = "1f4c161949b3616516745781fb91444e6443cc25"
MODEL_REPO_ID = "yuvraj108c/Depth-Anything-2-Onnx"
MODEL_FILENAME = "depth_anything_v2_vits.onnx"

def compile_depth(
    output_dir: str = "engines/depth-anything",
    workspace_dir: str = "workspace",
    force_rebuild: bool = False,
):
    """
    Compile Depth Anything TensorRT engine.
    
    Args:
        output_dir: Directory to save the engine.
        workspace_dir: Directory to clone the repository.
        force_rebuild: Force rebuild even if engine exists.
    """
    output_path = Path(output_dir)
    workspace_path = Path(workspace_dir)
    engine_path = output_path / "depth_anything_v2_vits.engine"
    
    if engine_path.exists() and not force_rebuild:
        logger.info(f"Engine already exists: {engine_path}")
        return

    # 1. Setup Workspace
    repo_dir = workspace_path / "ComfyUI-Depth-Anything-Tensorrt"
    if not repo_dir.exists():
        logger.info(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    
    logger.info(f"Checking out commit {REPO_COMMIT}...")
    subprocess.run(["git", "-C", str(repo_dir), "checkout", REPO_COMMIT], check=True)

    # 2. Download ONNX
    logger.info(f"Downloading {MODEL_FILENAME} from {MODEL_REPO_ID}...")
    onnx_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
    )
    
    # 3. Build Engine
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Building TensorRT engine...")
    
    # Install requirements if needed
    if (repo_dir / "requirements.txt").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_dir, check=True)

    subprocess.run(
        [
            sys.executable,
            "export_trt.py",
            "--trt-path",
            str(engine_path.absolute()),
            "--onnx-path",
            str(onnx_path),
        ],
        cwd=repo_dir,
        check=True,
    )
    
    logger.info(f"Successfully built engine: {engine_path}")

if __name__ == "__main__":
    fire.Fire(compile_depth)
