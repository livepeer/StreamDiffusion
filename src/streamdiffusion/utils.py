import torch
from pathlib import Path
from diffusers.models import ControlNetModel

def load_pytorch_controlnet_model(model_id: str, dtype: torch.dtype, device: str):
        """Load a ControlNet model from HuggingFace or local path"""
        try:
            # Check if it's a local path
            if Path(model_id).exists():
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    local_files_only=True
                )
            else:
                # Try as HuggingFace model ID
                if "/" in model_id and model_id.count("/") > 1:
                    # Handle subfolder case (e.g., "repo/model/subfolder")
                    parts = model_id.split("/")
                    repo_id = "/".join(parts[:2])
                    subfolder = "/".join(parts[2:])
                    controlnet = ControlNetModel.from_pretrained(
                        repo_id,
                        subfolder=subfolder,
                        torch_dtype=dtype
                    )
                else:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=dtype
                    )
            
            # Move to device
            controlnet = controlnet.to(device=device, dtype=dtype)
            return controlnet
            
        except Exception as e:
            raise ValueError(f"Failed to load ControlNet model '{model_id}': {e}")