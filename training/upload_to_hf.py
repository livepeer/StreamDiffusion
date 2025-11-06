#!/usr/bin/env python3
"""
Upload TemporalNet2 ControlNet model to HuggingFace Hub
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_to_huggingface(model_dir, repo_id, private=False):
    """
    Upload a model directory to HuggingFace Hub
    
    Args:
        model_dir: Path to the model directory containing config.json and safetensors
        repo_id: Repository ID on HuggingFace (e.g., 'username/model-name')
        private: Whether to create a private repository
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Check for required files
    required_files = ["config.json"]
    safetensors_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.safetensors.index.json"))
    
    for req_file in required_files:
        if not (model_dir / req_file).exists():
            raise ValueError(f"Required file not found: {req_file}")
    
    if not safetensors_files:
        raise ValueError("No safetensors files found in model directory")
    
    print(f"Uploading model from {model_dir} to {repo_id}")
    print(f"Found {len(safetensors_files)} safetensors file(s)")
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Create repository
    print(f"\nCreating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print("✓ Repository created/verified")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload files
    print(f"\nUploading files...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload TemporalNet2 ControlNet SDXL model"
    )
    
    print("\n" + "="*80)
    print("✓ Upload complete!")
    print("="*80)
    print(f"\nYour model is now available at:")
    print(f"https://huggingface.co/{repo_id}")
    print("\nYou can load it using:")
    print(f"""
from diffusers import ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.float16
)
""")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory (e.g., checkpoint-25000_hf)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repository ID (e.g., 'username/temporalnet2-sdxl-controlnet')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    args = parser.parse_args()
    
    upload_to_huggingface(args.model_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()


