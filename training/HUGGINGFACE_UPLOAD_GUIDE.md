# HuggingFace Upload Guide for TemporalNet2 ControlNet

This guide explains how to convert your DeepSpeed checkpoint to safetensors format and upload it to HuggingFace.

## Step 1: Convert DeepSpeed Checkpoint to PyTorch

First, convert the DeepSpeed checkpoint to a consolidated PyTorch model:

```bash
cd /workspace/StreamDiffusion/training/temporalnet2-sdxl-controlnet/checkpoint-25000
python zero_to_fp32.py . pytorch_model.bin
```

This will create `pytorch_model.bin/` directory with the consolidated model weights.

## Step 2: Convert to Safetensors with Config

Use the provided conversion script to convert to safetensors format with proper HuggingFace config:

```bash
cd /workspace/StreamDiffusion/training
python convert_to_safetensors.py temporalnet2-sdxl-controlnet/checkpoint-25000
```

This will create a new directory `checkpoint-25000_hf/` with:
- `config.json` - Model configuration for diffusers
- `diffusion_pytorch_model-*.safetensors` - Model weights in safetensors format (sharded)
- `diffusion_pytorch_model.safetensors.index.json` - Index for sharded weights
- `README.md` - Model card with usage instructions

You can also specify a custom output directory:
```bash
python convert_to_safetensors.py temporalnet2-sdxl-controlnet/checkpoint-25000 --output-dir my_model
```

## Step 3: Upload to HuggingFace

### Option A: Using the Upload Script

First, login to HuggingFace:
```bash
huggingface-cli login
```

Then upload using the provided script:
```bash
python upload_to_hf.py checkpoint-25000_hf YOUR_USERNAME/temporalnet2-sdxl-controlnet
```

Add `--private` flag to create a private repository:
```bash
python upload_to_hf.py checkpoint-25000_hf YOUR_USERNAME/temporalnet2-sdxl-controlnet --private
```

### Option B: Manual Upload with Python

```python
from huggingface_hub import HfApi

api = HfApi()

# Create repository (only needed once)
api.create_repo('YOUR_USERNAME/temporalnet2-sdxl-controlnet', repo_type='model')

# Upload folder
api.upload_folder(
    folder_path='checkpoint-25000_hf',
    repo_id='YOUR_USERNAME/temporalnet2-sdxl-controlnet',
    repo_type='model'
)
```

### Option C: Using Web Interface

1. Go to https://huggingface.co/new
2. Create a new model repository
3. Use the "Files" tab to upload files manually or use `git` to push

## Using the Model

Once uploaded, anyone can load your model with:

```python
from diffusers import ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained(
    "YOUR_USERNAME/temporalnet2-sdxl-controlnet",
    torch_dtype=torch.float16
)
```

## Model Details

- **Architecture**: ControlNet for SDXL
- **Conditioning Channels**: 6 (3 for previous frame + 3 for optical flow)
- **Base Model**: stabilityai/stable-diffusion-xl-base-1.0
- **Parameters**: ~1.25 billion
- **Size**: ~4.7 GB (sharded into 2 files for efficient loading)

## File Structure

```
checkpoint-25000_hf/
├── config.json                                           # Model configuration
├── diffusion_pytorch_model-00001-of-00002.safetensors   # Model weights (shard 1)
├── diffusion_pytorch_model-00002-of-00002.safetensors   # Model weights (shard 2)
├── diffusion_pytorch_model.safetensors.index.json       # Sharding index
└── README.md                                             # Model card
```

## Troubleshooting

### "huggingface_hub not found"
Install it with:
```bash
pip install huggingface_hub
```

### "You are not authenticated"
Login first:
```bash
huggingface-cli login
```

### Model is too large
The conversion script automatically shards models larger than 4GB. No action needed.

### Want to test before uploading?
Load the model locally first:
```python
from diffusers import ControlNetModel
controlnet = ControlNetModel.from_pretrained(
    "/workspace/StreamDiffusion/training/checkpoint-25000_hf"
)
```

## Notes

- The model uses SDXL architecture with 6 conditioning channels
- Make sure to update `YOUR_USERNAME` with your actual HuggingFace username
- Consider adding example images to your model card after uploading
- You can edit the README.md before uploading to customize the model card


