#!/bin/bash
# Example training script for Temporal Prior ControlNet with SD-Turbo
#
# This script demonstrates how to train a 6-channel Temporal Prior ControlNet
# that learns temporal consistency for real-time video stylization.
#
# Prerequisites:
# 1. JSONL dataset with temporal frame pairs (see data format below)
# 2. SD-Turbo model (will be downloaded automatically from HuggingFace)
# 3. RAFT optical flow model (will be downloaded automatically via torchvision)
#
# Data format (JSONL):
# Each line should be a JSON object:
# {
#   "video_id": "video_name_frame_id",
#   "prompt": "description of the stylized appearance",
#   "negative_prompt": "unwanted artifacts",
#   "prev_img_path": "/path/to/frame_t-1.jpg",
#   "curr_img_path": "/path/to/frame_t.jpg"
# }

# Configuration
export MODEL_NAME="stabilityai/sd-turbo"
export JSONL_PATH="/home/user/StreamDiffusion/temporalnet2_celebs_exp.jsonl"  # UPDATE THIS
export OUTPUT_DIR="/home/user/StreamDiffusion/training/temporal_controlnet"
export RESOLUTION=512
export BATCH_SIZE=16  # Adjust based on GPU memory
export LEARNING_RATE=1e-5
export MAX_TRAIN_STEPS=15000
export CHECKPOINTING_STEPS=2500
export GRADIENT_ACCUMULATION_STEPS=4

# Validation samples (automatically extracted from dataset)
export VALIDATION_PROMPT="woman with long hair, arched eyebrows, pointy nose, no beard, wearing lipstick, talking to someone in an indoor setting with dim lighting, side profile view, dark background, focused expression"
export VALIDATION_PREV1="/home/user/datasets/processed_celebs_data/LMGAxyUnURI_0/54.jpg"
export VALIDATION_CURR1="/home/user/datasets/processed_celebs_data/LMGAxyUnURI_0/63.jpg"
export VALIDATION_PREV2="/home/user/datasets/processed_celebs_data/LMGAxyUnURI_0/181.jpg"
export VALIDATION_CURR2="/home/user/datasets/processed_celebs_data/LMGAxyUnURI_0/184.jpg"

# Optional: Limit training samples for quick testing
# export MAX_TRAIN_SAMPLES=1000

# Run training
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$JSONL_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --seed=42 \
  --report_to="wandb" \
  --tracker_project_name="temporalnet2-celebs" \
  --validation_prompt "$VALIDATION_PROMPT" "$VALIDATION_PROMPT" \
  --validation_prev_image "$VALIDATION_PREV1" "$VALIDATION_PREV2" \
  --validation_curr_image "$VALIDATION_CURR1" "$VALIDATION_CURR2" \
  --validation_steps 500 \
  --num_validation_images 2

# Optional parameters:
# --max_train_samples=1000  # Limit dataset size for testing
# --proportion_empty_prompts=0.1  # Randomly replace 10% of prompts with empty strings for CFG
# --use_8bit_adam  # Use 8-bit Adam optimizer to save memory
# --report_to="wandb"  # Use Weights & Biases instead of TensorBoard

echo "Training complete! Model saved to: $OUTPUT_DIR"
echo ""
echo "To use the trained ControlNet:"
echo "1. Load it with: ControlNetModel.from_pretrained('$OUTPUT_DIR')"
echo "2. Build 6-channel temporal prior using the utility functions in the script"
echo "3. Pass it to your StreamDiffusion pipeline with SD-Turbo"


