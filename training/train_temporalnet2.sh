#!/bin/bash

# TemporalNet2 ControlNet Training Script
# This script trains a TemporalNet2 ControlNet on SDXL base model
# with 6-channel input (prev_frame + optical_flow)

export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export TRAIN_DATA="/home/user/StreamDiffusion/temporalnet2_celebs_single_prompt_20.jsonl"
export OUTPUT_DIR="./temporalnet2-sdxl-controlnet"

# Validation images (randomly selected from dataset)
export VALIDATION_PREV_IMG_1="/home/user/datasets/processed_celebs_data/IiUmOrBplcM_9/160.jpg"
export VALIDATION_FLOW_1="/home/user/datasets/processed_celebs_data/IiUmOrBplcM_9/optical_flow_160_163.jpg"
export VALIDATION_PROMPT_1="a young woman with black wavy hair and eyeglasses, talking in a dimly lit room, serious expression, close-up shot, dark ambiance, soft shadows, focused attention"

export VALIDATION_PREV_IMG_2="/home/user/datasets/processed_celebs_data/7hZ0K2nFJCU_1_0/112.jpg"
export VALIDATION_FLOW_2="/home/user/datasets/processed_celebs_data/7hZ0K2nFJCU_1_0/optical_flow_112_114.jpg"
export VALIDATION_PROMPT_2="chubby man with a receding hairline, black hair, mustache, sideburns, holding a microphone, talking, medium shot, casual indoor setting, warm and soft lighting, relaxed posture"

export VALIDATION_PREV_IMG_3="/home/user/datasets/processed_celebs_data/AZxn9md_aZI_1/18.jpg"
export VALIDATION_FLOW_3="/home/user/datasets/processed_celebs_data/AZxn9md_aZI_1/optical_flow_18_20.jpg"
export VALIDATION_PROMPT_3="man with pointy nose and sideburns, looking worried, dark indoor environment, close-up, tense expression, dim lighting, shadows on face, somber atmosphere"

accelerate launch train_controlnet_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --train_data_dir=$TRAIN_DATA \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=300000 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 \
    --checkpoints_total_limit=40 \
    --validation_prompt "$VALIDATION_PROMPT_1" "$VALIDATION_PROMPT_2" "$VALIDATION_PROMPT_3" \
    --validation_prev_image "$VALIDATION_PREV_IMG_1" "$VALIDATION_PREV_IMG_2" "$VALIDATION_PREV_IMG_3" \
    --validation_optical_flow "$VALIDATION_FLOW_1" "$VALIDATION_FLOW_2" "$VALIDATION_FLOW_3" \
    --validation_steps=2000 \
    --num_validation_images=3 \
    --tracker_project_name="temporalnet2-sdxl" \
    --enable_xformers_memory_efficient_attention \
    --report_to="wandb" \
    --seed=42 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=1000 \
    --dataloader_num_workers=8

# Notes:
# - The script now supports multi-resolution training with resolutions: 512, 640, 768, 896, 1024
# - Resolutions are randomly selected per batch with probabilities: 0.40, 0.15, 0.25, 0.05, 0.15
# - The base resolution (512) is used as reference for embeddings
# - ControlNet is initialized with 6 conditioning channels (prev_frame + optical_flow)
# - Dataset format: JSONL with fields: video_id, prompt, negative_prompt, prev_img_path, curr_img_path, optical_flow_path
# - Adjust batch_size and gradient_accumulation_steps based on your GPU memory
# - Total effective batch size = train_batch_size * gradient_accumulation_steps * num_gpus

