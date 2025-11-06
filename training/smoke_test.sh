#!/bin/bash

# Smoke Test for TemporalNet2 ControlNet Training
# This script runs a minimal training to verify everything works correctly

export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export TRAIN_DATA="/workspace/StreamDiffusion/training/smoke_test_data.jsonl"
export OUTPUT_DIR="./smoke_test_output"

# Use the same validation images as the full training
export VALIDATION_PREV_IMG_1="/workspace/datasets/processed_celebs_data/IiUmOrBplcM_9/160.jpg"
export VALIDATION_FLOW_1="/workspace/datasets/processed_celebs_data/IiUmOrBplcM_9/optical_flow_160_163.jpg"
export VALIDATION_PROMPT_1="a young woman with black wavy hair and eyeglasses, talking in a dimly lit room"

echo "=========================================="
echo "Starting Smoke Test"
echo "=========================================="
echo "Training Data: $TRAIN_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "Max Steps: 6"
echo "Validation Steps: 3 (will run at steps 3 and 6)"
echo "Checkpointing Steps: 3"
echo "=========================================="

# Clean up previous smoke test output if it exists
rm -rf $OUTPUT_DIR

accelerate launch train_controlnet_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --train_data_dir=$TRAIN_DATA \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=6 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --checkpointing_steps=3 \
    --validation_prompt "$VALIDATION_PROMPT_1" \
    --validation_prev_image "$VALIDATION_PREV_IMG_1" \
    --validation_optical_flow "$VALIDATION_FLOW_1" \
    --validation_steps=3 \
    --num_validation_images=1 \
    --tracker_project_name="temporalnet2-smoke-test" \
    --enable_xformers_memory_efficient_attention \
    --report_to="tensorboard" \
    --seed=42 \
    --lr_scheduler="constant" \
    --dataloader_num_workers=0

echo ""
echo "=========================================="
echo "Smoke Test Complete!"
echo "=========================================="
echo "If this completed without errors, your setup is working correctly."
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "To view validation images in TensorBoard:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "Look for these log messages to confirm validation ran:"
echo "  - 'Running validation...'"
echo "  - 'Generating validation image 1/1'"
echo "  - 'Validation complete! Generated X validation samples'"
echo ""

