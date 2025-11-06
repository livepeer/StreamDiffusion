# Training Troubleshooting Guide

## Quick Diagnostics

### Check Current Training Status

```bash
# Check if training is running
ps aux | grep train_controlnet_sdxl

# Check GPU memory usage
nvidia-smi

# Check latest checkpoint
ls -lth /workspace/StreamDiffusion/training/temporalnet2-sdxl-controlnet/checkpoint-* | head -5

# Check training logs
tail -f /workspace/StreamDiffusion/training/temporalnet2-sdxl-controlnet/logs/*.log
```

## Common Issues & Solutions

### 1. Still Getting OOM Errors

#### Option A: Reduce batch size further
Edit `train_temporalnet2.sh`:
```bash
--train_batch_size=2 \
--gradient_accumulation_steps=16 \
```

#### Option B: Disable multi-resolution training
Edit `train_controlnet_sdxl.py` line 820-821:
```python
resolutions = [512]
resolution_probs = [1.0]
```

#### Option C: Reduce resolution
Edit `train_temporalnet2.sh`:
```bash
--resolution=384 \  # instead of 512
```

#### Option D: Kill other GPU processes
```bash
# Check what's using GPU memory
nvidia-smi

# Kill specific process (replace PID with actual process ID)
kill -9 <PID>
```

### 2. Blank Validation Images

#### Is this normal?
**YES**, if you're in early training (< 500 steps). The model needs time to learn.

#### When to worry?
Only worry if validation images are still blank/gray after:
- 2000 steps for simple patterns
- 5000 steps for recognizable faces

#### Debug steps:
1. Check validation logs:
```bash
grep "Validation image shape" temporalnet2-sdxl-controlnet/logs/*.log | tail -10
grep "Successfully generated" temporalnet2-sdxl-controlnet/logs/*.log | tail -10
```

2. Verify image shape is correct: `[1, 6, H, W]`

3. Verify value range: `[0.000, 1.000]`

4. Check if validation files exist:
```bash
ls -lh /workspace/datasets/processed_celebs_data/IiUmOrBplcM_9/160.jpg
ls -lh /workspace/datasets/processed_celebs_data/IiUmOrBplcM_9/optical_flow_160_163.jpg
```

### 3. Training Not Resuming from Checkpoint

#### Check checkpoint exists:
```bash
ls -ld temporalnet2-sdxl-controlnet/checkpoint-*
```

#### Manually specify checkpoint:
Edit `train_temporalnet2.sh`:
```bash
--resume_from_checkpoint="temporalnet2-sdxl-controlnet/checkpoint-2500" \
```

### 4. Loss Not Decreasing

#### Check learning rate:
```bash
# Look for "lr" in logs
grep '"lr":' temporalnet2-sdxl-controlnet/logs/*.log | tail -20
```

#### Possible causes:
- Learning rate too low (increase to 2e-5)
- Learning rate too high (decrease to 5e-6)
- Bad data (check a few samples manually)
- Need more training time (wait longer)

#### Try different learning rate:
Edit `train_temporalnet2.sh`:
```bash
--learning_rate=2e-5 \  # or 5e-6
```

### 5. Validation Takes Too Long / OOM

#### Reduce inference steps:
Edit `train_controlnet_sdxl.py` line 193:
```python
num_inference_steps=10,  # instead of 20
```

#### Skip validation temporarily:
Comment out validation in `train_temporalnet2.sh`:
```bash
# --validation_prompt "$VALIDATION_PROMPT_1" "$VALIDATION_PROMPT_2" "$VALIDATION_PROMPT_3" \
# --validation_prev_image "$VALIDATION_PREV_IMG_1" "$VALIDATION_PREV_IMG_2" "$VALIDATION_PREV_IMG_3" \
# --validation_optical_flow "$VALIDATION_FLOW_1" "$VALIDATION_FLOW_2" "$VALIDATION_FLOW_3" \
# --validation_steps=500 \
# --num_validation_images=1 \
```

### 6. Training is Very Slow

#### Current bottlenecks:
- High resolution images (1024) - reduced frequency in updated code
- Validation - reduced to 1 image per prompt
- Data loading - reduced to 8 workers

#### Speed up options:

1. **Reduce validation frequency:**
```bash
--validation_steps=1000 \  # instead of 500
```

2. **Disable xformers if causing issues:**
```bash
# --enable_xformers_memory_efficient_attention \
```

3. **Use fp16 instead of bf16:**
```bash
--mixed_precision="fp16" \  # instead of bf16
```

### 7. WandB Not Logging

#### Check WandB login:
```bash
wandb login
```

#### Check WandB status:
```bash
wandb status
```

#### Use tensorboard instead:
Edit `train_temporalnet2.sh`:
```bash
--report_to="tensorboard" \  # instead of "wandb"
```

Then view tensorboard:
```bash
tensorboard --logdir=temporalnet2-sdxl-controlnet/logs
```

## Emergency: Kill and Restart Training

```bash
# Kill all training processes
pkill -f train_controlnet_sdxl

# Clear CUDA cache (optional, if stuck)
python3 -c "import torch; torch.cuda.empty_cache()"

# Wait a moment
sleep 5

# Restart training
bash train_temporalnet2.sh
```

## Understanding Training Metrics

### Good Signs:
- Loss decreasing (even if slowly)
- Loss around 0.05-0.15 after 10k steps
- Validation images show some structure after 1k steps
- Validation images look recognizable after 5k steps

### Warning Signs:
- Loss increasing consistently
- Loss stuck at same value for >1000 steps
- Loss becomes NaN or Inf
- Validation images still blank after 5k steps

### What's Normal:
- Loss fluctuates (this is OK)
- Some validation steps fail with OOM (now skipped automatically)
- First ~500 steps produce gray/blank images
- Training takes hours/days for good results

## Optimal Training Settings

Based on your 80GB GPU:

```bash
# Conservative (most stable)
--train_batch_size=2 --gradient_accumulation_steps=16

# Balanced (current setting)
--train_batch_size=4 --gradient_accumulation_steps=8

# Aggressive (if you have free memory)
--train_batch_size=6 --gradient_accumulation_steps=6
```

## Getting Help

### Information to Provide:
1. Current training step
2. GPU model and memory
3. Error message (full stack trace)
4. Output of `nvidia-smi`
5. Last 50 lines of training log
6. Sample validation image (if available)

### Useful commands:
```bash
# Last error in logs
grep -i "error\|exception" temporalnet2-sdxl-controlnet/logs/*.log | tail -20

# Current training progress
grep -i "steps:" temporalnet2-sdxl-controlnet/logs/*.log | tail -1

# GPU usage
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# Checkpoint sizes
du -sh temporalnet2-sdxl-controlnet/checkpoint-*/
```

