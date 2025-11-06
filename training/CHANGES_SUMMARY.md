# Training Script Fixes - Summary

## Issues Addressed

### 1. CUDA Out of Memory (OOM) Error
**Problem:** Training was failing with CUDA OOM errors during forward pass.

**Solutions Implemented:**

#### Memory Optimization Changes:
1. **Reduced Batch Size:** Changed from `train_batch_size=8` to `train_batch_size=4`
2. **Increased Gradient Accumulation:** Changed from `gradient_accumulation_steps=4` to `gradient_accumulation_steps=8`
   - This keeps the same effective batch size (32) but uses less memory per step
3. **Environment Variables:** Added memory optimization flags:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export CUDA_LAUNCH_BLOCKING=0
   ```
4. **Multi-Resolution Training Adjusted:** Changed resolution sampling probabilities to favor lower resolutions:
   - Before: `[0.40, 0.15, 0.25, 0.05, 0.15]` for `[512, 640, 768, 896, 1024]`
   - After: `[0.60, 0.20, 0.15, 0.04, 0.01]` for `[512, 640, 768, 896, 1024]`
   - This means 60% of batches use 512x512 resolution (vs 40% before)
5. **Reduced Dataloader Workers:** Changed from `dataloader_num_workers=16` to `dataloader_num_workers=8`
6. **Reduced Validation Images:** Changed from `num_validation_images=3` to `num_validation_images=1` per prompt
7. **VAE Optimizations:** Added VAE slicing and tiling in validation:
   ```python
   pipeline.vae.enable_slicing()
   pipeline.vae.enable_tiling()
   ```
8. **Better Memory Cleanup:** Improved CUDA cache clearing and garbage collection

#### Error Handling:
- Added try-catch around validation to prevent OOM during validation from crashing training
- Training will skip validation if OOM occurs and continue

### 2. Automatic Checkpoint Resumption
**Added:** `--resume_from_checkpoint="latest"` flag to automatically resume from the most recent checkpoint.

### 3. Blank Validation Images

**Potential Causes & Fixes:**

1. **Early Training (Normal):** Blank or mostly gray images are **normal in early training** (first few hundred steps). The model hasn't learned meaningful patterns yet.

2. **ControlNet Conditioning:** Improved validation to ensure 6-channel input is properly handled:
   - Added explicit dtype conversion: `.to(weight_dtype)`
   - Added `controlnet_conditioning_scale=1.0` parameter
   - Added `guidance_scale=7.5` for better generation

3. **Better Logging:** Added detailed logging during validation:
   - Logs image shape, dtype, and value range
   - Logs prompt and generation status
   - Error handling with fallback to red image on failure

4. **Image Format:** Ensured validation images are in correct format `[0, 1]` range matching training

## What to Expect

### Training Progress:
- **Steps 0-500:** Likely to see blank/gray/noisy images (this is NORMAL)
- **Steps 500-2000:** Should start seeing some structure/colors
- **Steps 2000-5000:** Should see recognizable but low-quality images
- **Steps 5000+:** Quality should progressively improve

### Memory Usage:
- The training should now use less peak memory
- If you still get OOM, you can:
  1. Reduce `train_batch_size` to 2 (and increase `gradient_accumulation_steps` to 16)
  2. Remove the 1024 resolution entirely from `collate_fn`
  3. Set resolution to 512 only (no multi-resolution training)

### Validation:
- Validation now happens every 500 steps with 1 image per prompt (3 prompts = 3 images total)
- If validation OOMs, it will be skipped and training continues
- Check the logs for detailed information about each validation run

## Files Modified

1. **`train_temporalnet2.sh`**
   - Memory optimization environment variables
   - Reduced batch size, increased gradient accumulation
   - Reduced dataloader workers
   - Reduced validation images
   - Added automatic checkpoint resumption

2. **`train_controlnet_sdxl.py`**
   - Improved validation image handling for 6-channel input
   - Added VAE slicing and tiling
   - Better memory cleanup
   - OOM error handling during validation
   - Detailed logging during validation
   - Adjusted multi-resolution training probabilities
   - Better CUDA cache management

## How to Run

Simply execute the training script:

```bash
cd /workspace/StreamDiffusion/training
bash train_temporalnet2.sh
```

The script will automatically:
- Resume from the latest checkpoint if available
- Use memory-optimized settings
- Handle OOM errors during validation gracefully
- Log detailed information about validation images

## Monitoring Tips

1. **Watch the logs for:**
   - "Validation image shape" - should be `[1, 6, 512, 512]` or similar
   - "Successfully generated validation image" - confirms image generation worked
   - Loss values - should decrease over time (though may be noisy)

2. **Check WandB for:**
   - Loss curve trending downward
   - Validation images improving over time
   - Learning rate schedule

3. **If you see:**
   - Red images in validation → Generation failed, check logs for errors
   - All gray images in early training → Normal, keep training
   - OOM warnings → Training skipped validation but continues, this is OK

## Further Memory Reduction (If Needed)

If you still encounter OOM errors, edit `train_temporalnet2.sh` and change:

```bash
--train_batch_size=2 \
--gradient_accumulation_steps=16 \
```

Or in `train_controlnet_sdxl.py`, line 821, change to only use 512 resolution:

```python
resolutions = [512]
resolution_probs = [1.0]
```

