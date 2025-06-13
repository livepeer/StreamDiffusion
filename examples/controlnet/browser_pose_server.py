#!/usr/bin/env python3
"""
StreamDiffusion Browser Pose Server

Simple Flask server that demonstrates:
- Browser preprocessor integration
- Client-side MediaPipe pose processing
- Real-time pose-controlled generation
- Minimal server-side processing

Usage:
    python browser_pose_server.py
    
Then open browser demo at: http://localhost:5000
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional
import io
import argparse

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from PIL import Image
from flask import Flask, request, send_file, jsonify, send_from_directory
from utils.wrapper import StreamDiffusionWrapper

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Server configuration"""
    
    # Model paths - EDIT THESE FOR YOUR SETUP
    MODEL_PATH = r"C:\_dev\comfy\ComfyUI\models\checkpoints\sd_turbo.safetensors"
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    
    # Default generation settings
    DEFAULT_PROMPT = "anime style character, vibrant colors, masterpiece"
    DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, flat, 2d"
    DEFAULT_GUIDANCE_SCALE = 1.041
    DEFAULT_CONTROL_STRENGTH = 1.0
    
    # Pipeline settings
    RESOLUTION = 512
    FRAME_BUFFER_SIZE = 1
    DELTA = 0.7
    T_INDEX_LIST = [24, 27]
    
    # Performance settings
    ACCELERATION = "tensorrt"
    USE_LCM_LORA = True
    USE_TINY_VAE = True
    CFG_TYPE = "self"
    SEED = 789
    
    # Browser preprocessor ControlNet config
    CONTROLNET_CONFIG = {
        "model_id": "thibaud/controlnet-sd21-openpose-diffusers",
        "preprocessor": "browser",  # Our new browser preprocessor
        "conditioning_scale": 0.711,
        "enabled": True,
        "preprocessor_params": {
            "image_resolution": 512,
            "validate_input": True,
            "normalize_brightness": False
        },
        "pipeline_type": "sdturbo",
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
    }


# ============================================================================
# STREAMDIFFUSION PIPELINE
# ============================================================================

class BrowserPoseStreamDiffusion:
    """StreamDiffusion pipeline optimized for browser pose input"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.wrapper = None
        self.is_ready = False
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize StreamDiffusion with browser preprocessor"""
        print("Initializing StreamDiffusion with Browser Preprocessor...")
        print(f"Model: {self.config.MODEL_PATH}")
        print(f"Acceleration: {self.config.ACCELERATION}")
        print(f"ControlNet: {self.config.CONTROLNET_CONFIG['model_id']}")
        print(f"Preprocessor: {self.config.CONTROLNET_CONFIG['preprocessor']} (client-side processing)")
        
        try:
            # Initialize wrapper with browser preprocessor
            self.wrapper = StreamDiffusionWrapper(
                model_id_or_path=self.config.MODEL_PATH,
                t_index_list=self.config.T_INDEX_LIST,
                mode="img2img",
                output_type="pil",
                dtype=torch.float16,
                frame_buffer_size=self.config.FRAME_BUFFER_SIZE,
                width=self.config.RESOLUTION,
                height=self.config.RESOLUTION,
                warmup=10,
                acceleration=self.config.ACCELERATION,
                do_add_noise=True,
                use_lcm_lora=self.config.USE_LCM_LORA,
                use_tiny_vae=self.config.USE_TINY_VAE,
                use_denoising_batch=True,
                cfg_type=self.config.CFG_TYPE,
                seed=self.config.SEED,
                use_safety_checker=False,
                use_controlnet=True,
                controlnet_config=self.config.CONTROLNET_CONFIG,
            )
            
            # Prepare pipeline with default settings
            self.wrapper.prepare(
                prompt=self.config.DEFAULT_PROMPT,
                negative_prompt=self.config.DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=50,
                guidance_scale=self.config.DEFAULT_GUIDANCE_SCALE,
                delta=self.config.DELTA,
            )
            
            print("✓ Pipeline ready for browser pose input!")
            
            # Check acceleration status
            if hasattr(self.wrapper.stream, 'unet') and hasattr(self.wrapper.stream.unet, 'engine'):
                print("✓ TensorRT acceleration active")
            else:
                print("⚠ Running in PyTorch mode")
            
            self.is_ready = True
            
        except Exception as e:
            print(f"ERROR initializing pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.is_ready = False
    
    def generate(self, input_image: Image.Image, control_image: Image.Image, 
                prompt: Optional[str] = None, control_strength: float = 1.0) -> Image.Image:
        """
        Generate image with pose control
        
        Args:
            input_image: Camera frame from browser
            control_image: Pre-processed pose skeleton from browser MediaPipe
            prompt: Text prompt (optional)
            control_strength: ControlNet strength
            
        Returns:
            Generated image
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready")
        
        # Update prompt if provided
        if prompt and prompt.strip():
            self.wrapper.stream.update_prompt(prompt.strip())
        
        # Update control strength if different
        if abs(control_strength - self.config.DEFAULT_CONTROL_STRENGTH) > 0.01:
            self.wrapper.update_controlnet_scale(0, control_strength)
        
        # Update control image (browser-preprocessed pose)
        # The browser preprocessor will validate and resize as needed
        self.wrapper.update_control_image_efficient(control_image)
        
        # Generate with input image
        return self.wrapper(input_image)


# ============================================================================
# FLASK SERVER
# ============================================================================

app = Flask(__name__)
pipeline = None

@app.route('/')
def index():
    """Serve the browser demo"""
    demo_dir = Path(__file__).parent.parent.parent / "demo" / "streamdiffusion-browser"
    return send_from_directory(demo_dir, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files from the demo directory"""
    demo_dir = Path(__file__).parent.parent.parent / "demo" / "streamdiffusion-browser"
    return send_from_directory(demo_dir, filename)

@app.route('/api/status')
def status():
    """Check pipeline status"""
    global pipeline
    return jsonify({
        'ready': pipeline is not None and pipeline.is_ready,
        'message': 'Pipeline ready' if pipeline and pipeline.is_ready else 'Pipeline not ready'
    })

@app.route('/api/streamdiffusion/generate', methods=['POST'])
def generate():
    """Generate image with browser-processed pose control"""
    global pipeline
    
    if not pipeline or not pipeline.is_ready:
        return jsonify({'error': 'Pipeline not ready'}), 503
    
    try:
        # Parse form data
        if 'input_image' not in request.files or 'control_image' not in request.files:
            return jsonify({'error': 'Missing input_image or control_image'}), 400
        
        # Load images
        input_file = request.files['input_image']
        control_file = request.files['control_image']
        
        input_image = Image.open(input_file.stream).convert('RGB')
        control_image = Image.open(control_file.stream).convert('RGB')
        
        # Get parameters
        prompt = request.form.get('prompt', '').strip()
        try:
            control_strength = float(request.form.get('control_strength', '1.0'))
        except ValueError:
            control_strength = 1.0
        
        print(f"Generating with prompt: '{prompt}', strength: {control_strength}")
        
        # Generate image
        start_time = time.time()
        result_image = pipeline.generate(
            input_image=input_image,
            control_image=control_image,
            prompt=prompt if prompt else None,
            control_strength=control_strength
        )
        inference_time = time.time() - start_time
        
        print(f"Generation completed in {inference_time:.2f}s")
        
        # Return image
        img_io = io.BytesIO()
        result_image.save(img_io, format='JPEG', quality=90)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StreamDiffusion Browser Pose Server")
    parser.add_argument("--model", help="Path to model file", default=Config.MODEL_PATH)
    parser.add_argument("--port", type=int, help="Server port", default=Config.PORT)
    parser.add_argument("--host", help="Server host", default=Config.HOST)
    parser.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT acceleration")
    args = parser.parse_args()
    
    # Update config from args
    config = Config()
    config.MODEL_PATH = args.model
    config.PORT = args.port
    config.HOST = args.host
    
    if args.no_tensorrt:
        config.ACCELERATION = "none"
    
    print("=" * 70)
    print("StreamDiffusion Browser Pose Server")
    print("=" * 70)
    print(f"Model: {config.MODEL_PATH}")
    print(f"ControlNet: {config.CONTROLNET_CONFIG['model_id']}")
    print(f"Preprocessor: {config.CONTROLNET_CONFIG['preprocessor']} (client-side)")
    print(f"Acceleration: {config.ACCELERATION}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print("=" * 70)
    
    # Validate model path
    if not os.path.exists(config.MODEL_PATH):
        print(f"ERROR: Model not found at {config.MODEL_PATH}")
        print("Please update the MODEL_PATH or use --model argument")
        return
    
    # Initialize pipeline
    print("Initializing pipeline...")
    global pipeline
    pipeline = BrowserPoseStreamDiffusion(config)
    
    if not pipeline.is_ready:
        print("ERROR: Pipeline initialization failed")
        return
    
    print("\n✓ Server ready!")
    print(f"Open browser to: http://localhost:{config.PORT}")
    print("\nThe browser will:")
    print("1. Process pose detection client-side with MediaPipe")
    print("2. Send preprocessed pose skeleton to server")  
    print("3. Server uses 'browser' preprocessor (minimal processing)")
    print("4. Generate pose-controlled images with StreamDiffusion")
    print("\nThis demonstrates client-side preprocessing advantages!")
    
    # Start Flask server
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )


if __name__ == "__main__":
    main() 