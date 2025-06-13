# StreamDiffusion Browser Pose Integration

Complete example demonstrating **client-side MediaPipe preprocessing** with **server-side StreamDiffusion generation**.

## Overview

This integration showcases the advantages of browser-based preprocessing:

**Client Side (Browser):**
- MediaPipe pose detection 
- Real-time pose visualization
- Minimal data sent to server

**Server Side (Python):**
- Browser preprocessor (minimal processing)
- StreamDiffusion generation
- ControlNet pose conditioning

## Files

- `browser_pose_server.py` - Flask server with StreamDiffusion integration
- `sdturbo_browser_pose_example.yaml` - Configuration using browser preprocessor
- `requirements_browser_demo.txt` - Python dependencies
- `../../demo/streamdiffusion-browser/` - Browser interface

## Quick Start

### 1. Install Dependencies

```bash
# Install Flask for the demo server
pip install Flask>=2.3.0

# Or install all demo requirements
pip install -r requirements_browser_demo.txt
```

### 2. Update Model Path

Edit the model path in `browser_pose_server.py`:
```python
MODEL_PATH = r"C:\_dev\comfy\ComfyUI\models\checkpoints\sd_turbo.safetensors"
```

### 3. Run the Server

```bash
cd StreamDiffusion/examples/controlnet/
python browser_pose_server.py
```

### 4. Open Browser

Navigate to: `http://localhost:5000`

### 5. Test the Demo

1. Click "Start Camera" and grant permissions
2. Click "Start Generation" 
3. Move around - your pose controls the AI generation!

## How It Works

```
Browser                    Server
┌─────────────────────┐   ┌──────────────────────┐
│ 1. Webcam Input     │   │                      │
│ 2. MediaPipe Pose   │   │ 4. Browser           │
│ 3. Send Images ────────► │    Preprocessor      │
│                     │   │ 5. StreamDiffusion   │
│ 6. Display Result  ◄──── │    Generation        │
└─────────────────────┘   └──────────────────────┘
```

**Step by Step:**
1. **Browser**: Captures webcam input
2. **Browser**: Processes pose with MediaPipe (client-side)
3. **Browser**: Sends camera frame + pose skeleton to server
4. **Server**: Browser preprocessor validates pose (minimal processing)
5. **Server**: StreamDiffusion generates with pose control
6. **Browser**: Displays generated result

## Browser Preprocessor Advantages

### vs Server-side MediaPipe:
- ✅ **No server MediaPipe installation** 
- ✅ **Reduced server CPU/GPU** usage
- ✅ **Lower latency** (no pose analysis round-trip)
- ✅ **Better scaling** (processing distributed to clients)
- ✅ **Mobile support** (works on phones with HTTPS)

### Technical Benefits:
- ✅ **Privacy** - Raw video never leaves client
- ✅ **Real-time feedback** - See pose detection immediately  
- ✅ **Offline pose** - Detection works without internet
- ✅ **Device utilization** - Uses client GPU/WebGL

## Configuration

### Server Options

```bash
# Custom model path
python browser_pose_server.py --model /path/to/your/model.safetensors

# Different port
python browser_pose_server.py --port 8080

# Disable TensorRT (if needed)
python browser_pose_server.py --no-tensorrt
```

### Browser Preprocessor Settings

In the config file (`sdturbo_browser_pose_example.yaml`):

```yaml
controlnets:
  - model_id: "thibaud/controlnet-sd21-openpose-diffusers"
    preprocessor: "browser"  # ← Key: uses browser preprocessor
    conditioning_scale: 0.711
    preprocessor_params:
      image_resolution: 512
      validate_input: true          # Validate pose format
      normalize_brightness: false   # Skip brightness normalization
```

## API Endpoints

The server provides these endpoints:

### `GET /` 
Serves the browser demo interface

### `GET /api/status`
Returns pipeline status:
```json
{"ready": true, "message": "Pipeline ready"}
```

### `POST /api/streamdiffusion/generate`
Generates image with pose control:

**Form Data:**
- `input_image`: Camera frame (JPEG)
- `control_image`: Pose skeleton (JPEG, pre-processed)
- `prompt`: Text prompt (optional)
- `control_strength`: Float 0.0-2.0 (optional)

**Returns:** Generated image (JPEG)

## Extending the Demo

### Add More Control Types

The browser preprocessor can handle any client-processed control:

```javascript
// In browser: send different control types
formData.append('control_type', 'pose');  // or 'hands', 'face', 'depth'
```

```python
# In server: handle different control types
control_type = request.form.get('control_type', 'pose')
```

### Multiple ControlNets

Add multiple ControlNets in the config:

```yaml
controlnets:
  - model_id: "thibaud/controlnet-sd21-openpose-diffusers"
    preprocessor: "browser"
    conditioning_scale: 0.7
  - model_id: "thibaud/controlnet-sd21-depth-diffusers" 
    preprocessor: "browser"
    conditioning_scale: 0.3
```

### WebSocket Streaming

For even lower latency, replace HTTP with WebSockets:

```python
from flask_socketio import SocketIO, emit

# Real-time bidirectional communication
@socketio.on('generate_pose')
def handle_pose_generation(data):
    # Process and emit result
    emit('generation_result', result_data)
```

## Troubleshooting

### "Pipeline not ready"
- Check model path exists
- Ensure ControlNet model downloads
- Wait for TensorRT engine compilation (first run)

### Camera not working
- Use HTTPS for mobile devices
- Check browser permissions
- Try different browsers (Chrome recommended)

### Slow generation
- Reduce image resolution
- Use TensorRT acceleration  
- Lower ControlNet conditioning scale

### Memory issues
- Reduce frame buffer size
- Use smaller models
- Enable gradient checkpointing

## Performance Notes

**First Run:**
- TensorRT engine compilation: 5-10 minutes
- Subsequent runs are much faster

**Typical Performance:**
- Pose detection: ~16ms (client-side)
- StreamDiffusion: ~100-200ms (TensorRT)
- Total latency: ~200-300ms

**Optimization Tips:**
- Use TensorRT acceleration
- Reduce unnecessary preprocessing
- Optimize network payload size
- Cache models in GPU memory

## Integration Guide

To integrate this into your own application:

1. **Use the browser preprocessor** in your ControlNet config
2. **Implement the API endpoints** in your web framework
3. **Add client-side MediaPipe** processing
4. **Handle the image upload/display** flow

The browser preprocessor design makes it easy to add client-side preprocessing for any control type while keeping the server-side code simple and efficient. 