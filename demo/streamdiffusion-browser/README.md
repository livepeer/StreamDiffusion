# StreamDiffusion Browser Demo

A pose-controlled AI generation demo showcasing **client-side MediaPipe processing** with **server-side StreamDiffusion**.

## Overview

This demo demonstrates the advantages of client-side preprocessing:
- **Reduced server load** - No MediaPipe processing on server
- **Lower latency** - Pose detection happens locally 
- **Better performance** - Utilizes client device capabilities
- **Real-time streaming** - Continuous pose control

## Architecture

```
Client Browser          Server
┌─────────────────┐    ┌──────────────────┐
│ Webcam Input    │    │                  │
│ ↓               │    │ StreamDiffusion  │
│ MediaPipe Pose  │────┤ + Browser        │
│ ↓               │    │   Preprocessor   │
│ Pose Skeleton   │    │ ↓                │
└─────────────────┘    │ Generated Image  │
                       └──────────────────┘
```

## Features

- **Client-side pose detection** using MediaPipe
- **Real-time pose visualization** with skeleton overlay  
- **Customizable prompts** and control strength
- **Three-panel interface**: Input → Pose → Generated
- **Browser preprocessor** integration

## How to Run

### Frontend Demo
1. Navigate to the `streamdiffusion-browser` directory
2. Start HTTP server: `python -m http.server 8080`
3. Open browser to `http://localhost:8080`
4. Click "Start Camera" and grant permissions
5. Click "Start Generation" (shows placeholder without server)

### With StreamDiffusion Server
You'll need to integrate this with a StreamDiffusion server that:

1. **Uses the browser preprocessor**:
   ```python
   controlnet_config = {
       'model_id': 'lllyasviel/control_v11p_sd15_openpose',
       'preprocessor': 'browser',  # ← Our new preprocessor
       'conditioning_scale': 1.0
   }
   ```

2. **Accepts the API format**:
   ```
   POST /api/streamdiffusion/generate
   - input_image: Camera frame
   - control_image: Pose skeleton (pre-processed!)  
   - prompt: Text prompt
   - control_strength: Float value
   - preprocessor: 'browser'
   ```

3. **Returns generated images** as JPEG blobs

## Browser Preprocessor

The new `browser` preprocessor (`StreamDiffusion/src/streamdiffusion/controlnet/preprocessors/browser.py`):

- **Validates** client-processed control images
- **Resizes** to target resolution
- **Minimal processing** - trusts client preprocessing
- **Demonstrates advantages** of client-side work

## Advantages Demonstrated

### vs Server-side MediaPipe:
- ✅ **No server MediaPipe installation** required
- ✅ **Reduced server CPU/GPU** usage  
- ✅ **Lower network latency** (no image analysis round-trip)
- ✅ **Better scaling** (processing distributed to clients)
- ✅ **Mobile compatibility** (works on phones with HTTPS)

### Client-side Benefits:
- ✅ **Real-time feedback** - See pose detection immediately
- ✅ **Privacy** - No raw video sent to server
- ✅ **Offline capable** - Pose detection works without internet
- ✅ **Device utilization** - Uses client GPU/CPU efficiently

## Integration Guide

To integrate with your StreamDiffusion setup:

1. **Use the browser preprocessor**:
   ```python
   from streamdiffusion.controlnet.preprocessors import BrowserPreprocessor
   
   preprocessor = BrowserPreprocessor()
   control_image = preprocessor.process(client_pose_image)
   ```

2. **Create API endpoint** that accepts the format shown above

3. **Configure ControlNet** with browser preprocessor:
   ```python
   wrapper = StreamDiffusionWrapper(
       model_id_or_path="your-model",
       use_controlnet=True,
       controlnet_config={
           'model_id': 'lllyasviel/control_v11p_sd15_openpose',
           'preprocessor': 'browser',
           'conditioning_scale': 1.0
       }
   )
   ```

## Files

- `index.html` - Main interface with three-panel layout
- `style.css` - Responsive styling with generation indicators  
- `script.js` - MediaPipe integration + StreamDiffusion API calls
- `README.md` - This documentation

## Future Enhancements

- Support for other control types (hands, face, depth)
- WebSocket streaming for even lower latency
- Client-side image preprocessing (filters, effects)
- Progressive enhancement for different device capabilities

## Technical Details

- **MediaPipe**: Client-side pose detection
- **Canvas API**: Image processing and pose rendering
- **Fetch API**: Server communication
- **WebRTC**: Camera access
- **Responsive**: Works on desktop and mobile (with HTTPS) 