# MediaPipe Demo

A simple web application demonstrating real-time MediaPipe capabilities including pose detection, hand tracking, face mesh, and selfie segmentation.

## Features

- **Pose Detection**: Real-time pose landmark detection with skeleton overlay
- **Hand Tracking**: Multi-hand detection and tracking with landmarks
- **Face Mesh**: Detailed facial landmark detection
- **Selfie Segmentation**: Person segmentation with background replacement
- Side-by-side comparison of input and processed output
- Works on desktop and mobile browsers

## How to Run

### Desktop Testing
1. Navigate to the `mediapipe` directory in terminal/command prompt
2. Start HTTP server: `python -m http.server 8080`
3. Open browser to `http://localhost:8080`
4. Click "Start Camera" and grant camera permissions
5. Use dropdown to switch between detection modes

### Mobile Testing (Requires HTTPS)
1. Start HTTP server: `python -m http.server 8080`
2. Create HTTPS tunnel: `npx localtunnel --port 8080`
3. Use the provided HTTPS URL on your mobile device
4. Follow password instructions on the tunnel page if prompted

## Requirements

- Python 3.x (for HTTP server)
- Modern web browser with camera support
- Internet connection (for MediaPipe models)
- HTTPS required for mobile camera access

## Technical Details

- Client-side MediaPipe JavaScript processing
- Compatible MediaPipe versions pinned for stability
- WebRTC camera access with mobile-optimized constraints 