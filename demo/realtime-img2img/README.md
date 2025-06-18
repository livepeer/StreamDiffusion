# Img2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Features

- **Standard Mode**: Basic image-to-image generation with SD-Turbo
- **ControlNet Mode**: Enhanced generation with ControlNet support (depth, canny, pose, etc.)
- **Real-time streaming**: WebRTC/WebSocket based streaming for low latency
- **Web interface**: No desktop app required, runs in browser

## Usage
You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

### Standard Mode (Default)
```bash
cd frontend
npm i
npm run build
cd ..
pip install -r requirements.txt
python main.py --acceleration tensorrt   
```

or 

### Quick Start Script
```
chmod +x start.sh
./start.sh
```

Then open `http://0.0.0.0:7860` in your browser.
(*If `http://0.0.0:7860` does not work well, try `http://localhost:7860`)

## ControlNet Configuration

When using ControlNet mode, you can specify:
- **Model**: Base diffusion model (SD1.5, SD-Turbo, SDXL-Turbo)
- **ControlNets**: One or multiple ControlNet models with preprocessors
- **Parameters**: Generation settings, temporal consistency, acceleration options

See the [ControlNet configuration examples](../../configs/controlnet_examples/) for detailed YAML configuration options.

### ControlNet Mode
To use ControlNet, provide a YAML configuration file:

```bash
python main.py --acceleration tensorrt --controlnet-config /path/to/config.yaml
```

### Running with Docker

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

Where `ENGINE_DIR` and `HF_HOME` set a local cache directory, making it faster to restart the docker container.

## Command Line Options

```
--host HOST                    Host address (default: 0.0.0.0)
--port PORT                    Port number (default: 7860)
--controlnet-config PATH       Path to ControlNet YAML configuration (optional)
--acceleration ACCEL           Acceleration type: none, xformers, sfast, tensorrt
--taesd / --no-taesd          Use Tiny Autoencoder (default: enabled)
--engine-dir DIR              TensorRT engine directory
--debug                       Enable debug mode
```
