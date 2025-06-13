class StreamDiffusionPoseDemo {
    constructor() {
        this.camera = null;
        this.pose = null;
        this.isRunning = false;
        this.isGenerating = false;
        this.generationInterval = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupMediaPipe();
    }

    initializeElements() {
        this.startButton = document.getElementById('startButton');
        this.stopButton = document.getElementById('stopButton');
        this.generateButton = document.getElementById('generateButton');
        this.stopGenButton = document.getElementById('stopGenButton');
        
        this.promptInput = document.getElementById('promptInput');
        this.strengthSlider = document.getElementById('strengthSlider');
        this.strengthValue = document.getElementById('strengthValue');
        
        this.inputVideo = document.getElementById('inputVideo');
        this.poseCanvas = document.getElementById('poseCanvas');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.statusText = document.getElementById('statusText');
        
        this.poseCtx = this.poseCanvas.getContext('2d');
        this.outputCtx = this.outputCanvas.getContext('2d');
        
        // Set canvas dimensions
        this.poseCanvas.width = 512;
        this.poseCanvas.height = 512;
        this.outputCanvas.width = 512;
        this.outputCanvas.height = 512;
    }

    setupEventListeners() {
        this.startButton.addEventListener('click', () => this.startCamera());
        this.stopButton.addEventListener('click', () => this.stopCamera());
        this.generateButton.addEventListener('click', () => this.startGeneration());
        this.stopGenButton.addEventListener('click', () => this.stopGeneration());
        
        this.strengthSlider.addEventListener('input', (e) => {
            this.strengthValue.textContent = e.target.value;
        });
    }

    updateStatus(message) {
        console.log('StreamDiffusionPoseDemo updateStatus:', message);
        this.statusText.textContent = message;
    }

    async setupMediaPipe() {
        try {
            console.log('StreamDiffusionPoseDemo setupMediaPipe: Initializing MediaPipe Pose');
            
            this.pose = new Pose({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`;
                }
            });
            
            this.pose.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: false,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            this.pose.onResults((results) => this.onPoseResults(results));
            await this.pose.initialize();
            
        } catch (error) {
            console.error('StreamDiffusionPoseDemo setupMediaPipe error:', error);
            this.updateStatus('Error initializing MediaPipe: ' + error.message);
        }
    }

    async startCamera() {
        try {
            this.updateStatus('Starting camera...');
            
            const constraints = {
                video: {
                    width: { ideal: 640, max: 640 },
                    height: { ideal: 480, max: 480 },
                    facingMode: { ideal: 'user' }
                },
                audio: false
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.inputVideo.srcObject = stream;
            
            this.camera = new Camera(this.inputVideo, {
                onFrame: async () => {
                    if (this.pose && this.isRunning) {
                        await this.pose.send({image: this.inputVideo});
                    }
                },
                width: 640,
                height: 480
            });
            
            await this.camera.start();
            
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            this.generateButton.disabled = false;
            this.isRunning = true;
            
            this.updateStatus('Camera started - Ready for generation');
            
        } catch (error) {
            console.error('StreamDiffusionPoseDemo startCamera error:', error);
            this.updateStatus('Error starting camera: ' + error.message);
        }
    }

    stopCamera() {
        console.log('StreamDiffusionPoseDemo stopCamera: Stopping camera');
        
        this.stopGeneration();
        
        if (this.inputVideo.srcObject) {
            const tracks = this.inputVideo.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.inputVideo.srcObject = null;
        }
        
        if (this.camera) {
            this.camera.stop();
            this.camera = null;
        }
        
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        this.generateButton.disabled = true;
        this.isRunning = false;
        
        this.poseCtx.clearRect(0, 0, this.poseCanvas.width, this.poseCanvas.height);
        this.outputCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        this.updateStatus('Camera stopped');
    }

    startGeneration() {
        if (!this.isRunning) {
            this.updateStatus('Please start camera first');
            return;
        }
        
        console.log('StreamDiffusionPoseDemo startGeneration: Starting AI generation');
        this.isGenerating = true;
        this.generateButton.disabled = true;
        this.stopGenButton.disabled = false;
        
        this.outputCanvas.classList.add('generating');
        this.updateStatus('AI generation started');
        
        // Start generation loop (every 500ms for demo)
        this.generationInterval = setInterval(() => {
            if (this.isGenerating) {
                this.sendToStreamDiffusion();
            }
        }, 500);
    }

    stopGeneration() {
        console.log('StreamDiffusionPoseDemo stopGeneration: Stopping AI generation');
        this.isGenerating = false;
        this.generateButton.disabled = false;
        this.stopGenButton.disabled = true;
        
        if (this.generationInterval) {
            clearInterval(this.generationInterval);
            this.generationInterval = null;
        }
        
        this.outputCanvas.classList.remove('generating');
        this.updateStatus('AI generation stopped');
    }

    onPoseResults(results) {
        // Clear pose canvas
        this.poseCtx.clearRect(0, 0, this.poseCanvas.width, this.poseCanvas.height);
        
        // Set canvas to match input video dimensions for processing
        this.poseCanvas.width = results.image.width;
        this.poseCanvas.height = results.image.height;
        
        // Create black background for pose
        this.poseCtx.fillStyle = 'black';
        this.poseCtx.fillRect(0, 0, this.poseCanvas.width, this.poseCanvas.height);
        
        // Draw pose skeleton if detected
        if (results.poseLandmarks) {
            this.drawPoseSkeleton(results.poseLandmarks);
        }
        
        // Resize back to display size
        this.resizePoseCanvas();
    }

    drawPoseSkeleton(landmarks) {
        // Draw pose connections
        drawConnectors(this.poseCtx, landmarks, POSE_CONNECTIONS, 
                      {color: '#00FF00', lineWidth: 4});
        
        // Draw pose landmarks
        drawLandmarks(this.poseCtx, landmarks, 
                     {color: '#FF0000', lineWidth: 2});
    }

    resizePoseCanvas() {
        // Create temporary canvas to resize
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 512;
        tempCanvas.height = 512;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw resized pose
        tempCtx.drawImage(this.poseCanvas, 0, 0, 512, 512);
        
        // Update main pose canvas
        this.poseCanvas.width = 512;
        this.poseCanvas.height = 512;
        this.poseCtx.drawImage(tempCanvas, 0, 0);
    }

    async sendToStreamDiffusion() {
        try {
            // Get current frame from video
            const inputCanvas = document.createElement('canvas');
            inputCanvas.width = 512;
            inputCanvas.height = 512;
            const inputCtx = inputCanvas.getContext('2d');
            inputCtx.drawImage(this.inputVideo, 0, 0, 512, 512);
            
            // Convert canvases to blobs
            const inputBlob = await this.canvasToBlob(inputCanvas);
            const poseBlob = await this.canvasToBlob(this.poseCanvas);
            
            // Prepare form data
            const formData = new FormData();
            formData.append('input_image', inputBlob, 'input.jpg');
            formData.append('control_image', poseBlob, 'pose.jpg');
            formData.append('prompt', this.promptInput.value);
            formData.append('control_strength', this.strengthSlider.value);
            formData.append('preprocessor', 'browser');
            
            // Send to StreamDiffusion server
            const response = await fetch('/api/streamdiffusion/generate', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const img = new Image();
                img.onload = () => {
                    this.outputCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
                    this.outputCtx.drawImage(img, 0, 0, 512, 512);
                    URL.revokeObjectURL(img.src);
                };
                img.src = URL.createObjectURL(blob);
            } else {
                console.error('StreamDiffusionPoseDemo sendToStreamDiffusion: Server error:', response.status);
            }
            
        } catch (error) {
            console.error('StreamDiffusionPoseDemo sendToStreamDiffusion error:', error);
            // For demo purposes, show placeholder
            this.showPlaceholderGeneration();
        }
    }

    async canvasToBlob(canvas) {
        return new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
    }

    showPlaceholderGeneration() {
        // Show a placeholder for demo purposes
        this.outputCtx.fillStyle = '#2a2a2a';
        this.outputCtx.fillRect(0, 0, 512, 512);
        
        this.outputCtx.fillStyle = '#ffffff';
        this.outputCtx.font = '24px Arial';
        this.outputCtx.textAlign = 'center';
        this.outputCtx.fillText('StreamDiffusion', 256, 240);
        this.outputCtx.fillText('Server Required', 256, 280);
        
        this.outputCtx.font = '16px Arial';
        this.outputCtx.fillText('Connect to StreamDiffusion API', 256, 320);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('StreamDiffusionPoseDemo: DOM loaded, initializing demo');
    new StreamDiffusionPoseDemo();
}); 