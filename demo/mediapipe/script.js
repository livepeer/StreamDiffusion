class MediaPipeDemo {
    constructor() {
        this.camera = null;
        this.currentModel = null;
        this.currentMode = 'pose';
        this.isRunning = false;
        
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.startButton = document.getElementById('startButton');
        this.stopButton = document.getElementById('stopButton');
        this.detectionMode = document.getElementById('detectionMode');
        this.inputVideo = document.getElementById('inputVideo');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.statusText = document.getElementById('statusText');
        
        this.canvasCtx = this.outputCanvas.getContext('2d');
    }

    setupEventListeners() {
        this.startButton.addEventListener('click', () => this.startCamera());
        this.stopButton.addEventListener('click', () => this.stopCamera());
        this.detectionMode.addEventListener('change', (e) => this.switchMode(e.target.value));
    }

    updateStatus(message) {
        console.log('MediaPipeDemo updateStatus:', message);
        this.statusText.textContent = message;
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
            
            await this.initializeMediaPipe();
            
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            this.isRunning = true;
            
            this.updateStatus('Camera started successfully');
            
        } catch (error) {
            console.error('MediaPipeDemo startCamera error:', error);
            this.updateStatus('Error starting camera: ' + error.message);
        }
    }

    stopCamera() {
        console.log('MediaPipeDemo stopCamera: Stopping camera');
        
        if (this.inputVideo.srcObject) {
            const tracks = this.inputVideo.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.inputVideo.srcObject = null;
        }
        
        if (this.camera) {
            this.camera.stop();
            this.camera = null;
        }
        
        if (this.currentModel) {
            this.currentModel.close();
            this.currentModel = null;
        }
        
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        this.isRunning = false;
        
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        this.updateStatus('Camera stopped');
    }

    async switchMode(mode) {
        console.log('MediaPipeDemo switchMode:', mode);
        this.currentMode = mode;
        
        if (this.isRunning) {
            this.updateStatus('Switching to ' + mode + ' mode...');
            await this.initializeMediaPipe();
        }
    }

    async initializeMediaPipe() {
        try {
            console.log('MediaPipeDemo initializeMediaPipe: Initializing', this.currentMode);
            
            if (this.currentModel) {
                this.currentModel.close();
            }
            
            if (this.camera) {
                this.camera.stop();
            }
            
            switch (this.currentMode) {
                case 'pose':
                    await this.initializePose();
                    break;
                case 'hands':
                    await this.initializeHands();
                    break;
                case 'face':
                    await this.initializeFaceMesh();
                    break;
                case 'segmentation':
                    await this.initializeSegmentation();
                    break;
                default:
                    throw new Error('Unknown mode: ' + this.currentMode);
            }
            
            this.camera = new Camera(this.inputVideo, {
                onFrame: async () => {
                    if (this.currentModel && this.isRunning) {
                        await this.currentModel.send({image: this.inputVideo});
                    }
                },
                width: 640,
                height: 480
            });
            
            await this.camera.start();
            
        } catch (error) {
            console.error('MediaPipeDemo initializeMediaPipe error:', error);
            let errorMessage = 'Error initializing MediaPipe';
            
            if (error.message.includes('Module.arguments')) {
                errorMessage = 'MediaPipe compatibility issue. Try refreshing the page or use a different browser.';
            } else if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Failed to load MediaPipe models. Check your internet connection.';
            } else {
                errorMessage = 'Error initializing ' + this.currentMode + ' detection: ' + error.message;
            }
            
            this.updateStatus(errorMessage);
        }
    }

    async initializePose() {
        console.log('MediaPipeDemo initializePose: Setting up pose detection');
        
        this.currentModel = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`;
            }
        });
        
        this.currentModel.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        this.currentModel.onResults((results) => this.onPoseResults(results));
        await this.currentModel.initialize();
    }

    async initializeHands() {
        console.log('MediaPipeDemo initializeHands: Setting up hand detection');
        
        this.currentModel = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`;
            }
        });
        
        this.currentModel.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        this.currentModel.onResults((results) => this.onHandsResults(results));
        await this.currentModel.initialize();
    }

    async initializeFaceMesh() {
        console.log('MediaPipeDemo initializeFaceMesh: Setting up face mesh');
        
        this.currentModel = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`;
            }
        });
        
        this.currentModel.setOptions({
            maxNumFaces: 1,
            refineLandmarks: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        this.currentModel.onResults((results) => this.onFaceMeshResults(results));
        await this.currentModel.initialize();
    }

    async initializeSegmentation() {
        console.log('MediaPipeDemo initializeSegmentation: Setting up selfie segmentation');
        
        this.currentModel = new SelfieSegmentation({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@0.1/${file}`;
            }
        });
        
        this.currentModel.setOptions({
            modelSelection: 0,
            selfieMode: true,
        });
        
        this.currentModel.onResults((results) => this.onSegmentationResults(results));
        await this.currentModel.initialize();
    }

    onPoseResults(results) {
        this.outputCanvas.width = results.image.width;
        this.outputCanvas.height = results.image.height;
        
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        if (results.poseLandmarks) {
            drawConnectors(this.canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, 
                          {color: '#00FF00', lineWidth: 4});
            drawLandmarks(this.canvasCtx, results.poseLandmarks, 
                         {color: '#FF0000', lineWidth: 2});
        }
        
        this.canvasCtx.restore();
    }

    onHandsResults(results) {
        this.outputCanvas.width = results.image.width;
        this.outputCanvas.height = results.image.height;
        
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        if (results.multiHandLandmarks) {
            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(this.canvasCtx, landmarks, HAND_CONNECTIONS, 
                              {color: '#00CC00', lineWidth: 5});
                drawLandmarks(this.canvasCtx, landmarks, 
                             {color: '#FF0000', lineWidth: 2});
            }
        }
        
        this.canvasCtx.restore();
    }

    onFaceMeshResults(results) {
        this.outputCanvas.width = results.image.width;
        this.outputCanvas.height = results.image.height;
        
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        if (results.multiFaceLandmarks) {
            for (const landmarks of results.multiFaceLandmarks) {
                try {
                    drawConnectors(this.canvasCtx, landmarks, FACEMESH_TESSELATION, 
                                  {color: '#C0C0C070', lineWidth: 1});
                    drawConnectors(this.canvasCtx, landmarks, FACEMESH_RIGHT_EYE, 
                                  {color: '#FF3030'});
                    drawConnectors(this.canvasCtx, landmarks, FACEMESH_LEFT_EYE, 
                                  {color: '#30FF30'});
                    drawConnectors(this.canvasCtx, landmarks, FACEMESH_LIPS, 
                                  {color: '#E0E0E0'});
                } catch (error) {
                    console.log('MediaPipeDemo onFaceMeshResults: Drawing fallback landmarks');
                    drawLandmarks(this.canvasCtx, landmarks, {color: '#00FF00', lineWidth: 2});
                }
            }
        }
        
        this.canvasCtx.restore();
    }

    onSegmentationResults(results) {
        this.outputCanvas.width = results.image.width;
        this.outputCanvas.height = results.image.height;
        
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        this.canvasCtx.globalCompositeOperation = 'source-over';
        this.canvasCtx.fillStyle = '#00FF00';
        this.canvasCtx.fillRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        this.canvasCtx.globalCompositeOperation = 'source-in';
        this.canvasCtx.drawImage(results.segmentationMask, 0, 0, 
                                this.outputCanvas.width, this.outputCanvas.height);
        
        this.canvasCtx.globalCompositeOperation = 'destination-atop';
        this.canvasCtx.drawImage(results.image, 0, 0, 
                                this.outputCanvas.width, this.outputCanvas.height);
        
        this.canvasCtx.restore();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('MediaPipeDemo: DOM loaded, initializing demo');
    new MediaPipeDemo();
}); 