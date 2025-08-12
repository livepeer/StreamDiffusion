// Shared hand tracking service to avoid multiple MediaPipe instances
// MediaPipe types
interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

interface HandResults {
  multiHandLandmarks?: HandLandmark[][];
}

interface HandsConfig {
  locateFile: (file: string) => string;
}

interface HandsOptions {
  maxNumHands: number;
  modelComplexity: number;
  minDetectionConfidence: number;
  minTrackingConfidence: number;
}

interface HandsInstance {
  setOptions(options: HandsOptions): void;
  onResults(callback: (results: HandResults) => void): void;
  send(inputs: { image: HTMLVideoElement }): Promise<void>;
  close(): void;
}

declare const Hands: new (config: HandsConfig) => HandsInstance;

export interface HandTrackingCallback {
  handIndex: number;
  sensitivity: number;
  onValueChange: (value: number) => void;
  onHandsData?: (results: HandResults) => void;
}

class HandTrackingService {
  private hands: HandsInstance | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private mediaStream: MediaStream | null = null;
  private isInitialized: boolean = false;
  private isActive: boolean = false;
  private callbacks: Map<string, HandTrackingCallback> = new Map();
  private processingFrame: boolean = false;
  private lastValues: Map<string, number> = new Map();
  private lastHandsDetected: Map<string, boolean> = new Map();
  private smoothedValues: Map<string, number> = new Map();
  private lastUpdateTime: Map<string, number> = new Map();
  private readonly VALUE_CHANGE_THRESHOLD = 0.003; // Very small threshold
  private readonly SMOOTHING_FACTOR = 0.6; // More responsive smoothing
  private readonly MAX_UPDATE_INTERVAL = 50; // Max 50ms between updates when hands are active

  async loadMediaPipeScript(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (typeof Hands !== 'undefined') {
        resolve();
        return;
      }
      
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load MediaPipe Hands script'));
      document.head.appendChild(script);
    });
  }

  async initialize(): Promise<boolean> {
    if (this.isInitialized) return true;

    try {
      // Load MediaPipe Hands dynamically
      await this.loadMediaPipeScript();
      
      // Initialize MediaPipe Hands
      this.hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      this.hands.setOptions({
        maxNumHands: 4,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      this.hands.onResults(this.onResults.bind(this));

      // Get camera access
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access not supported in this browser');
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user'
        }
      });

      // Create video element
      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = this.mediaStream;
      this.videoElement.autoplay = true;
      this.videoElement.muted = true;
      this.videoElement.playsInline = true;
      
      await new Promise((resolve) => {
        this.videoElement!.onloadedmetadata = resolve;
      });

      this.isInitialized = true;
      return true;
      
    } catch (error) {
      console.error('HandTrackingService: Failed to initialize:', error);
      this.isInitialized = false;
      return false;
    }
  }

  registerCallback(id: string, callback: HandTrackingCallback): void {
    this.callbacks.set(id, callback);
    // Initialize tracking state for this callback
    this.lastValues.set(id, -1); // Use -1 to ensure first valid value triggers callback
    this.lastHandsDetected.set(id, false);
    this.smoothedValues.set(id, 0);
    this.lastUpdateTime.set(id, 0);
    
    // Start processing if this is the first callback and we're initialized
    if (this.callbacks.size === 1 && this.isInitialized && !this.isActive) {
      this.start();
    }
  }

  unregisterCallback(id: string): void {
    this.callbacks.delete(id);
    this.lastValues.delete(id);
    this.lastHandsDetected.delete(id);
    this.smoothedValues.delete(id);
    this.lastUpdateTime.delete(id);
    
    // Stop processing if no more callbacks
    if (this.callbacks.size === 0) {
      this.stop();
    }
  }

  private onResults(results: HandResults): void {
    const currentTime = Date.now();
    
    // Calculate distances for each registered callback
    this.callbacks.forEach((callback, callbackId) => {
      const lastValue = this.lastValues.get(callbackId) ?? -1;
      const lastHandsDetected = this.lastHandsDetected.get(callbackId) ?? false;
      const lastSmoothedValue = this.smoothedValues.get(callbackId) ?? 0;
      const lastUpdateTime = this.lastUpdateTime.get(callbackId) ?? 0;
      
      let rawValue = 0;
      let handsCurrentlyDetected = false;
      
      if (results.multiHandLandmarks && callback.handIndex < results.multiHandLandmarks.length) {
        const landmarks = results.multiHandLandmarks[callback.handIndex];
        handsCurrentlyDetected = true;
        
        // Get thumb tip (landmark 4) and index finger tip (landmark 8)
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        
        if (thumbTip && indexTip) {
          // Calculate distance
          const distance = Math.sqrt(
            Math.pow(thumbTip.x - indexTip.x, 2) + 
            Math.pow(thumbTip.y - indexTip.y, 2)
          );
          
          // Normalize distance (typical max distance is around 0.2-0.3)
          const normalizedDistance = Math.min(1.0, Math.max(0.0, distance / 0.25));
          
          // Apply sensitivity
          rawValue = Math.min(1.0, normalizedDistance * callback.sensitivity);
        }
      }
      
      // Apply exponential smoothing when hands are detected
      let smoothedValue: number;
      if (handsCurrentlyDetected) {
        smoothedValue = (this.SMOOTHING_FACTOR * lastSmoothedValue) + ((1 - this.SMOOTHING_FACTOR) * rawValue);
      } else {
        smoothedValue = 0; // Reset immediately when hands disappear
      }
      
      // Update smoothed value
      this.smoothedValues.set(callbackId, smoothedValue);
      
      // Send value updates when:
      // 1. Hands detection state changes (appeared/disappeared)
      // 2. Hands are detected AND (value changed OR enough time has passed)
      const handsStateChanged = handsCurrentlyDetected !== lastHandsDetected;
      const valueChangedSignificantly = Math.abs(smoothedValue - lastValue) >= this.VALUE_CHANGE_THRESHOLD;
      const timeSinceLastUpdate = currentTime - lastUpdateTime;
      const timeForUpdate = timeSinceLastUpdate >= this.MAX_UPDATE_INTERVAL;
      
      if (handsStateChanged || (handsCurrentlyDetected && (valueChangedSignificantly || timeForUpdate))) {
        callback.onValueChange(smoothedValue);
        this.lastValues.set(callbackId, smoothedValue);
        this.lastUpdateTime.set(callbackId, currentTime);
      }
      
      // Update hands detection state
      this.lastHandsDetected.set(callbackId, handsCurrentlyDetected);

      // Send hands data for visualization if callback wants it
      if (callback.onHandsData) {
        callback.onHandsData(results);
      }
    });
  }

  async start(): Promise<void> {
    if (!this.isInitialized || this.isActive) return;
    
    this.isActive = true;
    
    if (this.videoElement) {
      this.videoElement.play();
    }
    
    this.processFrame();
  }

  stop(): void {
    this.isActive = false;
    
    if (this.videoElement) {
      this.videoElement.pause();
    }
  }

  private async processFrame(): Promise<void> {
    if (!this.hands || !this.videoElement || !this.isActive || this.processingFrame) return;
    
    this.processingFrame = true;
    
    try {
      await this.hands.send({ image: this.videoElement });
    } catch (error) {
      console.error('HandTrackingService: Error processing frame:', error);
    }
    
    this.processingFrame = false;
    
    if (this.isActive) {
      requestAnimationFrame(this.processFrame.bind(this));
    }
  }

  cleanup(): void {
    this.stop();
    
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    if (this.hands) {
      this.hands.close();
      this.hands = null;
    }
    
    if (this.videoElement) {
      this.videoElement.remove();
      this.videoElement = null;
    }
    
    this.callbacks.clear();
    this.lastValues.clear();
    this.lastHandsDetected.clear();
    this.smoothedValues.clear();
    this.lastUpdateTime.clear();
    this.isInitialized = false;
  }

  getVideoElement(): HTMLVideoElement | null {
    return this.videoElement;
  }

  getActiveHandsCount(): number {
    return this.callbacks.size;
  }

  isServiceActive(): boolean {
    return this.isActive;
  }
}

// Singleton instance
export const handTrackingService = new HandTrackingService(); 