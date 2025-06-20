<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  export let videoElement: HTMLVideoElement | null = null;
  export let width = 512;
  export let height = 512;
  export let enabled = true;
  export let showOverlay = true;
  
  // Pose detection results callback
  export let onPoseResults: (poseImage: Blob | null, landmarks: any) => void = () => {};

  let pose: any = null;
  let poseCanvas: HTMLCanvasElement;
  let poseCtx: CanvasRenderingContext2D;
  let overlayCanvas: HTMLCanvasElement;
  let overlayCtx: CanvasRenderingContext2D;
  let isInitialized = false;

  // Pose rendering configuration
  const POSE_CONFIG = {
    pointRadius: 3,
    connectionLineWidth: 2,
    pointColor: '#00FF00',
    connectionColor: '#00FF00',
    backgroundColor: '#000000'
  };

  onMount(async () => {
    await initializePose();
  });

  onDestroy(() => {
    if (pose) {
      pose.close();
    }
  });

  async function initializePose() {
    try {
      // @ts-ignore - MediaPipe loaded via CDN
      pose = new window.Pose({
        locateFile: (file: string) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
      });

      pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      pose.onResults(onResults);
      
      // Initialize canvases
      if (poseCanvas) {
        poseCtx = poseCanvas.getContext('2d') as CanvasRenderingContext2D;
        poseCanvas.width = width;
        poseCanvas.height = height;
      }
      
      if (overlayCanvas) {
        overlayCtx = overlayCanvas.getContext('2d') as CanvasRenderingContext2D;
        overlayCanvas.width = width;
        overlayCanvas.height = height;
      }

      isInitialized = true;
      console.log('PoseDetector: MediaPipe Pose initialized successfully');
    } catch (error) {
      console.error('PoseDetector: Failed to initialize MediaPipe Pose:', error);
    }
  }

  async function onResults(results: any) {
    if (!poseCtx || !overlayCtx) return;

    // Clear canvases
    poseCtx.fillStyle = POSE_CONFIG.backgroundColor;
    poseCtx.fillRect(0, 0, width, height);
    overlayCtx.clearRect(0, 0, width, height);

    if (results.poseLandmarks) {
      // Draw pose on black background for ControlNet
      drawPose(poseCtx, results.poseLandmarks, results.poseWorldLandmarks);
      
      // Draw overlay on transparent canvas for visualization
      if (showOverlay) {
        drawPose(overlayCtx, results.poseLandmarks, results.poseWorldLandmarks);
      }

      // Convert pose canvas to blob for server
      try {
        const poseBlob = await new Promise<Blob>((resolve) => {
          poseCanvas.toBlob(
            (blob) => {
              resolve(blob as Blob);
            },
            'image/jpeg',
            0.9
          );
        });
        
        onPoseResults(poseBlob, results.poseLandmarks);
      } catch (error) {
        console.error('PoseDetector: Failed to create pose blob:', error);
        onPoseResults(null, results.poseLandmarks);
      }
    } else {
      onPoseResults(null, null);
    }
  }

  function drawPose(ctx: CanvasRenderingContext2D, landmarks: any, worldLandmarks?: any) {
    // Draw connections
    ctx.strokeStyle = POSE_CONFIG.connectionColor;
    ctx.lineWidth = POSE_CONFIG.connectionLineWidth;
    
    // @ts-ignore - MediaPipe loaded via CDN  
    for (const connection of window.POSE_CONNECTIONS) {
      const startIdx = connection[0];
      const endIdx = connection[1];
      
      const startLandmark = landmarks[startIdx];
      const endLandmark = landmarks[endIdx];
      
      if (startLandmark && endLandmark && 
          startLandmark.visibility > 0.5 && endLandmark.visibility > 0.5) {
        ctx.beginPath();
        ctx.moveTo(startLandmark.x * width, startLandmark.y * height);
        ctx.lineTo(endLandmark.x * width, endLandmark.y * height);
        ctx.stroke();
      }
    }

    // Draw landmarks
    ctx.fillStyle = POSE_CONFIG.pointColor;
    
    for (const landmark of landmarks) {
      if (landmark.visibility > 0.5) {
        ctx.beginPath();
        ctx.arc(
          landmark.x * width,
          landmark.y * height,
          POSE_CONFIG.pointRadius,
          0,
          2 * Math.PI
        );
        ctx.fill();
      }
    }
  }

  export async function processFrame(imageSource: HTMLVideoElement | HTMLCanvasElement) {
    if (!pose || !enabled || !isInitialized) {
      onPoseResults(null, null);
      return;
    }

    try {
      await pose.send({ image: imageSource });
    } catch (error) {
      console.error('PoseDetector: Error processing frame:', error);
      onPoseResults(null, null);
    }
  }

  // Reactive updates
  $: if (poseCanvas && poseCtx) {
    poseCanvas.width = width;
    poseCanvas.height = height;
  }
  
  $: if (overlayCanvas && overlayCtx) {
    overlayCanvas.width = width;
    overlayCanvas.height = height;
  }
</script>

<!-- Hidden canvas for pose rendering (black background for ControlNet) -->
<canvas
  bind:this={poseCanvas}
  style="display: none;"
  width={width}
  height={height}
></canvas>

<!-- Overlay canvas for visualization (transparent background) -->
{#if showOverlay}
  <canvas
    bind:this={overlayCanvas}
    class="absolute left-0 top-0 aspect-square w-full object-cover pointer-events-none"
    style="z-index: 15;"
    width={width}
    height={height}
  ></canvas>
{/if} 