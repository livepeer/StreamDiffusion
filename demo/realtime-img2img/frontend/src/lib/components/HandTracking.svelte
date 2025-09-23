<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { handTrackingService, type HandTrackingCallback } from '$lib/handTrackingService';

  export let isActive: boolean = false;
  export let onValueChange: (value: number) => void = () => {};
  export let sensitivity: number = 1.0;
  export let handIndex: number = 0;
  export let showVisualizer: boolean = true;

  let canvasElement: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let controlId: string;
  
  let currentDistance: number = 0;
  let handTrackingError: string = '';
  let lastHandsData: any = null;
  
  onMount(() => {
    if (canvasElement) {
      ctx = canvasElement.getContext('2d')!;
    }
    controlId = `hand_${Date.now()}_${Math.random()}`;
  });

  onDestroy(() => {
    cleanup();
  });

  async function initializeHandTracking(): Promise<boolean> {
    try {
      handTrackingError = '';
      
      const initialized = await handTrackingService.initialize();
      if (!initialized) {
        throw new Error('Failed to initialize shared hand tracking service');
      }

      return true;
      
    } catch (error) {
      console.error('initializeHandTracking: Failed to initialize hand tracking:', error);
      handTrackingError = error instanceof Error ? error.message : 'Failed to initialize hand tracking';
      return false;
    }
  }

  function onHandsData(results: any) {
    lastHandsData = results;
    
    if (!showVisualizer || !ctx || !canvasElement) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw video frame if available
    const videoElement = handTrackingService.getVideoElement();
    if (videoElement) {
      ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    }

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      // Draw all hands for visual feedback
      results.multiHandLandmarks.forEach((landmarks: any, index: number) => {
        const isTrackedHand = index === handIndex;
        drawLandmarks(landmarks, index, isTrackedHand);
      });
    }
  }

  function drawLandmarks(landmarks: any[], index: number, isTrackedHand: boolean) {
    if (!ctx) return;
    
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    
    if (isTrackedHand) {
      // Draw tracked hand with bright colors
      // Draw thumb tip (red)
      ctx.fillStyle = 'red';
      ctx.beginPath();
      ctx.arc(thumbTip.x * canvasElement.width, thumbTip.y * canvasElement.height, 8, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw index finger tip (blue)
      ctx.fillStyle = 'blue';
      ctx.beginPath();
      ctx.arc(indexTip.x * canvasElement.width, indexTip.y * canvasElement.height, 8, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw line between them
      ctx.strokeStyle = 'green';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(thumbTip.x * canvasElement.width, thumbTip.y * canvasElement.height);
      ctx.lineTo(indexTip.x * canvasElement.width, indexTip.y * canvasElement.height);
      ctx.stroke();
      
      // Draw hand index label
      ctx.fillStyle = 'white';
      ctx.font = '16px Arial';
      ctx.fillText(`Hand ${index} (TRACKING)`, thumbTip.x * canvasElement.width + 15, thumbTip.y * canvasElement.height - 10);
    } else {
      // Draw other hands with dimmed colors
      // Draw thumb tip (dimmed red)
      ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.beginPath();
      ctx.arc(thumbTip.x * canvasElement.width, thumbTip.y * canvasElement.height, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw index finger tip (dimmed blue)
      ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
      ctx.beginPath();
      ctx.arc(indexTip.x * canvasElement.width, indexTip.y * canvasElement.height, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw dimmed line between them
      ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(thumbTip.x * canvasElement.width, thumbTip.y * canvasElement.height);
      ctx.lineTo(indexTip.x * canvasElement.width, indexTip.y * canvasElement.height);
      ctx.stroke();
      
      // Draw hand index label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = '14px Arial';
      ctx.fillText(`Hand ${index}`, thumbTip.x * canvasElement.width + 10, thumbTip.y * canvasElement.height - 10);
    }
  }

  async function startHandTracking(): Promise<boolean> {
    const initialized = await initializeHandTracking();
    if (!initialized) return false;

    // Register with the shared service
    const callback: HandTrackingCallback = {
      handIndex,
      sensitivity,
      onValueChange: (value) => {
        currentDistance = value;
        onValueChange(value);
      },
      onHandsData: showVisualizer ? onHandsData : undefined
    };

    handTrackingService.registerCallback(controlId, callback);
    return true;
  }

  function stopHandTracking() {
    handTrackingService.unregisterCallback(controlId);
  }

  function cleanup() {
    handTrackingService.unregisterCallback(controlId);
  }

  // Update callback when sensitivity or handIndex changes
  $: if (isActive && controlId) {
    const callback: HandTrackingCallback = {
      handIndex,
      sensitivity,
      onValueChange: (value) => {
        currentDistance = value;
        onValueChange(value);
      },
      onHandsData: showVisualizer ? onHandsData : undefined
    };
    handTrackingService.registerCallback(controlId, callback);
  }

  $: if (isActive) {
    startHandTracking();
  } else {
    stopHandTracking();
  }

  $: canvasWidth = 320;
  $: canvasHeight = 240;
</script>

{#if showVisualizer}
  <div class="hand-tracking-container">
    <div class="relative">
      <canvas
        bind:this={canvasElement}
        width={canvasWidth}
        height={canvasHeight}
        class="border border-gray-300 rounded"
      ></canvas>
      
      <div class="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        Hand {handIndex} Distance: {(currentDistance || 0).toFixed(3)}
      </div>
    </div>
    
    {#if handTrackingError}
      <p class="text-red-500 text-sm mt-2">{handTrackingError}</p>
    {/if}
  </div>
{:else}
  <div class="hand-tracking-info">
    <div class="text-sm text-gray-400">
      Hand {handIndex} Distance: {(currentDistance || 0).toFixed(3)}
    </div>
    {#if handTrackingError}
      <p class="text-red-500 text-sm mt-1">{handTrackingError}</p>
    {/if}
  </div>
{/if}

<style>
  .hand-tracking-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .hand-tracking-info {
    padding: 0.5rem;
    text-align: center;
  }
</style> 