<script lang="ts">
  import 'rvfc-polyfill';

  import { onDestroy, onMount } from 'svelte';
  import {
    mediaStreamStatus,
    MediaStreamStatusEnum,
    onFrameChangeStore,
    mediaStream,
    mediaDevices
  } from '$lib/mediaStream';
  import { pipelineValues } from '$lib/store';
  import { parseResolution, calculateCropRegion, type ResolutionInfo } from '$lib/utils';
  import MediaListSwitcher from './MediaListSwitcher.svelte';
  
  export let width = 512;
  export let height = 512;

  let videoEl: HTMLVideoElement;
  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let videoFrameCallbackId: number;

  // ajust the throttle time to your needs
  const THROTTLE = 1000 / 120;
  let selectedDevice: string = '';
  let videoIsReady = false;
  let currentResolution: ResolutionInfo;

  // Reactive resolution parsing
  $: {
    if ($pipelineValues.resolution) {
      currentResolution = parseResolution($pipelineValues.resolution);
    } else {
      // Fallback to props
      currentResolution = {
        width,
        height,
        aspectRatio: width / height,
        aspectRatioString: "1:1"
      };
    }
  }

  // Update canvas size when resolution changes
  $: if (canvasEl && currentResolution) {
    canvasEl.width = currentResolution.width;
    canvasEl.height = currentResolution.height;
  }

  onMount(() => {
    ctx = canvasEl.getContext('2d') as CanvasRenderingContext2D;
    if (currentResolution) {
      canvasEl.width = currentResolution.width;
      canvasEl.height = currentResolution.height;
    } else {
      canvasEl.width = width;
      canvasEl.height = height;
    }
  });
  
  $: {
    console.log(selectedDevice);
  }
  
  onDestroy(() => {
    if (videoFrameCallbackId) videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
  });

  $: if (videoEl) {
    videoEl.srcObject = $mediaStream;
  }
  
  let lastMillis = 0;
  async function onFrameChange(now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) {
    if (now - lastMillis < THROTTLE) {
      videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
      return;
    }
    
    if (!currentResolution) return;
    
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;
    
    // Calculate crop region to maintain target aspect ratio
    const cropRegion = calculateCropRegion(
      videoWidth,
      videoHeight,
      currentResolution.width,
      currentResolution.height
    );
    
    // Clear canvas and draw the cropped/scaled video
    ctx.clearRect(0, 0, currentResolution.width, currentResolution.height);
    ctx.drawImage(
      videoEl,
      cropRegion.x,
      cropRegion.y,
      cropRegion.width,
      cropRegion.height,
      0,
      0,
      currentResolution.width,
      currentResolution.height
    );
    
    const blob = await new Promise<Blob>((resolve) => {
      canvasEl.toBlob(
        (blob) => {
          resolve(blob as Blob);
        },
        'image/jpeg',
        1
      );
    });
    onFrameChangeStore.set({ blob });
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  $: if ($mediaStreamStatus == MediaStreamStatusEnum.CONNECTED && videoIsReady) {
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }
</script>

<div class="relative mx-auto w-full max-w-2xl overflow-hidden rounded-lg border border-slate-300">
  <div 
    class="relative z-10 w-full object-cover"
    style="aspect-ratio: {currentResolution?.aspectRatio || 1}"
  >
    {#if $mediaDevices.length > 0}
      <div class="absolute bottom-0 right-0 z-10">
        <MediaListSwitcher />
      </div>
    {/if}
    <video
      class="pointer-events-none w-full h-full object-cover"
      bind:this={videoEl}
      on:loadeddata={() => {
        videoIsReady = true;
      }}
      playsinline
      autoplay
      muted
      loop
    ></video>
    <canvas 
      bind:this={canvasEl} 
      class="absolute left-0 top-0 w-full h-full object-cover"
      style="aspect-ratio: {currentResolution?.aspectRatio || 1}"
    ></canvas>
    
    <!-- Resolution indicator -->
    {#if currentResolution}
      <div class="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        {currentResolution.width}×{currentResolution.height} ({currentResolution.aspectRatioString})
      </div>
    {/if}
  </div>
  <div class="absolute left-0 top-0 flex w-full h-full items-center justify-center" style="aspect-ratio: {currentResolution?.aspectRatio || 1}">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 448" class="w-40 p-5 opacity-20">
      <path
        fill="currentColor"
        d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
      />
    </svg>
  </div>
</div>
