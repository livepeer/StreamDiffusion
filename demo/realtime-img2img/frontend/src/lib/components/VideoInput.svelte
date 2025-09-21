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
  import { appState } from '$lib/store';
  import { parseResolution, calculateCropRegion, type ResolutionInfo } from '$lib/utils';
  import MediaListSwitcher from './MediaListSwitcher.svelte';
  
  export let width = 512;
  export let height = 512;
  export let currentResolution: ResolutionInfo | undefined = undefined;

  let videoEl: HTMLVideoElement;
  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let videoFrameCallbackId: number;

  // ajust the throttle time to your needs
  const THROTTLE = 1000 / 120;
  let selectedDevice: string = '';
  let videoIsReady = false;
  let localResolution: ResolutionInfo;

  // Reactive resolution parsing
  $: {
    if (currentResolution) {
      // Use prop if provided
      localResolution = currentResolution;
    } else {
      // Fallback to props
      localResolution = {
        width,
        height,
        aspectRatio: width / height,
        aspectRatioString: "1:1"
      };
    }
  }

  // Update canvas size when resolution changes
  $: if (canvasEl && localResolution) {
    canvasEl.width = localResolution.width;
    canvasEl.height = localResolution.height;
  }

  onMount(() => {
    ctx = canvasEl.getContext('2d') as CanvasRenderingContext2D;
    if (localResolution) {
      canvasEl.width = localResolution.width;
      canvasEl.height = localResolution.height;
    } else {
      canvasEl.width = width;
      canvasEl.height = height;
    }
  });
  
  $: {
    console.log('VideoInput: selectedDevice', selectedDevice);
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
    
    if (!localResolution) return;
    
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;
    
    // Calculate crop region to maintain target aspect ratio
    const cropRegion = calculateCropRegion(
      videoWidth,
      videoHeight,
      localResolution.width,
      localResolution.height
    );
    
    // Clear canvas and draw the cropped/scaled video
    ctx.clearRect(0, 0, localResolution.width, localResolution.height);
    ctx.drawImage(
      videoEl,
      cropRegion.x,
      cropRegion.y,
      cropRegion.width,
      cropRegion.height,
      0,
      0,
      localResolution.width,
      localResolution.height
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

<div 
  class="relative w-full max-w-xs mx-auto overflow-hidden rounded-lg border border-slate-300 bg-gray-100 dark:bg-gray-800"
  style="aspect-ratio: {localResolution?.aspectRatio || 1}"
>
  <div class="relative z-10 w-full h-full">
    {#if $mediaDevices.length > 0}
      <div class="absolute bottom-1 right-1 z-10">
        <MediaListSwitcher />
      </div>
    {/if}
    <video
      class="pointer-events-none w-full h-full object-cover rounded-lg"
      bind:this={videoEl}
      on:loadeddata={() => {
        videoIsReady = true;
      }}
      playsinline
      autoplay
      muted
      loop
    ></video>
    <canvas bind:this={canvasEl} class="absolute left-0 top-0 w-full h-full object-cover rounded-lg opacity-0"></canvas>
    
    <!-- Resolution indicator -->
    {#if localResolution}
      <div class="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        {localResolution.width}×{localResolution.height} ({localResolution.aspectRatioString})
      </div>
    {/if}
  </div>
  <div class="absolute left-0 top-0 flex w-full h-full items-center justify-center">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 448" class="w-16 h-16 opacity-20 text-gray-400">
      <path
        fill="currentColor"
        d="M224 256a128 128 0 1 0 0-256 128 128 0 1 0 0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6a29.7 29.7 0 0 0 29.7-29.7c0-98.5-79.8-178.3-178.3-178.3h-91.4z"
      />
    </svg>
  </div>
</div>
