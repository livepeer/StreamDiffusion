<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues, pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';
  import { onDestroy } from 'svelte';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  let isFullscreen = false;

  export let currentResolution: ResolutionInfo | undefined = undefined;

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('ImagePlayer: isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  let localResolution: ResolutionInfo;

  // Reactive resolution parsing
  $: {
    if (currentResolution) {
      // Use prop if provided
      localResolution = currentResolution;
    } else if ($pipelineValues.resolution) {
      // Fallback to pipeline values
      localResolution = parseResolution($pipelineValues.resolution);
    } else {
      // Default fallback
      localResolution = {
        width: 512,
        height: 512,
        aspectRatio: 1,
        aspectRatioString: "1:1"
      };
    }
  }
  
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }

  async function toggleFullscreen() {
    if (!imageEl) return;
    
    try {
      if (!document.fullscreenElement) {
        await imageEl.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error('toggleFullscreen: Error toggling fullscreen:', err);
    }
  }

  function handleFullscreenChange() {
    isFullscreen = !!document.fullscreenElement;
  }

  // Listen for fullscreen changes
  if (typeof window !== 'undefined') {
    document.addEventListener('fullscreenchange', handleFullscreenChange);
  }

  onDestroy(() => {
    if (typeof window !== 'undefined') {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }
  });
</script>

<div 
  class="relative w-full h-full flex items-center justify-center overflow-hidden rounded-lg border border-slate-300 bg-gray-50 dark:bg-gray-900"
  style="aspect-ratio: {localResolution?.aspectRatio || 1}"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="max-w-full max-h-full object-contain rounded-lg"
      src={'/api/stream/' + $streamId}
      alt="Generated output stream"
    />
    
    <!-- Resolution indicator -->
    {#if localResolution}
      <div class="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        Output: {localResolution.width}×{localResolution.height} ({localResolution.aspectRatioString})
      </div>
    {/if}
    
    <div class="absolute bottom-2 right-2 flex gap-2">
      <Button
        on:click={toggleFullscreen}
        disabled={!isLCMRunning}
        title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
        classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {#if isFullscreen}
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 9V6a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v3M6 15v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-3M9 12h6"></path>
          {:else}
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0-5 5M4 16v4m0 0h4m-4 0 5-5m11 1v4m0 0h-4m4 0-5-5"></path>
          {/if}
        </svg>
      </Button>
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm text-white bg-black bg-opacity-50 hover:bg-opacity-70 p-2 shadow-lg rounded-lg backdrop-blur-sm transition-all'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <div class="w-full h-full flex flex-col items-center justify-center text-gray-400 dark:text-gray-600">
      <div class="w-24 h-24 mb-4 opacity-30">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <circle cx="9" cy="9" r="2"/>
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
        </svg>
      </div>
      <p class="text-lg font-medium">Generated output will appear here</p>
      <p class="text-sm opacity-75">Click "Start Stream" to begin</p>
      {#if localResolution}
        <div class="text-xs mt-2 opacity-50">
          Ready for {localResolution.width}×{localResolution.height} ({localResolution.aspectRatioString})
        </div>
      {/if}
    </div>
  {/if}
</div>
