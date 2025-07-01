<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues, pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  
  let imageEl: HTMLImageElement;
  let currentResolution: ResolutionInfo;

  // Reactive resolution parsing
  $: {
    if ($pipelineValues.resolution) {
      currentResolution = parseResolution($pipelineValues.resolution);
    } else {
      // Default fallback
      currentResolution = {
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
</script>

<div
  class="relative mx-auto w-full max-w-2xl self-center overflow-hidden rounded-lg border border-slate-300"
  style="aspect-ratio: {currentResolution?.aspectRatio || 1}"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="w-full h-full rounded-lg object-cover"
      style="aspect-ratio: {currentResolution?.aspectRatio || 1}"
      src={'/api/stream/' + $streamId}
    />
    
    <!-- Resolution indicator -->
    {#if currentResolution}
      <div class="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        Output: {currentResolution.width}×{currentResolution.height} ({currentResolution.aspectRatioString})
      </div>
    {/if}
    
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <div 
      class="w-full h-full rounded-lg flex items-center justify-center"
      style="aspect-ratio: {currentResolution?.aspectRatio || 1}"
    >
      {#if currentResolution}
        <div class="text-center text-gray-500">
          <div class="text-sm font-medium">Ready for {currentResolution.width}×{currentResolution.height}</div>
          <div class="text-xs">Aspect Ratio: {currentResolution.aspectRatioString}</div>
        </div>
      {:else}
        <div class="text-center text-gray-500 text-sm">
          Ready
        </div>
      {/if}
    </div>
  {/if}
</div>
