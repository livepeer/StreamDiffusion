<script lang="ts">
  import { onMount } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import ControlNetConfig from '$lib/components/ControlNetConfig.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues, pipelineValues } from '$lib/store';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let controlnetInfo: any = null;
  let tIndexList: number[] = [35, 45];
  let guidanceScale: number = 1.1;
  let delta: number = 0.7;
  let numInferenceSteps: number = 50;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch('/api/settings').then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    controlnetInfo = settings.controlnet || null;
    tIndexList = settings.t_index_list || [35, 45];
    guidanceScale = settings.guidance_scale || 1.1;
    delta = settings.delta || 0.7;
    numInferenceSteps = settings.num_inference_steps || 50;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    
    // Update prompt in store if config prompt is available
    if (settings.config_prompt) {
      pipelineValues.update(values => ({
        ...values,
        prompt: settings.config_prompt
      }));
    }
    
    console.log(pipelineParams);
    console.log('ControlNet Info:', controlnetInfo);
    console.log('T-Index List:', tIndexList);
    toggleQueueChecker(true);
  }

  function handleControlNetUpdate(event: CustomEvent) {
    controlnetInfo = event.detail.controlnet;
    
    // Update prompt if config prompt is available
    if (event.detail.config_prompt) {
      pipelineValues.update(values => ({
        ...values,
        prompt: event.detail.config_prompt
      }));
    }
    
    // Update t_index_list if available
    if (event.detail.t_index_list) {
      tIndexList = [...event.detail.t_index_list];
    }
    
    console.log('ControlNet updated:', controlnetInfo);
    console.log('T-Index List updated:', tIndexList);
  }

  async function handleTIndexListUpdate(newTIndexList: number[]) {
    try {
      const response = await fetch('/api/update-t-index-list', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          t_index_list: newTIndexList
        }),
      });

      if (response.ok) {
        tIndexList = [...newTIndexList]; // Update local state
        console.log('T-Index List updated:', tIndexList);
      } else {
        const result = await response.json();
        console.error('Failed to update t_index_list:', result.detail);
      }
    } catch (error) {
      console.error('Failed to update t_index_list:', error);
    }
  }

  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch('/api/queue').then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }

  function getSreamdata() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }
  let disabled = false;
  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();
        }
        disabled = true;
        await lcmLiveActions.start(getSreamdata);
        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        lcmLiveActions.stop();
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      toggleQueueChecker(true);
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <Warning bind:message={warningMessage}></Warning>
  <article class="text-center">
    {#if pageContent}
      {@html pageContent}
    {/if}
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {#if isImageMode}
        <div class="sm:col-start-1">
          <VideoInput
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
          ></VideoInput>
        </div>
      {/if}
      <div class={isImageMode ? 'sm:col-start-2' : 'col-span-2'}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-2">
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg my-1 p-2'}>
          {#if isLCMRunning}
            Stop
          {:else}
            Start
          {/if}
        </Button>
        <PipelineOptions {pipelineParams}></PipelineOptions>
      </div>
      <!-- ControlNet Configuration Section -->
      <div class="sm:col-span-2">
        <ControlNetConfig 
          {controlnetInfo} 
          {tIndexList} 
          {guidanceScale}
          {delta}
          {numInferenceSteps}
          on:controlnetUpdated={handleControlNetUpdate}
          on:tIndexListUpdated={(e) => handleTIndexListUpdate(e.detail)}
        ></ControlNetConfig>
      </div>
    </article>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
