<script lang="ts">
  import { onMount } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import ControlNetConfig from '$lib/components/ControlNetConfig.svelte';
  import PromptBlendingControl from '$lib/components/PromptBlendingControl.svelte';
  import SeedBlendingControl from '$lib/components/SeedBlendingControl.svelte';
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
  let seed: number = 2;
  let promptBlendingConfig: any = null;
  let seedBlendingConfig: any = null;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  
  // Panel state management
  let showBasicControls: boolean = true;
  let showAdvancedControls: boolean = true;
  let showBlendingControls: boolean = false;
  let showControlNetConfig: boolean = false;
  let leftPanelCollapsed: boolean = false;
  let rightPanelCollapsed: boolean = false;

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
    seed = settings.seed || 2;
    promptBlendingConfig = settings.prompt_blending || null;
    seedBlendingConfig = settings.seed_blending || null;
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
    console.log('handleControlNetUpdate: ControlNet Info:', controlnetInfo);
    console.log('handleControlNetUpdate: T-Index List:', tIndexList);
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
    
    console.log('handleControlNetUpdate: ControlNet updated:', controlnetInfo);
    console.log('handleControlNetUpdate: T-Index List updated:', tIndexList);
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
        console.log('handleTIndexListUpdate: T-Index List updated:', tIndexList);
      } else {
        const result = await response.json();
        console.error('handleTIndexListUpdate: Failed to update t_index_list:', result.detail);
      }
    } catch (error) {
      console.error('handleTIndexListUpdate: Failed to update t_index_list:', error);
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

<main class="h-screen flex flex-col overflow-hidden">
  <Warning bind:message={warningMessage}></Warning>
  
  <!-- Header Section -->
  <header class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex-shrink-0">
    <div class="flex items-center justify-between">
      <div class="flex-1">
        {#if pageContent}
          <div class="text-center">
            {@html pageContent}
          </div>
        {/if}
        {#if maxQueueSize > 0}
          <p class="text-sm text-center mt-2">
            There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
            user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
            <a
              href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
              target="_blank"
              class="text-blue-500 underline hover:no-underline">Duplicate</a
            > and run it on your own GPU.
          </p>
        {/if}
      </div>
      
      <!-- Main Control Button -->
      <div class="ml-4">
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg px-6 py-3 font-semibold'}>
          {#if isLCMRunning}
            Stop Stream
          {:else}
            Start Stream
          {/if}
        </Button>
      </div>
    </div>
  </header>

  {#if pipelineParams}
    <!-- Main Content Grid -->
    <div class="flex-1 grid grid-cols-12 gap-4 p-4 overflow-hidden">
      
      <!-- Left Panel - Input and Basic Controls -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-y-auto">
        <!-- Panel Header -->
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold">Input & Controls</h2>
          <button 
            on:click={() => leftPanelCollapsed = !leftPanelCollapsed}
            class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {leftPanelCollapsed ? '→' : '←'}
          </button>
        </div>
        
        {#if !leftPanelCollapsed}
          <!-- Video Input (Image Mode Only) -->
          {#if isImageMode}
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <h3 class="text-md font-medium mb-3">Video Input</h3>
              <VideoInput
                width={Number(pipelineParams.width.default)}
                height={Number(pipelineParams.height.default)}
              />
            </div>
          {/if}

          <!-- Basic Pipeline Controls -->
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <button 
              on:click={() => showBasicControls = !showBasicControls}
              class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
            >
              <h3 class="text-md font-medium">Pipeline Settings</h3>
              <span class="text-sm">{showBasicControls ? '−' : '+'}</span>
            </button>
            {#if showBasicControls}
              <div class="p-4 pt-0">
                <PipelineOptions {pipelineParams} />
              </div>
            {/if}
          </div>

          <!-- Blending Controls -->
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <button 
              on:click={() => showBlendingControls = !showBlendingControls}
              class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
            >
              <h3 class="text-md font-medium">Prompt & Seed Blending</h3>
              <span class="text-sm">{showBlendingControls ? '−' : '+'}</span>
            </button>
            {#if showBlendingControls}
              <div class="p-4 pt-0 space-y-4">
                <PromptBlendingControl {promptBlendingConfig} />
                <SeedBlendingControl {seedBlendingConfig} />
              </div>
            {/if}
          </div>
        {/if}
      </div>

      <!-- Center Panel - Main Image Output -->
      <div class="col-span-12 lg:col-span-6 flex flex-col">
        <div class="flex-1 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-lg font-semibold">Generated Output</h2>
            <div class="text-sm text-gray-600 dark:text-gray-400">
              Status: {isLCMRunning ? 'Running' : 'Stopped'}
            </div>
          </div>
          <div class="flex-1 flex items-center justify-center">
            <div class="w-full max-w-2xl">
              <ImagePlayer />
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel - Advanced Controls -->
      <div class="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-y-auto">
        <!-- Panel Header -->
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold">Advanced Settings</h2>
          <button 
            on:click={() => rightPanelCollapsed = !rightPanelCollapsed}
            class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {rightPanelCollapsed ? '←' : '→'}
          </button>
        </div>
        
        {#if !rightPanelCollapsed}
          <!-- ControlNet Configuration -->
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <button 
              on:click={() => showControlNetConfig = !showControlNetConfig}
              class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
            >
              <h3 class="text-md font-medium">ControlNet & Inference</h3>
              <span class="text-sm">{showControlNetConfig ? '−' : '+'}</span>
            </button>
            {#if showControlNetConfig}
              <div class="p-4 pt-0">
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
            {/if}
          </div>
        {/if}
      </div>
    </div>
  {:else}
    <!-- Loading State -->
    <div class="flex-1 flex items-center justify-center">
      <div class="flex items-center gap-3 text-2xl">
        <Spinner classList={'animate-spin opacity-50'} />
        <p>Loading StreamDiffusion...</p>
      </div>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
  
  /* Custom scrollbar styling */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    @apply bg-gray-100 dark:bg-gray-800;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    @apply bg-gray-300 dark:bg-gray-600 rounded-full;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    @apply bg-gray-400 dark:bg-gray-500;
  }
</style>
