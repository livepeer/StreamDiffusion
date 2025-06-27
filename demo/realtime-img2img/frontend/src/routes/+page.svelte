<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
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
  let showPromptBlending: boolean = false;
  let showSeedBlending: boolean = false;
  let leftPanelCollapsed: boolean = false;
  let rightPanelCollapsed: boolean = false;

  // FPS tracking
  let fps = 0;
  let fpsInterval: number | null = null;

  onMount(() => {
    getSettings();
    updateFPS();
    fpsInterval = setInterval(updateFPS, 1000);
  });

  onDestroy(() => {
    if (fpsInterval) {
      clearInterval(fpsInterval);
    }
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

  async function updateFPS() {
    try {
      const response = await fetch('/api/fps');
      const data = await response.json();
      fps = data.fps;
    } catch (error) {
      console.error('updateFPS: Failed to fetch FPS:', error);
    }
  }

  // Pipeline configuration upload
  let fileInput: HTMLInputElement;
  let uploading = false;
  let uploadStatus = '';

  async function uploadConfig() {
    if (!fileInput.files || fileInput.files.length === 0) {
      uploadStatus = 'Please select a YAML file';
      return;
    }

    const file = fileInput.files[0];
    if (!file.name.endsWith('.yaml') && !file.name.endsWith('.yml')) {
      uploadStatus = 'Please select a YAML file (.yaml or .yml)';
      return;
    }

    uploading = true;
    uploadStatus = 'Uploading configuration...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/controlnet/upload-config', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = 'Configuration uploaded successfully! Pipeline will load when you start streaming.';
        fileInput.value = '';
        
        if (result.controlnet) {
          controlnetInfo = result.controlnet;
          if (result.t_index_list) {
            tIndexList = [...result.t_index_list];
          }
          if (result.config_prompt) {
            pipelineValues.update(values => ({
              ...values,
              prompt: result.config_prompt
            }));
          }
        }
        
        setTimeout(() => {
          uploadStatus = '';
        }, 4000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to load configuration'}`;
      }
    } catch (error) {
      console.error('uploadConfig: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploading = false;
    }
  }

  function selectFile() {
    fileInput.click();
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
      
      <!-- Pipeline Configuration and Main Controls -->
      <div class="flex items-center gap-4">
        <!-- Pipeline Configuration -->
        <div class="flex items-center gap-2">
          <Button on:click={selectFile} disabled={uploading} classList="text-sm px-3 py-2">
            {uploading ? 'Uploading...' : 'Load YAML Config'}
          </Button>
        </div>
        
        <input
          bind:this={fileInput}
          type="file"
          accept=".yaml,.yml"
          class="hidden"
          on:change={uploadConfig}
        />
        
        <!-- Main Control Button -->
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg px-6 py-3 font-semibold'}>
          {#if isLCMRunning}
            Stop Stream
          {:else}
            Start Stream
          {/if}
        </Button>
      </div>
    </div>
    
    {#if uploadStatus}
      <div class="mt-2 text-center">
        <p class="text-sm {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
          {uploadStatus}
        </p>
      </div>
    {/if}
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

          <!-- Prompt Blending -->
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <button 
              on:click={() => showPromptBlending = !showPromptBlending}
              class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
            >
              <h3 class="text-md font-medium">Prompt Blending</h3>
              <span class="text-sm">{showPromptBlending ? '−' : '+'}</span>
            </button>
            {#if showPromptBlending}
              <div class="p-4 pt-0">
                <PromptBlendingControl {promptBlendingConfig} />
              </div>
            {/if}
          </div>

          <!-- Seed Blending -->
          <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <button 
              on:click={() => showSeedBlending = !showSeedBlending}
              class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
            >
              <h3 class="text-md font-medium">Seed Blending</h3>
              <span class="text-sm">{showSeedBlending ? '−' : '+'}</span>
            </button>
            {#if showSeedBlending}
              <div class="p-4 pt-0">
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
            <div class="flex items-center gap-4">
              {#if isLCMRunning}
                <div class="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900 rounded-lg">
                  <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span class="text-sm font-medium text-green-800 dark:text-green-200">
                    {fps.toFixed(1)} FPS
                  </span>
                </div>
              {/if}
              <div class="text-sm text-gray-600 dark:text-gray-400">
                Status: {isLCMRunning ? 'Streaming' : 'Stopped'}
              </div>
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
          <ControlNetConfig 
            {controlnetInfo} 
            {tIndexList} 
            {guidanceScale}
            {delta}
            {numInferenceSteps}
            on:controlnetUpdated={handleControlNetUpdate}
            on:tIndexListUpdated={(e) => handleTIndexListUpdate(e.detail)}
          ></ControlNetConfig>
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
