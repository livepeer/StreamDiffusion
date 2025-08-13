<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import ControlNetConfig from '$lib/components/ControlNetConfig.svelte';
  import IPAdapterConfig from '$lib/components/IPAdapterConfig.svelte';
  import BlendingControl from '$lib/components/BlendingControl.svelte';
  import ResolutionPicker from '$lib/components/ResolutionPicker.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues, pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';
  import TextArea from '$lib/components/TextArea.svelte';
  import InputControl from '$lib/components/InputControl.svelte';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let controlnetInfo: any = null;
  let ipadapterInfo: any = null;
  let ipadapterScale: number = 1.0;
  let ipadapterWeightType: string = "linear";
  let tIndexList: number[] = [35, 45];
  let guidanceScale: number = 1.1;
  let delta: number = 0.7;
  let numInferenceSteps: number = 50;
  let seed: number = 2;
  let promptBlendingConfig: any = null;
  let seedBlendingConfig: any = null;
  let normalizePromptWeights: boolean = true;
  let normalizeSeedWeights: boolean = true;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';

  let currentResolution: ResolutionInfo;
  let apiError: string = '';
  let isRetrying: boolean = false;
  
  // Reactive resolution parsing
  $: {
    if ($pipelineValues.resolution) {
      currentResolution = parseResolution($pipelineValues.resolution);
    } else if (pipelineParams?.width?.default && pipelineParams?.height?.default) {
      // Fallback to pipeline params
      currentResolution = {
        width: Number(pipelineParams.width.default),
        height: Number(pipelineParams.height.default),
        aspectRatio: Number(pipelineParams.width.default) / Number(pipelineParams.height.default),
        aspectRatioString: "1:1"
      };
    }
  }
  
  // Panel state management
  let showPromptBlending: boolean = true; // Default to expanded since it's the unified blending interface
  let showResolutionPicker: boolean = true; // Default to expanded
  let leftPanelCollapsed: boolean = false;
  let rightPanelCollapsed: boolean = false;

  // Column resizing
  let leftColumnWidth: number = 25; // Percentage
  let rightColumnWidth: number = 25; // Percentage
  let isDragging: boolean = false;
  let dragTarget: 'left' | 'right' | null = null;

  // Floating video input state
  let floatingVideoPosition = { x: 20, y: 100 };
  let isDraggingVideo = false;
  let videoOffsetX = 0;
  let videoOffsetY = 0;

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
    try {
      apiError = '';
      isRetrying = false;
      
      const response = await fetch('/api/settings');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const settings = await response.json();

      pipelineParams = settings.input_params.properties;
      pipelineInfo = settings.info.properties;
      
      // Initialize prompt value in store if not already set
      if (!($pipelineValues.prompt)) {
        pipelineValues.update(values => ({
          ...values,
          prompt: pipelineParams.prompt?.default || "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        }));
      }
      
      controlnetInfo = settings.controlnet || null;
      ipadapterInfo = settings.ipadapter || null;
      ipadapterScale = settings.ipadapter?.scale || 1.0;
      ipadapterWeightType = settings.ipadapter?.weight_type || "linear";
      tIndexList = settings.t_index_list || [35, 45];
      guidanceScale = settings.guidance_scale || 1.1;
      delta = settings.delta || 0.7;
      numInferenceSteps = settings.num_inference_steps || 50;
      seed = settings.seed || 2;
      promptBlendingConfig = settings.prompt_blending || null;
      seedBlendingConfig = settings.seed_blending || null;
      normalizePromptWeights = settings.normalize_prompt_weights ?? true;
      normalizeSeedWeights = settings.normalize_seed_weights ?? true;
      isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
      maxQueueSize = settings.max_queue_size;
      pageContent = settings.page_content;
      
      console.log('getSettings: promptBlendingConfig:', promptBlendingConfig);
      console.log('getSettings: current prompt in store:', $pipelineValues.prompt);
      
      // Update prompt in store if config prompt is available
      if (settings.config_prompt) {
        pipelineValues.update(values => ({
          ...values,
          prompt: settings.config_prompt
        }));
        console.log('getSettings: Updated prompt from config_prompt:', settings.config_prompt);
      }
      
      // Set initial resolution value if available
      if (settings.current_resolution) {
        pipelineValues.update(values => ({
          ...values,
          resolution: settings.current_resolution
        }));
      }
      
      console.log(pipelineParams);
      console.log('handleControlNetUpdate: ControlNet Info:', controlnetInfo);
      console.log('handleControlNetUpdate: T-Index List:', tIndexList);
      toggleQueueChecker(true);
      
    } catch (error) {
      console.error('Failed to load settings:', error);
      apiError = error instanceof Error ? error.message : 'Failed to connect to the API. Please check if the server is running.';
    }
  }

  async function retryConnection() {
    isRetrying = true;
    await getSettings();
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
      const response = await fetch('/api/params', {
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

  async function handleResolutionUpdate(resolution: string) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ resolution }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('handleResolutionUpdate: Resolution updated successfully:', result.message);
        
        // Show success message - no restart needed for real-time updates
        if (result.message) {
          warningMessage = result.message;
          // Clear message after a few seconds
          setTimeout(() => {
            warningMessage = '';
          }, 3000);
        }
      } else {
        const result = await response.json();
        console.error('handleResolutionUpdate: Failed to update resolution:', result.detail);
        warningMessage = 'Failed to update resolution: ' + result.detail;
      }
    } catch (error: unknown) {
      console.error('handleResolutionUpdate: Failed to update resolution:', error);
      warningMessage = 'Failed to update resolution: ' + (error instanceof Error ? error.message : String(error));
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
    
    try {
      const response = await fetch('/api/queue');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      currentQueueSize = data.queue_size;
    } catch (error) {
      console.error('Failed to get queue size:', error);
      // Don't show error to user for queue size, just log it
      // This is a background operation that shouldn't interrupt the main flow
    }
    
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
  
  // Watch for resolution changes
  let previousResolution: string = '';
  $: {
    if ($pipelineValues.resolution && $pipelineValues.resolution !== previousResolution && previousResolution !== '') {
      previousResolution = $pipelineValues.resolution;
      handleResolutionUpdate($pipelineValues.resolution);
    } else if ($pipelineValues.resolution && previousResolution === '') {
      previousResolution = $pipelineValues.resolution;
    }
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

  async function refreshBlendingConfigs() {
    try {
      const response = await fetch('/api/blending/current');
      const data = await response.json();
      
      if (data.prompt_blending) {
        promptBlendingConfig = data.prompt_blending;
        console.log('refreshBlendingConfigs: Updated prompt blending:', promptBlendingConfig);
      }
      
      if (data.seed_blending) {
        seedBlendingConfig = data.seed_blending;
        console.log('refreshBlendingConfigs: Updated seed blending:', seedBlendingConfig);
      }
      
      if (data.normalize_prompt_weights !== undefined) {
        normalizePromptWeights = data.normalize_prompt_weights;
      }
      
      if (data.normalize_seed_weights !== undefined) {
        normalizeSeedWeights = data.normalize_seed_weights;
      }
      
      console.log('refreshBlendingConfigs: Blending configs refreshed');
    } catch (error) {
      console.error('refreshBlendingConfigs: Failed to refresh blending configs:', error);
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
        
        // Update ControlNet info
        if (result.controlnet) {
          controlnetInfo = result.controlnet;
        }
        
        // Update IPAdapter info
        if (result.ipadapter) {
          ipadapterInfo = result.ipadapter;
          ipadapterScale = result.ipadapter.scale || 1.0;
        }
        
        // Update streaming parameters
        if (result.t_index_list) {
          tIndexList = [...result.t_index_list];
        }
        if (result.guidance_scale !== undefined) {
          guidanceScale = result.guidance_scale;
        }
        if (result.delta !== undefined) {
          delta = result.delta;
        }
        if (result.num_inference_steps !== undefined) {
          numInferenceSteps = result.num_inference_steps;
        }
        if (result.seed !== undefined) {
          seed = result.seed;
        }
        
        // Update normalization settings
        if (result.normalize_prompt_weights !== undefined) {
          normalizePromptWeights = result.normalize_prompt_weights;
        }
        if (result.normalize_seed_weights !== undefined) {
          normalizeSeedWeights = result.normalize_seed_weights;
        }
        
        // Update blending configurations
        if (result.prompt_blending) {
          promptBlendingConfig = result.prompt_blending;
          showPromptBlending = true;  // Auto-expand if config has blending data
          console.log('uploadConfig: Updated prompt blending config:', promptBlendingConfig);
        }
        if (result.seed_blending) {
          seedBlendingConfig = result.seed_blending;
          console.log('uploadConfig: Updated seed blending config:', seedBlendingConfig);
        }
        
        // Update main prompt if config prompt is available
        if (result.config_prompt) {
          pipelineValues.update(values => ({
            ...values,
            prompt: result.config_prompt
          }));
        }
        
        // Update resolution if config resolution is available
        if (result.current_resolution) {
          pipelineValues.update(values => ({
            ...values,
            resolution: result.current_resolution
          }));
          console.log('uploadConfig: Updated resolution to:', result.current_resolution);
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

  // Column resizing functions
  function startDrag(event: MouseEvent, target: 'left' | 'right') {
    isDragging = true;
    dragTarget = target;
    event.preventDefault();
    
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('mouseup', stopDrag);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  function handleDrag(event: MouseEvent) {
    if (!isDragging || !dragTarget) return;
    
    const containerWidth = document.querySelector('.main-grid')?.clientWidth || 1200;
    const mouseX = event.clientX;
    const containerRect = document.querySelector('.main-grid')?.getBoundingClientRect();
    
    if (containerRect) {
      const relativeX = mouseX - containerRect.left;
      const percentage = (relativeX / containerWidth) * 100;
      
      if (dragTarget === 'left') {
        leftColumnWidth = Math.max(15, Math.min(40, percentage));
      } else if (dragTarget === 'right') {
        const rightPercentage = 100 - percentage;
        rightColumnWidth = Math.max(15, Math.min(40, rightPercentage));
      }
    }
  }

  function stopDrag() {
    isDragging = false;
    dragTarget = null;
    document.removeEventListener('mousemove', handleDrag);
    document.removeEventListener('mouseup', stopDrag);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }

  // Calculate center column width
  $: centerColumnWidth = 100 - (leftPanelCollapsed ? 0 : leftColumnWidth) - (rightPanelCollapsed ? 0 : rightColumnWidth);

  // Floating video input drag functions
  function startVideoDrag(event: MouseEvent) {
    isDraggingVideo = true;
    const target = event.currentTarget as HTMLElement;
    if (target) {
      const rect = target.getBoundingClientRect();
      videoOffsetX = event.clientX - rect.left;
      videoOffsetY = event.clientY - rect.top;
    }
    
    document.addEventListener('mousemove', handleVideoDrag);
    document.addEventListener('mouseup', stopVideoDrag);
    document.body.style.userSelect = 'none';
    event.preventDefault();
  }

  function handleVideoDrag(event: MouseEvent) {
    if (!isDraggingVideo) return;
    
    const newX = event.clientX - videoOffsetX;
    const newY = event.clientY - videoOffsetY;
    
    // Keep within viewport bounds
    const maxX = window.innerWidth - 320; // Assuming 300px width + some margin
    const maxY = window.innerHeight - 240; // Assuming 200px height + some margin
    
    floatingVideoPosition = {
      x: Math.max(0, Math.min(maxX, newX)),
      y: Math.max(0, Math.min(maxY, newY))
    };
  }

  function stopVideoDrag() {
    isDraggingVideo = false;
    document.removeEventListener('mousemove', handleVideoDrag);
    document.removeEventListener('mouseup', stopVideoDrag);
    document.body.style.userSelect = '';
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
  <header class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-2 flex-shrink-0">
    <div class="flex items-center justify-between">
      <div class="flex-1">
        {#if pageContent}
          <div class="text-left">
            {@html pageContent}
          </div>
        {/if}
        {#if maxQueueSize > 0}
          <p class="text-xs text-center mt-1">
            <span id="queue_size" class="font-bold">{currentQueueSize}</span> users sharing GPU.
            <a
              href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
              target="_blank"
              class="text-blue-500 underline hover:no-underline">Duplicate</a
            > to run on your own GPU.
          </p>
        {/if}
      </div>

      <!-- Pipeline Configuration and Main Controls -->
      <div class="flex items-center gap-2">
        <!-- Pipeline Configuration -->
        <Button on:click={selectFile} disabled={uploading} classList={'text-sm px-4 py-2 font-semibold'}>
          {uploading ? 'Uploading...' : 'Load YAML Config'}
        </Button>

        <input
          bind:this={fileInput}
          type="file"
          accept=".yaml,.yml"
          class="hidden"
          on:change={uploadConfig}
        />
        
        <!-- Main Control Button -->
        <Button on:click={toggleLcmLive} {disabled} classList={'text-sm px-4 py-2 font-semibold'}>
          {#if isLCMRunning}
            Stop Stream
          {:else}
            Start Stream
          {/if}
        </Button>
      </div>
    </div>
      
    {#if uploadStatus}
      <div class="mt-1 text-center">
        <p class="text-xs {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
          {uploadStatus}
        </p>
      </div>
    {/if}
  </header>

  {#if pipelineParams}
    <!-- Main Content Grid with Resizable Columns -->
    <div class="flex-1 flex p-4 overflow-hidden main-grid" style="gap: 0;">
      
      <!-- Left Panel - Input and Basic Controls -->
      {#if !leftPanelCollapsed}
        <div
          class="flex flex-col gap-4 overflow-hidden pr-2"
          style="width: {leftColumnWidth}%; min-width: 250px;"
        >
          <!-- Panel Header -->
          <div class="flex items-center justify-between flex-shrink-0">
            <h2 class="text-lg font-semibold">Input & Controls</h2>
            <button
              on:click={() => leftPanelCollapsed = !leftPanelCollapsed}
              class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
              title="Collapse panel"
            >
              ←
            </button>
          </div>
          
          <!-- Fixed Video Input Section (Image Mode Only) -->
          {#if isImageMode}
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex-shrink-0">
              <h3 class="text-md font-medium mb-3">Video Input</h3>
              <VideoInput
                width={Number(pipelineParams.width.default)}
                height={Number(pipelineParams.height.default)}
                {currentResolution}
              />
            </div>
          {/if}

          <!-- Scrollable Controls Section -->
          <div class="flex-1 overflow-y-auto space-y-4">
            <!-- Resolution Picker -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button
                on:click={() => showResolutionPicker = !showResolutionPicker}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Resolution</h3>
                <span class="text-sm">{showResolutionPicker ? '−' : '+'}</span>
              </button>
              {#if showResolutionPicker}
                <div class="p-4 pt-0">
                  <ResolutionPicker {currentResolution} {pipelineParams} />
                </div>
              {/if}
            </div>

            <!-- Unified Blending Control -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button 
                on:click={() => showPromptBlending = !showPromptBlending}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Blending Controls</h3>
                <span class="text-sm">{showPromptBlending ? '−' : '+'}</span>
              </button>
              {#if showPromptBlending}
                <div class="p-4 pt-0">
                  <BlendingControl
                    {promptBlendingConfig}
                    {seedBlendingConfig}
                    {normalizePromptWeights}
                    {normalizeSeedWeights}
                    currentPrompt={$pipelineValues.prompt}
                  />
                </div>
              {/if}
            </div>

            <!-- Input Control Section -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <InputControl />
            </div>
          </div>
        </div>

        <!-- Left Resizer -->
        <button
          type="button"
          class="w-1 bg-gray-300 dark:bg-gray-600 hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize left panel"
          on:mousedown={(e) => startDrag(e, 'left')}
          title="Drag to resize"
        ></button>
      {:else}
        <!-- Collapsed Left Panel Toggle -->
        <div class="flex flex-col items-center py-4 pr-2">
          <button
            on:click={() => leftPanelCollapsed = !leftPanelCollapsed}
            class="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 writing-mode-vertical"
            title="Expand Input & Controls"
          >
            →
          </button>
        </div>
      {/if}

      <!-- Center Panel - Main Image Output -->
      <div
        class="flex flex-col px-2"
        style="width: {centerColumnWidth}%; min-width: 300px;"
      >
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
              <ImagePlayer {currentResolution} />
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel - Advanced Controls -->
      {#if !rightPanelCollapsed}
        <!-- Right Resizer -->
        <button
          type="button"
          class="w-1 bg-gray-300 dark:bg-gray-600 hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize right panel"
          on:mousedown={(e) => startDrag(e, 'right')}
          title="Drag to resize"
        ></button>

        <div
          class="flex flex-col gap-4 overflow-y-auto pl-2"
          style="width: {rightColumnWidth}%; min-width: 250px;"
        >
          <!-- Panel Header -->
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">Advanced Settings</h2>
            <button
              on:click={() => rightPanelCollapsed = !rightPanelCollapsed}
              class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
              title="Collapse panel"
            >
              →
            </button>
          </div>
          
          <ControlNetConfig 
            {controlnetInfo} 
            {tIndexList} 
            {guidanceScale}
            {delta}
            {numInferenceSteps}
            on:controlnetUpdated={handleControlNetUpdate}
            on:tIndexListUpdated={(e) => handleTIndexListUpdate(e.detail)}
            on:controlnetConfigChanged={getSettings}
          ></ControlNetConfig>
          
          <IPAdapterConfig 
            {ipadapterInfo} 
            currentScale={ipadapterScale}
            currentWeightType={ipadapterWeightType}
          ></IPAdapterConfig>
        </div>
      {:else}
        <!-- Collapsed Right Panel Toggle -->
        <div class="flex flex-col items-center py-4 pl-2">
          <button
            on:click={() => rightPanelCollapsed = !rightPanelCollapsed}
            class="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            title="Expand Advanced Settings"
          >
            ←
          </button>
        </div>
      {/if}
    </div>
  {:else if apiError}
    <!-- API Error -->
    <div class="flex-1 flex flex-col items-center justify-center gap-6 py-48 text-center">
      <div>
        <h2 class="text-2xl font-bold text-red-600 mb-2">API Connection Failed</h2>
        <p class="text-gray-600 dark:text-gray-400 mb-4 max-w-md">
          {apiError}
        </p>
        <Button 
          on:click={retryConnection} 
          disabled={isRetrying} 
          classList="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2"
        >
          {#if isRetrying}
            <Spinner classList="w-4 h-4 mr-2 animate-spin" />
            Retrying...
          {:else}
            Retry Connection
          {/if}
        </Button>
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

  <!-- Floating Video Input (when left panel is collapsed and in image mode) -->
  {#if leftPanelCollapsed && isImageMode && pipelineParams}
    <div
      class="fixed z-50 bg-white dark:bg-gray-800 rounded-lg border-2 border-gray-300 dark:border-gray-600 shadow-lg"
      style="left: {floatingVideoPosition.x}px; top: {floatingVideoPosition.y}px; width: 320px;"
    >
      <!-- Drag Handle -->
      <div
        class="bg-gray-100 dark:bg-gray-700 px-3 py-2 rounded-t-lg cursor-move border-b border-gray-200 dark:border-gray-600 flex items-center justify-between"
        role="button"
        tabindex="0"
        on:mousedown={startVideoDrag}
      >
        <div class="flex items-center gap-2 text-sm font-medium">
          <span>📹</span>
          <span>Video Input</span>
        </div>
        <span class="text-xs text-gray-500 dark:text-gray-400">Drag to move</span>
      </div>
      
      <!-- Video Input Content -->
      <div class="p-3">
        <VideoInput
          width={Number(pipelineParams.width.default)}
          height={Number(pipelineParams.height.default)}
          {currentResolution}
        />
      </div>
    </div>
  {/if}
</main>

<style lang="postcss">
  @reference "tailwindcss";
  
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

  /* Resizer styling */
  .main-grid {
    position: relative;
  }

  /* Prevent text selection during drag */
  :global(body.dragging) {
    user-select: none;
    cursor: col-resize !important;
  }

  /* Removed unused .resizer:hover selector */
</style>
