<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import InputRange from './InputRange.svelte';
  import PreprocessorDocs from './PreprocessorDocs.svelte';

  export let controlnetInfo: any = null;
  export let tIndexList: number[] = [35, 45];
  export let guidanceScale: number = 1.1;
  export let delta: number = 0.7;
  export let numInferenceSteps: number = 50;

  const dispatch = createEventDispatcher();

  let fileInput: HTMLInputElement;
  let uploading = false;
  let uploadStatus = '';
  let fps = 0;
  let fpsInterval: number | null = null;
  let showDocs = false;

  // Initialize FPS tracking
  onMount(() => {
    updateFPS();
    fpsInterval = setInterval(updateFPS, 1000); // Update FPS every second
  });

  onDestroy(() => {
    if (fpsInterval) {
      clearInterval(fpsInterval);
    }
  });

  async function updateFPS() {
    try {
      const response = await fetch('/api/fps');
      const data = await response.json();
      fps = data.fps;
    } catch (error) {
      console.error('updateFPS: Failed to fetch FPS:', error);
    }
  }

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
        uploadStatus = 'Configuration uploaded successfully! Pipeline will load with these settings when you start streaming.';
        // Clear file input
        fileInput.value = '';
        
        // Update controlnet info from the response
        if (result.controlnet) {
          controlnetInfo = result.controlnet;
          // Update t_index_list if provided in response
          if (result.t_index_list) {
            tIndexList = [...result.t_index_list];
          }
          // Dispatch event to parent to update its controlnetInfo
          // Include both controlnet info, config prompt, and t_index_list if available
          dispatch('controlnetUpdated', {
            controlnet: result.controlnet,
            config_prompt: result.config_prompt || null,
            t_index_list: result.t_index_list || null
          });
        }
        
        // Clear status after a delay
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

  async function updateControlNetStrength(index: number, strength: number) {
    try {
      const response = await fetch('/api/controlnet/update-strength', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          index: index,
          strength: strength,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateControlNetStrength: Failed to update strength:', result.detail);
      }
    } catch (error) {
      console.error('updateControlNetStrength: Update failed:', error);
    }
  }

  function handleStrengthChange(index: number, event: Event) {
    const target = event.target as HTMLInputElement;
    const strength = parseFloat(target.value);
    
    // Update local state immediately for responsiveness
    if (controlnetInfo && controlnetInfo.controlnets) {
      controlnetInfo.controlnets[index].strength = strength;
      controlnetInfo = { ...controlnetInfo }; // Trigger reactivity
    }
    
    updateControlNetStrength(index, strength);
  }

  function selectFile() {
    fileInput.click();
  }

  function handleTIndexChange(index: number, event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseInt(target.value);
    
    // Update local tIndexList
    const newTIndexList = [...tIndexList];
    newTIndexList[index] = value;
    
    // Update local state immediately for UI responsiveness
    tIndexList = newTIndexList;
    
    // Dispatch the update event
    dispatch('tIndexListUpdated', newTIndexList);
  }

  // Parameter controls - now props from parent

  async function updateGuidanceScale(value: number) {
    try {
      guidanceScale = value;
      const response = await fetch('/api/update-guidance-scale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ guidance_scale: value })
      });
      if (!response.ok) {
        const result = await response.json();
        console.error('updateGuidanceScale: Failed to update guidance_scale:', result.detail);
      }
    } catch (error) {
      console.error('updateGuidanceScale: Update failed:', error);
    }
  }

  async function updateDelta(value: number) {
    try {
      delta = value;
      const response = await fetch('/api/update-delta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta: value })
      });
      if (!response.ok) {
        const result = await response.json();
        console.error('updateDelta: Failed to update delta:', result.detail);
      }
    } catch (error) {
      console.error('updateDelta: Update failed:', error);
    }
  }

  async function updateNumInferenceSteps(value: number) {
    try {
      numInferenceSteps = value;
      const response = await fetch('/api/update-num-inference-steps', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_inference_steps: value })
      });
      if (!response.ok) {
        const result = await response.json();
        console.error('updateNumInferenceSteps: Failed to update num_inference_steps:', result.detail);
      }
    } catch (error) {
      console.error('updateNumInferenceSteps: Update failed:', error);
    }
  }

  function handleGuidanceScaleChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseFloat(target.value);
    updateGuidanceScale(value);
  }

  function handleDeltaChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseFloat(target.value);
    updateDelta(value);
  }

  function handleNumInferenceStepsChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseInt(target.value);
    updateNumInferenceSteps(value);
  }
</script>

<div class="controlnet-config space-y-4">
  <!-- FPS Display -->
  <div class="flex items-center justify-between bg-gray-100 dark:bg-gray-800 rounded p-2">
    <span class="text-sm font-medium">Performance</span>
    <span class="text-base font-bold text-green-600 dark:text-green-400">
      {fps.toFixed(1)} FPS
    </span>
  </div>

  <!-- Compact Single Column Layout -->
  <div class="space-y-4">
    
    <!-- ControlNet Configuration Section -->
    <div class="space-y-3">
      <div class="flex justify-between items-center">
        <h4 class="text-base font-semibold">ControlNet Configuration</h4>
        <Button on:click={() => showDocs = true} classList="text-xs px-2 py-1">
          Help
        </Button>
      </div>
    
      <!-- File Upload -->
      <div class="space-y-2">
        <div class="flex items-center gap-2">
          <Button on:click={selectFile} disabled={uploading} classList="text-xs px-3 py-1">
            {uploading ? 'Uploading...' : 'Load YAML'}
          </Button>
          <span class="text-xs text-gray-600 dark:text-gray-400">
            {controlnetInfo?.enabled ? 'Ready' : 'Standard'}
          </span>
        </div>
        
        <input
          bind:this={fileInput}
          type="file"
          accept=".yaml,.yml"
          class="hidden"
          on:change={uploadConfig}
        />
        
        {#if uploadStatus}
          <p class="text-xs {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
            {uploadStatus}
          </p>
        {/if}
      </div>

      <!-- ControlNet Strength Controls -->
      {#if controlnetInfo?.enabled && controlnetInfo?.controlnets?.length > 0}
        <div class="space-y-2">
          <h5 class="text-sm font-medium">ControlNet Strengths</h5>
          {#each controlnetInfo.controlnets as controlnet}
            <div class="bg-gray-50 dark:bg-gray-700 rounded p-2 space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-xs font-medium truncate">
                  {controlnet.name}
                </span>
                <span class="text-xs text-gray-600 dark:text-gray-400">
                  {controlnet.strength.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={controlnet.strength}
                on:input={(e) => handleStrengthChange(controlnet.index, e)}
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
              />
              <p class="text-xs text-gray-500">{controlnet.preprocessor}</p>
            </div>
          {/each}
        </div>
      {:else if controlnetInfo?.enabled}
        <p class="text-xs text-gray-600 dark:text-gray-400">
          ControlNet enabled but no networks found.
        </p>
      {:else}
        <p class="text-xs text-gray-600 dark:text-gray-400">
          Upload YAML to enable ControlNet features.
        </p>
      {/if}
    </div>

    <!-- Timestep Configuration -->
    <div class="space-y-3">
      <h4 class="text-base font-semibold">Timestep Configuration</h4>
      
      <div class="space-y-2">
        <h5 class="text-sm font-medium">Timestep Indices</h5>
        <div class="bg-gray-50 dark:bg-gray-700 rounded p-2 space-y-2">
          <p class="text-xs text-gray-600 dark:text-gray-400">
            Controls denoising steps (0-49)
          </p>
          <div class="space-y-2">
            {#each tIndexList as tIndex, index}
              <div class="space-y-1">
                <div class="flex items-center justify-between">
                  <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Step {index + 1}</label>
                  <span class="text-xs text-gray-600 dark:text-gray-400">{tIndex}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="49"
                  step="1"
                  value={tIndex}
                  on:input={(e) => handleTIndexChange(index, e)}
                  class="w-full appearance-none cursor-pointer h-2"
                />
              </div>
            {/each}
          </div>
          <p class="text-xs text-gray-500">
            Current: [{tIndexList.join(', ')}]
          </p>
        </div>
      </div>
    </div>

    <!-- Streaming Parameters -->
    <div class="space-y-3">
      <h4 class="text-base font-semibold">Streaming Parameters</h4>
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-2 space-y-3">
        <p class="text-xs text-gray-600 dark:text-gray-400">
          Real-time adjustments during inference
        </p>
        <div class="space-y-3">
          <!-- Guidance Scale -->
          <div class="space-y-1">
            <div class="flex items-center justify-between">
              <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Guidance Scale</label>
              <span class="text-xs text-gray-600 dark:text-gray-400">{guidanceScale.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="3.0"
              step="0.01"
              value={guidanceScale}
              on:input={handleGuidanceScaleChange}
              class="w-full appearance-none cursor-pointer h-2"
            />
            <p class="text-xs text-gray-500">CFG guidance strength</p>
          </div>
          
          <!-- Delta -->
          <div class="space-y-1">
            <div class="flex items-center justify-between">
              <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Delta</label>
              <span class="text-xs text-gray-600 dark:text-gray-400">{delta.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.01"
              value={delta}
              on:input={handleDeltaChange}
              class="w-full appearance-none cursor-pointer h-2"
            />
            <p class="text-xs text-gray-500">Virtual residual noise multiplier</p>
          </div>
          
          <!-- Inference Steps -->
          <div class="space-y-1">
            <div class="flex items-center justify-between">
              <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Inference Steps</label>
              <span class="text-xs text-gray-600 dark:text-gray-400">{numInferenceSteps}</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={numInferenceSteps}
              on:input={handleNumInferenceStepsChange}
              class="w-full appearance-none cursor-pointer h-2"
            />
            <p class="text-xs text-gray-500">Number of denoising steps</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Preprocessor Documentation Modal -->
<PreprocessorDocs bind:visible={showDocs} />

<style>
  .controlnet-config {
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    padding: 0.75rem;
  }
  
  .dark .controlnet-config {
    border-color: #374151;
  }

  /* Compact range slider styling */
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  input[type="range"]::-moz-range-thumb {
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  input[type="range"]::-webkit-slider-track {
    height: 8px;
    border-radius: 4px;
    background: #e5e7eb;
  }

  input[type="range"]::-moz-range-track {
    height: 8px;
    border-radius: 4px;
    background: #e5e7eb;
    border: none;
  }

  .dark input[type="range"]::-webkit-slider-track {
    background: #4b5563;
  }

  .dark input[type="range"]::-moz-range-track {
    background: #4b5563;
  }
</style> 