<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import InputRange from './InputRange.svelte';

  export let controlnetInfo: any = null;
  export let tIndexList: number[] = [35, 45];

  const dispatch = createEventDispatcher();

  let fileInput: HTMLInputElement;
  let uploading = false;
  let uploadStatus = '';
  let fps = 0;
  let fpsInterval: NodeJS.Timeout | null = null;

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
          // Dispatch event to parent to update its controlnetInfo
          // Include both controlnet info and config prompt if available
          dispatch('controlnetUpdated', {
            controlnet: result.controlnet,
            config_prompt: result.config_prompt || null
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
</script>

<div class="controlnet-config space-y-4">
  <!-- FPS Display -->
  <div class="flex items-center justify-between bg-gray-100 dark:bg-gray-800 rounded p-3">
    <span class="text-sm font-medium">Performance</span>
    <span class="text-lg font-bold text-green-600 dark:text-green-400">
      {fps.toFixed(1)} FPS
    </span>
  </div>

  <!-- ControlNet Configuration Section -->
  <div class="space-y-3">
    <h3 class="text-lg font-semibold">ControlNet Configuration</h3>
    
    <!-- File Upload -->
    <div class="space-y-2">
      <div class="flex items-center gap-2">
        <Button on:click={selectFile} disabled={uploading} classList="text-sm">
          {uploading ? 'Uploading...' : 'Load YAML Config'}
        </Button>
        <span class="text-sm text-gray-600 dark:text-gray-400">
          {controlnetInfo?.enabled ? 'ControlNet Ready' : 'Standard Mode'}
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
        <p class="text-sm {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
          {uploadStatus}
        </p>
      {/if}
    </div>

    <!-- ControlNet Strength Controls -->
    {#if controlnetInfo?.enabled && controlnetInfo?.controlnets?.length > 0}
      <div class="space-y-3">
        <h4 class="font-medium">ControlNet Strengths <span class="text-sm text-gray-500">(Pipeline loads when you start streaming)</span></h4>
        {#each controlnetInfo.controlnets as controlnet}
          <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium">
                {controlnet.name} ({controlnet.preprocessor})
              </span>
              <span class="text-sm text-gray-600 dark:text-gray-400">
                {controlnet.strength.toFixed(3)}
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
          </div>
        {/each}
      </div>
    {:else if controlnetInfo?.enabled}
      <p class="text-sm text-gray-600 dark:text-gray-400">
        ControlNet enabled but no control networks found in configuration.
      </p>
    {:else}
      <p class="text-sm text-gray-600 dark:text-gray-400">
        Upload a YAML configuration file to enable ControlNet features. All pipelines load only when you start streaming.
      </p>
    {/if}

    <!-- T-Index List Controls -->
    <div class="space-y-3">
      <h4 class="font-medium">Timestep Indices (t_index_list) <span class="text-sm text-gray-500">Controls denoising steps - lower = less denoising, higher = more denoising</span></h4>
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 space-y-3">
        <p class="text-xs text-gray-600 dark:text-gray-400">
          These values control which timesteps are used for denoising. You can adjust the values but not the number of elements.
        </p>
        <div class="space-y-3">
          {#each tIndexList as tIndex, index}
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <label class="text-sm font-medium text-gray-600 dark:text-gray-400">Step {index + 1}</label>
                <span class="text-sm text-gray-600 dark:text-gray-400">{tIndex}</span>
              </div>
              <input
                type="range"
                min="0"
                max="49"
                step="1"
                value={tIndex}
                on:input={(e) => handleTIndexChange(index, e)}
                class="w-full appearance-none cursor-pointer"
              />
            </div>
          {/each}
        </div>
        <p class="text-xs text-gray-500">
          Current: [{tIndexList.join(', ')}] | Range: 0-49 (50 total inference steps)
        </p>
      </div>
    </div>
  </div>
</div>

<style>
  .controlnet-config {
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    padding: 1rem;
  }
  
  .dark .controlnet-config {
    border-color: #374151;
  }

  /* Custom range slider styling */
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  input[type="range"]::-moz-range-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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