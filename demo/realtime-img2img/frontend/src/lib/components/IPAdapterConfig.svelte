<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import InputSourceSelector from './InputSourceSelector.svelte';

  export let ipadapterInfo: any = null;
  export let currentScale: number = 1.0;
  export let currentWeightType: string = "linear";
  export let currentEnabled: boolean = true;

  const dispatch = createEventDispatcher();


  // Collapsible toggle handled internally for consistency
  let showIPAdapter: boolean = true;

  // Available weight types
  const weightTypes = [
    "linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
    "weak input", "weak output", "weak middle", "strong middle", 
    "style transfer", "composition", "strong style transfer", 
    "style and composition", "style transfer precise", "composition precise"
  ];

  async function updateIPAdapterScale(scale: number) {
    try {
      const response = await fetch('/api/ipadapter/update-scale', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          scale: scale,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateIPAdapterScale: Failed to update scale:', result.detail);
      }
    } catch (error) {
      console.error('updateIPAdapterScale: Update failed:', error);
    }
  }

  function handleScaleChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const scale = parseFloat(target.value);
    
    // Update local state immediately for responsiveness
    currentScale = scale;
    
    updateIPAdapterScale(scale);
  }

  async function updateIPAdapterWeightType(weightType: string) {
    try {
      const response = await fetch('/api/ipadapter/update-weight-type', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          weight_type: weightType,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateIPAdapterWeightType: Failed to update weight type:', result.detail);
      }
    } catch (error) {
      console.error('updateIPAdapterWeightType: Update failed:', error);
    }
  }

  function handleWeightTypeChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const weightType = target.value;
    
    // Update local state immediately for responsiveness
    currentWeightType = weightType;
    
    updateIPAdapterWeightType(weightType);
  }

  async function updateIPAdapterEnabled(enabled: boolean) {
    try {
      const response = await fetch('/api/ipadapter/update-enabled', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          enabled: enabled,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateIPAdapterEnabled: Failed to update enabled state:', result.detail);
      }
    } catch (error) {
      console.error('updateIPAdapterEnabled: Update failed:', error);
    }
  }

  function handleEnabledChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const enabled = target.checked;
    
    // Update local state immediately for responsiveness
    currentEnabled = enabled;
    
    updateIPAdapterEnabled(enabled);
  }

  function handleInputSourceChanged(event: CustomEvent) {
    const { componentType, sourceType, sourceData } = event.detail;
    console.log('IPAdapterConfig: Input source changed:', event.detail);
    
    // The InputSourceSelector handles all input routing
    // The backend will automatically handle:
    // - Image mode: uses config image or default demo image
    // - Webcam mode: uses live camera feed
    // - Video mode: cycles through video frames
  }

  // Store reference to InputSourceSelector component
  let inputSourceSelector: any;

  // Expose reset function for parent components
  export function resetInputSource() {
    console.log('IPAdapterConfig: resetInputSource called');
    
    if (inputSourceSelector && inputSourceSelector.resetToDefaults) {
      inputSourceSelector.resetToDefaults();
    }
  }

  // Update current scale, weight type, and enabled state when props change
  $: if (ipadapterInfo?.scale !== undefined) {
    currentScale = ipadapterInfo.scale;
  }
  $: if (ipadapterInfo?.weight_type !== undefined) {
    currentWeightType = ipadapterInfo.weight_type;
  }
  $: if (ipadapterInfo?.enabled !== undefined) {
    currentEnabled = ipadapterInfo.enabled;
  }
</script>

<div class="space-y-4">
  <!-- IPAdapter Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showIPAdapter = !showIPAdapter}
      class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
    >
      <h3 class="text-md font-medium">IPAdapter</h3>
      <span class="text-sm">{showIPAdapter ? '−' : '+'}</span>
    </button>
    {#if showIPAdapter}
    <div class="p-4 pt-1">
        <!-- IPAdapter Status -->
        <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded mb-3">
          {#if currentEnabled}
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-sm font-medium text-green-800 dark:text-green-200">IPAdapter Enabled</span>
          {:else}
            <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">IPAdapter Disabled</span>
          {/if}
        </div>

        <!-- Enable/Disable Toggle (only show if IPAdapter is available) -->
        {#if ipadapterInfo}
          <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 mb-3">
            <div class="flex items-center justify-between">
              <label class="text-sm font-medium text-gray-700 dark:text-gray-300" for="ipadapter-enabled">
                Enable IPAdapter
              </label>
              <label class="relative inline-flex items-center cursor-pointer">
                <input
                  id="ipadapter-enabled"
                  type="checkbox"
                  class="sr-only peer"
                  checked={currentEnabled}
                  on:change={handleEnabledChange}
                />
                <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
            <p class="text-xs text-gray-500 mt-2">
              Toggle to enable or disable IPAdapter style influence on the generated images.
            </p>
          </div>
        {/if}

        <!-- Style Input Source -->
        <div class="space-y-3">
          <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <h5 class="text-sm font-medium mb-3">Style Input Source</h5>
            
            <InputSourceSelector
              bind:this={inputSourceSelector}
              componentType="ipadapter"
              on:sourceChanged={handleInputSourceChanged}
            />
            <p class="text-xs text-gray-500 mt-3">
              Choose the input source for IPAdapter style conditioning. Upload images/videos or use webcam for dynamic styling.
            </p>
          </div>

          <!-- Scale Control -->
          <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <h5 class="text-sm font-medium mb-2">IPAdapter Scale</h5>
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Strength</label>
                <span class="text-xs text-gray-600 dark:text-gray-400">{(currentScale || 0).toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={currentScale}
                on:input={handleScaleChange}
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
              />
              <p class="text-xs text-gray-500">
                Controls how strongly the style image influences the generation. Higher values = stronger style influence.
              </p>
            </div>
          </div>

          <!-- Weight Type Control -->
          <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <h5 class="text-sm font-medium mb-2">Weight Type</h5>
            <div class="space-y-2">
              <select
                value={currentWeightType}
                on:change={handleWeightTypeChange}
                class="w-full px-3 py-2 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {#each weightTypes as weightType}
                  <option value={weightType}>{weightType}</option>
                {/each}
              </select>
              <p class="text-xs text-gray-500">
                Controls how the IPAdapter influence is distributed across different layers of the model.
              </p>
            </div>
          </div>

          <!-- IPAdapter Info -->
          {#if ipadapterInfo?.model_path}
            <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
              <h5 class="text-sm font-medium mb-2">Model Information</h5>
              <p class="text-xs text-gray-600 dark:text-gray-400 font-mono break-all">
                {ipadapterInfo.model_path}
              </p>
            </div>
          {/if}
        </div>
    </div>
    {/if}
  </div>
</div>

<style>
  /* Range slider styling */
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