<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import InputRange from './InputRange.svelte';
  import PreprocessorDocs from './PreprocessorDocs.svelte';
  import PreprocessorSelector from './PreprocessorSelector.svelte';
  import PreprocessorParams from './PreprocessorParams.svelte';

  export let controlnetInfo: any = null;
  export let tIndexList: number[] = [35, 45];
  export let guidanceScale: number = 1.1;
  export let delta: number = 0.7;
  export let numInferenceSteps: number = 50;

  const dispatch = createEventDispatcher();

  let showDocs = false;
  
  // Collapsible section state
  let showControlNet: boolean = true;
  let showTimesteps: boolean = true;
  let showStreamingParams: boolean = true;
  let showPreprocessorParams: boolean = true;
  
  // Preprocessor state
  let currentPreprocessors: { [index: number]: string } = {};
  let preprocessorInfos: { [index: number]: any } = {};
  let preprocessorParams: { [index: number]: { [key: string]: any } } = {};

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

  function handleHelpClick(event: Event) {
    event.stopPropagation();
    showDocs = true;
  }

  function handlePreprocessorChanged(event: CustomEvent) {
    const { controlnet_index, preprocessor, preprocessor_info, current_params } = event.detail;
    console.log('ControlNetConfig: handlePreprocessorChanged called with:', event.detail);
    
    currentPreprocessors[controlnet_index] = preprocessor;
    preprocessorInfos[controlnet_index] = preprocessor_info;
    
    // Initialize parameters with current values or defaults
    if (preprocessor_info && preprocessor_info.parameters) {
      const newParams: { [key: string]: any } = {};
      for (const [paramName, paramInfo] of Object.entries(preprocessor_info.parameters)) {
        const paramData = paramInfo as any;
        
        // Use current value if available, otherwise use default
        if (current_params && current_params[paramName] !== undefined) {
          newParams[paramName] = current_params[paramName];
        } else if (paramData.default !== undefined) {
          newParams[paramName] = paramData.default;
        } else {
          // Set reasonable defaults based on type
          switch (paramData.type) {
            case 'bool': newParams[paramName] = false; break;
            case 'int': newParams[paramName] = paramData.range ? paramData.range[0] : 0; break;
            case 'float': newParams[paramName] = paramData.range ? paramData.range[0] : 0.0; break;
            default: newParams[paramName] = ''; break;
          }
        }
      }
      preprocessorParams[controlnet_index] = newParams;
      console.log('ControlNetConfig: Initialized params for CN', controlnet_index, ':', newParams);
    }
    
    // Force reactivity by creating new objects
    currentPreprocessors = { ...currentPreprocessors };
    preprocessorInfos = { ...preprocessorInfos };
    preprocessorParams = { ...preprocessorParams };
    
    console.log('ControlNetConfig: State after change:', { 
      currentPreprocessors, 
      preprocessorInfos: Object.keys(preprocessorInfos), 
      preprocessorParams: Object.keys(preprocessorParams) 
    });
  }

  function handleParametersUpdated(event: CustomEvent) {
    const { controlnet_index, parameters } = event.detail;
    preprocessorParams[controlnet_index] = { ...preprocessorParams[controlnet_index], ...parameters };
    console.log('ControlNetConfig: Parameters updated:', { controlnet_index, parameters });
  }
  
  // Initialize preprocessor states when controlnet info is available
  $: if (controlnetInfo && controlnetInfo.controlnets) {
    controlnetInfo.controlnets.forEach(async (controlnet: any, index: number) => {
      if (controlnet.preprocessor && !currentPreprocessors[index]) {
        currentPreprocessors[index] = controlnet.preprocessor;
        
        // Also initialize parameters by fetching current values
        try {
          const response = await fetch(`/api/preprocessors/current-params/${index}`);
          if (response.ok) {
            const data = await response.json();
            if (data.parameters && Object.keys(data.parameters).length > 0) {
              preprocessorParams[index] = { ...data.parameters };
              // Force reactivity
              preprocessorParams = { ...preprocessorParams };
              console.log('ControlNetConfig: Loaded initial params for CN', index, ':', data.parameters);
            }
          }
        } catch (err) {
          console.warn('ControlNetConfig: Failed to load initial params for CN', index, ':', err);
        }
      }
    });
  }
</script>

<div class="space-y-4">
  
  <!-- ControlNet Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showControlNet = !showControlNet}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <div class="flex justify-between items-center w-full">
        <h4 class="text-sm font-semibold">ControlNet</h4>
        <div class="flex items-center gap-2">
          <Button on:click={handleHelpClick} classList="text-xs px-2 py-1">
            Help
          </Button>
          <span class="text-sm">{showControlNet ? '−' : '+'}</span>
        </div>
      </div>
    </button>
    {#if showControlNet}
      <div class="p-3">
        <!-- ControlNet Status -->
        <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded mb-3">
          {#if controlnetInfo?.enabled}
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-sm font-medium text-green-800 dark:text-green-200">ControlNet Enabled</span>
          {:else}
            <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">Standard Mode</span>
          {/if}
        </div>

        <!-- ControlNet Strength Controls -->
        {#if controlnetInfo?.enabled && controlnetInfo?.controlnets?.length > 0}
          <div class="space-y-3">
            <h5 class="text-sm font-medium">ControlNet Configuration</h5>
            {#each controlnetInfo.controlnets as controlnet}
              <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-sm font-semibold truncate">
                    {controlnet.name}
                  </span>
                  <span class="text-xs text-gray-600 dark:text-gray-400">
                    Index: {controlnet.index}
                  </span>
                </div>
                
                <!-- Strength Control -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-medium">Strength</span>
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
                </div>
                
                <!-- Preprocessor Selector -->
                <div class="space-y-2">
                  <PreprocessorSelector
                    controlnetIndex={controlnet.index}
                    currentPreprocessor={currentPreprocessors[controlnet.index] || controlnet.preprocessor || 'passthrough'}
                    on:preprocessorChanged={handlePreprocessorChanged}
                  />
                </div>
                
                <!-- Preprocessor Parameters -->
                <div class="border-t border-gray-200 dark:border-gray-600 pt-2">
                  <PreprocessorParams
                    controlnetIndex={controlnet.index}
                    preprocessorInfo={preprocessorInfos[controlnet.index] || {}}
                    currentParams={preprocessorParams[controlnet.index] || {}}
                    on:parametersUpdated={handleParametersUpdated}
                  />
                </div>
              </div>
            {/each}
          </div>
        {:else if controlnetInfo?.enabled}
          <p class="text-xs text-gray-600 dark:text-gray-400">
            ControlNet enabled but no networks found.
          </p>
        {:else}
          <p class="text-xs text-gray-600 dark:text-gray-400">
            Load pipeline configuration to enable ControlNet.
          </p>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Timesteps Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showTimesteps = !showTimesteps}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <h4 class="text-sm font-semibold">Timesteps</h4>
      <span class="text-sm">{showTimesteps ? '−' : '+'}</span>
    </button>
    {#if showTimesteps}
      <div class="p-3">
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
    {/if}
  </div>

  <!-- Streaming Parameters Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showStreamingParams = !showStreamingParams}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <h4 class="text-sm font-semibold">Streaming Parameters</h4>
      <span class="text-sm">{showStreamingParams ? '−' : '+'}</span>
    </button>
    {#if showStreamingParams}
      <div class="p-3">
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
                max="3.0"
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
    {/if}
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