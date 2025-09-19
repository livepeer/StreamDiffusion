<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  // Simple debounce function
  function debounce(func: Function, wait: number) {
    let timeout: any;
    return function executedFunction(...args: any[]) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  export let processorIndex: number = 0;
  export let processorInfo: any = {};
  export let currentParams: { [key: string]: any } = {};
  export let apiEndpoint: string = '/api/preprocessors'; // e.g., '/api/preprocessors', '/api/pipeline-preprocessing', '/api/postprocessing'
  export let processorType: string = 'preprocessor'; // Used for logging
  
  const dispatch = createEventDispatcher();

  let error = '';
  
  // Immediate UI update with debounced server update
  const debouncedServerUpdate = debounce(async (updatedParams: { [key: string]: any }) => {
    try {
      error = '';
      console.log(`ProcessorParams: Updating params for ${processorType} ${processorIndex}:`, updatedParams);
      
      // Use different parameter names based on the endpoint
      let requestBody: any;
      if (apiEndpoint === '/api/preprocessors') {
        // ControlNet preprocessors expect different parameter names
        requestBody = {
          controlnet_index: processorIndex,
          params: updatedParams
        };
      } else {
        // Pipeline hooks use the standard parameter names
        requestBody = {
          processor_index: processorIndex,
          processor_params: updatedParams
        };
      }

      const response = await fetch(`${apiEndpoint}/update-params`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const result = await response.json();
        
        // Check if this is a "no processors configured" error
        if (result.detail && result.detail.includes('No processors configured')) {
          // This is expected - processor hasn't been added yet, just show a helpful message
          console.log(`ProcessorParams: Processor not configured yet - ${result.detail}`);
          error = 'Add a processor first before configuring parameters';
          return; // Don't throw error, just show message
        }
        
        throw new Error(result.detail || 'Failed to update parameters');
      }

      const result = await response.json();
      console.log(`ProcessorParams: Successfully updated parameters:`, result);

    } catch (err) {
      console.error(`ProcessorParams: Failed to update parameters:`, err);
      error = `Failed to update parameters: ${err instanceof Error ? err.message : 'Unknown error'}`;
    }
  }, 100); // Reduced to 100ms for better responsiveness

  function handleParameterChange(paramName: string, value: any) {
    // Update UI immediately
    currentParams = { ...currentParams, [paramName]: value };
    
    // Notify parent immediately for UI updates
    dispatch('parametersUpdated', {
      processor_index: processorIndex,
      parameters: { [paramName]: value }
    });
    
    // Send debounced update to server
    debouncedServerUpdate(currentParams);
  }

  function getInputType(paramInfo: any): string {
    if (paramInfo.type === 'bool') return 'checkbox';
    if (paramInfo.options && Array.isArray(paramInfo.options)) return 'select';
    if (paramInfo.type === 'int' || paramInfo.type === 'float') return 'range';
    return 'text';
  }

  function formatParameterName(paramName: string): string {
    return paramName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  function getStepValue(paramInfo: any): number {
    if (paramInfo.step !== undefined) return paramInfo.step;
    if (paramInfo.type === 'int') return 1;
    if (paramInfo.type === 'float') return 0.1;
    return 1;
  }

  function getMinValue(paramInfo: any): number {
    return paramInfo.range ? paramInfo.range[0] : 0;
  }

  function getMaxValue(paramInfo: any): number {
    return paramInfo.range ? paramInfo.range[1] : 100;
  }

  function getDisplayValue(paramName: string, paramInfo: any): any {
    if (currentParams[paramName] !== undefined) {
      return currentParams[paramName];
    }
    
    // Use default value if not set
    const defaultValue = paramInfo.default !== undefined ? paramInfo.default : getDefaultValue(paramInfo);
    return defaultValue;
  }

  // Create reactive parameter values for template binding - depends on currentParams for reactivity
  $: parameterValues = processorInfo?.parameters && currentParams ? 
    Object.fromEntries(
      Object.entries(processorInfo.parameters).map(([paramName, paramInfo]) => {
        // Explicitly reference currentParams to ensure reactivity
        const value = currentParams[paramName] !== undefined ? 
          currentParams[paramName] : 
          (paramInfo as any).default !== undefined ? 
            (paramInfo as any).default : 
            getDefaultValue(paramInfo as any);
        return [paramName, value];
      })
    ) : {};
  
  // Initialize parameters when component mounts or processorInfo changes
  let lastProcessorName = '';
  let initialized = false;
  
  async function initializeParams() {
    if (!processorInfo || !processorInfo.parameters || initialized) return;
    
    console.log(`ProcessorParams: Initializing parameters for processor ${processorIndex}...`);
    
    // Always try to fetch current values from server first (config is source of truth)
    let serverParams: { [key: string]: any } = {};
    try {
      const response = await fetch(`${apiEndpoint}/current-params/${processorIndex}`);
      if (response.ok) {
        const data = await response.json();
        if (data.parameters && Object.keys(data.parameters).length > 0) {
          serverParams = { ...data.parameters };
          console.log(`ProcessorParams: Loaded current params from server:`, serverParams);
        }
      }
    } catch (err) {
      console.warn(`ProcessorParams: Failed to fetch current params:`, err);
    }
    
    // Build complete parameter set: server params take priority, then defaults
    const completeParams: { [key: string]: any } = {};
    
    for (const [paramName, paramInfo] of Object.entries(processorInfo.parameters)) {
      const paramData = paramInfo as any;
      
      // Priority: server value > current value > default value > type default
      if (serverParams[paramName] !== undefined) {
        completeParams[paramName] = serverParams[paramName];
        console.log(`ProcessorParams: Using server value for ${paramName}:`, serverParams[paramName]);
      } else if (currentParams[paramName] !== undefined) {
        completeParams[paramName] = currentParams[paramName];
      } else if (paramData.default !== undefined) {
        completeParams[paramName] = paramData.default;
        console.log(`ProcessorParams: Using default value for ${paramName}:`, paramData.default);
      } else {
        // Fallback to type-based defaults
        completeParams[paramName] = getDefaultValue(paramData);
      }
    }
    
    // Update currentParams completely
    currentParams = { ...completeParams };
    console.log(`ProcessorParams: Final initialized params:`, currentParams);
    
    initialized = true;
  }
  
  $: {
    if (processorInfo && processorInfo.display_name && processorInfo.display_name !== lastProcessorName) {
      console.log(`ProcessorParams: processorInfo changed:`, processorInfo);
      console.log(`ProcessorParams: currentParams:`, currentParams);
      if (processorInfo.parameters) {
        console.log(`ProcessorParams: Available parameters:`, Object.keys(processorInfo.parameters));
      }
      lastProcessorName = processorInfo.display_name;
      initialized = false;
      initializeParams();
    }
  }
  
  // Also trigger initialization when component is mounted fresh (config upload scenario)
  $: if (processorInfo && processorInfo.parameters && !initialized) {
    console.log(`ProcessorParams: Fresh initialization triggered for processor ${processorIndex}`);
    initializeParams();
  }

  function getDefaultValue(paramInfo: any): any {
    switch (paramInfo.type) {
      case 'bool': return false;
      case 'int': return paramInfo.range ? paramInfo.range[0] : 0;
      case 'float': return paramInfo.range ? paramInfo.range[0] : 0.0;
      default: return '';
    }
  }
</script>

{#if processorInfo && processorInfo.parameters && Object.keys(processorInfo.parameters).length > 0}
  <div class="space-y-3">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
      Processor Parameters
    </h4>
    
    {#each Object.entries(processorInfo.parameters) as [paramName, paramInfo]}
      {#if paramName !== 'device' && paramName !== 'dtype' && paramName !== 'image_width' && paramName !== 'image_height'}
        <div class="parameter-control">
          {#if getInputType(paramInfo as any) === 'checkbox'}
            <!-- Boolean parameter -->
            <div class="flex items-center gap-3">
              <input
                type="checkbox"
                id="param-{processorIndex}-{paramName}"
                checked={(parameterValues[paramName] !== undefined ? parameterValues[paramName] : getDisplayValue(paramName, paramInfo as any)) || false}
                on:change={(e) => handleParameterChange(paramName, (e.target as HTMLInputElement).checked)}
                class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label for="param-{processorIndex}-{paramName}" class="text-sm font-medium">
                {formatParameterName(paramName)}
              </label>
            </div>
            
          {:else if getInputType(paramInfo as any) === 'select'}
            <!-- Select parameter with options -->
            <div class="space-y-1">
              <label for="param-{processorIndex}-{paramName}" class="text-sm font-medium block">
                {formatParameterName(paramName)}
              </label>
              <select
                id="param-{processorIndex}-{paramName}"
                value={parameterValues[paramName] !== undefined ? parameterValues[paramName] : getDisplayValue(paramName, paramInfo as any)}
                on:change={(e) => handleParameterChange(paramName, (e.target as HTMLSelectElement).value)}
                class="w-full rounded-md border border-gray-300 dark:border-gray-600 px-2 py-1 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white cursor-pointer"
              >
                {#each (paramInfo as any).options as option}
                  <option value={option} class="bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                    {option}
                  </option>
                {/each}
              </select>
            </div>
            
          {:else if getInputType(paramInfo as any) === 'range'}
            <!-- Numeric parameter with range slider -->
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <label for="param-{processorIndex}-{paramName}" class="text-sm font-medium">
                  {formatParameterName(paramName)}
                </label>
                <input
                  type="number"
                  step={getStepValue(paramInfo as any)}
                  value={parameterValues[paramName] !== undefined ? parameterValues[paramName] : getDisplayValue(paramName, paramInfo as any)}
                  on:input={(e) => handleParameterChange(paramName, (paramInfo as any).type === 'int' ? parseInt((e.target as HTMLInputElement).value) : parseFloat((e.target as HTMLInputElement).value))}
                  class="w-20 rounded-md border border-gray-300 dark:border-gray-600 px-2 py-1 text-center text-xs bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min={getMinValue(paramInfo as any)}
                  max={getMaxValue(paramInfo as any)}
                />
              </div>
              <input
                class="w-full h-2 cursor-pointer appearance-none rounded-lg bg-gray-200 dark:bg-gray-600"
                value={parameterValues[paramName] !== undefined ? parameterValues[paramName] : getDisplayValue(paramName, paramInfo as any)}
                on:input={(e) => handleParameterChange(paramName, (paramInfo as any).type === 'int' ? parseInt((e.target as HTMLInputElement).value) : parseFloat((e.target as HTMLInputElement).value))}
                type="range"
                id="param-{processorIndex}-{paramName}"
                min={getMinValue(paramInfo as any)}
                max={getMaxValue(paramInfo as any)}
                step={getStepValue(paramInfo as any)}
              />
            </div>
            
          {:else}
            <!-- Text parameter -->
            <div class="space-y-1">
              <label for="param-{processorIndex}-{paramName}" class="text-sm font-medium block">
                {formatParameterName(paramName)}
              </label>
              <input
                type="text"
                id="param-{processorIndex}-{paramName}"
                value={(parameterValues[paramName] !== undefined ? parameterValues[paramName] : getDisplayValue(paramName, paramInfo as any)) || ''}
                on:input={(e) => handleParameterChange(paramName, (e.target as HTMLInputElement).value)}
                class="w-full rounded-md border border-gray-300 dark:border-gray-600 px-2 py-1 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder={(paramInfo as any).default || ''}
              />
            </div>
          {/if}
          
          {#if (paramInfo as any).description}
            <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {(paramInfo as any).description}
            </div>
          {/if}
        </div>
      {/if}
    {/each}
  </div>
{:else}
  <div class="text-sm text-gray-500 dark:text-gray-400">
    No configurable parameters for this processor.
  </div>
{/if}

{#if error}
  <div class="text-sm text-red-500 mt-2">{error}</div>
{/if}

<!-- Parameter initialization handled by parent component -->

<style lang="postcss">
  .parameter-control {
    @apply mb-4 last:mb-0;
  }
  
  /* Custom range slider styling */
  input[type='range']::-webkit-slider-runnable-track {
    @apply h-2 cursor-pointer rounded-lg bg-gray-200 dark:bg-gray-600;
  }
  input[type='range']::-webkit-slider-thumb {
    @apply cursor-pointer rounded-lg bg-blue-500 dark:bg-blue-400 h-4 w-4 appearance-none;
  }
  input[type='range']::-moz-range-track {
    @apply cursor-pointer rounded-lg bg-gray-200 dark:bg-gray-600;
  }
  input[type='range']::-moz-range-thumb {
    @apply cursor-pointer rounded-lg bg-blue-500 dark:bg-blue-400 h-4 w-4 border-none;
  }
</style>
