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

  export let controlnetIndex: number = 0;
  export let preprocessorInfo: any = {};
  export let currentParams: { [key: string]: any } = {};
  
  const dispatch = createEventDispatcher();

  let error = '';
  
  // Immediate UI update with debounced server update
  const debouncedServerUpdate = debounce(async (updatedParams: { [key: string]: any }) => {
    try {
      error = '';
      console.log(`PreprocessorParams: Updating params for ControlNet ${controlnetIndex}:`, updatedParams);
      
      const response = await fetch('/api/preprocessors/update-params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          controlnet_index: controlnetIndex,
          preprocessor_params: updatedParams
        })
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || 'Failed to update parameters');
      }

      const result = await response.json();
      console.log('PreprocessorParams: Successfully updated parameters:', result);

    } catch (err) {
      console.error('PreprocessorParams: Failed to update parameters:', err);
      error = `Failed to update parameters: ${err instanceof Error ? err.message : 'Unknown error'}`;
    }
  }, 100); // Reduced to 100ms for better responsiveness

  function handleParameterChange(paramName: string, value: any) {
    // Update UI immediately
    currentParams = { ...currentParams, [paramName]: value };
    
    // Notify parent immediately for UI updates
    dispatch('parametersUpdated', {
      controlnet_index: controlnetIndex,
      parameters: { [paramName]: value }
    });
    
    // Send debounced update to server
    debouncedServerUpdate(currentParams);
  }

  function getInputType(paramInfo: any): string {
    if (paramInfo.type === 'bool') return 'checkbox';
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
  $: parameterValues = preprocessorInfo?.parameters ? 
    Object.fromEntries(
      Object.entries(preprocessorInfo.parameters).map(([paramName, paramInfo]) => {
        // Explicitly reference currentParams to ensure reactivity
        const value = currentParams[paramName] !== undefined ? 
          currentParams[paramName] : 
          (paramInfo as any).default !== undefined ? 
            (paramInfo as any).default : 
            getDefaultValue(paramInfo as any);
        return [paramName, value];
      })
    ) : {};
  
  // Initialize parameters when component mounts or preprocessorInfo changes
  let lastPreprocessorName = '';
  let initialized = false;
  
  async function initializeParams() {
    if (!preprocessorInfo || !preprocessorInfo.parameters || initialized) return;
    
    console.log('PreprocessorParams: Initializing parameters...');
    
    // Check if currentParams is empty or missing values
    const hasCurrentValues = Object.keys(currentParams).length > 0;
    
    if (!hasCurrentValues) {
      // Try to fetch current values from server
      try {
        const response = await fetch(`/api/preprocessors/current-params/${controlnetIndex}`);
        if (response.ok) {
          const data = await response.json();
          if (data.parameters && Object.keys(data.parameters).length > 0) {
            currentParams = { ...data.parameters };
            console.log('PreprocessorParams: Loaded current params from server:', currentParams);
          }
        }
      } catch (err) {
        console.warn('PreprocessorParams: Failed to fetch current params:', err);
      }
    }
    
    // Fill in any missing parameters with defaults
    const updatedParams = { ...currentParams };
    let hasUpdates = false;
    
    for (const [paramName, paramInfo] of Object.entries(preprocessorInfo.parameters)) {
      if (!(paramName in updatedParams)) {
        const paramData = paramInfo as any;
        if (paramData.default !== undefined) {
          updatedParams[paramName] = paramData.default;
          hasUpdates = true;
        }
      }
    }
    
    if (hasUpdates) {
      currentParams = updatedParams;
      console.log('PreprocessorParams: Added default values:', updatedParams);
    }
    
    initialized = true;
  }
  
  $: {
    if (preprocessorInfo && preprocessorInfo.display_name && preprocessorInfo.display_name !== lastPreprocessorName) {
      console.log('PreprocessorParams: preprocessorInfo changed:', preprocessorInfo);
      console.log('PreprocessorParams: currentParams:', currentParams);
      if (preprocessorInfo.parameters) {
        console.log('PreprocessorParams: Available parameters:', Object.keys(preprocessorInfo.parameters));
      }
      lastPreprocessorName = preprocessorInfo.display_name;
      initialized = false;
      initializeParams();
    }
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

{#if preprocessorInfo && preprocessorInfo.parameters && Object.keys(preprocessorInfo.parameters).length > 0}
  <div class="space-y-3">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
      Preprocessor Parameters
    </h4>
    
    {#each Object.entries(preprocessorInfo.parameters) as [paramName, paramInfo]}
      {#if paramName !== 'device' && paramName !== 'dtype' && paramName !== 'image_width' && paramName !== 'image_height'}
        <div class="parameter-control">
          {#if getInputType(paramInfo as any) === 'checkbox'}
            <!-- Boolean parameter -->
            <div class="flex items-center gap-3">
              <input
                type="checkbox"
                id="param-{controlnetIndex}-{paramName}"
                checked={parameterValues[paramName] || false}
                on:change={(e) => handleParameterChange(paramName, (e.target as HTMLInputElement).checked)}
                class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label for="param-{controlnetIndex}-{paramName}" class="text-sm font-medium">
                {formatParameterName(paramName)}
              </label>
            </div>
            
          {:else if getInputType(paramInfo as any) === 'range'}
            <!-- Numeric parameter with range slider -->
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <label for="param-{controlnetIndex}-{paramName}" class="text-sm font-medium">
                  {formatParameterName(paramName)}
                </label>
                <input
                  type="number"
                  step={getStepValue(paramInfo as any)}
                  value={parameterValues[paramName] || 0}
                  on:input={(e) => handleParameterChange(paramName, (paramInfo as any).type === 'int' ? parseInt((e.target as HTMLInputElement).value) : parseFloat((e.target as HTMLInputElement).value))}
                  class="w-20 rounded-md border border-gray-300 dark:border-gray-600 px-2 py-1 text-center text-xs bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min={getMinValue(paramInfo as any)}
                  max={getMaxValue(paramInfo as any)}
                />
              </div>
              <input
                class="w-full h-2 cursor-pointer appearance-none rounded-lg bg-gray-200 dark:bg-gray-600"
                value={parameterValues[paramName] || 0}
                on:input={(e) => handleParameterChange(paramName, (paramInfo as any).type === 'int' ? parseInt((e.target as HTMLInputElement).value) : parseFloat((e.target as HTMLInputElement).value))}
                type="range"
                id="param-{controlnetIndex}-{paramName}"
                min={getMinValue(paramInfo as any)}
                max={getMaxValue(paramInfo as any)}
                step={getStepValue(paramInfo as any)}
              />
            </div>
            
          {:else}
            <!-- Text parameter -->
            <div class="space-y-1">
              <label for="param-{controlnetIndex}-{paramName}" class="text-sm font-medium block">
                {formatParameterName(paramName)}
              </label>
              <input
                type="text"
                id="param-{controlnetIndex}-{paramName}"
                value={parameterValues[paramName] || ''}
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
    No configurable parameters for this preprocessor.
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