<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';

  export let processorIndex: number = 0;
  export let currentProcessor: string = 'passthrough';
  export let apiEndpoint: string = '/api/preprocessors'; // e.g., '/api/preprocessors', '/api/pipeline-preprocessing', '/api/postprocessing'
  export let processorType: string = 'preprocessor'; // Used for API calls and logging
  
  const dispatch = createEventDispatcher();

  let processorsInfo: any = {};
  let availableProcessors: string[] = [];
  let loading = true;
  let error = '';

  onMount(async () => {
    await loadProcessorsInfo();
  });

  async function loadProcessorsInfo() {
    try {
      loading = true;
      error = '';
      const response = await fetch(`${apiEndpoint}/info`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      processorsInfo = data.preprocessors || {};
      availableProcessors = data.available || [];
      
      console.log(`ProcessorSelector: Loaded ${processorType}s:`, availableProcessors);
      console.log(`ProcessorSelector: ${processorType}s info:`, processorsInfo);
      
      // Emit initial processor info if we have a current processor
      if (currentProcessor && processorsInfo[currentProcessor]) {
        console.log(`ProcessorSelector: Emitting initial ${processorType} info for:`, currentProcessor);
        
        // Fetch current parameter values
        try {
          const paramsResponse = await fetch(`${apiEndpoint}/current-params/${processorIndex}`);
          if (paramsResponse.ok) {
            const paramsData = await paramsResponse.json();
            console.log(`ProcessorSelector: Current params loaded:`, paramsData);
            
            dispatch('processorChanged', {
              processor_index: processorIndex,
              processor: currentProcessor,
              processor_info: processorsInfo[currentProcessor],
              current_params: paramsData.parameters || {}
            });
          } else {
            // Fallback without current params
            dispatch('processorChanged', {
              processor_index: processorIndex,
              processor: currentProcessor,
              processor_info: processorsInfo[currentProcessor]
            });
          }
        } catch (err) {
          console.warn(`ProcessorSelector: Failed to load current params:`, err);
          // Fallback without current params
          dispatch('processorChanged', {
            processor_index: processorIndex,
            processor: currentProcessor,
            processor_info: processorsInfo[currentProcessor]
          });
        }
      }
    } catch (err) {
      console.error(`ProcessorSelector: Failed to load ${processorType}s:`, err);
      error = `Failed to load ${processorType}s: ${err instanceof Error ? err.message : 'Unknown error'}`;
    } finally {
      loading = false;
    }
  }

  async function handleProcessorChange() {
    try {
      console.log(`ProcessorSelector: Switching to ${currentProcessor} for ${processorType} ${processorIndex}`);
      
      const response = await fetch(`${apiEndpoint}/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          processor_index: processorIndex,
          processor: currentProcessor,
          processor_params: {}
        })
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || `Failed to switch ${processorType}`);
      }

      const result = await response.json();
      console.log(`ProcessorSelector: Successfully switched ${processorType}:`, result);
      
      // Fetch current parameter values after switching
      let currentParams = {};
      try {
        const paramsResponse = await fetch(`${apiEndpoint}/current-params/${processorIndex}`);
        if (paramsResponse.ok) {
          const paramsData = await paramsResponse.json();
          currentParams = paramsData.parameters || {};
          console.log(`ProcessorSelector: Current params after switch:`, currentParams);
        }
      } catch (paramErr) {
        console.warn(`ProcessorSelector: Failed to load current params after switch:`, paramErr);
      }
      
      // Emit event to parent with new processor info and current params
      dispatch('processorChanged', {
        processor_index: processorIndex,
        processor: currentProcessor,
        processor_info: processorsInfo[currentProcessor] || {},
        current_params: currentParams
      });

    } catch (err) {
      console.error(`ProcessorSelector: Failed to switch ${processorType}:`, err);
      error = `Failed to switch ${processorType}: ${err instanceof Error ? err.message : 'Unknown error'}`;
    }
  }

  function getProcessorDisplayName(processorName: string): string {
    return processorsInfo[processorName]?.display_name || processorName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
</script>

<div class="space-y-2">
  <div class="flex items-center gap-3">
    <label for="processor-selector-{processorIndex}" class="text-sm font-medium flex-shrink-0">
      Processor:
    </label>
    
    {#if loading}
      <div class="text-sm text-gray-500">Loading...</div>
    {:else if error}
      <div class="text-sm text-red-500" title={error}>Error</div>
    {:else if availableProcessors.length > 0}
      <select
        bind:value={currentProcessor}
        on:change={handleProcessorChange}
        id="processor-selector-{processorIndex}"
        class="text-sm flex-1 cursor-pointer rounded-md border border-gray-300 dark:border-gray-600 p-2 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
      >
        {#each availableProcessors as processor}
          <option value={processor} class="bg-white dark:bg-gray-800 text-gray-900 dark:text-white ">
            {getProcessorDisplayName(processor)}
          </option>
        {/each}
      </select>
    {:else}
      <div class="text-sm text-gray-500">No processors available</div>
    {/if}
  </div>
  
  {#if currentProcessor && processorsInfo[currentProcessor] && processorsInfo[currentProcessor].description}
    <div class="text-xs text-gray-600 dark:text-gray-400 pl-1">
      {processorsInfo[currentProcessor].description}
    </div>
  {/if}
</div>

{#if error}
  <div class="text-sm text-red-500 mt-1">{error}</div>
{/if}
