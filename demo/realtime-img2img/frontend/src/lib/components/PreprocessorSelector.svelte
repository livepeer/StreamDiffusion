<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { get } from 'svelte/store';

  export let controlnetIndex: number = 0;
  export let currentPreprocessor: string = 'passthrough';
  
  const dispatch = createEventDispatcher();

  let preprocessorsInfo: any = {};
  let availablePreprocessors: string[] = [];
  let loading = true;
  let error = '';

  onMount(async () => {
    await loadPreprocessorsInfo();
  });

  async function loadPreprocessorsInfo() {
    try {
      loading = true;
      error = '';
      const response = await fetch('/api/preprocessors/info');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      preprocessorsInfo = data.preprocessors || {};
      availablePreprocessors = data.available || [];
      
      console.log('PreprocessorSelector: Loaded preprocessors:', availablePreprocessors);
      console.log('PreprocessorSelector: Preprocessors info:', preprocessorsInfo);
      
      // Emit initial preprocessor info if we have a current preprocessor
      if (currentPreprocessor && preprocessorsInfo[currentPreprocessor]) {
        console.log('PreprocessorSelector: Emitting initial preprocessor info for:', currentPreprocessor);
        
        // Fetch current parameter values
        try {
          const paramsResponse = await fetch(`/api/preprocessors/current-params/${controlnetIndex}`);
          if (paramsResponse.ok) {
            const paramsData = await paramsResponse.json();
            console.log('PreprocessorSelector: Current params loaded:', paramsData);
            
            dispatch('preprocessorChanged', {
              controlnet_index: controlnetIndex,
              preprocessor: currentPreprocessor,
              preprocessor_info: preprocessorsInfo[currentPreprocessor],
              current_params: paramsData.parameters || {}
            });
          } else {
            // Fallback without current params
            dispatch('preprocessorChanged', {
              controlnet_index: controlnetIndex,
              preprocessor: currentPreprocessor,
              preprocessor_info: preprocessorsInfo[currentPreprocessor]
            });
          }
        } catch (err) {
          console.warn('PreprocessorSelector: Failed to load current params:', err);
          // Fallback without current params
          dispatch('preprocessorChanged', {
            controlnet_index: controlnetIndex,
            preprocessor: currentPreprocessor,
            preprocessor_info: preprocessorsInfo[currentPreprocessor]
          });
        }
      }
    } catch (err) {
      console.error('PreprocessorSelector: Failed to load preprocessors:', err);
      error = `Failed to load preprocessors: ${err instanceof Error ? err.message : 'Unknown error'}`;
    } finally {
      loading = false;
    }
  }

  async function handlePreprocessorChange() {
    try {
      console.log(`PreprocessorSelector: Switching to ${currentPreprocessor} for ControlNet ${controlnetIndex}`);
      
      const response = await fetch('/api/preprocessors/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          controlnet_index: controlnetIndex,
          preprocessor: currentPreprocessor,
          preprocessor_params: {}
        })
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || 'Failed to switch preprocessor');
      }

      const result = await response.json();
      console.log('PreprocessorSelector: Successfully switched preprocessor:', result);
      
      // Fetch current parameter values after switching
      let currentParams = {};
      try {
        const paramsResponse = await fetch(`/api/preprocessors/current-params/${controlnetIndex}`);
        if (paramsResponse.ok) {
          const paramsData = await paramsResponse.json();
          currentParams = paramsData.parameters || {};
          console.log('PreprocessorSelector: Current params after switch:', currentParams);
        }
      } catch (paramErr) {
        console.warn('PreprocessorSelector: Failed to load current params after switch:', paramErr);
      }
      
      // Emit event to parent with new preprocessor info and current params
      dispatch('preprocessorChanged', {
        controlnet_index: controlnetIndex,
        preprocessor: currentPreprocessor,
        preprocessor_info: preprocessorsInfo[currentPreprocessor] || {},
        current_params: currentParams
      });

    } catch (err) {
      console.error('PreprocessorSelector: Failed to switch preprocessor:', err);
      error = `Failed to switch preprocessor: ${err instanceof Error ? err.message : 'Unknown error'}`;
    }
  }

  function getPreprocessorDisplayName(preprocessorName: string): string {
    return preprocessorsInfo[preprocessorName]?.display_name || preprocessorName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
</script>

<div class="space-y-2">
  <div class="flex items-center gap-3">
    <label for="preprocessor-selector-{controlnetIndex}" class="text-sm font-medium flex-shrink-0">
      Preprocessor:
    </label>
    
    {#if loading}
      <div class="text-sm text-gray-500">Loading...</div>
    {:else if error}
      <div class="text-sm text-red-500" title={error}>Error</div>
    {:else if availablePreprocessors.length > 0}
      <select
        bind:value={currentPreprocessor}
        on:change={handlePreprocessorChange}
        id="preprocessor-selector-{controlnetIndex}"
        class="text-sm flex-1 cursor-pointer rounded-md border border-gray-300 dark:border-gray-600 p-2 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
      >
        {#each availablePreprocessors as preprocessor}
          <option value={preprocessor} class="bg-white dark:bg-gray-800 text-gray-900 dark:text-white ">
            {getPreprocessorDisplayName(preprocessor)}
          </option>
        {/each}
      </select>
    {:else}
      <div class="text-sm text-gray-500">No preprocessors available</div>
    {/if}
  </div>
  
  {#if currentPreprocessor && preprocessorsInfo[currentPreprocessor] && preprocessorsInfo[currentPreprocessor].description}
    <div class="text-xs text-gray-600 dark:text-gray-400 pl-1">
      {preprocessorsInfo[currentPreprocessor].description}
    </div>
  {/if}
</div>

{#if error}
  <div class="text-sm text-red-500 mt-1">{error}</div>
{/if}