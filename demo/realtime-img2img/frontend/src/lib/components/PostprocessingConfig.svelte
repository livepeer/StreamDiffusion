<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import InputRange from './InputRange.svelte';
  import ProcessorSelector from './ProcessorSelector.svelte';
  import ProcessorParams from './ProcessorParams.svelte';

  export let postprocessingInfo: any = null;

  const dispatch = createEventDispatcher();

  let showPostprocessing: boolean = true;
  let showProcessorParams: boolean = true;
  
  // Processor state
  let currentProcessors: { [index: number]: string } = {};
  let processorInfos: { [index: number]: any } = {};
  let processorParams: { [index: number]: { [key: string]: any } } = {};

  async function addPostprocessor() {
    try {
      const response = await fetch('/api/postprocessing/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          processor: 'passthrough',
          enabled: true,
          processor_params: {}
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('addPostprocessor: Failed to add processor:', result.detail);
      } else {
        console.log('addPostprocessor: Successfully added processor');
        // Refresh the postprocessing info
        dispatch('refresh');
      }
    } catch (error) {
      console.error('addPostprocessor: Add failed:', error);
    }
  }

  async function removePostprocessor(index: number) {
    try {
      const response = await fetch(`/api/postprocessing/remove/${index}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('removePostprocessor: Failed to remove processor:', result.detail);
      } else {
        console.log('removePostprocessor: Successfully removed processor');
        // Clean up local state
        delete currentProcessors[index];
        delete processorInfos[index];
        delete processorParams[index];
        
        // Force reactivity
        currentProcessors = { ...currentProcessors };
        processorInfos = { ...processorInfos };
        processorParams = { ...processorParams };
        
        // Refresh the postprocessing info
        dispatch('refresh');
      }
    } catch (error) {
      console.error('removePostprocessor: Remove failed:', error);
    }
  }

  async function togglePostprocessorEnabled(index: number, enabled: boolean) {
    try {
      const response = await fetch('/api/postprocessing/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          processor_index: index,
          enabled: enabled
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('togglePostprocessorEnabled: Failed to toggle processor:', result.detail);
      } else {
        console.log('togglePostprocessorEnabled: Successfully toggled processor');
      }
    } catch (error) {
      console.error('togglePostprocessorEnabled: Toggle failed:', error);
    }
  }

  function handleProcessorChanged(event: CustomEvent) {
    const { processor_index, processor, processor_info, current_params } = event.detail;
    console.log('PostprocessingConfig: handleProcessorChanged called with:', event.detail);
    
    currentProcessors[processor_index] = processor;
    processorInfos[processor_index] = processor_info;
    
    // Initialize parameters with current values or defaults
    if (processor_info && processor_info.parameters) {
      const newParams: { [key: string]: any } = {};
      for (const [paramName, paramInfo] of Object.entries(processor_info.parameters)) {
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
      processorParams[processor_index] = newParams;
      console.log('PostprocessingConfig: Initialized params for processor', processor_index, ':', newParams);
    }
    
    // Force reactivity by creating new objects
    currentProcessors = { ...currentProcessors };
    processorInfos = { ...processorInfos };
    processorParams = { ...processorParams };
    
    console.log('PostprocessingConfig: State after change:', { 
      processorInfos: Object.keys(processorInfos), 
      processorParams: Object.keys(processorParams) 
    });
  }

  function handleParametersUpdated(event: CustomEvent) {
    const { processor_index, parameters } = event.detail;
    processorParams[processor_index] = { ...processorParams[processor_index], ...parameters };
    console.log('PostprocessingConfig: Parameters updated:', { processor_index, parameters });
  }
  
  // Clear processor state when postprocessing info changes (e.g., new YAML uploaded)
  let lastPostprocessingSignature = '';
  
  // Initialize processor states when postprocessing info is available
  $: if (postprocessingInfo && postprocessingInfo.processors) {
    // Create a signature based on processor names and indices to detect changes
    const currentSignature = postprocessingInfo.processors.map((p: any) => `${p.index}:${p.name}`).join(',');
    
    // If the signature changed, clear state (new YAML or reordering)
    if (currentSignature !== lastPostprocessingSignature && lastPostprocessingSignature !== '') {
      console.log('PostprocessingConfig: Postprocessing configuration changed, clearing processor state');
      console.log('PostprocessingConfig: Old signature:', lastPostprocessingSignature);
      console.log('PostprocessingConfig: New signature:', currentSignature);
      currentProcessors = {};
      processorInfos = {};
      processorParams = {};
    }
    lastPostprocessingSignature = currentSignature;
    
    postprocessingInfo.processors.forEach(async (processor: any, index: number) => {
      if (processor.name && !currentProcessors[index]) {
        currentProcessors[index] = processor.name;
        
        // Also initialize parameters by fetching current values
        try {
          const response = await fetch(`/api/postprocessing/current-params/${index}`);
          if (response.ok) {
            const data = await response.json();
            if (data.parameters && Object.keys(data.parameters).length > 0) {
              processorParams[index] = { ...data.parameters };
              // Force reactivity
              processorParams = { ...processorParams };
              console.log('PostprocessingConfig: Loaded initial params for processor', index, ':', data.parameters);
            }
          }
        } catch (err) {
          console.warn('PostprocessingConfig: Failed to load initial params for processor', index, ':', err);
        }
      }
    });
  }
</script>

<div class="space-y-4">
  
  <!-- Postprocessing Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
    <button
      class="flex items-center justify-between w-full text-left focus:outline-none"
      on:click={() => showPostprocessing = !showPostprocessing}
    >
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        Postprocessing
      </h3>
      <svg class="w-5 h-5 text-gray-500 transform transition-transform {showPostprocessing ? 'rotate-180' : ''}" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
      </svg>
    </button>
    
    {#if showPostprocessing}
      <div class="mt-4 space-y-4">
        {#if postprocessingInfo && postprocessingInfo.processors && postprocessingInfo.processors.length > 0}
          {#each postprocessingInfo.processors as processor, index}
            <div class="border border-gray-200 dark:border-gray-600 rounded-lg p-4 bg-gray-50 dark:bg-gray-700">
              <div class="flex items-center justify-between mb-4">
                <h4 class="text-md font-medium text-gray-900 dark:text-white">
                  Postprocessor {index + 1}
                </h4>
                <div class="flex items-center gap-2">
                  <!-- Enabled Toggle -->
                  <label class="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={processor.enabled}
                      on:change={(e) => togglePostprocessorEnabled(index, (e.target as HTMLInputElement).checked)}
                      class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    Enabled
                  </label>
                  
                  <!-- Remove Button -->
                  <Button
                    classList="bg-red-600 hover:bg-red-700 text-white px-3 py-1 text-sm"
                    on:click={() => removePostprocessor(index)}
                  >
                    Remove
                  </Button>
                </div>
              </div>
              
              <!-- Processor Selector -->
              <div class="mb-4">
                <ProcessorSelector
                  processorIndex={index}
                  currentProcessor={currentProcessors[index] || processor.name || 'passthrough'}
                  apiEndpoint="/api/postprocessing"
                  processorType="postprocessor"
                  on:processorChanged={handleProcessorChanged}
                />
              </div>
              
              <!-- Processor Parameters -->
              {#if processorInfos[index] && showProcessorParams}
                <div class="mt-4">
                  <ProcessorParams
                    processorIndex={index}
                    processorInfo={processorInfos[index]}
                    currentParams={processorParams[index] || {}}
                    apiEndpoint="/api/postprocessing"
                    processorType="postprocessor"
                    on:parametersUpdated={handleParametersUpdated}
                  />
                </div>
              {/if}
            </div>
          {/each}
        {:else}
          <div class="text-sm text-gray-500 dark:text-gray-400 py-4 text-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
            No postprocessors configured. Add one to get started.
          </div>
        {/if}
        
        <!-- Add Postprocessor Button -->
        <div class="pt-2">
          <Button
            classList="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 text-sm"
            on:click={addPostprocessor}
          >
            Add Postprocessor
          </Button>
        </div>
      </div>
    {/if}
  </div>
</div>
