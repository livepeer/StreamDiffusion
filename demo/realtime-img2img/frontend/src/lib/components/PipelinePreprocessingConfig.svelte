<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import InputRange from './InputRange.svelte';
  import ProcessorSelector from './ProcessorSelector.svelte';
  import ProcessorParams from './ProcessorParams.svelte';

  export let pipelinePreprocessingInfo: any = null;

  const dispatch = createEventDispatcher();

  let showPipelinePreprocessing: boolean = true;
  let showPreprocessorParams: boolean = true;
  
  // Processor state
  let currentProcessors: { [index: number]: string } = {};
  let processorInfos: { [index: number]: any } = {};
  let processorParams: { [index: number]: { [key: string]: any } } = {};

  async function addPipelinePreprocessor() {
    try {
      const response = await fetch('/api/pipeline-preprocessing/add', {
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
        console.error('addPipelinePreprocessor: Failed to add processor:', result.detail);
      } else {
        console.log('addPipelinePreprocessor: Successfully added processor');
        // Refresh the pipeline preprocessing info
        dispatch('refresh');
      }
    } catch (error) {
      console.error('addPipelinePreprocessor: Add failed:', error);
    }
  }

  async function removePipelinePreprocessor(index: number) {
    try {
      const response = await fetch(`/api/pipeline-preprocessing/remove/${index}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('removePipelinePreprocessor: Failed to remove processor:', result.detail);
      } else {
        console.log('removePipelinePreprocessor: Successfully removed processor');
        // Clean up local state
        delete currentProcessors[index];
        delete processorInfos[index];
        delete processorParams[index];
        
        // Force reactivity
        currentProcessors = { ...currentProcessors };
        processorInfos = { ...processorInfos };
        processorParams = { ...processorParams };
        
        // Refresh the pipeline preprocessing info
        dispatch('refresh');
      }
    } catch (error) {
      console.error('removePipelinePreprocessor: Remove failed:', error);
    }
  }

  async function togglePipelinePreprocessorEnabled(index: number, enabled: boolean) {
    try {
      const response = await fetch('/api/pipeline-preprocessing/toggle', {
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
        console.error('togglePipelinePreprocessorEnabled: Failed to toggle processor:', result.detail);
      } else {
        console.log('togglePipelinePreprocessorEnabled: Successfully toggled processor');
      }
    } catch (error) {
      console.error('togglePipelinePreprocessorEnabled: Toggle failed:', error);
    }
  }

  function handleProcessorChanged(event: CustomEvent) {
    const { processor_index, processor, processor_info, current_params } = event.detail;
    console.log('PipelinePreprocessingConfig: handleProcessorChanged called with:', event.detail);
    
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
      console.log('PipelinePreprocessingConfig: Initialized params for processor', processor_index, ':', newParams);
    }
    
    // Force reactivity by creating new objects
    currentProcessors = { ...currentProcessors };
    processorInfos = { ...processorInfos };
    processorParams = { ...processorParams };
    
    console.log('PipelinePreprocessingConfig: State after change:', { 
      processorInfos: Object.keys(processorInfos), 
      processorParams: Object.keys(processorParams) 
    });
  }

  function handleParametersUpdated(event: CustomEvent) {
    const { processor_index, parameters } = event.detail;
    processorParams[processor_index] = { ...processorParams[processor_index], ...parameters };
    console.log('PipelinePreprocessingConfig: Parameters updated:', { processor_index, parameters });
  }
  
  // Clear processor state when pipeline preprocessing info changes (e.g., new YAML uploaded)
  let lastPipelinePreprocessingSignature = '';
  
  // Initialize processor states when pipeline preprocessing info is available
  $: if (pipelinePreprocessingInfo && pipelinePreprocessingInfo.processors) {
    // Create a signature based on processor names and indices to detect changes
    const currentSignature = pipelinePreprocessingInfo.processors.map((p: any) => `${p.index}:${p.name}`).join(',');
    
    // If the signature changed, clear state (new YAML or reordering)
    if (currentSignature !== lastPipelinePreprocessingSignature && lastPipelinePreprocessingSignature !== '') {
      console.log('PipelinePreprocessingConfig: Pipeline preprocessing configuration changed, clearing processor state');
      console.log('PipelinePreprocessingConfig: Old signature:', lastPipelinePreprocessingSignature);
      console.log('PipelinePreprocessingConfig: New signature:', currentSignature);
      currentProcessors = {};
      processorInfos = {};
      processorParams = {};
    }
    lastPipelinePreprocessingSignature = currentSignature;
    
    pipelinePreprocessingInfo.processors.forEach(async (processor: any, index: number) => {
      if (processor.name && !currentProcessors[index]) {
        currentProcessors[index] = processor.name;
        
        // Also initialize parameters by fetching current values
        try {
          const response = await fetch(`/api/pipeline-preprocessing/current-params/${index}`);
          if (response.ok) {
            const data = await response.json();
            if (data.parameters && Object.keys(data.parameters).length > 0) {
              processorParams[index] = { ...data.parameters };
              // Force reactivity
              processorParams = { ...processorParams };
              console.log('PipelinePreprocessingConfig: Loaded initial params for processor', index, ':', data.parameters);
            }
          }
        } catch (err) {
          console.warn('PipelinePreprocessingConfig: Failed to load initial params for processor', index, ':', err);
        }
      }
    });
  }
</script>

<div class="space-y-4">
  
  <!-- Pipeline Preprocessing Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
    <button
      class="flex items-center justify-between w-full text-left focus:outline-none"
      on:click={() => showPipelinePreprocessing = !showPipelinePreprocessing}
    >
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        Pipeline Preprocessing
      </h3>
      <svg class="w-5 h-5 text-gray-500 transform transition-transform {showPipelinePreprocessing ? 'rotate-180' : ''}" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
      </svg>
    </button>
    
    {#if showPipelinePreprocessing}
      <div class="mt-4 space-y-4">
        {#if pipelinePreprocessingInfo && pipelinePreprocessingInfo.processors && pipelinePreprocessingInfo.processors.length > 0}
          {#each pipelinePreprocessingInfo.processors as processor, index}
            <div class="border border-gray-200 dark:border-gray-600 rounded-lg p-4 bg-gray-50 dark:bg-gray-700">
              <div class="flex items-center justify-between mb-4">
                <h4 class="text-md font-medium text-gray-900 dark:text-white">
                  Pipeline Preprocessor {index + 1}
                </h4>
                <div class="flex items-center gap-2">
                  <!-- Enabled Toggle -->
                  <label class="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={processor.enabled}
                      on:change={(e) => togglePipelinePreprocessorEnabled(index, (e.target as HTMLInputElement).checked)}
                      class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    Enabled
                  </label>
                  
                  <!-- Remove Button -->
                  <Button
                    classList="bg-red-600 hover:bg-red-700 text-white px-3 py-1 text-sm"
                    on:click={() => removePipelinePreprocessor(index)}
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
                  apiEndpoint="/api/pipeline-preprocessing"
                  processorType="pipeline preprocessor"
                  on:processorChanged={handleProcessorChanged}
                />
              </div>
              
              <!-- Processor Parameters -->
              {#if processorInfos[index] && showPreprocessorParams}
                <div class="mt-4">
                  <ProcessorParams
                    processorIndex={index}
                    processorInfo={processorInfos[index]}
                    currentParams={processorParams[index] || {}}
                    apiEndpoint="/api/pipeline-preprocessing"
                    processorType="pipeline preprocessor"
                    on:parametersUpdated={handleParametersUpdated}
                  />
                </div>
              {/if}
            </div>
          {/each}
        {:else}
          <div class="text-sm text-gray-500 dark:text-gray-400 py-4 text-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
            No pipeline preprocessors configured. Add one to get started.
          </div>
        {/if}
        
        <!-- Add Pipeline Preprocessor Button -->
        <div class="pt-2">
          <Button
            classList="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 text-sm"
            on:click={addPipelinePreprocessor}
          >
            Add Pipeline Preprocessor
          </Button>
        </div>
      </div>
    {/if}
  </div>
</div>
