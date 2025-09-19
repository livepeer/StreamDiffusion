<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import ProcessorSelector from './ProcessorSelector.svelte';
  import ProcessorParams from './ProcessorParams.svelte';

  export let hookType: string = 'image_preprocessing'; // image_preprocessing, image_postprocessing, latent_preprocessing, latent_postprocessing
  export let hookInfo: any = null;
  export let skipDiffusion: boolean = false; // Passed from parent

  const dispatch = createEventDispatcher();

  let showHookConfig: boolean = true;
  let showProcessorParams: boolean = true;
  
  // Processor state
  let currentProcessors: { [index: number]: string } = {};
  let processorInfos: { [index: number]: any } = {};
  let processorParams: { [index: number]: { [key: string]: any } } = {};

  // Get display names for different hook types
  function getHookDisplayName(hookType: string): string {
    const displayNames: { [key: string]: string } = {
      'image_preprocessing': 'Image Preprocessing',
      'image_postprocessing': 'Image Postprocessing', 
      'latent_preprocessing': 'Latent Preprocessing',
      'latent_postprocessing': 'Latent Postprocessing'
    };
    return displayNames[hookType] || hookType;
  }

  async function addProcessor() {
    try {
      const response = await fetch(`/api/pipeline-hooks/${hookType}/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          processor_type: 'passthrough',
          enabled: true,
          processor_params: {}
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error(`addProcessor: Failed to add processor:`, result.detail);
      } else {
        console.log(`addProcessor: Successfully added processor`);
        // Refresh the hook info
        dispatch('refresh');
      }
    } catch (error) {
      console.error(`addProcessor: Add failed:`, error);
    }
  }

  async function removeProcessor(index: number) {
    try {
      const response = await fetch(`/api/pipeline-hooks/${hookType}/remove/${index}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const result = await response.json();
        console.error(`removeProcessor: Failed to remove processor:`, result.detail);
      } else {
        console.log(`removeProcessor: Successfully removed processor`);
        // Clean up local state
        delete currentProcessors[index];
        delete processorInfos[index];
        delete processorParams[index];
        
        // Force reactivity
        currentProcessors = { ...currentProcessors };
        processorInfos = { ...processorInfos };
        processorParams = { ...processorParams };
        
        // Refresh the hook info
        dispatch('refresh');
      }
    } catch (error) {
      console.error(`removeProcessor: Remove failed:`, error);
    }
  }

  async function updateProcessorEnabled(index: number, enabled: boolean) {
    try {
      const response = await fetch(`/api/pipeline-hooks/${hookType}/update-params`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          processor_index: index,
          processor_params: { enabled: enabled }
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error(`updateProcessorEnabled: Failed to update processor enabled state:`, result.detail);
      } else {
        console.log(`updateProcessorEnabled: Successfully updated processor enabled state`);
        
        // Update local state immediately for UI responsiveness
        if (hookInfo && hookInfo.processors && hookInfo.processors[index]) {
          hookInfo.processors[index].enabled = enabled;
          // Force reactivity
          hookInfo = { ...hookInfo };
        }
      }
    } catch (error) {
      console.error(`updateProcessorEnabled: Update failed:`, error);
    }
  }

  function toggleSkipDiffusion(enabled: boolean) {
    console.log(`PipelineHooksConfig: Skip diffusion toggle requested:`, enabled);
    dispatch('skipDiffusionChanged', enabled);
  }

  function handleProcessorChanged(event: CustomEvent) {
    const { processor_index, processor, processor_info, current_params } = event.detail;
    console.log(`PipelineHooksConfig: handleProcessorChanged called with:`, event.detail);
    
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
      console.log(`PipelineHooksConfig: Initialized params for processor`, processor_index, ':', newParams);
    }
    
    // Force reactivity by creating new objects
    currentProcessors = { ...currentProcessors };
    processorInfos = { ...processorInfos };
    processorParams = { ...processorParams };
    
    console.log(`PipelineHooksConfig: State after change:`, { 
      processorInfos: Object.keys(processorInfos), 
      processorParams: Object.keys(processorParams) 
    });
  }

  function handleParametersUpdated(event: CustomEvent) {
    const { processor_index, parameters } = event.detail;
    processorParams[processor_index] = { ...processorParams[processor_index], ...parameters };
    console.log(`PipelineHooksConfig: Parameters updated:`, { processor_index, parameters });
  }
  
  // Clear processor state when hook info changes (e.g., new YAML uploaded)
  let lastHookSignature = '';
  

  // Initialize processor states when hook info is available
  $: if (hookInfo && hookInfo.processors) {
    // Create a signature based on processor names and indices to detect changes
    const currentSignature = hookInfo.processors.map((p: any) => `${p.index}:${p.name}`).join(',');
    
    // If the signature changed, clear state (including initial load)
    if (currentSignature !== lastHookSignature) {
      console.log(`PipelineHooksConfig: Hook configuration changed, clearing processor state`);
      console.log(`PipelineHooksConfig: Old signature:`, lastHookSignature);
      console.log(`PipelineHooksConfig: New signature:`, currentSignature);
      currentProcessors = {};
      processorInfos = {};
      processorParams = {};
      lastHookSignature = currentSignature;
      
      // Initialize all processors from config (synchronously)
      hookInfo.processors.forEach((processor: any, index: number) => {
        if (processor.name) {
          currentProcessors[index] = processor.name;
          console.log(`PipelineHooksConfig: Initialized processor ${index} with name:`, processor.name);
        }
      });
      
      // Force reactivity for processors after all are set
      currentProcessors = { ...currentProcessors };
      console.log(`PipelineHooksConfig: Final currentProcessors state:`, currentProcessors);
    }
  }
</script>

<div class="space-y-4">
  
  <!-- Pipeline Hook Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
    <button
      class="flex items-center justify-between w-full text-left focus:outline-none"
      on:click={() => showHookConfig = !showHookConfig}
    >
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        {getHookDisplayName(hookType)}
      </h3>
      <svg class="w-5 h-5 text-gray-500 transform transition-transform {showHookConfig ? 'rotate-180' : ''}" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
      </svg>
    </button>
    
    {#if showHookConfig}
      <div class="mt-4 space-y-4">
        <!-- Skip Diffusion Toggle - only show for image preprocessing -->
        {#if hookType === 'image_preprocessing'}
          <div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg p-4">
            <div class="flex items-center justify-between">
              <div>
                <h4 class="text-sm font-semibold text-yellow-800 dark:text-yellow-200">Skip Diffusion Mode</h4>
                <p class="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                  Bypass diffusion process and only run pre/post processing pipelines
                </p>
              </div>
              <label class="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={skipDiffusion}
                  on:change={(e) => toggleSkipDiffusion((e.target as HTMLInputElement).checked)}
                  class="rounded border-gray-300 text-yellow-600 focus:ring-yellow-500"
                />
                <span class="text-yellow-800 dark:text-yellow-200 font-medium">Enable</span>
              </label>
            </div>
          </div>
        {/if}
        
        {#if hookInfo && hookInfo.processors && hookInfo.processors.length > 0}
          {#each hookInfo.processors as processor, index}
            <div class="border border-gray-200 dark:border-gray-600 rounded-lg p-4 bg-gray-50 dark:bg-gray-700">
              <div class="flex items-center justify-between mb-4">
                <h4 class="text-md font-medium text-gray-900 dark:text-white">
                  {getHookDisplayName(hookType)} Processor {index + 1}
                </h4>
                <div class="flex items-center gap-2">
                  <!-- Enabled Toggle -->
                  <label class="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={processor.enabled}
                      on:change={(e) => updateProcessorEnabled(index, (e.target as HTMLInputElement).checked)}
                      class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    Enabled
                  </label>
                  
                  <!-- Remove Button -->
                  <Button
                    classList="bg-red-600 hover:bg-red-700 text-white px-3 py-1 text-sm"
                    on:click={() => removeProcessor(index)}
                  >
                    Remove
                  </Button>
                </div>
              </div>
              
              <!-- Processor Selector -->
              <div class="mb-4">
                <ProcessorSelector
                  processorIndex={index}
                  currentProcessor={processor.name || processor.type}
                  apiEndpoint="/api/pipeline-hooks/{hookType}"
                  processorType="{getHookDisplayName(hookType)} processor"
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
                    apiEndpoint={`/api/pipeline-hooks/${hookType}`}
                    processorType="{getHookDisplayName(hookType)} processor"
                    on:parametersUpdated={handleParametersUpdated}
                  />
                </div>
              {/if}
            </div>
          {/each}
        {:else}
          <div class="text-sm text-gray-500 dark:text-gray-400 py-4 text-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg">
            No {getHookDisplayName(hookType).toLowerCase()} processors configured. Add one to get started.
          </div>
        {/if}
        
        <!-- Add Processor Button -->
        <div class="pt-2">
          <Button
            classList="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 text-sm"
            on:click={addProcessor}
          >
            Add {getHookDisplayName(hookType)} Processor
          </Button>
        </div>
      </div>
    {/if}
  </div>
</div>
