<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import ProcessorSelector from './ProcessorSelector.svelte';

  export let controlnetIndex: number = 0;
  export let currentPreprocessor: string = 'passthrough';
  
  const dispatch = createEventDispatcher();

  function handleProcessorChanged(event: CustomEvent) {
    // Convert generic processor event to preprocessor-specific event
    const { processor_index, processor, processor_info, current_params } = event.detail;
    
    dispatch('preprocessorChanged', {
      controlnet_index: processor_index,
      preprocessor: processor,
      preprocessor_info: processor_info,
      current_params: current_params
    });
  }
</script>

<ProcessorSelector
  processorIndex={controlnetIndex}
  currentProcessor={currentPreprocessor}
  apiEndpoint="/api/preprocessors"
  processorType="preprocessor"
  on:processorChanged={handleProcessorChanged}
/>