<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import ProcessorParams from './ProcessorParams.svelte';

  export let controlnetIndex: number = 0;
  export let preprocessorInfo: any = {};
  export let currentParams: { [key: string]: any } = {};
  
  const dispatch = createEventDispatcher();

  function handleParametersUpdated(event: CustomEvent) {
    // Convert generic processor event to preprocessor-specific event
    const { processor_index, parameters } = event.detail;
    
    dispatch('parametersUpdated', {
      controlnet_index: processor_index,
      parameters: parameters
    });
  }


</script>

<ProcessorParams
  processorIndex={controlnetIndex}
  processorInfo={preprocessorInfo}
  {currentParams}
  apiEndpoint="/api/preprocessors"
  processorType="preprocessor"
  on:parametersUpdated={handleParametersUpdated}
/>