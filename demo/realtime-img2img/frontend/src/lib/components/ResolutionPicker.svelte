<script lang="ts">
  import { appState, fetchAppState } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';
  import { createEventDispatcher } from 'svelte';

  export let currentResolution: ResolutionInfo;
  
  const dispatch = createEventDispatcher();

  // Generate resolution options from 384 to 1024, divisible by 64
  const resolutionValues = Array.from({ length: 11 }, (_, i) => 384 + (i * 64));

  // Local state for selected values (not yet applied)
  // Initialize once from currentResolution, then let user control independently
  let selectedWidth = currentResolution?.width || 512;
  let selectedHeight = currentResolution?.height || 512;
  let hasInitialized = false;

  // Only initialize once when component mounts, don't reset on every currentResolution change
  $: if (currentResolution && !hasInitialized) {
    selectedWidth = currentResolution.width;
    selectedHeight = currentResolution.height;
    hasInitialized = true;
  }

  function handleWidthChange(event: Event) {
    selectedWidth = parseInt((event.target as HTMLSelectElement).value);
  }

  function handleHeightChange(event: Event) {
    selectedHeight = parseInt((event.target as HTMLSelectElement).value);
  }

  async function updateResolution() {
    console.log('updateResolution: Starting update with selectedWidth:', selectedWidth, 'selectedHeight:', selectedHeight);
    
    // Notify parent that this is a user-initiated resolution change
    dispatch('userResolutionChange');
    
    const aspectRatio = selectedWidth / selectedHeight;
    let aspectRatioString = "1:1";
    
    if (aspectRatio > 1.1) {
      aspectRatioString = "Landscape";
    } else if (aspectRatio < 0.9) {
      aspectRatioString = "Portrait";
    }
    
    const resolutionString = `${selectedWidth}x${selectedHeight} (${aspectRatioString})`;
    console.log('updateResolution: Sending resolution string:', resolutionString);
    
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resolution: resolutionString
        }),
      });
      
      console.log('updateResolution: API response status:', response.status);
      
      if (response.ok) {
        const result = await response.json();
        console.log('updateResolution: API response result:', result);
        
        // State will be updated automatically by the polling mechanism
        // No need to manually fetch state here to avoid potential double-triggering
      } else {
        const result = await response.json();
        console.error('updateResolution: Failed to update resolution:', result.detail);
      }
    } catch (error) {
      console.error('updateResolution: Error updating resolution:', error);
    }
  }
</script>

<div class="space-y-3">
  <div class="flex items-center justify-between">
    <label class="text-sm font-medium text-gray-700 dark:text-gray-300">
      Resolution
    </label>
    {#if currentResolution}
      <div class="flex items-center gap-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">
          {currentResolution.width}×{currentResolution.height}
        </span>
      </div>
    {/if}
  </div>

  <div class="flex gap-2">
    <div class="flex-1">
      <label class="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
        Width
      </label>
      <select
        class="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
        value={selectedWidth}
        on:change={handleWidthChange}
      >
        {#each resolutionValues as value}
          <option value={value}>{value}</option>
        {/each}
      </select>
    </div>
    
    <div class="flex-1">
      <label class="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
        Height
      </label>
      <select
        class="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
        value={selectedHeight}
        on:change={handleHeightChange}
      >
        {#each resolutionValues as value}
          <option value={value}>{value}</option>
        {/each}
      </select>
    </div>
  </div>

  <!-- Update Button -->
  <div class="flex justify-end">
    <button
      on:click={updateResolution}
      class="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
    >
      Update Resolution
    </button>
  </div>
</div> 