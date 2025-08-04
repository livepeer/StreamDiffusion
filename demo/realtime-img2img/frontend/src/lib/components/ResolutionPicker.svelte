<script lang="ts">
  import { pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';

  export let currentResolution: ResolutionInfo;
  export let pipelineParams: any;

  // Generate resolution options from 384 to 1024, divisible by 64
  const resolutionValues = Array.from({ length: 11 }, (_, i) => 384 + (i * 64));

  // Local state for selected values (not yet applied)
  let selectedWidth = currentResolution?.width || 512;
  let selectedHeight = currentResolution?.height || 512;

  // Update local state when currentResolution changes
  $: if (currentResolution) {
    selectedWidth = currentResolution.width;
    selectedHeight = currentResolution.height;
  }

  function handleWidthChange(event: Event) {
    selectedWidth = parseInt((event.target as HTMLSelectElement).value);
  }

  function handleHeightChange(event: Event) {
    selectedHeight = parseInt((event.target as HTMLSelectElement).value);
  }

  function updateResolution() {
    const aspectRatio = selectedWidth / selectedHeight;
    let aspectRatioString = "1:1";
    
    if (aspectRatio > 1.1) {
      aspectRatioString = "Landscape";
    } else if (aspectRatio < 0.9) {
      aspectRatioString = "Portrait";
    }
    
    const resolutionString = `${selectedWidth}x${selectedHeight} (${aspectRatioString})`;
    
    pipelineValues.update(values => ({
      ...values,
      resolution: resolutionString
    }));
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