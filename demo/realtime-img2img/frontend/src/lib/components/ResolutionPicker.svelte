<script lang="ts">
  import { pipelineValues } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';

  export let currentResolution: ResolutionInfo;
  export let pipelineParams: any; // Add pipeline params prop

  // Parse resolution values from pipeline params
  $: resolutionOptions = pipelineParams?.resolution?.values || [];

  function handleResolutionChange(value: string) {
    pipelineValues.update(values => ({
      ...values,
      resolution: value
    }));
  }



  // Parse resolution string to get width, height, and aspect ratio
  function parseResolutionOption(option: string) {
    const match = option.match(/^(\d+)x(\d+)\s*\(([^)]+)\)$/);
    if (match) {
      const [, width, height, aspectRatio] = match;
      return {
        label: `${width}×${height}`,
        value: option,
        aspectRatio: aspectRatio,
        width: parseInt(width),
        height: parseInt(height)
      };
    }
    return null;
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

  {#if resolutionOptions.length > 0}
    <div class="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
      {#each resolutionOptions as option}
        {@const parsed = parseResolutionOption(option)}
        {#if parsed}
          <button
            on:click={() => handleResolutionChange(parsed.value)}
            class="p-2 text-xs border rounded-lg transition-colors {$pipelineValues.resolution === parsed.value 
              ? 'bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700 text-blue-800 dark:text-blue-200' 
              : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700'}"
            title="Aspect Ratio: {parsed.aspectRatio}"
          >
            <div class="font-medium">{parsed.label}</div>
            <div class="text-xs opacity-75">{parsed.aspectRatio}</div>
          </button>
        {/if}
      {/each}
    </div>
  {:else}
    <!-- Fallback to basic presets if no pipeline options -->
    <div class="grid grid-cols-2 gap-2">
      <button
        on:click={() => handleResolutionChange('512x512 (1:1)')}
        class="p-2 text-xs border rounded-lg transition-colors {$pipelineValues.resolution === '512x512 (1:1)' 
          ? 'bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700 text-blue-800 dark:text-blue-200' 
          : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700'}"
      >
        <div class="font-medium">512×512</div>
        <div class="text-xs opacity-75">1:1</div>
      </button>
      <button
        on:click={() => handleResolutionChange('768x768 (1:1)')}
        class="p-2 text-xs border rounded-lg transition-colors {$pipelineValues.resolution === '768x768 (1:1)' 
          ? 'bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700 text-blue-800 dark:text-blue-200' 
          : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700'}"
      >
        <div class="font-medium">768×768</div>
        <div class="text-xs opacity-75">1:1</div>
      </button>
    </div>
  {/if}

  <!-- Custom resolution input -->
  <div class="space-y-2">
    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">
      Custom Resolution
    </label>
    <div class="flex gap-2">
      <input
        type="number"
        placeholder="Width"
        class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
        min="64"
        max="2048"
        step="64"
        on:change={(e) => {
          const width = e.target.value;
          const height = currentResolution?.height || 512;
          if (width) {
            handleResolutionChange(`${width}x${height} (${width}:${height})`);
          }
        }}
      />
      <span class="text-xs text-gray-500 dark:text-gray-400 self-center">×</span>
      <input
        type="number"
        placeholder="Height"
        class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
        min="64"
        max="2048"
        step="64"
        on:change={(e) => {
          const height = e.target.value;
          const width = currentResolution?.width || 512;
          if (height) {
            handleResolutionChange(`${width}x${height} (${width}:${height})`);
          }
        }}
      />
    </div>
  </div>
</div> 