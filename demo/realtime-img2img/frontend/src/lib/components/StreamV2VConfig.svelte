<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let streamv2vInfo: any = null;

  const dispatch = createEventDispatcher();

  // Collapsible section state
  let showStreamV2V: boolean = true;
  let showTemporalSettings: boolean = true;
  let showOptimizationSettings: boolean = true;

  // Local state for parameters
  let useFeatureInjection: boolean = true;
  let featureInjectionStrength: number = 0.8;
  let featureSimilarityThreshold: number = 0.98;
  let interval: number = 4;
  let maxFrames: number = 1;
  let useTomeCache: boolean = true;
  let tomeRatio: number = 0.5;
  let useGrid: boolean = false;

  // Initialize local state when streamv2vInfo changes
  $: if (streamv2vInfo) {
    useFeatureInjection = streamv2vInfo.use_feature_injection ?? true;
    featureInjectionStrength = streamv2vInfo.feature_injection_strength ?? 0.8;
    featureSimilarityThreshold = streamv2vInfo.feature_similarity_threshold ?? 0.98;
    interval = streamv2vInfo.interval ?? 4;
    maxFrames = streamv2vInfo.max_frames ?? 1;
    useTomeCache = streamv2vInfo.use_tome_cache ?? true;
    tomeRatio = streamv2vInfo.tome_ratio ?? 0.5;
    useGrid = streamv2vInfo.use_grid ?? false;
  }

  async function updateStreamV2VParam(paramName: string, value: any) {
    try {
      const response = await fetch('/api/streamv2v/update-param', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          [paramName]: value,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error(`updateStreamV2VParam: Failed to update ${paramName}:`, result.detail);
      }
    } catch (error) {
      console.error(`updateStreamV2VParam: Update failed for ${paramName}:`, error);
    }
  }

  function handleUseFeatureInjectionChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = target.checked;
    useFeatureInjection = value;
    updateStreamV2VParam('use_feature_injection', value);
  }

  function handleFeatureInjectionStrengthChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseFloat(target.value);
    featureInjectionStrength = value;
    updateStreamV2VParam('feature_injection_strength', value);
  }

  function handleFeatureSimilarityThresholdChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseFloat(target.value);
    featureSimilarityThreshold = value;
    updateStreamV2VParam('feature_similarity_threshold', value);
  }

  function handleIntervalChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseInt(target.value);
    interval = value;
    updateStreamV2VParam('interval', value);
  }

  function handleMaxFramesChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseInt(target.value);
    maxFrames = value;
    updateStreamV2VParam('max_frames', value);
  }

  function handleUseTomeCacheChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = target.checked;
    useTomeCache = value;
    updateStreamV2VParam('use_tome_cache', value);
  }

  function handleTomeRatioChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = parseFloat(target.value);
    tomeRatio = value;
    updateStreamV2VParam('tome_ratio', value);
  }

  function handleUseGridChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const value = target.checked;
    useGrid = value;
    updateStreamV2VParam('use_grid', value);
  }

  async function clearFeatureBanks() {
    try {
      const response = await fetch('/api/streamv2v/clear-feature-banks', {
        method: 'POST',
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('clearFeatureBanks: Failed to clear feature banks:', result.detail);
      }
    } catch (error) {
      console.error('clearFeatureBanks: Clear failed:', error);
    }
  }
</script>

<div class="space-y-4">
  <!-- StreamV2V Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showStreamV2V = !showStreamV2V}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <h4 class="text-sm font-semibold">StreamV2V Temporal Consistency</h4>
      <span class="text-sm">{showStreamV2V ? '−' : '+'}</span>
    </button>
    {#if showStreamV2V}
      <div class="p-3">
        <!-- StreamV2V Status -->
        <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded mb-3">
          {#if streamv2vInfo?.enabled}
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-sm font-medium text-green-800 dark:text-green-200">StreamV2V Enabled</span>
          {:else}
            <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">Standard Mode</span>
          {/if}
        </div>

        {#if streamv2vInfo?.enabled}
          <!-- Temporal Settings Section -->
          <div class="bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 mb-3">
            <button 
              on:click={() => showTemporalSettings = !showTemporalSettings}
              class="w-full p-2 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded border-b border-gray-200 dark:border-gray-700"
            >
              <h5 class="text-sm font-medium">Temporal Consistency</h5>
              <span class="text-xs">{showTemporalSettings ? '−' : '+'}</span>
            </button>
            {#if showTemporalSettings}
              <div class="p-3 space-y-3">
                <!-- Feature Injection Toggle -->
                <div class="space-y-2">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Feature Injection</label>
                    <input
                      type="checkbox"
                      checked={useFeatureInjection}
                      on:change={handleUseFeatureInjectionChange}
                      class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
                    />
                  </div>
                  <p class="text-xs text-gray-500">Enable temporal feature injection for consistency</p>
                </div>

                <!-- Feature Injection Strength -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Injection Strength</label>
                    <span class="text-xs text-gray-600 dark:text-gray-400">{featureInjectionStrength.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-2"
                    max="3"
                    step="0.01"
                    value={featureInjectionStrength}
                    on:input={handleFeatureInjectionStrengthChange}
                    disabled={!useFeatureInjection}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600 disabled:opacity-50"
                  />
                  <p class="text-xs text-gray-500">Strength of temporal feature blending (-2.0-3.0)</p>
                </div>

                <!-- Feature Similarity Threshold -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Similarity Threshold</label>
                    <span class="text-xs text-gray-600 dark:text-gray-400">{featureSimilarityThreshold.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.001"
                    value={featureSimilarityThreshold}
                    on:input={handleFeatureSimilarityThresholdChange}
                    disabled={!useFeatureInjection}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600 disabled:opacity-50"
                  />
                  <p class="text-xs text-gray-500">Threshold for feature matching (0.0-1.0)</p>
                </div>

                <!-- Interval -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Update Interval</label>
                    <span class="text-xs text-gray-600 dark:text-gray-400">{interval} frames</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={interval}
                    on:input={handleIntervalChange}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                  />
                  <p class="text-xs text-gray-500">Update cache every N frames</p>
                </div>

                <!-- Max Frames -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Max Cached Frames</label>
                    <span class="text-xs text-gray-600 dark:text-gray-400">{maxFrames}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    step="1"
                    value={maxFrames}
                    on:input={handleMaxFramesChange}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                  />
                  <p class="text-xs text-gray-500">Maximum frames to cache for temporal consistency</p>
                </div>
              </div>
            {/if}
          </div>

          <!-- Optimization Settings Section -->
          <div class="bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 mb-3">
            <button 
              on:click={() => showOptimizationSettings = !showOptimizationSettings}
              class="w-full p-2 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded border-b border-gray-200 dark:border-gray-700"
            >
              <h5 class="text-sm font-medium">Optimization Settings</h5>
              <span class="text-xs">{showOptimizationSettings ? '−' : '+'}</span>
            </button>
            {#if showOptimizationSettings}
              <div class="p-3 space-y-3">
                <!-- ToMe Cache Toggle -->
                <div class="space-y-2">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Token Merging</label>
                    <input
                      type="checkbox"
                      checked={useTomeCache}
                      on:change={handleUseTomeCacheChange}
                      class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
                    />
                  </div>
                  <p class="text-xs text-gray-500">Enable dynamic feature merging for performance</p>
                </div>

                <!-- ToMe Ratio -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Merging Ratio</label>
                    <span class="text-xs text-gray-600 dark:text-gray-400">{tomeRatio.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={tomeRatio}
                    on:input={handleTomeRatioChange}
                    disabled={!useTomeCache}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600 disabled:opacity-50"
                  />
                  <p class="text-xs text-gray-500">Token merging compression ratio (0.0-1.0)</p>
                </div>

                <!-- Grid Sampling Toggle -->
                <div class="space-y-2">
                  <div class="flex items-center justify-between">
                    <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Grid Sampling</label>
                    <input
                      type="checkbox"
                      checked={useGrid}
                      on:change={handleUseGridChange}
                      class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
                    />
                  </div>
                  <p class="text-xs text-gray-500">Use grid-based sampling for merging</p>
                </div>
              </div>
            {/if}
          </div>

          <!-- Controls -->
          <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <h5 class="text-sm font-medium mb-2">Actions</h5>
            <div class="flex gap-2">
              <Button 
                on:click={clearFeatureBanks} 
                classList="text-xs px-3 py-2"
              >
                Clear Feature Banks
              </Button>
            </div>
            <p class="text-xs text-gray-500 mt-2">
              Clear feature banks to reset temporal state and start fresh.
            </p>
          </div>
        {:else}
          <p class="text-xs text-gray-600 dark:text-gray-400">
            Load a configuration with StreamV2V settings to enable temporal consistency features.
          </p>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Range slider styling */
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  input[type="range"]::-moz-range-thumb {
    height: 16px;
    width: 16px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  input[type="range"]::-webkit-slider-track {
    height: 8px;
    border-radius: 4px;
    background: #e5e7eb;
  }

  input[type="range"]::-moz-range-track {
    height: 8px;
    border-radius: 4px;
    background: #e5e7eb;
    border: none;
  }

  .dark input[type="range"]::-webkit-slider-track {
    background: #4b5563;
  }

  .dark input[type="range"]::-moz-range-track {
    background: #4b5563;
  }

  /* Disabled state styling */
  input[type="range"]:disabled::-webkit-slider-thumb {
    background: #9ca3af;
  }

  input[type="range"]:disabled::-moz-range-thumb {
    background: #9ca3af;
  }
</style> 