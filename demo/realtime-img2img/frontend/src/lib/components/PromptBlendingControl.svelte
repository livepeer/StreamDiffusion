<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let promptBlendingConfig: any = null;
  export let normalizePromptWeights: boolean = true;
  export let currentPrompt: string = '';

  const dispatch = createEventDispatcher();

  let promptList: Array<[string, number]> = [];
  let interpolationMethod = 'slerp';
  let initialized = false;
  let lastConfigHash = '';

  // Helper function to create a simple hash of the config for comparison
  function getConfigHash(config: any, current: string): string {
    return JSON.stringify(config) + '|' + current;
  }

  // Reactive logic to update promptList when config actually changes
  $: {
    const currentConfigHash = getConfigHash(promptBlendingConfig, currentPrompt);
    
    if (currentConfigHash !== lastConfigHash) {
      lastConfigHash = currentConfigHash;
      
      if (promptBlendingConfig && Array.isArray(promptBlendingConfig)) {
        // Handle normalized config format from backend [[prompt, weight], ...]
        promptList = [...promptBlendingConfig];
        initialized = true;
        console.log('PromptBlendingControl: Updated from config:', promptList);
      } else if (promptBlendingConfig && promptBlendingConfig.prompt_list) {
        // Handle legacy format with prompt_list property
        promptList = [...promptBlendingConfig.prompt_list];
        interpolationMethod = promptBlendingConfig.interpolation_method || 'slerp';
        initialized = true;
        console.log('PromptBlendingControl: Updated from legacy config:', promptList);
      } else if (!initialized) {
        // Initialize with current prompt if no blending config and not yet initialized
        const initPrompt = currentPrompt && currentPrompt.trim() ? currentPrompt : 'a beautiful landscape';
        promptList = [[initPrompt, 1.0]];
        initialized = true;
        console.log('PromptBlendingControl: Initialized with current prompt:', initPrompt);
      }
    }
  }

  function addPrompt() {
    promptList = [...promptList, ['new prompt', 0.5]];
    console.log('PromptBlendingControl: Added prompt, new list:', promptList);
    updateBlending();
  }

  function removePrompt(index: number) {
    if (promptList.length > 1) {
      promptList = promptList.filter((_, i) => i !== index);
      console.log('PromptBlendingControl: Removed prompt, new list:', promptList);
      updateBlending();
    }
  }

  function updatePromptText(index: number, value: string) {
    console.log(`updatePromptText: Updating prompt ${index} to: "${value}"`);
    promptList[index][0] = value;
    promptList = [...promptList];
    updateBlending();
  }

  function updatePromptWeight(index: number, value: number) {
    console.log(`updatePromptWeight: Updating weight ${index} to: ${value}`);
    promptList[index][1] = value;
    promptList = [...promptList];
    updateBlending();
  }

  function updateInterpolationMethod(value: string) {
    console.log(`updateInterpolationMethod: Updating method to: ${value}`);
    interpolationMethod = value;
    updateBlending();
  }

  async function updateNormalizeWeights(normalize: boolean) {
    console.log(`updateNormalizeWeights: Updating normalize to: ${normalize}`);
    try {
      const response = await fetch('/api/update-normalize-prompt-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ normalize })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateNormalizeWeights: Failed to update normalize prompt weights:', result.detail);
      } else {
        normalizePromptWeights = normalize;
      }
    } catch (error) {
      console.error('updateNormalizeWeights: Update failed:', error);
    }
  }

  function normalizeWeights() {
    console.log('normalizeWeights: Normalizing weights');
    const total = promptList.reduce((sum, [, weight]) => sum + weight, 0);
    if (total > 0) {
      promptList = promptList.map(([prompt, weight]) => [prompt, weight / total]);
      updateBlending();
    }
  }

  async function updateBlending() {
    console.log('updateBlending: Sending update to backend:', {
      prompt_list: promptList,
      interpolation_method: interpolationMethod
    });
    
    try {
      const response = await fetch('/api/prompt-blending/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt_list: promptList,
          interpolation_method: interpolationMethod
        })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateBlending: Failed to update prompt blending:', result.detail);
      } else {
        console.log('updateBlending: Successfully updated prompt blending');
        // Refresh the blending config to get the current weights from the backend
        await refreshCurrentWeights();
      }
    } catch (error) {
      console.error('updateBlending: Update failed:', error);
    }
  }

  async function refreshCurrentWeights() {
    try {
      const response = await fetch('/api/blending/current');
      const data = await response.json();
      
      if (data.prompt_blending && Array.isArray(data.prompt_blending)) {
        console.log('refreshCurrentWeights: Received updated weights:', data.prompt_blending);
        // Update the local promptList with the weights from the backend
        promptList = [...data.prompt_blending];
        // Trigger reactivity
        promptList = promptList;
      }
    } catch (error) {
      console.error('refreshCurrentWeights: Failed to refresh weights:', error);
    }
  }
</script>

<div class="space-y-4">
  <div class="flex items-center justify-between">
    <h3 class="text-lg font-semibold">Prompt Configuration</h3>
    <div class="flex gap-2">
      <Button on:click={addPrompt} classList="text-sm">
        + Add Prompt
      </Button>
    </div>
  </div>

  <div class="space-y-3">
    <!-- Normalize Weights Checkbox -->
    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
      <label class="flex items-center gap-2 text-sm font-medium">
        <input
          type="checkbox"
          bind:checked={normalizePromptWeights}
          on:change={() => updateNormalizeWeights(normalizePromptWeights)}
          class="cursor-pointer"
        />
        Normalize Prompt Weights
      </label>
      <p class="text-xs text-gray-600 dark:text-gray-400 mt-1">
        When enabled, weights are normalized to sum to 1. When disabled, weights > 1 amplify embeddings.
      </p>
    </div>

    <!-- Interpolation Method -->
    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
      <label class="block text-sm font-medium mb-2">Interpolation Method</label>
              <select
          bind:value={interpolationMethod}
          on:change={() => updateBlending()}
          class="w-full p-2 border rounded dark:bg-gray-600 dark:border-gray-500"
        >
          <option value="slerp">SLERP (Spherical Linear)</option>
          <option value="linear">Linear</option>
        </select>
    </div>

    <!-- Prompt List -->
    {#each promptList as [prompt, weight], index (index)}
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Prompt {index + 1}</span>
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-600 dark:text-gray-400">
              Weight: {promptList[index][1].toFixed(3)}
            </span>
            {#if promptList.length > 1}
              <Button on:click={() => removePrompt(index)} classList="text-xs text-red-600">
                Remove
              </Button>
            {/if}
          </div>
        </div>

        <textarea
          bind:value={promptList[index][0]}
          on:input={() => updatePromptText(index, promptList[index][0])}
          placeholder="Enter prompt..."
          rows="2"
          class="w-full p-2 border rounded resize-none dark:bg-gray-600 dark:border-gray-500"
        ></textarea>

        <div class="space-y-1">
          <input
            type="range"
            min="0"
            max="2"
            step="0.01"
            bind:value={promptList[index][1]}
            on:input={(e) => { const target = e.target; updatePromptWeight(index, parseFloat(target.value)); }}
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
          />
          <div class="flex justify-between text-xs text-gray-500">
            <span>0.0</span>
            <span>2.0</span>
          </div>
        </div>
      </div>
    {/each}

    <!-- Total Weight Display -->
    <div class="text-sm text-gray-600 dark:text-gray-400 text-center">
      Total Weight: {promptList.reduce((sum, [, weight]) => sum + weight, 0).toFixed(3)}
    </div>
  </div>
</div>

<style>
  /* Custom range slider styling */
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  input[type="range"]::-moz-range-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
  }
</style>