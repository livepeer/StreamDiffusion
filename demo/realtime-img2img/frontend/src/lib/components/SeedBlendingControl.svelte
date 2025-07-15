<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let seedBlendingConfig: any = null;
  export let normalizeSeedWeights: boolean = true;

  const dispatch = createEventDispatcher();

  let seedList: Array<[number, number]> = [];
  let seedInterpolationMethod = 'linear';

  $: if (seedBlendingConfig && Array.isArray(seedBlendingConfig)) {
    // Handle normalized config format from backend [[seed, weight], ...]
    seedList = [...seedBlendingConfig];
    console.log('SeedBlendingControl: Updated from config:', seedList);
  } else if (seedBlendingConfig && seedBlendingConfig.seed_list) {
    // Handle legacy format with seed_list property
    seedList = [...seedBlendingConfig.seed_list];
    seedInterpolationMethod = seedBlendingConfig.seed_interpolation_method || 'linear';
    console.log('SeedBlendingControl: Updated from legacy config:', seedList);
  } else {
    // Initialize with single seed if no blending config
    if (seedList.length === 0) {
      seedList = [[2, 1.0]];
    }
  }

  function addSeed() {
    const newSeed = Math.floor(Math.random() * 999999) + 1;
    seedList = [...seedList, [newSeed, 0.5]];
    updateBlending();
  }

  function removeSeed(index: number) {
    if (seedList.length > 1) {
      seedList = seedList.filter((_, i) => i !== index);
      updateBlending();
    }
  }

  function updateSeedValue(index: number, value: number) {
    seedList[index][0] = value;
    seedList = [...seedList];
    updateBlending();
  }

  function updateSeedWeight(index: number, value: number) {
    seedList[index][1] = value;
    seedList = [...seedList];
    updateBlending();
  }

  function updateSeedInterpolationMethod(value: string) {
    seedInterpolationMethod = value;
    updateBlending();
  }

  async function updateNormalizeWeights(normalize: boolean) {
    try {
      const response = await fetch('/api/update-normalize-seed-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ normalize })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateNormalizeWeights: Failed to update normalize seed weights:', result.detail);
      } else {
        normalizeSeedWeights = normalize;
      }
    } catch (error) {
      console.error('updateNormalizeWeights: Update failed:', error);
    }
  }

  function normalizeWeights() {
    const total = seedList.reduce((sum, [, weight]) => sum + weight, 0);
    if (total > 0) {
      seedList = seedList.map(([seed, weight]) => [seed, weight / total]);
      updateBlending();
    }
  }

  function randomizeSeed(index: number) {
    const newSeed = Math.floor(Math.random() * 999999) + 1;
    updateSeedValue(index, newSeed);
  }

  async function updateBlending() {
    try {
      const response = await fetch('/api/seed-blending/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seed_list: seedList,
          seed_interpolation_method: seedInterpolationMethod
        })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateBlending: Failed to update seed blending:', result.detail);
      } else {
        console.log('updateBlending: Successfully updated seed blending');
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
      
      if (data.seed_blending && Array.isArray(data.seed_blending)) {
        console.log('refreshCurrentWeights: Received updated seed weights:', data.seed_blending);
        // Update the local seedList with the weights from the backend
        seedList = [...data.seed_blending];
        // Trigger reactivity
        seedList = seedList;
      }
    } catch (error) {
      console.error('refreshCurrentWeights: Failed to refresh seed weights:', error);
    }
  }
</script>

<div class="space-y-4">
  <div class="flex items-center justify-between">
    <h3 class="text-lg font-semibold">Seed Blending</h3>
    <div class="flex gap-2">
      <Button on:click={addSeed} classList="text-sm">
        + Add Seed
      </Button>
    </div>
  </div>

  <div class="space-y-3">
    <!-- Normalize Weights Checkbox -->
    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
      <label class="flex items-center gap-2 text-sm font-medium">
        <input
          type="checkbox"
          bind:checked={normalizeSeedWeights}
          on:change={() => updateNormalizeWeights(normalizeSeedWeights)}
          class="cursor-pointer"
        />
        Normalize Seed Weights
      </label>
      <p class="text-xs text-gray-600 dark:text-gray-400 mt-1">
        When enabled, weights are normalized to sum to 1. When disabled, weights > 1 amplify noise.
      </p>
    </div>

    <!-- Interpolation Method -->
    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
      <label class="block text-sm font-medium mb-2">Interpolation Method</label>
      <select
        bind:value={seedInterpolationMethod}
        on:change={() => updateBlending()}
        class="w-full p-2 border rounded dark:bg-gray-600 dark:border-gray-500"
      >
        <option value="linear">Linear</option>
        <option value="slerp">SLERP (Spherical Linear)</option>
      </select>
    </div>

    <!-- Seed List -->
    {#each seedList as [seed, weight], index}
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Seed {index + 1}</span>
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-600 dark:text-gray-400">
              Weight: {weight.toFixed(3)}
            </span>
            {#if seedList.length > 1}
              <Button on:click={() => removeSeed(index)} classList="text-xs text-red-600">
                Remove
              </Button>
            {/if}
          </div>
        </div>

        <div class="flex items-center gap-2">
          <input
            type="number"
            bind:value={seed}
            on:input={() => updateSeedValue(index, seed)}
            min="1"
            max="999999"
            class="flex-1 p-2 border rounded dark:bg-gray-600 dark:border-gray-500"
          />
          <Button on:click={() => randomizeSeed(index)} classList="text-xs">
            Random
          </Button>
        </div>

        <div class="space-y-1">
          <input
            type="range"
            min="0"
            max="2"
            step="0.01"
            bind:value={weight}
            on:input={() => updateSeedWeight(index, weight)}
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
      Total Weight: {seedList.reduce((sum, [, weight]) => sum + weight, 0).toFixed(3)}
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