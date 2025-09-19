<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let promptBlendingConfig: any = null;
  export let seedBlendingConfig: any = null;
  export let normalizePromptWeights: boolean = true;
  export let normalizeSeedWeights: boolean = true;
  export let currentPrompt: string = '';

  const dispatch = createEventDispatcher();

  // Prompt blending state
  let promptList: Array<[string, number]> = [];
  let promptInterpolationMethod = 'slerp';
  let promptInitialized = false;
  let lastPromptConfigHash = '';
  let hasPendingPromptChanges = false;

  // Seed blending state
  let seedList: Array<[number, number]> = [];
  let seedInterpolationMethod = 'linear';
  let seedInitialized = false;
  let lastSeedConfigHash = '';
  let hasPendingSeedChanges = false;

  // UI state
  let activeTab: 'prompts' | 'seeds' = 'prompts';

  // Helper functions for config hashing
  function getPromptConfigHash(config: any, current: string): string {
    return JSON.stringify(config) + '|' + current;
  }

  function getSeedConfigHash(config: any): string {
    return JSON.stringify(config);
  }

  // Reactive logic for prompt blending - only update when config structure changes, not weights
  $: {
    const currentPromptConfigHash = getPromptConfigHash(promptBlendingConfig, currentPrompt);
    
    if (currentPromptConfigHash !== lastPromptConfigHash) {
      lastPromptConfigHash = currentPromptConfigHash;
      
      // Only update from config if not initialized yet or if the config has different prompt structure
      if (!promptInitialized) {
        if (promptBlendingConfig && Array.isArray(promptBlendingConfig)) {
          promptList = [...promptBlendingConfig];
          promptInitialized = true;
          console.log('BlendingControl: Initialized prompt list from config:', promptList);
        } else if (promptBlendingConfig && promptBlendingConfig.prompt_list) {
          promptList = [...promptBlendingConfig.prompt_list];
          promptInterpolationMethod = promptBlendingConfig.interpolation_method || 'slerp';
          promptInitialized = true;
          console.log('BlendingControl: Initialized prompt list from legacy config:', promptList);
        } else {
          const initPrompt = currentPrompt && currentPrompt.trim() ? currentPrompt : 'a beautiful landscape';
          promptList = [[initPrompt, 1.0]];
          promptInitialized = true;
          console.log('BlendingControl: Initialized with current prompt:', initPrompt);
        }
      } else if (promptBlendingConfig && Array.isArray(promptBlendingConfig) && !hasPendingPromptChanges) {
         // Update when structure changed OR when prompt texts changed (but avoid overwriting local edits)
         const structureChanged = promptBlendingConfig.length !== promptList.length;
         let textsChanged = false;
         if (!structureChanged) {
           for (let i = 0; i < promptList.length; i++) {
             const newText = promptBlendingConfig[i]?.[0] ?? '';
             const currText = promptList[i]?.[0] ?? '';
             if (newText !== currText) {
               textsChanged = true;
               break;
             }
           }
         }
         if (structureChanged || textsChanged) {
           promptList = [...promptBlendingConfig];
           console.log('BlendingControl: Updated prompt list from config change:', promptList);
         }
       } else if (!promptBlendingConfig && !hasPendingPromptChanges) {
         // No external prompt blending structure provided. If only the base prompt changed
         // (e.g., via YAML upload), reflect it in the first prompt entry to keep UI in sync.
         const newPrompt = currentPrompt && currentPrompt.trim() ? currentPrompt : 'a beautiful landscape';
         if (promptList.length === 0) {
           promptList = [[newPrompt, 1.0]];
         } else if (promptList[0][0] !== newPrompt) {
           promptList[0][0] = newPrompt;
           promptList = [...promptList];
           console.log('BlendingControl: Synced first prompt with current prompt:', newPrompt);
         }
       }
    }
  }

  // Reactive logic for seed blending - only update when config structure changes, not weights
  $: {
    const currentSeedConfigHash = getSeedConfigHash(seedBlendingConfig);
    
    if (currentSeedConfigHash !== lastSeedConfigHash) {
      lastSeedConfigHash = currentSeedConfigHash;
      
      // Only update from config if not initialized yet or if the config has different seed structure
      if (!seedInitialized) {
        if (seedBlendingConfig && Array.isArray(seedBlendingConfig)) {
          seedList = [...seedBlendingConfig];
          seedInitialized = true;
          console.log('BlendingControl: Initialized seed list from config:', seedList);
        } else if (seedBlendingConfig && seedBlendingConfig.seed_list) {
          seedList = [...seedBlendingConfig.seed_list];
          seedInterpolationMethod = seedBlendingConfig.seed_interpolation_method || 'linear';
          seedInitialized = true;
          console.log('BlendingControl: Initialized seed list from legacy config:', seedList);
        } else {
          seedList = [[2, 1.0]];
          seedInitialized = true;
          console.log('BlendingControl: Initialized with default seed');
        }
             } else if (seedBlendingConfig && Array.isArray(seedBlendingConfig) && !hasPendingSeedChanges) {
         // Only update if the number of seeds changed (structural change, not just weight change)
         // and we don't have pending local changes
         if (seedBlendingConfig.length !== seedList.length) {
           seedList = [...seedBlendingConfig];
           console.log('BlendingControl: Updated seed list due to structure change:', seedList);
         }
       }
    }
  }

  // Prompt blending functions
  function addPrompt() {
    promptList = [...promptList, ['new prompt', 0.5]];
    hasPendingPromptChanges = true;
    console.log('BlendingControl: Added prompt, new list:', promptList);
    updatePromptBlendingWithoutRefresh();
  }

  function removePrompt(index: number) {
    if (promptList.length > 1) {
      promptList = promptList.filter((_, i) => i !== index);
      hasPendingPromptChanges = true;
      console.log('BlendingControl: Removed prompt, new list:', promptList);
      updatePromptBlendingWithoutRefresh();
    }
  }

  function updatePromptText(index: number, value: string) {
    promptList[index][0] = value;
    promptList = [...promptList];
    updatePromptBlendingWithoutRefresh();
  }

  function updatePromptWeight(index: number, value: number) {
    promptList[index][1] = value;
    promptList = [...promptList];
    updatePromptBlendingWithoutRefresh();
  }

  function updatePromptInterpolationMethod(value: string) {
    promptInterpolationMethod = value;
    updatePromptBlendingWithoutRefresh();
  }

  async function updateNormalizePromptWeights(normalize: boolean) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ normalize_prompt_weights: normalize })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateNormalizePromptWeights: Failed to update normalize prompt weights:', result.detail);
      } else {
        normalizePromptWeights = normalize;
        // Also update the blending to apply the normalization change
        await updatePromptBlendingWithoutRefresh();
      }
    } catch (error) {
      console.error('updateNormalizePromptWeights: Update failed:', error);
    }
  }

  async function updatePromptBlendingWithoutRefresh() {
    try {
      const response = await fetch('/api/blending', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt_list: promptList,
          prompt_interpolation_method: promptInterpolationMethod
        })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updatePromptBlendingWithoutRefresh: Failed to update prompt blending:', result.detail);
      } else {
        console.log('updatePromptBlendingWithoutRefresh: Successfully updated prompt blending');
        hasPendingPromptChanges = false; // Clear pending flag on success
      }
    } catch (error) {
      console.error('updatePromptBlendingWithoutRefresh: Update failed:', error);
    }
  }

  // Seed blending functions
  function addSeed() {
    const newSeed = Math.floor(Math.random() * 999999) + 1;
    seedList = [...seedList, [newSeed, 0.5]];
    hasPendingSeedChanges = true;
    console.log('BlendingControl: Added seed, new list:', seedList);
    updateSeedBlendingWithoutRefresh();
  }

  function removeSeed(index: number) {
    if (seedList.length > 1) {
      seedList = seedList.filter((_, i) => i !== index);
      hasPendingSeedChanges = true;
      console.log('BlendingControl: Removed seed, new list:', seedList);
      updateSeedBlendingWithoutRefresh();
    }
  }

  function updateSeedValue(index: number, value: number) {
    seedList[index][0] = value;
    seedList = [...seedList];
    updateSeedBlendingWithoutRefresh();
  }

  function updateSeedWeight(index: number, value: number) {
    seedList[index][1] = value;
    seedList = [...seedList];
    updateSeedBlendingWithoutRefresh();
  }

  function updateSeedInterpolationMethod(value: string) {
    seedInterpolationMethod = value;
    updateSeedBlendingWithoutRefresh();
  }

  async function updateNormalizeSeedWeights(normalize: boolean) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ normalize_seed_weights: normalize })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateNormalizeSeedWeights: Failed to update normalize seed weights:', result.detail);
      } else {
        normalizeSeedWeights = normalize;
        // Also update the blending to apply the normalization change
        await updateSeedBlendingWithoutRefresh();
      }
    } catch (error) {
      console.error('updateNormalizeSeedWeights: Update failed:', error);
    }
  }

  function randomizeSeed(index: number) {
    const newSeed = Math.floor(Math.random() * 999999) + 1;
    updateSeedValue(index, newSeed);
  }

  async function updateSeedBlendingWithoutRefresh() {
    try {
      const response = await fetch('/api/blending', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seed_list: seedList,
          seed_interpolation_method: seedInterpolationMethod
        })
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateSeedBlendingWithoutRefresh: Failed to update seed blending:', result.detail);
      } else {
        console.log('updateSeedBlendingWithoutRefresh: Successfully updated seed blending');
        hasPendingSeedChanges = false; // Clear pending flag on success
      }
    } catch (error) {
      console.error('updateSeedBlendingWithoutRefresh: Update failed:', error);
    }
  }
</script>

<div class="space-y-4">
  <!-- Tab Navigation -->
  <div class="flex border-b border-gray-200 dark:border-gray-700">
    <button
      on:click={() => activeTab = 'prompts'}
      class="px-4 py-2 text-sm font-medium border-b-2 transition-colors {activeTab === 'prompts' 
        ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
        : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'}"
    >
      Prompt Blending
    </button>
    <button
      on:click={() => activeTab = 'seeds'}
      class="px-4 py-2 text-sm font-medium border-b-2 transition-colors {activeTab === 'seeds' 
        ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
        : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'}"
    >
      Seed Blending
    </button>
  </div>

  <!-- Prompt Blending Tab -->
  {#if activeTab === 'prompts'}
    <div class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold">Prompt Configuration</h3>
        <Button on:click={addPrompt} classList="text-sm">
          + Add Prompt
        </Button>
      </div>

      <div class="space-y-3">
        <!-- Normalize Weights Checkbox -->
        <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
          <label class="flex items-center gap-2 text-sm font-medium">
            <input
              type="checkbox"
              bind:checked={normalizePromptWeights}
              on:change={() => updateNormalizePromptWeights(normalizePromptWeights)}
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
            bind:value={promptInterpolationMethod}
            on:change={() => updatePromptBlendingWithoutRefresh()}
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
                  Weight: {(promptList[index][1] || 0).toFixed(3)}
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
                on:input={() => updatePromptWeight(index, promptList[index][1])}
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
          Total Weight: {promptList.reduce((sum, [, weight]) => sum + (weight || 0), 0).toFixed(3)}
        </div>
      </div>
    </div>
  {/if}

  <!-- Seed Blending Tab -->
  {#if activeTab === 'seeds'}
    <div class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold">Seed Blending</h3>
        <Button on:click={addSeed} classList="text-sm">
          + Add Seed
        </Button>
      </div>

      <div class="space-y-3">
        <!-- Normalize Weights Checkbox -->
        <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
          <label class="flex items-center gap-2 text-sm font-medium">
            <input
              type="checkbox"
              bind:checked={normalizeSeedWeights}
              on:change={() => updateNormalizeSeedWeights(normalizeSeedWeights)}
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
            on:change={() => updateSeedBlendingWithoutRefresh()}
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
                  Weight: {(weight || 0).toFixed(3)}
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
                bind:value={seedList[index][0]}
                on:input={() => updateSeedValue(index, seedList[index][0])}
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
                bind:value={seedList[index][1]}
                on:input={() => updateSeedWeight(index, seedList[index][1])}
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
          Total Weight: {seedList.reduce((sum, [, weight]) => sum + (weight || 0), 0).toFixed(3)}
        </div>
      </div>
    </div>
  {/if}
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