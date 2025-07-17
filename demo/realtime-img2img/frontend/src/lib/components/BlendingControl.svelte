<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';
  import '../styles/blending-controls.css';
  import { normalizeWeights as normalizeWeightsUtil, calculateTotalWeight, updateNormalizeWeights, updateBlendingConfig } from '../utils/blending-utils';

  // Generic types
  type BlendingValue = string | number;
  type BlendingItem = [BlendingValue, number];

  // Props
  export let blendingType: 'prompt' | 'seed';
  export let blendingConfig: any = null;
  export let normalizeWeights: boolean = true;

  // Derived configuration
  $: isPromptMode = blendingType === 'prompt';
  $: title = isPromptMode ? 'Prompt Blending' : 'Seed Blending';
  $: addButtonText = isPromptMode ? '+ Add Prompt' : '+ Add Seed';
  $: defaultValue = isPromptMode ? 'new prompt' : Math.floor(Math.random() * 999999) + 1;
  $: normalizeEndpoint = isPromptMode ? '/api/update-normalize-prompt-weights' : '/api/update-normalize-seed-weights';
  $: updateEndpoint = isPromptMode ? '/api/prompt-blending/update' : '/api/seed-blending/update';
  $: listKey = isPromptMode ? 'prompt_list' : 'seed_list';
  $: weightDescription = isPromptMode ? 'weights > 1 amplify embeddings' : 'weights > 1 amplify noise';

  const dispatch = createEventDispatcher();

  let itemList: BlendingItem[] = [];
  let interpolationMethod = isPromptMode ? 'slerp' : 'linear';

  // Reactive configuration handling
  $: if (blendingConfig && Array.isArray(blendingConfig)) {
    // Handle normalized config format from backend [[value, weight], ...]
    itemList = [...blendingConfig];
    console.log(`BlendingControl (${blendingType}): Updated from config:`, itemList);
  } else if (blendingConfig && blendingConfig[listKey]) {
    // Handle legacy format with list property
    itemList = [...blendingConfig[listKey]];
    const methodKey = listKey.replace('_list', '_interpolation_method');
    interpolationMethod = blendingConfig[methodKey] || (isPromptMode ? 'slerp' : 'linear');
    console.log(`BlendingControl (${blendingType}): Updated from legacy config:`, itemList);
  } else {
    // Initialize with single item if no blending config
    if (itemList.length === 0) {
      itemList = isPromptMode 
        ? [['a beautiful landscape', 1.0]] 
        : [[2, 1.0]];
    }
  }

  function addItem() {
    const newValue = isPromptMode 
      ? 'new prompt' 
      : Math.floor(Math.random() * 999999) + 1;
    itemList = [...itemList, [newValue, 0.5]];
    updateBlending();
  }

  function removeItem(index: number) {
    if (itemList.length > 1) {
      itemList = itemList.filter((_, i) => i !== index);
      updateBlending();
    }
  }

  function updateItemValue(index: number, value: BlendingValue) {
    itemList[index][0] = value;
    itemList = [...itemList];
    updateBlending();
  }

  function updateItemWeight(index: number, weight: number) {
    itemList[index][1] = weight;
    itemList = [...itemList];
    updateBlending();
  }

  function updateInterpolationMethod(value: string) {
    interpolationMethod = value;
    updateBlending();
  }

  async function handleNormalizeWeights(normalize: boolean) {
    const success = await updateNormalizeWeights(normalizeEndpoint, normalize);
    if (success) {
      normalizeWeights = normalize;
      // Trigger reblending to immediately apply the new normalization setting
      updateBlending();
    }
  }

  function normalizeWeightsManually() {
    itemList = normalizeWeightsUtil(itemList);
    updateBlending();
  }

  function randomizeValue(index: number) {
    if (!isPromptMode) {
      const newValue = Math.floor(Math.random() * 999999) + 1;
      updateItemValue(index, newValue);
    }
  }

  async function updateBlending() {
    const success = await updateBlendingConfig(
      updateEndpoint,
      listKey,
      itemList,
      interpolationMethod
    );
    
    if (!success) {
      console.error(`BlendingControl (${blendingType}): Failed to update blending config`);
    }
  }

  // Calculate total weight
  $: totalWeight = calculateTotalWeight(itemList);
</script>

<div class="space-y-4">
  <div class="flex items-center justify-between">
    <h3 class="text-lg font-semibold">{title}</h3>
    <div class="flex gap-2">
      <Button on:click={addItem} classList="text-sm">
        {addButtonText}
      </Button>
    </div>
  </div>

  <div class="space-y-3">
    <!-- Normalize Weights Checkbox -->
    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
      <label class="flex items-center gap-2 text-sm font-medium">
        <input
          type="checkbox"
          bind:checked={normalizeWeights}
          on:change={() => handleNormalizeWeights(normalizeWeights)}
          class="cursor-pointer"
        />
        Normalize {isPromptMode ? 'Prompt' : 'Seed'} Weights
      </label>
      <p class="text-xs text-gray-600 dark:text-gray-400 mt-1">
        When enabled, weights are normalized to sum to 1. When disabled, {weightDescription}.
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
        {#if isPromptMode}
          <option value="slerp">SLERP (Spherical Linear)</option>
          <option value="linear">Linear</option>
        {:else}
          <option value="linear">Linear</option>
          <option value="slerp">SLERP (Spherical Linear)</option>
        {/if}
      </select>
    </div>

    <!-- Item List -->
    {#each itemList as [value, weight], index}
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">{isPromptMode ? 'Prompt' : 'Seed'} {index + 1}</span>
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-600 dark:text-gray-400">
              Weight: {weight.toFixed(3)}
            </span>
            {#if itemList.length > 1}
              <Button on:click={() => removeItem(index)} classList="text-xs text-red-600">
                Remove
              </Button>
            {/if}
          </div>
        </div>

        {#if isPromptMode}
          <!-- Prompt input -->
          <textarea
            bind:value={value}
            on:input={() => updateItemValue(index, value)}
            placeholder="Enter prompt..."
            rows="2"
            class="w-full p-2 border rounded resize-none dark:bg-gray-600 dark:border-gray-500"
          ></textarea>
        {:else}
          <!-- Seed input -->
          <div class="flex items-center gap-2">
            <input
              type="number"
              bind:value={value}
              on:input={() => updateItemValue(index, value)}
              min="1"
              max="999999"
              class="flex-1 p-2 border rounded dark:bg-gray-600 dark:border-gray-500"
            />
            <Button on:click={() => randomizeValue(index)} classList="text-xs">
              Random
            </Button>
          </div>
        {/if}

        <div class="space-y-1">
          <input
            type="range"
            min="0"
            max="2"
            step="0.01"
            bind:value={weight}
            on:input={() => updateItemWeight(index, weight)}
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
      Total Weight: {totalWeight.toFixed(3)}
    </div>
  </div>
</div>

<!-- Styles imported from shared CSS file --> 