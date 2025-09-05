<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  const dispatch = createEventDispatcher();

  let availableControlNets: any[] = [];
  let isLoading = false;
  let showSelector = false;
  let modelType = "sd15";

  async function loadAvailableControlNets() {
    try {
      isLoading = true;
      const response = await fetch('/api/controlnet/available');
      if (response.ok) {
        const data = await response.json();
        availableControlNets = data.available_controlnets || [];
        modelType = data.model_type || "sd15";
      } else {
        console.error('ControlNetSelector: Failed to load available ControlNets');
      }
    } catch (error) {
      console.error('ControlNetSelector: Error loading available ControlNets:', error);
    } finally {
      isLoading = false;
    }
  }

  async function addControlNet(controlnetId: string, defaultScale: number) {
    try {
      const response = await fetch('/api/controlnet/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          controlnet_id: controlnetId,
          conditioning_scale: defaultScale,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('ControlNetSelector: Successfully added ControlNet:', result.message);
        
        // Refresh available list and notify parent with updated info
        await loadAvailableControlNets();
        dispatch('controlnetAdded', { 
          controlnetId, 
          controlnet_info: result.controlnet_info 
        });
        showSelector = false;
      } else {
        const error = await response.json();
        console.error('ControlNetSelector: Failed to add ControlNet:', error.detail);
      }
    } catch (error) {
      console.error('ControlNetSelector: Error adding ControlNet:', error);
    }
  }

  function handleAddClick() {
    showSelector = true;
    loadAvailableControlNets();
  }

  onMount(() => {
    loadAvailableControlNets();
  });
</script>

<div class="controlnet-selector">
  {#if !showSelector}
    <Button on:click={handleAddClick} classList="w-full text-sm">
      + Add ControlNet
    </Button>
  {:else}
    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 space-y-3">
      <div class="flex items-center justify-between">
        <h6 class="text-sm font-medium">Add ControlNet ({modelType.toUpperCase()})</h6>
        <button 
          on:click={() => showSelector = false}
          class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
        >
          ✕
        </button>
      </div>
      
      {#if isLoading}
        <div class="text-center py-4">
          <div class="text-sm text-gray-600 dark:text-gray-400">Loading available ControlNets...</div>
        </div>
      {:else if availableControlNets.length === 0}
        <div class="text-center py-4">
          <div class="text-sm text-gray-600 dark:text-gray-400">No additional ControlNets available</div>
        </div>
      {:else}
        <div class="space-y-2">
          {#each availableControlNets as controlnet}
            <div class="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-600">
              <div class="flex items-center justify-between mb-2">
                <div>
                  <div class="font-medium text-sm">{controlnet.name}</div>
                  <div class="text-xs text-gray-600 dark:text-gray-400">{controlnet.model_id}</div>
                </div>
                <Button 
                  on:click={() => addControlNet(controlnet.id, controlnet.default_scale)}
                  classList="text-xs px-2 py-1"
                >
                  Add
                </Button>
              </div>
              <div class="text-xs text-gray-500">{controlnet.description}</div>
              <div class="text-xs text-gray-400 mt-1">
                Default: {controlnet.default_preprocessor} (scale: {controlnet.default_scale})
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .controlnet-selector {
    width: 100%;
  }
</style>