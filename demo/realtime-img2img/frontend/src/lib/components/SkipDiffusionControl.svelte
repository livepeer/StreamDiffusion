<script lang="ts">
  import { onMount } from 'svelte';

  let skipDiffusion: boolean = false;
  let loading: boolean = false;
  let error: string = '';

  onMount(async () => {
    await loadSkipDiffusionStatus();
  });

  async function loadSkipDiffusionStatus() {
    try {
      loading = true;
      error = '';
      const response = await fetch('/api/skip-diffusion/status');
      if (response.ok) {
        const data = await response.json();
        skipDiffusion = data.skip_diffusion || false;
      } else {
        console.warn('SkipDiffusionControl: Failed to load skip diffusion status');
      }
    } catch (err) {
      console.error('SkipDiffusionControl: Failed to load skip diffusion status:', err);
      error = 'Failed to load skip diffusion status';
    } finally {
      loading = false;
    }
  }

  async function toggleSkipDiffusion() {
    try {
      error = '';
      const response = await fetch('/api/skip-diffusion/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          skip_diffusion: skipDiffusion
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || 'Failed to toggle skip diffusion');
      }

      const result = await response.json();
      console.log('SkipDiffusionControl: Successfully toggled skip diffusion:', result);
      
    } catch (err) {
      console.error('SkipDiffusionControl: Failed to toggle skip diffusion:', err);
      error = err instanceof Error ? err.message : 'Failed to toggle skip diffusion';
      // Revert the toggle on error
      skipDiffusion = !skipDiffusion;
    }
  }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
  <div class="space-y-3">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
      Processing Mode
    </h3>
    
    <div class="flex items-center justify-between">
      <div class="flex-1">
        <label for="skip-diffusion-toggle" class="text-sm font-medium text-gray-900 dark:text-white">
          Skip Diffusion
        </label>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          When enabled, bypasses the diffusion process and applies only preprocessing and postprocessing to the input image.
        </p>
      </div>
      
      <div class="flex items-center gap-2 ml-4">
        {#if loading}
          <div class="text-sm text-gray-500">Loading...</div>
        {:else}
          <input
            type="checkbox"
            id="skip-diffusion-toggle"
            bind:checked={skipDiffusion}
            on:change={toggleSkipDiffusion}
            class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            disabled={loading}
          />
        {/if}
      </div>
    </div>
    
    {#if error}
      <div class="text-sm text-red-500 mt-2">
        {error}
      </div>
    {/if}
  </div>
</div>
