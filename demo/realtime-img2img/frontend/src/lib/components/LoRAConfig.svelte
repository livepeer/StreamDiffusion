<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let loraInfo: any = null;

  const dispatch = createEventDispatcher();

  // LoRA upload state
  let loraFile: HTMLInputElement;
  let uploadingLora = false;
  let uploadStatus = '';

  // Collapsible toggle
  let showLoRA: boolean = true;

  async function updateLoRAScale(index: number, scale: number) {
    try {
      const response = await fetch('/api/lora/update-scale', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          index: index,
          scale: scale,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateLoRAScale: Failed to update scale:', result.detail);
      }
    } catch (error) {
      console.error('updateLoRAScale: Update failed:', error);
    }
  }

  function handleScaleChange(index: number, event: Event) {
    const target = event.target as HTMLInputElement;
    const scale = parseFloat(target.value);
    
    // Validate that the LoRA still exists at this index
    if (!loraInfo || !loraInfo.loras || index >= loraInfo.loras.length) {
      console.warn('handleScaleChange: LoRA at index', index, 'no longer exists, skipping update');
      return;
    }
    
    // Update local state immediately for responsiveness
    loraInfo.loras[index].scale = scale;
    loraInfo = { ...loraInfo }; // Trigger reactivity
    
    updateLoRAScale(index, scale);
  }

  async function updateLoRAEnabled(index: number, enabled: boolean) {
    try {
      const response = await fetch('/api/lora/update-enabled', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          index: index,
          enabled: enabled,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateLoRAEnabled: Failed to update enabled state:', result.detail);
      }
    } catch (error) {
      console.error('updateLoRAEnabled: Update failed:', error);
    }
  }

  function handleEnabledChange(index: number, event: Event) {
    const target = event.target as HTMLInputElement;
    const enabled = target.checked;
    
    // Validate that the LoRA still exists at this index
    if (!loraInfo || !loraInfo.loras || index >= loraInfo.loras.length) {
      console.warn('handleEnabledChange: LoRA at index', index, 'no longer exists, skipping update');
      return;
    }
    
    // Update local state immediately for responsiveness
    loraInfo.loras[index].enabled = enabled;
    loraInfo = { ...loraInfo }; // Trigger reactivity
    
    updateLoRAEnabled(index, enabled);
  }

  async function removeLora(index: number) {
    try {
      const response = await fetch('/api/lora/remove', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          index: index,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('removeLora: Failed to remove LoRA:', result.detail);
        return false;
      }
      
      const result = await response.json();
      console.log('removeLora: Successfully removed LoRA at index', index);
      
      // Update local state immediately with response data
      if (result.lora_info) {
        loraInfo = result.lora_info;
        console.log('removeLora: Updated local loraInfo:', loraInfo);
      }
      
      // Also trigger config refresh for parent component
      dispatch('loraConfigChanged');
      return true;
    } catch (error) {
      console.error('removeLora: Remove failed:', error);
      return false;
    }
  }

  function handleDeleteLora(index: number) {
    removeLora(index);
  }

  async function uploadLora() {
    if (!loraFile.files || loraFile.files.length === 0) {
      uploadStatus = 'Please select a LoRA file';
      return;
    }

    const file = loraFile.files[0];
    if (!file.name.endsWith('.safetensors') && !file.name.endsWith('.bin')) {
      uploadStatus = 'Please select a valid LoRA file (.safetensors or .bin)';
      return;
    }

    uploadingLora = true;
    uploadStatus = 'Uploading LoRA...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/lora/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = 'LoRA uploaded successfully!';
        
        // Update local state with response data
        if (result.lora_info) {
          loraInfo = result.lora_info;
        }
        
        // Clear file input
        loraFile.value = '';
        
        // Trigger config refresh for parent component
        dispatch('loraConfigChanged');
        
        setTimeout(() => {
          uploadStatus = '';
        }, 3000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to upload LoRA'}`;
      }
    } catch (error) {
      console.error('uploadLora: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploadingLora = false;
    }
  }

  function selectLoraFile() {
    loraFile.click();
  }

  async function addLoraByPath() {
    const path = prompt('Enter LoRA path or HuggingFace model ID:');
    if (!path || path.trim() === '') {
      return;
    }

    try {
      const response = await fetch('/api/lora/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lora_path: path.trim(),
          scale: 1.0,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        // Update local state with response data
        if (result.lora_info) {
          loraInfo = result.lora_info;
        }
        
        // Trigger config refresh for parent component
        dispatch('loraConfigChanged');
      } else {
        alert(`Failed to add LoRA: ${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('addLoraByPath: Add failed:', error);
      alert('Failed to add LoRA. Please try again.');
    }
  }
</script>

<div class="space-y-4">
  <!-- LoRA Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showLoRA = !showLoRA}
      class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
    >
      <h3 class="text-md font-medium">LoRA</h3>
      <span class="text-sm">{showLoRA ? '−' : '+'}</span>
    </button>
    {#if showLoRA}
    <div class="p-4 pt-1">
        <!-- LoRA Status -->
        <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded mb-3">
          {#if loraInfo?.enabled}
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-sm font-medium text-green-800 dark:text-green-200">LoRA Enabled</span>
          {:else}
            <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">Standard Mode</span>
          {/if}
        </div>

        {#if loraInfo?.enabled && loraInfo?.loras?.length > 0}
          <!-- LoRA Controls -->
          <div class="space-y-3">
            <div class="flex items-center justify-between gap-2">
              <h5 class="text-sm font-medium">LoRA Configuration</h5>
              <div class="flex gap-2">
                <Button 
                  on:click={addLoraByPath} 
                  classList="text-xs px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white"
                >
                  Add by Path
                </Button>
                <Button 
                  on:click={selectLoraFile} 
                  disabled={uploadingLora} 
                  classList="text-xs px-2 py-1 bg-green-500 hover:bg-green-600 text-white"
                >
                  {uploadingLora ? 'Uploading...' : 'Upload File'}
                </Button>
              </div>
            </div>
            
            {#each loraInfo.loras as lora, index}
              <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 space-y-3">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="text-sm font-semibold truncate" title={lora.lora_path}>
                      {lora.display_name || lora.lora_path.split('/').pop() || `LoRA ${index}`}
                    </span>
                    {#if lora.lora_type}
                      <span class="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
                        {lora.lora_type}
                      </span>
                    {/if}
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-xs text-gray-600 dark:text-gray-400">
                      Index: {index}
                    </span>
                    <Button 
                      on:click={() => handleDeleteLora(index)}
                      classList="text-xs px-2 py-1 bg-red-500 hover:bg-red-600 text-white"
                    >
                      Delete
                    </Button>
                  </div>
                </div>
                
                <!-- Enable/Disable Toggle -->
                <div class="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="lora-enabled-{index}"
                    checked={lora.enabled}
                    on:change={(e) => handleEnabledChange(index, e)}
                    class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                  />
                  <label for="lora-enabled-{index}" class="text-sm font-medium text-gray-900 dark:text-gray-300">
                    Enabled
                  </label>
                </div>
                
                <!-- Scale Control -->
                <div class="space-y-1">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-medium">Scale</span>
                    <span class="text-xs text-gray-600 dark:text-gray-400">
                      {lora.scale.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.01"
                    value={lora.scale}
                    disabled={!lora.enabled}
                    on:input={(e) => handleScaleChange(index, e)}
                    class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600 disabled:opacity-50"
                  />
                  <p class="text-xs text-gray-500">
                    Controls LoRA strength. Higher values = stronger effect.
                  </p>
                </div>
                
                <!-- LoRA Info -->
                {#if lora.description}
                  <div class="text-xs text-gray-600 dark:text-gray-400">
                    {lora.description}
                  </div>
                {/if}
                <div class="text-xs text-gray-500 font-mono break-all">
                  {lora.lora_path}
                </div>
              </div>
            {/each}
          </div>
        {:else if loraInfo?.enabled}
          <div class="space-y-3">
            <p class="text-xs text-gray-600 dark:text-gray-400">
              No LoRAs active. Add one to get started:
            </p>
            <div class="flex gap-2">
              <Button 
                on:click={addLoraByPath} 
                classList="text-xs px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white"
              >
                Add by Path
              </Button>
              <Button 
                on:click={selectLoraFile} 
                disabled={uploadingLora} 
                classList="text-xs px-2 py-1 bg-green-500 hover:bg-green-600 text-white"
              >
                {uploadingLora ? 'Uploading...' : 'Upload File'}
              </Button>
            </div>
          </div>
        {:else}
          <div class="space-y-3">
            <p class="text-xs text-gray-600 dark:text-gray-400">
              Load a configuration with LoRA settings to enable LoRA support.
            </p>
            <div class="flex gap-2">
              <Button 
                on:click={addLoraByPath} 
                classList="text-xs px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white"
              >
                Add by Path
              </Button>
              <Button 
                on:click={selectLoraFile} 
                disabled={uploadingLora} 
                classList="text-xs px-2 py-1 bg-green-500 hover:bg-green-600 text-white"
              >
                {uploadingLora ? 'Uploading...' : 'Upload File'}
              </Button>
            </div>
          </div>
        {/if}
        
        <!-- Hidden file input -->
        <input
          bind:this={loraFile}
          type="file"
          accept=".safetensors,.bin"
          class="hidden"
          on:change={uploadLora}
        />
        
        <!-- Upload Status -->
        {#if uploadStatus}
          <p class="text-xs mt-2 {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
            {uploadStatus}
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
</style>

