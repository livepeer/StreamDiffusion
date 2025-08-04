<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  export let ipadapterInfo: any = null;
  export let currentScale: number = 1.0;

  const dispatch = createEventDispatcher();

  // Style image upload state
  let styleImageFile: HTMLInputElement;
  let uploadingImage = false;
  let uploadStatus = '';
  let currentStyleImage: string | null = null;

  // Collapsible section state
  let showIPAdapter: boolean = true;

  async function updateIPAdapterScale(scale: number) {
    try {
      const response = await fetch('/api/ipadapter/update-scale', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          scale: scale,
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('updateIPAdapterScale: Failed to update scale:', result.detail);
      }
    } catch (error) {
      console.error('updateIPAdapterScale: Update failed:', error);
    }
  }

  function handleScaleChange(event: Event) {
    const target = event.target as HTMLInputElement;
    const scale = parseFloat(target.value);
    
    // Update local state immediately for responsiveness
    currentScale = scale;
    
    updateIPAdapterScale(scale);
  }

  async function uploadStyleImage() {
    if (!styleImageFile.files || styleImageFile.files.length === 0) {
      uploadStatus = 'Please select an image file';
      return;
    }

    const file = styleImageFile.files[0];
    if (!file.type.startsWith('image/')) {
      uploadStatus = 'Please select a valid image file';
      return;
    }

    uploadingImage = true;
    uploadStatus = 'Uploading style image...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/ipadapter/upload-style-image', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = 'Style image uploaded successfully!';
        
        // Create preview URL for the uploaded image and keep it for display
        const reader = new FileReader();
        reader.onload = (e) => {
          currentStyleImage = e.target?.result as string;
        };
        reader.readAsDataURL(file);
        
        // Clear file input
        styleImageFile.value = '';
        
        setTimeout(() => {
          uploadStatus = '';
        }, 3000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to upload style image'}`;
      }
    } catch (error) {
      console.error('uploadStyleImage: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploadingImage = false;
    }
  }

  function selectStyleImage() {
    styleImageFile.click();
  }

  // Update current scale when prop changes
  $: if (ipadapterInfo?.scale !== undefined) {
    currentScale = ipadapterInfo.scale;
  }
</script>

<div class="space-y-4">
  <!-- IPAdapter Section -->
  <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
    <button 
      on:click={() => showIPAdapter = !showIPAdapter}
      class="w-full p-3 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-700"
    >
      <h4 class="text-sm font-semibold">IPAdapter</h4>
      <span class="text-sm">{showIPAdapter ? '−' : '+'}</span>
    </button>
    {#if showIPAdapter}
      <div class="p-3">
        <!-- IPAdapter Status -->
        <div class="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-700 rounded mb-3">
          {#if ipadapterInfo?.enabled}
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-sm font-medium text-green-800 dark:text-green-200">IPAdapter Enabled</span>
          {:else}
            <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
            <span class="text-sm text-gray-600 dark:text-gray-400">Standard Mode</span>
          {/if}
        </div>

        {#if ipadapterInfo?.enabled}
          <!-- Style Image Upload -->
          <div class="space-y-3">
            <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
              <h5 class="text-sm font-medium mb-2">Style Image</h5>
              
              <!-- Style Image Preview -->
              {#if currentStyleImage}
                <div class="mb-3">
                  <img 
                    src={currentStyleImage} 
                    alt="Uploaded style image" 
                    class="w-full max-w-32 h-32 object-cover rounded border border-gray-200 dark:border-gray-600"
                  />
                  <p class="text-xs text-gray-500 mt-1">Uploaded style image</p>
                </div>
              {:else if ipadapterInfo?.style_image_path}
                <div class="mb-3">
                  <img 
                    src={ipadapterInfo.style_image_path} 
                    alt="Style image" 
                    class="w-full max-w-32 h-32 object-cover rounded border border-gray-200 dark:border-gray-600"
                  />
                  <!-- Show different text for uploaded vs config vs default style images -->
                  <p class="text-xs text-gray-500 mt-1">
                    {#if ipadapterInfo.style_image_path.includes('/api/ipadapter/uploaded-style-image')}
                      Uploaded style image
                    {:else if ipadapterInfo.style_image_path.includes('/api/default-image')}
                      Default style image (input.png)
                    {:else}
                      From config: {ipadapterInfo.style_image_path}
                    {/if}
                  </p>
                </div>
              {/if}
              
              <!-- Upload Button -->
              <div class="flex items-center gap-2">
                <Button 
                  on:click={selectStyleImage} 
                  disabled={uploadingImage} 
                  classList="text-sm px-3 py-2"
                >
                  {uploadingImage ? 'Uploading...' : 'Upload Style Image'}
                </Button>
              </div>
              
              <!-- Hidden file input -->
              <input
                bind:this={styleImageFile}
                type="file"
                accept="image/*"
                class="hidden"
                on:change={uploadStyleImage}
              />
              
              <!-- Upload Status -->
              {#if uploadStatus}
                <p class="text-xs mt-2 {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : 'text-green-600'}">
                  {uploadStatus}
                </p>
              {/if}
              
              <p class="text-xs text-gray-500 mt-2">
                Upload an image to use as style reference for IPAdapter conditioning. If no image is uploaded, the default input.png will be used.
              </p>
            </div>

            <!-- Scale Control -->
            <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
              <h5 class="text-sm font-medium mb-2">IPAdapter Scale</h5>
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <label class="text-xs font-medium text-gray-600 dark:text-gray-400">Strength</label>
                  <span class="text-xs text-gray-600 dark:text-gray-400">{currentScale.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.01"
                  value={currentScale}
                  on:input={handleScaleChange}
                  class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                />
                <p class="text-xs text-gray-500">
                  Controls how strongly the style image influences the generation. Higher values = stronger style influence.
                </p>
              </div>
            </div>

            <!-- IPAdapter Info -->
            {#if ipadapterInfo?.model_path}
              <div class="bg-gray-50 dark:bg-gray-700 rounded p-3">
                <h5 class="text-sm font-medium mb-2">Model Information</h5>
                <p class="text-xs text-gray-600 dark:text-gray-400 font-mono break-all">
                  {ipadapterInfo.model_path}
                </p>
              </div>
            {/if}
          </div>
        {:else}
          <p class="text-xs text-gray-600 dark:text-gray-400">
            Load a configuration with IPAdapter settings to enable style-guided generation.
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