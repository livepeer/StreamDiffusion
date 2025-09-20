<script lang="ts">
  import { createEventDispatcher, onDestroy } from 'svelte';
  import Button from './Button.svelte';

  export let componentType: 'controlnet' | 'ipadapter' | 'base';
  export let componentIndex: number = 0; // for controlnet
  export let currentSource: 'webcam' | 'uploaded_image' | 'uploaded_video' = 'webcam';
  export let disabled: boolean = false;

  const dispatch = createEventDispatcher();

  // File input elements
  let imageFileInput: HTMLInputElement;
  let videoFileInput: HTMLInputElement;
  
  // Upload state
  let uploadingImage = false;
  let uploadingVideo = false;
  let uploadStatus = '';

  // Preview state
  let previewImage: string | null = null;
  let previewVideo: string | null = null;

  const sourceTypes = [
    { value: 'webcam', label: 'Webcam', icon: '📹' },
    { value: 'uploaded_image', label: 'Image', icon: '🖼️' },
    { value: 'uploaded_video', label: 'Video', icon: '🎬' }
  ];

  async function setInputSource(sourceType: string, sourceData?: any) {
    try {
      const response = await fetch('/api/input-sources/set', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          component: componentType,
          index: componentType === 'controlnet' ? componentIndex : undefined,
          source_type: sourceType,
          source_data: sourceData
        }),
      });

      if (!response.ok) {
        const result = await response.json();
        console.error('setInputSource: Failed to set input source:', result.detail);
        uploadStatus = `Error: ${result.detail}`;
        return false;
      }

      const result = await response.json();
      console.log('setInputSource: Successfully set input source:', result);
      
      // Update local state
      currentSource = sourceType as any;
      
      // Dispatch event to parent
      dispatch('sourceChanged', {
        componentType,
        componentIndex,
        sourceType,
        sourceData
      });

      return true;
    } catch (error) {
      console.error('setInputSource: Request failed:', error);
      uploadStatus = 'Request failed. Please try again.';
      return false;
    }
  }

  function selectWebcam() {
    setInputSource('webcam');
    clearPreviews();
    uploadStatus = '';
  }

  function selectImageFile() {
    imageFileInput.click();
  }

  function selectVideoFile() {
    videoFileInput.click();
  }

  async function uploadImage() {
    if (!imageFileInput.files || imageFileInput.files.length === 0) {
      uploadStatus = 'Please select an image file';
      return;
    }

    const file = imageFileInput.files[0];
    if (!file.type.startsWith('image/')) {
      uploadStatus = 'Please select a valid image file';
      return;
    }

    uploadingImage = true;
    uploadStatus = 'Uploading image...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const endpoint = componentType === 'controlnet' 
        ? `/api/input-sources/upload-image/${componentType}?index=${componentIndex}`
        : `/api/input-sources/upload-image/${componentType}`;

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = 'Image uploaded successfully!';
        
        // Create preview URL for the uploaded image
        const reader = new FileReader();
        reader.onload = (e) => {
          previewImage = e.target?.result as string;
          previewVideo = null; // Clear video preview
        };
        reader.readAsDataURL(file);
        
        // Update source type
        currentSource = 'uploaded_image';
        
        // Dispatch event to parent
        dispatch('sourceChanged', {
          componentType,
          componentIndex,
          sourceType: 'uploaded_image',
          sourceData: result
        });
        
        // Clear file input
        imageFileInput.value = '';
        
        setTimeout(() => {
          uploadStatus = '';
        }, 3000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to upload image'}`;
      }
    } catch (error) {
      console.error('uploadImage: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploadingImage = false;
    }
  }

  async function uploadVideo() {
    if (!videoFileInput.files || videoFileInput.files.length === 0) {
      uploadStatus = 'Please select a video file';
      return;
    }

    const file = videoFileInput.files[0];
    
    // Basic video file validation
    const videoExtensions = ['.mp4', '.webm', '.ogg', '.avi', '.mov'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    if (!videoExtensions.includes(fileExtension)) {
      uploadStatus = 'Please select a valid video file (.mp4, .webm, .ogg, .avi, .mov)';
      return;
    }

    uploadingVideo = true;
    uploadStatus = 'Uploading video...';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const endpoint = componentType === 'controlnet' 
        ? `/api/input-sources/upload-video/${componentType}?index=${componentIndex}`
        : `/api/input-sources/upload-video/${componentType}`;

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        uploadStatus = 'Video uploaded successfully!';
        
        // Create preview URL for the uploaded video
        const reader = new FileReader();
        reader.onload = (e) => {
          previewVideo = e.target?.result as string;
          previewImage = null; // Clear image preview
        };
        reader.readAsDataURL(file);
        
        // Update source type
        currentSource = 'uploaded_video';
        
        // Dispatch event to parent
        dispatch('sourceChanged', {
          componentType,
          componentIndex,
          sourceType: 'uploaded_video',
          sourceData: result
        });
        
        // Clear file input
        videoFileInput.value = '';
        
        setTimeout(() => {
          uploadStatus = '';
        }, 3000);
      } else {
        uploadStatus = `Error: ${result.detail || 'Failed to upload video'}`;
      }
    } catch (error) {
      console.error('uploadVideo: Upload failed:', error);
      uploadStatus = 'Upload failed. Please try again.';
    } finally {
      uploadingVideo = false;
    }
  }

  function clearPreviews() {
    // Clean up blob URLs to avoid memory leaks
    if (previewImage && previewImage.startsWith('blob:')) {
      URL.revokeObjectURL(previewImage);
    }
    if (previewVideo && previewVideo.startsWith('blob:')) {
      URL.revokeObjectURL(previewVideo);
    }
    previewImage = null;
    previewVideo = null;
  }

  // Clean up on component destroy
  onDestroy(() => {
    clearPreviews();
  });

  // Load current source info on mount
  async function loadCurrentSourceInfo() {
    try {
      const endpoint = componentType === 'controlnet' 
        ? `/api/input-sources/info/${componentType}?index=${componentIndex}`
        : `/api/input-sources/info/${componentType}`;

      const response = await fetch(endpoint);
      if (response.ok) {
        const result = await response.json();
        const sourceInfo = result.source_info;
        
        if (sourceInfo && sourceInfo.source_type !== 'fallback' && sourceInfo.source_type !== 'none') {
          currentSource = sourceInfo.source_type as any;
        } else if (componentType === 'ipadapter') {
          // For IPAdapter, default to uploaded_image mode to maintain existing behavior
          currentSource = 'uploaded_image';
        } else if (componentType === 'controlnet') {
          // For ControlNet, default to webcam mode
          currentSource = 'webcam';
        }
      } else if (componentType === 'ipadapter') {
        // If no source info exists for IPAdapter, default to uploaded_image
        currentSource = 'uploaded_image';
      }
      
      // Load preview image if in uploaded_image mode
      if (currentSource === 'uploaded_image') {
        await loadCurrentImagePreview();
      }
    } catch (error) {
      console.warn('loadCurrentSourceInfo: Failed to load current source info:', error);
      // For IPAdapter, always default to uploaded_image even if API fails
      if (componentType === 'ipadapter') {
        currentSource = 'uploaded_image';
        await loadCurrentImagePreview();
      } else if (componentType === 'controlnet') {
        // For ControlNet, default to webcam even if API fails
        currentSource = 'webcam';
      }
    }
  }

  // Load current image preview for uploaded_image mode
  async function loadCurrentImagePreview() {
    try {
      let imageEndpoint = '';
      
      if (componentType === 'ipadapter') {
        // For IPAdapter, use default image as fallback
        // The actual style image is managed through InputSourceManager
        imageEndpoint = '/api/default-image';
      } else {
        // For other components, try to get their uploaded image
        // (This would need backend support for per-component image storage)
        console.log('loadCurrentImagePreview: No preview loading implemented for', componentType);
        return;
      }
      
      const response = await fetch(imageEndpoint);
      if (response.ok) {
        const blob = await response.blob();
        previewImage = URL.createObjectURL(blob);
        previewVideo = null;
        console.log('loadCurrentImagePreview: Loaded default image preview for', componentType);
      }
    } catch (error) {
      console.warn('loadCurrentImagePreview: Failed to load image preview:', error);
    }
  }

  // Expose reset function for parent components
  export function resetToDefaults() {
    console.log('InputSourceSelector: resetToDefaults called for', componentType, componentIndex);
    
    // Clear previews
    clearPreviews();
    
    // Reset to default source type
    if (componentType === 'ipadapter') {
      currentSource = 'uploaded_image';
    } else {
      currentSource = 'webcam';
    }
    
    // Clear upload status
    uploadStatus = '';
    
    // Reload source info from backend
    loadCurrentSourceInfo();
  }

  // Load source info when component mounts or component parameters change
  $: if (componentType && (componentType !== 'controlnet' || componentIndex !== undefined)) {
    loadCurrentSourceInfo();
  }
</script>

<div class="space-y-3">
  <!-- Source Type Selector -->
  <div class="space-y-2">
    <h6 class="text-xs font-medium text-gray-700 dark:text-gray-300">Input Source</h6>
    <div class="grid grid-cols-3 gap-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
      {#each sourceTypes as sourceType}
        <button
          class="flex flex-col items-center gap-1 px-2 py-2 rounded-md text-xs transition-colors
            {currentSource === sourceType.value
              ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}"
          disabled={disabled || uploadingImage || uploadingVideo}
          on:click={() => {
            if (sourceType.value === 'webcam') {
              selectWebcam();
            } else if (sourceType.value === 'uploaded_image') {
              selectImageFile();
            } else if (sourceType.value === 'uploaded_video') {
              selectVideoFile();
            }
          }}
        >
          <span class="text-lg">{sourceType.icon}</span>
          <span class="font-medium">{sourceType.label}</span>
        </button>
      {/each}
    </div>
  </div>

  <!-- Preview Section -->
  {#if previewImage}
    <div class="space-y-2">
      <h6 class="text-xs font-medium text-gray-700 dark:text-gray-300">Preview</h6>
      <div class="relative">
        <img 
          src={previewImage} 
          alt="Uploaded preview" 
          class="w-full max-w-48 h-32 object-cover rounded border border-gray-200 dark:border-gray-600"
        />
        <button
          class="absolute top-1 right-1 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full text-xs flex items-center justify-center"
          on:click={selectWebcam}
          title="Remove and use webcam"
        >
          ×
        </button>
      </div>
    </div>
  {:else if previewVideo}
    <div class="space-y-2">
      <h6 class="text-xs font-medium text-gray-700 dark:text-gray-300">Preview</h6>
      <div class="relative">
        <video 
          src={previewVideo}
          class="w-full max-w-48 h-32 object-cover rounded border border-gray-200 dark:border-gray-600"
          controls
          muted
          loop
          autoplay
        ></video>
        <button
          class="absolute top-1 right-1 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full text-xs flex items-center justify-center"
          on:click={selectWebcam}
          title="Remove and use webcam"
        >
          ×
        </button>
      </div>
    </div>
  {/if}

  <!-- Status Message -->
  {#if uploadStatus}
    <p class="text-xs {uploadStatus.includes('Error') || uploadStatus.includes('Please') ? 'text-red-600' : uploadStatus.includes('successfully') ? 'text-green-600' : 'text-blue-600'}">
      {uploadStatus}
    </p>
  {/if}

  <!-- Current Source Info -->
  <div class="text-xs text-gray-500 dark:text-gray-400">
    Current: {sourceTypes.find(s => s.value === currentSource)?.label || 'Webcam'}
    {#if componentType === 'controlnet'}
      (CN-{componentIndex})
    {:else if componentType === 'ipadapter'}
      (IPAdapter)
    {:else}
      (Base)
    {/if}
  </div>

  <!-- Hidden file inputs -->
  <input
    bind:this={imageFileInput}
    type="file"
    accept="image/*"
    class="hidden"
    on:change={uploadImage}
  />

  <input
    bind:this={videoFileInput}
    type="file"
    accept="video/*"
    class="hidden"
    on:change={uploadVideo}
  />
</div>

<style>
  /* Custom styling for the toggle buttons */
  button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  video {
    background: #000;
  }
</style>
