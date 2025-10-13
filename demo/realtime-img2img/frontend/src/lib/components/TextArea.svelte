<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount, onDestroy } from 'svelte';
  export let value: string;
  export let params: FieldProps;
  
  let localValue: string = value;
  let debounceTimer: number | null = null;
  const DEBOUNCE_DELAY = 300;
  
  onMount(() => {
    if (!value || value === '') {
      value = String(params?.default ?? '');
      localValue = value;
    }
  });
  
  onDestroy(() => {
    if (debounceTimer !== null) {
      clearTimeout(debounceTimer);
    }
  });
  
  // Update localValue when external value changes (but not during typing)
  $: if (value !== localValue && debounceTimer === null) {
    localValue = value;
  }
  
  function handleInput(event: Event) {
    const target = event.currentTarget as HTMLTextAreaElement;
    localValue = target.value;
    
    // Clear existing timer
    if (debounceTimer !== null) {
      clearTimeout(debounceTimer);
    }
    
    // Set new debounced timer to update parent value
    debounceTimer = setTimeout(() => {
      value = localValue;
      debounceTimer = null;
    }, DEBOUNCE_DELAY) as unknown as number;
  }
</script>

<div class="">
  <label class="text-sm font-medium" for={params?.title}>
    {params?.title}
  </label>
  <div class="text-normal flex items-center rounded-md border border-gray-700">
    <textarea
      class="mx-1 w-full px-3 py-2 outline-none"
      title={params?.title}
      placeholder="Add your prompt here..."
      value={localValue}
      on:input={handleInput}
    ></textarea>
  </div>
</div>
