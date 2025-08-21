<script lang="ts">
  export let message: string = '';

  let timeout = 0;
  $: if (message !== '') {
    console.log('success message', message);
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      message = '';
    }, 5000);
  }

  function dismissMessage() {
    message = '';
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      dismissMessage();
    }
  }
</script>

{#if message}
  <div 
    class="fixed right-0 bottom-0 m-4 cursor-pointer" style="z-index: 1000;"
    on:click={dismissMessage}
    on:keydown={handleKeydown}
    role="button"
    tabindex="0"
    aria-label="Dismiss success message"
  >
    <div class="rounded bg-green-800 p-4 text-white">
      {message}
    </div>
    <div class="bar transition-all duration-500" style="width: 0;"></div>
  </div>
{/if}
