<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { gamepadService, type GamepadCallback } from '$lib/gamepadService';

  export let isActive: boolean = false;
  export let onValueChange: (value: number) => void = () => {};
  export let sensitivity: number = 1.0;
  export let gamepadIndex: number = 0;
  export let axisIndex: number = 0;
  export let deadzone: number = 0.1;
  export let showVisualizer: boolean = true;

  let canvasElement: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let controlId: string;
  
  let currentValue: number = 0;
  let gamepadError: string = '';
  let lastGamepadData: Gamepad | null = null;
  let connectedGamepads: Gamepad[] = [];
  
  onMount(() => {
    if (canvasElement) {
      ctx = canvasElement.getContext('2d')!;
    }
    controlId = `gamepad_${Date.now()}_${Math.random()}`;
    
    // Check for connected gamepads
    updateConnectedGamepads();
    
    // Listen for gamepad connection events
    window.addEventListener('gamepadconnected', handleGamepadConnected);
    window.addEventListener('gamepaddisconnected', handleGamepadDisconnected);
  });

  onDestroy(() => {
    cleanup();
    window.removeEventListener('gamepadconnected', handleGamepadConnected);
    window.removeEventListener('gamepaddisconnected', handleGamepadDisconnected);
  });

  function handleGamepadConnected(event: GamepadEvent) {
    updateConnectedGamepads();
  }

  function handleGamepadDisconnected(event: GamepadEvent) {
    updateConnectedGamepads();
  }

  function updateConnectedGamepads() {
    connectedGamepads = gamepadService.getConnectedGamepads();
  }

  function onGamepadData(gamepad: Gamepad) {
    lastGamepadData = gamepad;
    
    if (!showVisualizer || !ctx || !canvasElement) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw gamepad visualization
    drawGamepadVisualization(gamepad);
  }

  function drawGamepadVisualization(gamepad: Gamepad) {
    if (!ctx || !canvasElement) return;
    
    const canvasWidth = canvasElement.width;
    const canvasHeight = canvasElement.height;
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    
    // Draw background
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    
    // Draw gamepad name
    ctx.fillStyle = '#f3f4f6';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(gamepad.id || `Gamepad ${gamepad.index}`, centerX, 20);
    
    // Draw connection status
    ctx.fillStyle = gamepad.connected ? '#059669' : '#dc2626';
    ctx.fillText(gamepad.connected ? 'Connected' : 'Disconnected', centerX, 35);
    
    // Draw axis visualization
    if (axisIndex < gamepad.axes.length) {
      const axisValue = gamepad.axes[axisIndex];
      
      // Draw axis background circle
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 60, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Draw deadzone circle
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 60 * deadzone, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Draw axis position
      const radius = 60;
      const x = centerX + (axisValue * radius);
      const y = centerY;
      
      // Draw axis indicator
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw center line
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX - radius, centerY);
      ctx.lineTo(centerX + radius, centerY);
      ctx.stroke();
      
      // Draw value text
      ctx.fillStyle = '#f3f4f6';
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`Axis ${axisIndex}: ${(axisValue || 0).toFixed(3)}`, centerX, centerY + 100);
      ctx.fillText(`Normalized: ${(currentValue || 0).toFixed(3)}`, centerX, centerY + 115);
    }
    
    // Draw all axes as small bars
    const barWidth = 4;
    const barHeight = 40;
    const barSpacing = 8;
    const startX = 10;
    const startY = canvasHeight - 60;
    
    gamepad.axes.forEach((value, index) => {
      const x = startX + (index * (barWidth + barSpacing));
      const isActiveAxis = index === axisIndex;
      
      // Draw bar background
      ctx.fillStyle = isActiveAxis ? '#374151' : '#1f2937';
      ctx.fillRect(x, startY, barWidth, barHeight);
      
      // Draw bar value
      const normalizedValue = (value + 1) / 2;
      const barFillHeight = barHeight * normalizedValue;
      ctx.fillStyle = isActiveAxis ? '#3b82f6' : '#6b7280';
      ctx.fillRect(x, startY + barHeight - barFillHeight, barWidth, barFillHeight);
      
      // Draw axis label
      ctx.fillStyle = isActiveAxis ? '#f3f4f6' : '#6b7280';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(index.toString(), x + barWidth / 2, startY + barHeight + 12);
    });
  }

  async function startGamepadControl(): Promise<boolean> {
    try {
      gamepadError = '';
      
      // Check if gamepad is available
      updateConnectedGamepads();
      if (connectedGamepads.length === 0) {
        throw new Error('No gamepads connected. Please connect a gamepad and try again.');
      }
      
      if (gamepadIndex >= connectedGamepads.length) {
        throw new Error(`Gamepad ${gamepadIndex} not found. Available gamepads: ${connectedGamepads.length}`);
      }

      // Register with the shared service
      const callback: GamepadCallback = {
        gamepadIndex,
        axisIndex,
        deadzone,
        sensitivity,
        onValueChange: (value) => {
          currentValue = value;
          onValueChange(value);
        },
        onGamepadData: showVisualizer ? onGamepadData : undefined
      };

      gamepadService.registerCallback(controlId, callback);
      return true;
      
    } catch (error) {
      console.error('startGamepadControl: Failed to start gamepad control:', error);
      gamepadError = error instanceof Error ? error.message : 'Failed to start gamepad control';
      return false;
    }
  }

  function stopGamepadControl() {
    gamepadService.unregisterCallback(controlId);
  }

  function cleanup() {
    gamepadService.unregisterCallback(controlId);
  }

  // Update callback when parameters change
  $: if (isActive && controlId) {
    const callback: GamepadCallback = {
      gamepadIndex,
      axisIndex,
      deadzone,
      sensitivity,
      onValueChange: (value) => {
        currentValue = value;
        onValueChange(value);
      },
      onGamepadData: showVisualizer ? onGamepadData : undefined
    };
    gamepadService.registerCallback(controlId, callback);
  }

  $: if (isActive) {
    startGamepadControl();
  } else {
    stopGamepadControl();
  }

  $: canvasWidth = 320;
  $: canvasHeight = 240;
</script>

{#if showVisualizer}
  <div class="gamepad-control-container">
    <div class="relative">
      <canvas
        bind:this={canvasElement}
        width={canvasWidth}
        height={canvasHeight}
        class="border border-gray-300 rounded"
      ></canvas>
      
      <div class="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        Gamepad {gamepadIndex}, Axis {axisIndex}: {(currentValue || 0).toFixed(3)}
      </div>
    </div>
    
    {#if gamepadError}
      <p class="text-red-500 text-sm mt-2">{gamepadError}</p>
    {/if}
    
    <!-- Connected Gamepads Info -->
    <div class="gamepad-info">
      <div class="text-sm text-gray-400">
        Connected Gamepads: {connectedGamepads.length}
      </div>
      {#each connectedGamepads as gamepad, index}
        <div class="text-xs text-gray-500">
          {index}: {gamepad.id || `Gamepad ${gamepad.index}`} ({gamepad.axes.length} axes)
        </div>
      {/each}
    </div>
  </div>
{:else}
  <div class="gamepad-info">
    <div class="text-sm text-gray-400">
      Gamepad {gamepadIndex}, Axis {axisIndex}: {(currentValue || 0).toFixed(3)}
    </div>
    <div class="text-xs text-gray-500">
      Connected: {connectedGamepads.length}
    </div>
    {#if gamepadError}
      <p class="text-red-500 text-sm mt-1">{gamepadError}</p>
    {/if}
  </div>
{/if}

<style>
  .gamepad-control-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .gamepad-info {
    padding: 0.5rem;
    text-align: center;
  }
</style> 