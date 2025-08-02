<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Button from './Button.svelte';
  import HandTracking from './HandTracking.svelte';

  let showInputControls: boolean = false;
  
  // Microphone state
  let microphoneAccess: boolean = false;
  let mediaStream: MediaStream | null = null;
  let audioContext: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let microphoneError: string = '';
  
  // Hand tracking state
  let handTrackingAccess: boolean = false;
  let handTrackingError: string = '';
  
  // Active input controls - using array to show as forms
  let inputControlConfigs: Array<{
    id: string;
    type: string;
    parameter_name: string;
    min_value: number;
    max_value: number;
    sensitivity: number;
    update_rate: number;
    is_active: boolean;
    current_value: number;
    intervalId?: number;
    hand_index?: number;
    show_visualizer?: boolean;
  }> = [];
  
  let isLoading: boolean = false;
  let statusMessage: string = '';
  let controlnetInfo: any = null;
  let promptBlendingConfig: any = null;
  let seedBlendingConfig: any = null;
  
  // Base parameter options
  const baseParameterOptions = [
    { value: 'guidance_scale', label: 'Guidance Scale', min: 0.0, max: 2.0 },
    { value: 'delta', label: 'Delta', min: 0.0, max: 1.0 },
    { value: 'num_inference_steps', label: 'Inference Steps', min: 1, max: 100 },
    { value: 'seed', label: 'Seed', min: 0, max: 100000 },
    { value: 'ipadapter_scale', label: 'IPAdapter Scale', min: 0.0, max: 2.0 }
  ];

  // Dynamic parameter options including ControlNet strengths
  let parameterOptions: Array<{value: string, label: string, min: number, max: number}> = [...baseParameterOptions];

  onMount(() => {
    fetchControlNetInfo();
    fetchBlendingConfigs();
    const interval = setInterval(() => {
      fetchControlNetInfo();
      fetchBlendingConfigs();
    }, 3000); // Check for updates
    
    return () => clearInterval(interval);
  });

  onDestroy(() => {
    stopAllMicrophoneControls();
    stopAllHandTrackingControls();
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
      audioContext.close();
    }
  });

  async function fetchControlNetInfo() {
    try {
      const response = await fetch('/api/settings');
      if (response.ok) {
        const settings = await response.json();
        if (settings.controlnet) {
          controlnetInfo = settings.controlnet;
          updateParameterOptions();
        }
      }
    } catch (error) {
      console.error('fetchControlNetInfo: Failed to get ControlNet info:', error);
    }
  }

  async function fetchBlendingConfigs() {
    try {
      const response = await fetch('/api/blending/current');
      if (response.ok) {
        const data = await response.json();
        if (data.prompt_blending) {
          promptBlendingConfig = data.prompt_blending;
        }
        if (data.seed_blending) {
          seedBlendingConfig = data.seed_blending;
        }
        updateParameterOptions();
      }
    } catch (error) {
      console.error('fetchBlendingConfigs: Failed to get blending configs:', error);
    }
  }

  function updateParameterOptions() {
    // Reset to base options
    parameterOptions = [...baseParameterOptions];
    
    // Add ControlNet strength parameters
    if (controlnetInfo?.enabled && controlnetInfo?.controlnets) {
      controlnetInfo.controlnets.forEach((controlnet: any) => {
        parameterOptions.push({
          value: `controlnet_${controlnet.index}_strength`,
          label: `${controlnet.name} Strength`,
          min: 0.0,
          max: 2.0
        });
      });
    }

    // Add prompt blending weights
    if (promptBlendingConfig?.prompts) {
      promptBlendingConfig.prompts.forEach((prompt: any, index: number) => {
        parameterOptions.push({
          value: `prompt_weight_${index}`,
          label: `Prompt ${index + 1} Weight`,
          min: 0.0,
          max: 2.0
        });
      });
    }

    // Add seed blending weights
    if (seedBlendingConfig?.seeds) {
      seedBlendingConfig.seeds.forEach((seed: any, index: number) => {
        parameterOptions.push({
          value: `seed_weight_${index}`,
          label: `Seed ${index + 1} Weight`,
          min: 0.0,
          max: 2.0
        });
      });
    }
  }

  async function requestMicrophoneAccess(): Promise<boolean> {
    try {
      microphoneError = '';
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone access not supported in this browser');
      }

      mediaStream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });
      
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      
      const source = audioContext.createMediaStreamSource(mediaStream);
      source.connect(analyser);
      
      microphoneAccess = true;
      statusMessage = 'Microphone access granted';
      return true;
      
    } catch (error) {
      console.error('requestMicrophoneAccess: Failed to access microphone:', error);
      microphoneError = error instanceof Error ? error.message : 'Failed to access microphone';
      microphoneAccess = false;
      return false;
    }
  }

  function getCurrentMicrophoneLevel(): number {
    if (!analyser) return 0;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    
    // Calculate RMS volume
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / bufferLength);
    
    // Normalize to 0-1 range
    return Math.min(1.0, rms / 128.0);
  }

  async function addInputControl(type: 'microphone' | 'hand_tracking') {
    if (type === 'microphone' && !microphoneAccess) {
      const granted = await requestMicrophoneAccess();
      if (!granted) {
        statusMessage = 'Microphone access required';
        return;
      }
    }
    
    if (type === 'hand_tracking' && !handTrackingAccess) {
      const granted = await requestHandTrackingAccess();
      if (!granted) {
        statusMessage = 'Camera access required for hand tracking';
        return;
      }
    }
    
    isLoading = true;
    statusMessage = '';
    
    try {
      // Generate unique ID
      const newId = `input_${Date.now()}`;
      const selectedParam = parameterOptions.find(p => p.value === 'guidance_scale') || parameterOptions[0];
      
      const newControl = {
        id: newId,
        type: type,
        parameter_name: selectedParam.value,
        min_value: selectedParam.min,
        max_value: selectedParam.max,
        sensitivity: 1.0,
        update_rate: type === 'microphone' ? 0.1 : 0.05,
        is_active: false,
        current_value: 0,
        intervalId: type === 'microphone' ? 0 : undefined,
        hand_index: type === 'hand_tracking' ? 0 : undefined,
        show_visualizer: type === 'hand_tracking' ? true : undefined
      };
      
      inputControlConfigs = [...inputControlConfigs, newControl];
      statusMessage = `${type === 'microphone' ? 'Microphone' : 'Hand tracking'} input control added successfully`;
      
    } catch (error) {
      statusMessage = 'Failed to add input control';
      console.error('addInputControl: Error:', error);
    } finally {
      isLoading = false;
    }
  }

  async function startMicrophoneControl(control: any) {
    if (!microphoneAccess) return;
    
    const updateInterval = Math.max(50, control.update_rate * 1000);
    
    control.intervalId = setInterval(async () => {
      try {
        const level = getCurrentMicrophoneLevel();
        const sensitiveLevel = Math.min(1.0, level * control.sensitivity);
        const scaledValue = control.min_value + (sensitiveLevel * (control.max_value - control.min_value));
        
        control.current_value = scaledValue;
        
        // Send parameter update to backend
        if (control.parameter_name.startsWith('controlnet_') && control.parameter_name.endsWith('_strength')) {
          await updateControlNetParameter(control, scaledValue);
        } else if (control.parameter_name.startsWith('prompt_weight_')) {
          await updatePromptWeightParameter(control, scaledValue);
        } else if (control.parameter_name.startsWith('seed_weight_')) {
          await updateSeedWeightParameter(control, scaledValue);
        } else {
          const endpoint = getParameterUpdateEndpoint(control.parameter_name);
          if (endpoint) {
            await fetch(endpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ [getParameterKey(control.parameter_name)]: scaledValue })
            });
          }
        }
      } catch (error) {
        console.error('startMicrophoneControl: Update failed:', error);
      }
    }, updateInterval);
    
    control.is_active = true;
    inputControlConfigs = [...inputControlConfigs]; // Trigger reactivity
    statusMessage = `Started input control ${control.id}`;
  }

  function stopMicrophoneControl(control: any) {
    if (control.intervalId) {
      clearInterval(control.intervalId);
      control.intervalId = 0;
      control.is_active = false;
      inputControlConfigs = [...inputControlConfigs]; // Trigger reactivity
      statusMessage = `Stopped input control ${control.id}`;
    }
  }

  function stopAllMicrophoneControls() {
    inputControlConfigs.forEach(control => {
      if (control.type === 'microphone' && control.intervalId) {
        clearInterval(control.intervalId);
        control.intervalId = 0;
        control.is_active = false;
      }
    });
    inputControlConfigs = [...inputControlConfigs]; // Trigger reactivity
  }

  function stopAllHandTrackingControls() {
    inputControlConfigs.forEach(control => {
      if (control.type === 'hand_tracking' && control.is_active) {
        control.is_active = false;
      }
    });
    inputControlConfigs = [...inputControlConfigs]; // Trigger reactivity
  }

  async function requestHandTrackingAccess(): Promise<boolean> {
    try {
      handTrackingError = '';
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access not supported in this browser');
      }

      handTrackingAccess = true;
      statusMessage = 'Hand tracking access granted';
      return true;
      
    } catch (error) {
      console.error('requestHandTrackingAccess: Failed to access camera for hand tracking:', error);
      handTrackingError = error instanceof Error ? error.message : 'Failed to access camera for hand tracking';
      handTrackingAccess = false;
      return false;
    }
  }

  function handleHandTrackingValueChange(control: any, value: number) {
    const scaledValue = control.min_value + (value * (control.max_value - control.min_value));
    control.current_value = scaledValue;
    
    // Store pending update but don't send immediately - use same pattern as microphone control
    control.pendingValue = scaledValue;
  }

  function getParameterUpdateEndpoint(parameterName: string): string | null {
    // Handle ControlNet strength parameters
    if (parameterName.startsWith('controlnet_') && parameterName.endsWith('_strength')) {
      return '/api/controlnet/update-strength';
    }
    
    // Handle prompt weight parameters
    if (parameterName.startsWith('prompt_weight_')) {
      return '/api/blending/update-prompt-weight';
    }
    
    // Handle seed weight parameters
    if (parameterName.startsWith('seed_weight_')) {
      return '/api/blending/update-seed-weight';
    }
    
    const endpoints: Record<string, string> = {
      'guidance_scale': '/api/update-guidance-scale',
      'delta': '/api/update-delta', 
      'num_inference_steps': '/api/update-num-inference-steps',
      'seed': '/api/update-seed',
      'ipadapter_scale': '/api/ipadapter/update-scale'
    };
    return endpoints[parameterName] || null;
  }

  function getParameterKey(parameterName: string): string {
    // Handle ControlNet strength parameters
    if (parameterName.startsWith('controlnet_') && parameterName.endsWith('_strength')) {
      const match = parameterName.match(/controlnet_(\d+)_strength/);
      if (match) {
        return 'strength'; // API expects 'strength' key with 'index' key
      }
    }
    
    // Handle prompt weight parameters
    if (parameterName.startsWith('prompt_weight_')) {
      return 'weight'; // API expects 'weight' key with 'index' key
    }
    
    // Handle seed weight parameters
    if (parameterName.startsWith('seed_weight_')) {
      return 'weight'; // API expects 'weight' key with 'index' key
    }
    
    const keys: Record<string, string> = {
      'guidance_scale': 'guidance_scale',
      'delta': 'delta',
      'num_inference_steps': 'num_inference_steps', 
      'seed': 'seed',
      'ipadapter_scale': 'scale'
    };
    return keys[parameterName] || parameterName;
  }

  // Handle ControlNet parameter updates differently 
  async function updateControlNetParameter(control: any, scaledValue: number) {
    const match = control.parameter_name.match(/controlnet_(\d+)_strength/);
    if (match) {
      const index = parseInt(match[1]);
      await fetch('/api/controlnet/update-strength', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          index: index, 
          strength: scaledValue 
        })
      });
    }
  }

  // Handle prompt weight parameter updates
  async function updatePromptWeightParameter(control: any, scaledValue: number) {
    const match = control.parameter_name.match(/prompt_weight_(\d+)/);
    if (match) {
      const index = parseInt(match[1]);
      await fetch('/api/blending/update-prompt-weight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          index: index, 
          weight: scaledValue 
        })
      }).catch(error => {
        console.error('updatePromptWeightParameter: Update failed:', error);
      });
    }
  }

  // Handle seed weight parameter updates
  async function updateSeedWeightParameter(control: any, scaledValue: number) {
    const match = control.parameter_name.match(/seed_weight_(\d+)/);
    if (match) {
      const index = parseInt(match[1]);
      await fetch('/api/blending/update-seed-weight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          index: index, 
          weight: scaledValue 
        })
      }).catch(error => {
        console.error('updateSeedWeightParameter: Update failed:', error);
      });
    }
  }

  async function startHandTrackingControl(control: any) {
    // Initialize pending value tracking
    control.pendingValue = null;
    control.lastSentValue = null;
    
    // Set up controlled update interval like microphone control
    const updateInterval = Math.max(50, control.update_rate * 1000);
    control.intervalId = setInterval(async () => {
      if (control.pendingValue !== null && control.pendingValue !== control.lastSentValue) {
        try {
          // Send parameter update to backend
          if (control.parameter_name.startsWith('controlnet_') && control.parameter_name.endsWith('_strength')) {
            await updateControlNetParameter(control, control.pendingValue);
          } else if (control.parameter_name.startsWith('prompt_weight_')) {
            await updatePromptWeightParameter(control, control.pendingValue);
          } else if (control.parameter_name.startsWith('seed_weight_')) {
            await updateSeedWeightParameter(control, control.pendingValue);
          } else {
            const endpoint = getParameterUpdateEndpoint(control.parameter_name);
            if (endpoint) {
              await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [getParameterKey(control.parameter_name)]: control.pendingValue })
              });
            }
          }
          control.lastSentValue = control.pendingValue;
        } catch (error) {
          console.error('startHandTrackingControl: Update failed:', error);
        }
      }
    }, updateInterval);

    control.is_active = true;
    inputControlConfigs = [...inputControlConfigs];
    statusMessage = `Started hand tracking control ${control.id}`;
  }

  function stopHandTrackingControl(control: any) {
    // Clear interval like microphone control
    if (control.intervalId) {
      clearInterval(control.intervalId);
      control.intervalId = null;
    }
    
    // Clean up tracking state
    control.pendingValue = null;
    control.lastSentValue = null;
    
    control.is_active = false;
    inputControlConfigs = [...inputControlConfigs];
    statusMessage = `Stopped hand tracking control ${control.id}`;
  }

  function removeInputControl(index: number) {
    const control = inputControlConfigs[index];
    if (control.is_active) {
      if (control.type === 'microphone') {
        stopMicrophoneControl(control);
      } else if (control.type === 'hand_tracking') {
        control.is_active = false;
      }
    }
    inputControlConfigs = inputControlConfigs.filter((_, i) => i !== index);
    statusMessage = `Removed input control ${control.id}`;
  }

  function toggleInputControl(index: number) {
    const control = inputControlConfigs[index];
    if (control.is_active) {
      if (control.type === 'microphone') {
        stopMicrophoneControl(control);
      } else if (control.type === 'hand_tracking') {
        stopHandTrackingControl(control);
      }
    } else {
      if (control.type === 'microphone') {
        startMicrophoneControl(control);
      } else if (control.type === 'hand_tracking') {
        startHandTrackingControl(control);
      }
    }
  }

  function updateControlParameter(index: number, field: string, value: any) {
    const control = inputControlConfigs[index];
    
    if (field === 'parameter_name') {
      control.parameter_name = value;
      const selectedParam = parameterOptions.find(p => p.value === value);
      if (selectedParam) {
        control.min_value = selectedParam.min;
        control.max_value = selectedParam.max;
      }
    } else if (field === 'min_value') {
      control.min_value = value;
    } else if (field === 'max_value') {
      control.max_value = value;
    } else if (field === 'sensitivity') {
      control.sensitivity = value;
    } else if (field === 'update_rate') {
      control.update_rate = value;
    } else if (field === 'hand_index') {
      control.hand_index = value;
    } else if (field === 'show_visualizer') {
      control.show_visualizer = value;
    }
    
    inputControlConfigs = [...inputControlConfigs]; // Trigger reactivity
  }
</script>

<div class="input-control-panel">
  <div class="panel-header">
    <button 
      class="toggle-button"
      on:click={() => showInputControls = !showInputControls}
      class:expanded={showInputControls}
    >
      Input Controls {showInputControls ? '−' : '+'}
    </button>
  </div>

  {#if showInputControls}
    <div class="panel-content">
      <!-- Microphone Access Status -->
      <div class="mic-status">
        <div class="flex items-center justify-between">
          <span class="text-sm">Microphone Access:</span>
          <span class="status-badge" class:active={microphoneAccess}>
            {microphoneAccess ? 'Granted' : 'Not Granted'}
          </span>
        </div>
        {#if !microphoneAccess}
          <Button on:click={requestMicrophoneAccess} classList="mt-2">
            Request Microphone Access
          </Button>
        {/if}
        {#if microphoneError}
          <p class="text-red-500 text-sm mt-1">{microphoneError}</p>
        {/if}
      </div>

      <!-- Hand Tracking Access Status -->
      <div class="hand-tracking-status">
        <div class="flex items-center justify-between">
          <span class="text-sm">Hand Tracking Access:</span>
          <span class="status-badge" class:active={handTrackingAccess}>
            {handTrackingAccess ? 'Granted' : 'Not Granted'}
          </span>
        </div>
        {#if !handTrackingAccess}
          <Button on:click={requestHandTrackingAccess} classList="mt-2">
            Request Camera Access for Hand Tracking
          </Button>
        {/if}
        {#if handTrackingError}
          <p class="text-red-500 text-sm mt-1">{handTrackingError}</p>
        {/if}
      </div>

      <!-- Status Message -->
      {#if statusMessage}
        <div class="status-message" class:error={statusMessage.includes('Failed')}>
          {statusMessage}
        </div>
      {/if}

      <!-- Add New Input Control Buttons -->
      <div class="add-button-section">
        <div class="flex gap-2 flex-wrap">
          <Button 
            on:click={() => addInputControl('microphone')} 
            disabled={isLoading || !microphoneAccess}
          >
            {isLoading ? 'Adding...' : 'Add Microphone Control'}
          </Button>
          <Button 
            on:click={() => addInputControl('hand_tracking')} 
            disabled={isLoading || !handTrackingAccess}
          >
            {isLoading ? 'Adding...' : 'Add Hand Tracking Control'}
          </Button>
        </div>
      </div>

      <!-- Input Control Configurations -->
      {#if inputControlConfigs.length === 0}
        <div class="no-controls">
          <p class="text-gray-500 text-sm italic">No input controls configured</p>
        </div>
      {:else}
        <div class="controls-list">
          {#each inputControlConfigs as control, index}
            <div class="control-form">
              <div class="control-header">
                <div class="control-title">
                  <strong>
                    {control.type === 'hand_tracking' ? `Hand Tracking ${index + 1} (Hand ${control.hand_index || 0})` : `Input Control ${index + 1}`}
                  </strong>
                  <span class="status-badge" class:active={control.is_active}>
                    {control.is_active ? 'Active' : 'Inactive'}
                  </span>
                  <span class="value-display">
                    {control.current_value.toFixed(3)}
                  </span>
                </div>
                <div class="control-actions">
                  <Button on:click={() => toggleInputControl(index)}>
                    {control.is_active ? 'Stop' : 'Start'}
                  </Button>
                  <Button on:click={() => removeInputControl(index)}>
                    Remove
                  </Button>
                </div>
              </div>
              
              <div class="control-config">
                <div class="form-grid">
                  <div class="form-group">
                    <label>Parameter:</label>
                                         <select 
                       bind:value={control.parameter_name}
                       on:change={(e) => updateControlParameter(index, 'parameter_name', (e.target as HTMLSelectElement).value)}
                     >
                       {#each parameterOptions as option}
                         <option value={option.value}>{option.label}</option>
                       {/each}
                     </select>
                   </div>
                   
                   <div class="form-group">
                     <label>Min Value:</label>
                     <input 
                       type="number" 
                       bind:value={control.min_value}
                       on:input={(e) => updateControlParameter(index, 'min_value', parseFloat((e.target as HTMLInputElement).value))}
                       step="0.1" 
                     />
                   </div>
                   
                   <div class="form-group">
                     <label>Max Value:</label>
                     <input 
                       type="number" 
                       bind:value={control.max_value}
                       on:input={(e) => updateControlParameter(index, 'max_value', parseFloat((e.target as HTMLInputElement).value))}
                       step="0.1" 
                     />
                   </div>
                   
                   <div class="form-group">
                     <label>Sensitivity:</label>
                     <input 
                       type="number" 
                       bind:value={control.sensitivity}
                       on:input={(e) => updateControlParameter(index, 'sensitivity', parseFloat((e.target as HTMLInputElement).value))}
                       step="0.1" 
                       min="0.1" 
                       max="10" 
                     />
                   </div>
                   
                   <div class="form-group">
                     <label>Update Rate (s):</label>
                     <input 
                       type="number" 
                       bind:value={control.update_rate}
                       on:input={(e) => updateControlParameter(index, 'update_rate', parseFloat((e.target as HTMLInputElement).value))}
                       step="0.05" 
                       min="0.05" 
                       max="1.0" 
                     />
                   </div>
                   
                                       {#if control.type === 'hand_tracking'}
                      <div class="form-group">
                        <label>Hand Index:</label>
                        <select 
                          bind:value={control.hand_index}
                          on:change={(e) => updateControlParameter(index, 'hand_index', parseInt((e.target as HTMLSelectElement).value))}
                        >
                          <option value={0}>Hand 0</option>
                          <option value={1}>Hand 1</option>
                          <option value={2}>Hand 2</option>
                          <option value={3}>Hand 3</option>
                        </select>
                      </div>
                      
                      <div class="form-group">
                        <label>
                          <input 
                            type="checkbox" 
                            bind:checked={control.show_visualizer}
                            on:change={(e) => updateControlParameter(index, 'show_visualizer', (e.target as HTMLInputElement).checked)}
                          />
                          Show Visualizer
                        </label>
                      </div>
                    {/if}
                                 </div>
               </div>
               
                                <!-- Hand Tracking Component for hand tracking controls -->
                 {#if control.type === 'hand_tracking'}
                   <div class="hand-tracking-section">
                     <HandTracking 
                       isActive={control.is_active}
                       sensitivity={control.sensitivity}
                       handIndex={control.hand_index || 0}
                       showVisualizer={control.show_visualizer || false}
                       onValueChange={(value) => handleHandTrackingValueChange(control, value)}
                     />
                   </div>
                 {/if}
             </div>
           {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .input-control-panel {
    border: 1px solid #374151;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    background: #1f2937;
  }

  .panel-header {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #374151;
  }

  .toggle-button {
    background: none;
    border: none;
    color: #f3f4f6;
    cursor: pointer;
    font-weight: 500;
    width: 100%;
    text-align: left;
    font-size: 0.9rem;
  }

  .toggle-button:hover {
    color: #93c5fd;
  }

  .panel-content {
    padding: 1rem;
  }

  .mic-status {
    padding: 0.75rem;
    border: 1px solid #374151;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
    background: #111827;
  }

  .status-message {
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
    background: #059669;
    color: white;
    font-size: 0.875rem;
  }

  .status-message.error {
    background: #dc2626;
  }

  .add-button-section {
    margin-bottom: 1rem;
  }

  .no-controls {
    text-align: center;
    padding: 2rem;
  }

  .controls-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .control-form {
    border: 1px solid #374151;
    border-radius: 0.5rem;
    background: #111827;
  }

  .control-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    border-bottom: 1px solid #374151;
  }

  .control-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .control-title strong {
    color: #f3f4f6;
  }

  .control-actions {
    display: flex;
    gap: 0.5rem;
  }

  .control-config {
    padding: 0.75rem;
  }

  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
  }

  .form-group label {
    color: #d1d5db;
    font-size: 0.875rem;
    margin-bottom: 0.25rem;
  }

  .form-group input,
  .form-group select {
    padding: 0.5rem;
    border: 1px solid #374151;
    border-radius: 0.25rem;
    background: #1f2937;
    color: #f3f4f6;
    font-size: 0.875rem;
  }

  .status-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    background: #dc2626;
    color: white;
  }

  .status-badge.active {
    background: #059669;
  }

  .value-display {
    color: #d1d5db;
    font-size: 0.875rem;
    font-family: monospace;
  }

  .flex {
    display: flex;
  }

  .items-center {
    align-items: center;
  }

  .justify-between {
    justify-content: space-between;
  }

  .text-sm {
    font-size: 0.875rem;
  }

  .text-red-500 {
    color: #ef4444;
  }

  .text-gray-500 {
    color: #6b7280;
  }

  .mt-1 {
    margin-top: 0.25rem;
  }

  .mt-2 {
    margin-top: 0.5rem;
  }

  .italic {
    font-style: italic;
  }

  .hand-tracking-section {
    margin-top: 1rem;
    padding: 1rem;
    border: 1px solid #374151;
    border-radius: 0.5rem;
    background: #111827;
  }

  .hand-tracking-status {
    padding: 1rem;
    border: 1px solid #374151;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    background: #111827;
  }
</style> 