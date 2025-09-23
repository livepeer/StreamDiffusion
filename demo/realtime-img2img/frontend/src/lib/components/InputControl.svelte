<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Button from './Button.svelte';
  import HandTracking from './HandTracking.svelte';
  import GamepadControl from './GamepadControl.svelte';
  import { appState } from '$lib/store';

  // Toggle moved to parent page
  
  // Microphone state
  let microphoneAccess: boolean = false;
  let mediaStream: MediaStream | null = null;
  let audioContext: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let microphoneError: string = '';
  
  // Hand tracking state
  let handTrackingAccess: boolean = false;
  let handTrackingError: string = '';
  
  // Gamepad state
  let gamepadAccess: boolean = false;
  let gamepadError: string = '';
  
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
    gamepad_index?: number;
    axis_index?: number;
    deadzone?: number;
  }> = [];
  
  let isLoading: boolean = false;
  let statusMessage: string = '';
  let controlnetInfo: any = null;
  let promptBlendingConfig: any = null;
  let seedBlendingConfig: any = null;
  
  function nudgeControl(index: number, field: string, delta: number, step: number, min?: number, max?: number) {
    const control = inputControlConfigs[index] as any;
    const current = parseFloat(control[field] ?? 0) || 0;
    const next = current + delta * step;
    const clamped = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, parseFloat(next.toFixed(3))));
    updateControlParameter(index, field, clamped);
  }
  
  // Dynamic parameter options - built from API responses
  let parameterOptions: Array<{value: string, label: string, min: number, max: number, category?: string}> = [];

  onMount(() => {
    fetchAllParameterInfo();
    checkGamepadAccess();
    // Parameter info is now updated via centralized state polling
    // Remove local 3s polling interval
  });

  onDestroy(() => {
    stopAllMicrophoneControls();
    stopAllHandTrackingControls();
    stopAllGamepadControls();
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
      audioContext.close();
    }
  });

  async function fetchAllParameterInfo() {
    try {
      // Use centralized state if available, fallback to individual API calls
      if ($appState) {
        await updateParameterOptions($appState, {
          prompt_blending: $appState.prompt_blending,
          seed_blending: $appState.seed_blending
        });
      } else {
        // Fallback to individual API calls
        const settingsResponse = await fetch('/api/settings');
        if (settingsResponse.ok) {
          const settings = await settingsResponse.json();
          
          // Also fetch latest blending configs
          const blendingResponse = await fetch('/api/blending/current');
          let blendingData = null;
          if (blendingResponse.ok) {
            blendingData = await blendingResponse.json();
          }
          
          await updateParameterOptions(settings, blendingData);
        }
      }
    } catch (error) {
      console.error('fetchAllParameterInfo: Failed to get parameter info:', error);
    }
  }
  
  // Reactive update when appState changes
  $: if ($appState) {
    updateParameterOptions($appState, {
      prompt_blending: $appState.prompt_blending,
      seed_blending: $appState.seed_blending
    });
  }

  async function fetchPreprocessorParameters(controlnetIndex: number): Promise<any> {
    try {
      const response = await fetch(`/api/preprocessors/current-params/${controlnetIndex}`);
      if (response.ok) {
        const data = await response.json();
        return data.parameters || {};
      } else {
        console.warn(`fetchPreprocessorParameters: HTTP ${response.status} for ControlNet ${controlnetIndex} - preprocessor may not be initialized yet`);
        return {};
      }
    } catch (error) {
      console.warn(`fetchPreprocessorParameters: Failed to get preprocessor params for ControlNet ${controlnetIndex}:`, error);
    }
    return {};
  }

  async function fetchPreprocessorMetadata(): Promise<any> {
    try {
      const response = await fetch('/api/preprocessors/info');
      if (response.ok) {
        const data = await response.json();
        return data.preprocessors || {};
      } else {
        console.warn('fetchPreprocessorMetadata: Failed to get preprocessor metadata');
        return {};
      }
    } catch (error) {
      console.warn('fetchPreprocessorMetadata: Failed to get preprocessor metadata:', error);
    }
    return {};
  }

  async function updateParameterOptions(settings: any, blendingData: any = null) {
    const parameters: Array<{value: string, label: string, min: number, max: number, category?: string}> = [];
    
    // Base pipeline parameters from input_params schema
    if (settings.input_params?.properties) {
      const inputParams = settings.input_params.properties;
      
      // Core pipeline parameters with better defaults
      const coreParams = [
        { key: 'guidance_scale', label: 'Guidance Scale', min: 0.0, max: 2.0, current: settings.guidance_scale },
        { key: 'delta', label: 'Delta', min: 0.0, max: 1.0, current: settings.delta },
        { key: 'num_inference_steps', label: 'Inference Steps', min: 1, max: 100, current: settings.num_inference_steps },
        { key: 'seed', label: 'Seed', min: 0, max: 100000, current: settings.seed }
      ];
      
      coreParams.forEach(param => {
        parameters.push({
          value: param.key,
          label: param.label,
          min: param.min,
          max: param.max,
          category: 'pipeline'
        });
      });
    }

    // IPAdapter parameters
    if (settings.ipadapter) {
      parameters.push({
        value: 'ipadapter_scale',
        label: 'IPAdapter Scale',
        min: 0.0,
        max: 2.0,
        category: 'ipadapter'
      });
      
      parameters.push({
        value: 'ipadapter_weight_type',
        label: 'IPAdapter Weight Type',
        min: 0,
        max: 14, // 15 weight types (0-14)
        category: 'ipadapter'
      });
    }

    // ControlNet strength parameters
    if (settings.controlnet?.enabled && settings.controlnet?.controlnets) {
      for (const controlnet of settings.controlnet.controlnets) {
        parameters.push({
          value: `controlnet_${controlnet.index}_strength`,
          label: `${controlnet.name} Strength`,
          min: 0.0,
          max: 2.0,
          category: 'controlnet'
        });
      }
      
      // TODO: Fix preprocessor parameter controls - currently disabled due to feedback preprocessor detection issues
      // ControlNet preprocessor parameters - get metadata for proper ranges
      // const preprocessorMetadata = await fetchPreprocessorMetadata();
      // 
      // for (let i = 0; i < settings.controlnet.controlnets.length; i++) {
      //   try {
      //     const preprocessorParams = await fetchPreprocessorParameters(i);
      //     const controlnetName = settings.controlnet.controlnets[i]?.name || `ControlNet ${i}`;
      //     
      //     // Get current preprocessor info to find metadata
      //     const currentPreprocessorResponse = await fetch(`/api/preprocessors/current-params/${i}`);
      //     let currentPreprocessorName = null;
      //     if (currentPreprocessorResponse.ok) {
      //       const currentData = await currentPreprocessorResponse.json();
      //       currentPreprocessorName = currentData.preprocessor;
      //     }
      //     
      //     if (Object.keys(preprocessorParams).length > 0 && currentPreprocessorName && preprocessorMetadata[currentPreprocessorName]) {
      //       const preprocessorMeta = preprocessorMetadata[currentPreprocessorName];
      //       const paramMetadata = preprocessorMeta.parameters || {};
      //       
      //       Object.keys(preprocessorParams).forEach(paramName => {
      //         if (!['device', 'dtype', 'image_width', 'image_height'].includes(paramName)) {
      //           const currentValue = preprocessorParams[paramName];
      //           const paramMeta = paramMetadata[paramName];
      //           
      //           let min = 0, max = 100;
      //           
      //           if (paramMeta && paramMeta.range) {
      //             min = paramMeta.range[0];
      //             max = paramMeta.range[1];
      //           } else {
      //             // Fallback to smart defaults based on parameter name
      //             if (paramName.includes('threshold') || paramName.includes('scale')) {
      //               min = 0; max = 1;
      //             } else if (paramName.includes('sigma') || paramName.includes('blur')) {
      //               min = 0; max = 10;
      //             } else if (paramName.includes('steps')) {
      //               min = 1; max = 100;
      //             }
      //           }
      //           
      //           parameters.push({
      //             value: `controlnet_${i}_preprocessor_${paramName}`,
      //             label: `${controlnetName} ${paramName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}`,
      //             min,
      //             max,
      //             category: 'preprocessor'
      //           });
      //         }
      //       });
      //     }
      //   } catch (error) {
      //     console.warn(`updateParameterOptions: Failed to get preprocessor params for ControlNet ${i}:`, error);
      //   }
      // }
    }

    // Use blending data (prioritize the passed parameter, fallback to settings)
    const promptBlending = blendingData?.prompt_blending || settings.prompt_blending;
    const seedBlending = blendingData?.seed_blending || settings.seed_blending;

    // Prompt blending weights - check if it's an array directly or has prompts property
    if (Array.isArray(promptBlending) && promptBlending.length > 0) {
      promptBlending.forEach((prompt: any, index: number) => {
        parameters.push({
          value: `prompt_weight_${index}`,
          label: `Prompt ${index + 1} Weight (${prompt[0] ? prompt[0].substring(0, 20) + '...' : 'Prompt'})`,
          min: 0.0,
          max: 2.0,
          category: 'prompt_blending'
        });
      });
    } else if (promptBlending?.prompts) {
      promptBlending.prompts.forEach((prompt: any, index: number) => {
        parameters.push({
          value: `prompt_weight_${index}`,
          label: `Prompt ${index + 1} Weight`,
          min: 0.0,
          max: 2.0,
          category: 'prompt_blending'
        });
      });
    }

    // Seed blending weights - check if it's an array directly or has seeds property
    if (Array.isArray(seedBlending) && seedBlending.length > 0) {
      seedBlending.forEach((seed: any, index: number) => {
        parameters.push({
          value: `seed_weight_${index}`,
          label: `Seed ${index + 1} Weight (${seed[0]})`,
          min: 0.0,
          max: 2.0,
          category: 'seed_blending'
        });
      });
    } else if (seedBlending?.seeds) {
      seedBlending.seeds.forEach((seed: any, index: number) => {
        parameters.push({
          value: `seed_weight_${index}`,
          label: `Seed ${index + 1} Weight`,
          min: 0.0,
          max: 2.0,
          category: 'seed_blending'
        });
      });
    }

    // Update the parameter options
    parameterOptions = parameters;
    

    
    // Store the configs for later use in parameter updates
    controlnetInfo = settings.controlnet;
    promptBlendingConfig = promptBlending;
    seedBlendingConfig = seedBlending;
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

  async function addInputControl(type: 'microphone' | 'hand_tracking' | 'gamepad') {
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
    
    if (type === 'gamepad' && !gamepadAccess) {
      checkGamepadAccess();
      if (!gamepadAccess) {
        statusMessage = 'Gamepad API not available';
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
        show_visualizer: type === 'hand_tracking' ? true : undefined,
        gamepad_index: type === 'gamepad' ? 0 : undefined,
        axis_index: type === 'gamepad' ? 0 : undefined,
        deadzone: type === 'gamepad' ? 0.1 : undefined
      };
      
      inputControlConfigs = [...inputControlConfigs, newControl];
      const typeNames = {
        'microphone': 'Microphone',
        'hand_tracking': 'Hand tracking',
        'gamepad': 'Gamepad'
      };
      statusMessage = `${typeNames[type]} input control added successfully`;
      
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
        } else if (control.parameter_name.startsWith('controlnet_') && control.parameter_name.includes('_preprocessor_')) {
          // TODO: Re-enable when preprocessor parameter update is fixed
          // await updatePreprocessorParameter(control, scaledValue);
        } else if (control.parameter_name.startsWith('prompt_weight_')) {
          await updatePromptWeightParameter(control, scaledValue);
        } else if (control.parameter_name.startsWith('seed_weight_')) {
          await updateSeedWeightParameter(control, scaledValue);
        } else if (control.parameter_name === 'ipadapter_weight_type') {
          // Convert numeric value to weight type string
          const weightTypes = ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
                             "weak input", "weak output", "weak middle", "strong middle", 
                             "style transfer", "composition", "strong style transfer", 
                             "style and composition", "style transfer precise", "composition precise"];
          const index = Math.round(scaledValue) % weightTypes.length;
          const weightType = weightTypes[index];
          
          const endpoint = getParameterUpdateEndpoint(control.parameter_name);
          if (endpoint) {
            await fetch(endpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ weight_type: weightType })
            });
          }
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

  function stopAllGamepadControls() {
    inputControlConfigs.forEach(control => {
      if (control.type === 'gamepad' && control.is_active) {
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

  function checkGamepadAccess(): void {
    try {
      gamepadError = '';
      
      if (!navigator.getGamepads) {
        throw new Error('Gamepad API not supported in this browser');
      }

      gamepadAccess = true;
      
    } catch (error) {
      console.error('checkGamepadAccess: Gamepad API not available:', error);
      gamepadError = error instanceof Error ? error.message : 'Gamepad API not available';
      gamepadAccess = false;
    }
  }

  function handleHandTrackingValueChange(control: any, value: number) {
    const scaledValue = control.min_value + (value * (control.max_value - control.min_value));
    control.current_value = scaledValue;
    
    // Store pending update but don't send immediately - use same pattern as microphone control
    control.pendingValue = scaledValue;
  }

  function handleGamepadValueChange(control: any, value: number) {
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
    
    // TODO: Re-enable when preprocessor parameter update is fixed
    // Handle ControlNet preprocessor parameters
    // if (parameterName.startsWith('controlnet_') && parameterName.includes('_preprocessor_')) {
    //   return '/api/preprocessors/update-params';
    // }
    
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
      'ipadapter_scale': '/api/ipadapter/update-scale',
      'ipadapter_weight_type': '/api/ipadapter/update-weight-type'
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
      'ipadapter_scale': 'scale',
      'ipadapter_weight_type': 'weight_type'
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

  // Handle preprocessor parameter updates
  async function updatePreprocessorParameter(control: any, scaledValue: number) {
    const match = control.parameter_name.match(/controlnet_(\d+)_preprocessor_(.+)/);
    if (match) {
      const controlnetIndex = parseInt(match[1]);
      const paramName = match[2];
      
      await fetch('/api/preprocessors/update-params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          controlnet_index: controlnetIndex,
          params: {
            [paramName]: scaledValue
          }
        })
      }).catch(error => {
        console.error('updatePreprocessorParameter: Update failed:', error);
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
          } else if (control.parameter_name.startsWith('controlnet_') && control.parameter_name.includes('_preprocessor_')) {
            // TODO: Re-enable when preprocessor parameter update is fixed
            // await updatePreprocessorParameter(control, control.pendingValue);
          } else if (control.parameter_name.startsWith('prompt_weight_')) {
            await updatePromptWeightParameter(control, control.pendingValue);
          } else if (control.parameter_name.startsWith('seed_weight_')) {
            await updateSeedWeightParameter(control, control.pendingValue);
          } else if (control.parameter_name === 'ipadapter_weight_type') {
            // Convert numeric value to weight type string
            const weightTypes = ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
                               "weak input", "weak output", "weak middle", "strong middle", 
                               "style transfer", "composition", "strong style transfer", 
                               "style and composition", "style transfer precise", "composition precise"];
            const index = Math.round(control.pendingValue) % weightTypes.length;
            const weightType = weightTypes[index];
            
            const endpoint = getParameterUpdateEndpoint(control.parameter_name);
            if (endpoint) {
              await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ weight_type: weightType })
              });
            }
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

  async function startGamepadControl(control: any) {
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
          } else if (control.parameter_name.startsWith('controlnet_') && control.parameter_name.includes('_preprocessor_')) {
            // TODO: Re-enable when preprocessor parameter update is fixed
            // await updatePreprocessorParameter(control, control.pendingValue);
          } else if (control.parameter_name.startsWith('prompt_weight_')) {
            await updatePromptWeightParameter(control, control.pendingValue);
          } else if (control.parameter_name.startsWith('seed_weight_')) {
            await updateSeedWeightParameter(control, control.pendingValue);
          } else if (control.parameter_name === 'ipadapter_weight_type') {
            // Convert numeric value to weight type string
            const weightTypes = ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
                               "weak input", "weak output", "weak middle", "strong middle", 
                               "style transfer", "composition", "strong style transfer", 
                               "style and composition", "style transfer precise", "composition precise"];
            const index = Math.round(control.pendingValue) % weightTypes.length;
            const weightType = weightTypes[index];
            
            const endpoint = getParameterUpdateEndpoint(control.parameter_name);
            if (endpoint) {
              await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ weight_type: weightType })
              });
            }
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
          console.error('startGamepadControl: Update failed:', error);
        }
      }
    }, updateInterval);

    control.is_active = true;
    inputControlConfigs = [...inputControlConfigs];
    statusMessage = `Started gamepad control ${control.id}`;
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

  function stopGamepadControl(control: any) {
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
    statusMessage = `Stopped gamepad control ${control.id}`;
  }

  function removeInputControl(index: number) {
    const control = inputControlConfigs[index];
    if (control.is_active) {
      if (control.type === 'microphone') {
        stopMicrophoneControl(control);
      } else if (control.type === 'hand_tracking') {
        control.is_active = false;
      } else if (control.type === 'gamepad') {
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
      } else if (control.type === 'gamepad') {
        stopGamepadControl(control);
      }
    } else {
      if (control.type === 'microphone') {
        startMicrophoneControl(control);
      } else if (control.type === 'hand_tracking') {
        startHandTrackingControl(control);
      } else if (control.type === 'gamepad') {
        startGamepadControl(control);
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
    } else if (field === 'gamepad_index') {
      control.gamepad_index = value;
    } else if (field === 'axis_index') {
      control.axis_index = value;
    } else if (field === 'deadzone') {
      control.deadzone = value;
    }
    
    inputControlConfigs = [...inputControlConfigs]; // Trigger reactivity
  }
</script>

<div class="space-y-4">
      <!-- Microphone Access Status -->
      <div class="mic-status bg-gray-50 dark:bg-gray-700 rounded p-3">
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
      <div class="hand-tracking-status bg-gray-50 dark:bg-gray-700 rounded p-3">
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

      <!-- Gamepad Access Status -->
      <div class="gamepad-status bg-gray-50 dark:bg-gray-700 rounded p-3">
        <div class="flex items-center justify-between">
          <span class="text-sm">Gamepad Access:</span>
          <span class="status-badge" class:active={gamepadAccess}>
            {gamepadAccess ? 'Available' : 'Not Available'}
          </span>
        </div>
        {#if !gamepadAccess}
          <Button on:click={checkGamepadAccess} classList="mt-2">
            Check Gamepad Support
          </Button>
        {/if}
        {#if gamepadError}
          <p class="text-red-500 text-sm mt-1">{gamepadError}</p>
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
          <Button 
            on:click={() => addInputControl('gamepad')} 
            disabled={isLoading || !gamepadAccess}
          >
            {isLoading ? 'Adding...' : 'Add Gamepad Control'}
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
            <div class="control-form bg-gray-50 dark:bg-gray-700 rounded p-3 space-y-3">
              <div class="control-header">
                <div class="control-title">
                  <strong>
                    {#if control.type === 'hand_tracking'}
                      Hand Tracking {index + 1} (Hand {control.hand_index || 0})
                    {:else if control.type === 'gamepad'}
                      Gamepad {index + 1} (Gamepad {control.gamepad_index || 0}, Axis {control.axis_index || 0})
                    {:else}
                      Input Control {index + 1}
                    {/if}
                  </strong>
                  <span class="status-badge" class:active={control.is_active}>
                    {control.is_active ? 'Active' : 'Inactive'}
                  </span>
                  <span class="value-display">
                    {(control.current_value || 0).toFixed(3)}
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
                    <label for={`param-${index}`}>Parameter:</label>
                     <select 
                       id={`param-${index}`}
                       bind:value={control.parameter_name}
                       on:change={(e) => updateControlParameter(index, 'parameter_name', (e.target as HTMLSelectElement).value)}
                       class="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                     >
                       {#each parameterOptions as option}
                         <option value={option.value}>{option.label}</option>
                       {/each}
                     </select>
                   </div>
                   
                    <div class="form-group">
                      <label for={`min-${index}`}>Min Value:</label>
                      <div class="flex items-center gap-2">
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'min_value', -1, 0.1)}
                        >-</button>
                        <input 
                          type="number" 
                          id={`min-${index}`}
                          bind:value={control.min_value}
                          on:input={(e) => updateControlParameter(index, 'min_value', parseFloat((e.target as HTMLInputElement).value))}
                          step="0.1" 
                          class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        />
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'min_value', 1, 0.1)}
                        >+</button>
                      </div>
                    </div>
                   
                    <div class="form-group">
                      <label for={`max-${index}`}>Max Value:</label>
                      <div class="flex items-center gap-2">
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'max_value', -1, 0.1)}
                        >-</button>
                        <input 
                          type="number" 
                          id={`max-${index}`}
                          bind:value={control.max_value}
                          on:input={(e) => updateControlParameter(index, 'max_value', parseFloat((e.target as HTMLInputElement).value))}
                          step="0.1" 
                          class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        />
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'max_value', 1, 0.1)}
                        >+</button>
                      </div>
                    </div>
                   
                    <div class="form-group">
                      <label for={`sens-${index}`}>Sensitivity:</label>
                      <div class="flex items-center gap-2">
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'sensitivity', -1, 0.1, 0.1, 10)}
                        >-</button>
                        <input 
                          type="number" 
                          id={`sens-${index}`}
                          bind:value={control.sensitivity}
                          on:input={(e) => updateControlParameter(index, 'sensitivity', parseFloat((e.target as HTMLInputElement).value))}
                          step="0.1" 
                          min="0.1" 
                          max="10" 
                          class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        />
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'sensitivity', 1, 0.1, 0.1, 10)}
                        >+</button>
                      </div>
                    </div>
                   
                    <div class="form-group">
                      <label for={`rate-${index}`}>Update Rate (s):</label>
                      <div class="flex items-center gap-2">
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'update_rate', -1, 0.05, 0.05, 1.0)}
                        >-</button>
                        <input 
                          type="number" 
                          id={`rate-${index}`}
                          bind:value={control.update_rate}
                          on:input={(e) => updateControlParameter(index, 'update_rate', parseFloat((e.target as HTMLInputElement).value))}
                          step="0.05" 
                          min="0.05" 
                          max="1.0" 
                          class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        />
                        <button
                          class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          on:click={() => nudgeControl(index, 'update_rate', 1, 0.05, 0.05, 1.0)}
                        >+</button>
                      </div>
                    </div>
                   
                                       {#if control.type === 'hand_tracking'}
                      <div class="form-group">
                        <label for={`hand-${index}`}>Hand Index:</label>
                        <select 
                          id={`hand-${index}`}
                          bind:value={control.hand_index}
                          on:change={(e) => updateControlParameter(index, 'hand_index', parseInt((e.target as HTMLSelectElement).value))}
                          class="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
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
                    
                    {#if control.type === 'gamepad'}
                      <div class="form-group">
                        <label for={`gpad-${index}`}>Gamepad Index:</label>
                        <select 
                          id={`gpad-${index}`}
                          bind:value={control.gamepad_index}
                          on:change={(e) => updateControlParameter(index, 'gamepad_index', parseInt((e.target as HTMLSelectElement).value))}
                          class="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        >
                          <option value={0}>Gamepad 0</option>
                          <option value={1}>Gamepad 1</option>
                          <option value={2}>Gamepad 2</option>
                          <option value={3}>Gamepad 3</option>
                        </select>
                      </div>
                      
                      <div class="form-group">
                        <label for={`axis-${index}`}>Axis Index:</label>
                        <select 
                          id={`axis-${index}`}
                          bind:value={control.axis_index}
                          on:change={(e) => updateControlParameter(index, 'axis_index', parseInt((e.target as HTMLSelectElement).value))}
                          class="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                        >
                          <option value={0}>Axis 0 (Left Stick X)</option>
                          <option value={1}>Axis 1 (Left Stick Y)</option>
                          <option value={2}>Axis 2 (Right Stick X)</option>
                          <option value={3}>Axis 3 (Right Stick Y)</option>
                          <option value={4}>Axis 4 (Left Trigger)</option>
                          <option value={5}>Axis 5 (Right Trigger)</option>
                        </select>
                      </div>
                      
                      <div class="form-group">
                        <label for={`dead-${index}`}>Deadzone:</label>
                        <div class="flex items-center gap-2">
                          <button
                            class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                            on:click={() => nudgeControl(index, 'deadzone', -1, 0.01, 0.0, 0.5)}
                          >-</button>
                          <input 
                            type="number" 
                            id={`dead-${index}`}
                            bind:value={control.deadzone}
                            on:input={(e) => updateControlParameter(index, 'deadzone', parseFloat((e.target as HTMLInputElement).value))}
                            step="0.01" 
                            min="0.0" 
                            max="0.5" 
                            class="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                          />
                          <button
                            class="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
                            on:click={() => nudgeControl(index, 'deadzone', 1, 0.01, 0.0, 0.5)}
                          >+</button>
                        </div>
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
                 
                 <!-- Gamepad Component for gamepad controls -->
                 {#if control.type === 'gamepad'}
                   <div class="gamepad-section">
                     <GamepadControl 
                       isActive={control.is_active}
                       sensitivity={control.sensitivity}
                       gamepadIndex={control.gamepad_index || 0}
                       axisIndex={control.axis_index || 0}
                       deadzone={control.deadzone || 0.1}
                       showVisualizer={control.show_visualizer || false}
                       onValueChange={(value) => handleGamepadValueChange(control, value)}
                     />
                   </div>
                 {/if}
             </div>
           {/each}
        </div>
      {/if}
</div>

<style>
  /* removed legacy .panel-content */

  .mic-status {
    padding: 0.75rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
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
    border-radius: 0.5rem;
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
    border-radius: 0.5rem;
    margin-bottom: 1rem;
  }

  .gamepad-status {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
  }
</style> 