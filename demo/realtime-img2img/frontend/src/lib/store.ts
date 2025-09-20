
import { derived, writable, get, type Writable, type Readable } from 'svelte/store';

// Legacy pipeline values store for backward compatibility
export const pipelineValues: Writable<Record<string, any>> = writable({});
export const deboucedPipelineValues: Readable<Record<string, any>>
    = derived(pipelineValues, ($pipelineValues, set) => {
        const debounced = setTimeout(() => {
            set($pipelineValues);
        }, 100);
        return () => clearTimeout(debounced);
    });

// Comprehensive application state store
export interface AppState {
    // Pipeline state
    pipeline_active: boolean;
    pipeline_lifecycle: string;
    
    // Configuration
    config_needs_reload: boolean;
    
    // Resolution
    current_resolution: {
        width: number;
        height: number;
    };
    
    // Parameters
    pipeline_params: any;
    controlnet: any;
    ipadapter: any;
    prompt_blending: any;
    seed_blending: any;
    normalize_prompt_weights: boolean;
    normalize_seed_weights: boolean;
    
    // Core parameters
    guidance_scale: number;
    delta: number;
    num_inference_steps: number;
    seed: number;
    t_index_list: number[];
    negative_prompt: string;
    skip_diffusion: boolean;
    
    // UI state
    fps: number;
    queue_size: number;
    model_id: string;
    page_content: string;
    
    // Input sources
    input_sources: Record<string, any>;
    
    // Additional fields for backward compatibility
    info?: any;
    input_params?: any;
    max_queue_size?: number;
    acceleration?: string;
    config_prompt?: string;
    resolution?: string;
    image_preprocessing?: any;
    image_postprocessing?: any;
    latent_preprocessing?: any;
    latent_postprocessing?: any;
    config_values?: Record<string, any>;
}

// Create the comprehensive app state store
export const appState: Writable<AppState | null> = writable(null);

// Derived store for debounced pipeline values (maintains backward compatibility)
export const debouncedAppState: Readable<AppState | null> = derived(
    appState, 
    ($appState, set) => {
        const debounced = setTimeout(() => {
            set($appState);
        }, 100);
        return () => clearTimeout(debounced);
    }
);

// State management functions
let pollingInterval: number | null = null;

export async function fetchAppState(): Promise<AppState | null> {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) {
            throw new Error(`fetchAppState: HTTP ${response.status}: ${response.statusText}`);
        }
        
        const state = await response.json();
        
        // Update both stores for backward compatibility
        appState.set(state);
        
        // Update legacy pipelineValues store with relevant values
        pipelineValues.update(values => ({
            ...values,
            prompt: state.config_prompt || values.prompt,
            negative_prompt: state.negative_prompt,
            resolution: state.resolution,
            ...state.config_values
        }));
        
        return state;
    } catch (error) {
        console.error('fetchAppState: Failed to fetch app state:', error);
        return null;
    }
}

export function startStatePolling(intervalMs: number = 5000) {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    // Initial fetch
    fetchAppState();
    
    // Set up polling
    pollingInterval = setInterval(fetchAppState, intervalMs);
    console.log(`startStatePolling: Started polling every ${intervalMs}ms`);
}

export function stopStatePolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log('stopStatePolling: Stopped state polling');
    }
}

// Legacy function for backward compatibility
export const getPipelineValues = () => get(pipelineValues);

// New function to get current app state
export const getAppState = () => get(appState);