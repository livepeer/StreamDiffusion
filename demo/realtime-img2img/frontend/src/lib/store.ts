import { derived, writable, get, type Writable, type Readable } from 'svelte/store';

export const pipelineValues: Writable<Record<string, any>> = writable({});
export const deboucedPipelineValues: Readable<Record<string, any>>
    = derived(pipelineValues, ($pipelineValues, set) => {
        const debounced = setTimeout(() => {
            set($pipelineValues);
        }, 100);
        return () => clearTimeout(debounced);
    });

// FPS tracking store
export const fps = writable<number>(0);

// FPS calculator class
class FPSCalculator {
    private frameTimestamps: number[] = [];
    private readonly maxSamples = 10; // Keep last 10 frames for calculation
    private lastUpdateTime = 0;

    recordFrame() {
        const now = performance.now();
        this.frameTimestamps.push(now);
        
        // Remove old timestamps (keep only recent ones)
        if (this.frameTimestamps.length > this.maxSamples) {
            this.frameTimestamps.shift();
        }
        
        // Calculate FPS only if we have enough samples and enough time has passed
        if (this.frameTimestamps.length >= 2) {
            const timeSpan = this.frameTimestamps[this.frameTimestamps.length - 1] - this.frameTimestamps[0];
            if (timeSpan > 0) {
                const currentFPS = ((this.frameTimestamps.length - 1) / timeSpan) * 1000;
                
                // Update FPS display every 500ms to avoid flickering
                if (now - this.lastUpdateTime > 500) {
                    fps.set(Math.round(currentFPS * 10) / 10); // Round to 1 decimal place
                    this.lastUpdateTime = now;
                }
            }
        }
    }

    reset() {
        this.frameTimestamps = [];
        this.lastUpdateTime = 0;
        fps.set(0);
    }
}

export const fpsCalculator = new FPSCalculator();

export const getPipelineValues = () => get(pipelineValues);