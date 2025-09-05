// Shared gamepad service to handle multiple gamepad controls
export interface GamepadCallback {
  gamepadIndex: number;
  axisIndex: number;
  deadzone: number;
  sensitivity: number;
  onValueChange: (value: number) => void;
  onGamepadData?: (gamepad: Gamepad) => void;
}

class GamepadService {
  private isActive: boolean = false;
  private callbacks: Map<string, GamepadCallback> = new Map();
  private lastValues: Map<string, number> = new Map();
  private lastGamepadConnected: Map<string, boolean> = new Map();
  private smoothedValues: Map<string, number> = new Map();
  private lastUpdateTime: Map<string, number> = new Map();
  private readonly VALUE_CHANGE_THRESHOLD = 0.003;
  private readonly SMOOTHING_FACTOR = 0.6;
  private readonly MAX_UPDATE_INTERVAL = 50;
  private animationFrameId: number | null = null;

  registerCallback(id: string, callback: GamepadCallback): void {
    this.callbacks.set(id, callback);
    // Initialize tracking state for this callback
    this.lastValues.set(id, -1);
    this.lastGamepadConnected.set(id, false);
    this.smoothedValues.set(id, 0);
    this.lastUpdateTime.set(id, 0);
    
    // Start processing if this is the first callback
    if (this.callbacks.size === 1 && !this.isActive) {
      this.start();
    }
  }

  unregisterCallback(id: string): void {
    this.callbacks.delete(id);
    this.lastValues.delete(id);
    this.lastGamepadConnected.delete(id);
    this.smoothedValues.delete(id);
    this.lastUpdateTime.delete(id);
    
    // Stop processing if no more callbacks
    if (this.callbacks.size === 0) {
      this.stop();
    }
  }

  private processGamepads(): void {
    const currentTime = Date.now();
    
    // Get all connected gamepads
    const gamepads = navigator.getGamepads();
    
    // Process each registered callback
    this.callbacks.forEach((callback, callbackId) => {
      const lastValue = this.lastValues.get(callbackId) ?? -1;
      const lastGamepadConnected = this.lastGamepadConnected.get(callbackId) ?? false;
      const lastSmoothedValue = this.smoothedValues.get(callbackId) ?? 0;
      const lastUpdateTime = this.lastUpdateTime.get(callbackId) ?? 0;
      
      let rawValue = 0;
      let gamepadCurrentlyConnected = false;
      
      // Check if the target gamepad is connected
      const gamepad = gamepads[callback.gamepadIndex];
      if (gamepad && gamepad.connected && callback.axisIndex < gamepad.axes.length) {
        gamepadCurrentlyConnected = true;
        
        // Get axis value
        const axisValue = gamepad.axes[callback.axisIndex];
        
        // Apply deadzone
        let processedValue = axisValue;
        if (Math.abs(axisValue) < callback.deadzone) {
          processedValue = 0;
        }
        
        // Convert from [-1, 1] to [0, 1] range
        const normalizedValue = (processedValue + 1.0) / 2.0;
        
        // Apply sensitivity
        rawValue = Math.min(1.0, Math.max(0.0, normalizedValue * callback.sensitivity));
      }
      
      // Apply exponential smoothing when gamepad is connected
      let smoothedValue: number;
      if (gamepadCurrentlyConnected) {
        smoothedValue = (this.SMOOTHING_FACTOR * lastSmoothedValue) + ((1 - this.SMOOTHING_FACTOR) * rawValue);
      } else {
        smoothedValue = 0; // Reset when gamepad disconnects
      }
      
      // Update smoothed value
      this.smoothedValues.set(callbackId, smoothedValue);
      
      // Send value updates when:
      // 1. Gamepad connection state changes
      // 2. Gamepad is connected AND (value changed OR enough time has passed)
      const gamepadStateChanged = gamepadCurrentlyConnected !== lastGamepadConnected;
      const valueChangedSignificantly = Math.abs(smoothedValue - lastValue) >= this.VALUE_CHANGE_THRESHOLD;
      const timeSinceLastUpdate = currentTime - lastUpdateTime;
      const timeForUpdate = timeSinceLastUpdate >= this.MAX_UPDATE_INTERVAL;
      
      if (gamepadStateChanged || (gamepadCurrentlyConnected && (valueChangedSignificantly || timeForUpdate))) {
        callback.onValueChange(smoothedValue);
        this.lastValues.set(callbackId, smoothedValue);
        this.lastUpdateTime.set(callbackId, currentTime);
      }
      
      // Update gamepad connection state
      this.lastGamepadConnected.set(callbackId, gamepadCurrentlyConnected);
      
      // Send gamepad data for visualization if callback wants it
      if (callback.onGamepadData && gamepad) {
        callback.onGamepadData(gamepad);
      }
    });
    
    // Continue processing
    if (this.isActive) {
      this.animationFrameId = requestAnimationFrame(() => this.processGamepads());
    }
  }

  start(): void {
    if (this.isActive) return;
    
    this.isActive = true;
    this.processGamepads();
  }

  stop(): void {
    this.isActive = false;
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  getConnectedGamepads(): Gamepad[] {
    const gamepads = navigator.getGamepads();
    return Array.from(gamepads).filter((gamepad): gamepad is Gamepad => gamepad !== null && gamepad.connected);
  }

  getGamepadInfo(gamepadIndex: number): Gamepad | null {
    const gamepads = navigator.getGamepads();
    const gamepad = gamepads[gamepadIndex];
    return gamepad && gamepad.connected ? gamepad : null;
  }

  getActiveGamepadCount(): number {
    return this.callbacks.size;
  }

  isServiceActive(): boolean {
    return this.isActive;
  }

  cleanup(): void {
    this.stop();
    this.callbacks.clear();
    this.lastValues.clear();
    this.lastGamepadConnected.clear();
    this.smoothedValues.clear();
    this.lastUpdateTime.clear();
  }
}

// Singleton instance
export const gamepadService = new GamepadService(); 