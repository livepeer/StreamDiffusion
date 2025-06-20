// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
  namespace App {
    // interface Error {}
    // interface Locals {}
    // interface PageData {}
    // interface Platform {}
  }

  // MediaPipe global objects
  interface Window {
    Pose: any;
    POSE_CONNECTIONS: any;
    POSE_LANDMARKS: any;
  }
}

export {};
