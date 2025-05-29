#!/usr/bin/env python3
"""
General Purpose ControlNet Webcam Demo for StreamDiffusion

This script demonstrates real-time image generation using webcam input with any ControlNet configuration.
It loads a ControlNet config file and applies the specified preprocessing and conditioning to the webcam feed.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys
import time

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from streamdiffusion.controlnet import (
    load_controlnet_config, 
    create_controlnet_pipeline,
    ControlNetConfig
)
from streamdiffusion.image_utils import postprocess_image


def main():
    parser = argparse.ArgumentParser(description="General Purpose ControlNet Webcam Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent / "configs" / "controlnet_examples" / "depth_trt_example.yaml"
    
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str,
                       help="Override base model path from config")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--controlnet-scale", type=float,
                       help="Override ControlNet conditioning scale from config")
    parser.add_argument("--show-preprocessed", action="store_true",
                       help="Show the preprocessed control image in a separate window")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Camera and output resolution")
    
    args = parser.parse_args()
    
    print("🎨 Starting ControlNet Webcam Demo")
    
    # Load configuration
    config = load_controlnet_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")
    
    # Override parameters if provided
    if args.model:
        config.model_id = args.model
    if args.prompt:
        config.prompt = args.prompt
    if args.controlnet_scale is not None:
        config.controlnets[0].conditioning_scale = args.controlnet_scale
    
    # Create ControlNet pipeline
    print("🔄 Creating ControlNet pipeline...")
    print(f"📝 Using ControlNet: {config.controlnets[0].model_id}")
    print(f"🔧 Preprocessor: {config.controlnets[0].preprocessor}")
    pipeline = create_controlnet_pipeline(config)
    print("✓ Pipeline created successfully")
    
    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution)
    
    print("✓ Camera opened successfully")
    print(f"📝 Prompt: {config.prompt}")
    print(f"🎛️  ControlNet Scale: {config.controlnets[0].conditioning_scale}")
    print(f"📏 Resolution: {args.resolution}x{args.resolution}")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current output")
    print("  - Press 'c' to toggle control image preview")
    print("  - Press '+' to increase ControlNet scale")
    print("  - Press '-' to decrease ControlNet scale")
    print("  - Press 'p' to change prompt interactively")
    
    frame_count = 0
    show_preprocessed = args.show_preprocessed
    fps_counter = []
    
    # Get preprocessor name for display
    preprocessor_name = config.controlnets[0].preprocessor.replace("_", " ").title()
    
    # Profiling variables
    profile_times = {
        'preprocessing': [],
        'generation': [],
        'display': [],
        'total': []
    }
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Profile preprocessing
            prep_start = time.time()
            pipeline.update_control_image_efficient(frame_pil)
            prep_time = time.time() - prep_start
            profile_times['preprocessing'].append(prep_time)
            
            # Profile generation
            gen_start = time.time()
            x_output = pipeline(frame_pil)
            output_image = postprocess_image(x_output, output_type="pil")[0]
            gen_time = time.time() - gen_start
            profile_times['generation'].append(gen_time)
            
            # Profile display processing
            display_start = time.time()
            # Convert back to BGR for display
            output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # Get preprocessed control image from cache (avoid reprocessing)
            control_cv = None
            control_pil = None
            if show_preprocessed and len(pipeline.preprocessors) > 0:
                preprocessor = pipeline.preprocessors[0]
                if preprocessor is not None:
                    control_pil = pipeline.get_last_processed_image(0)
                    if control_pil is not None:
                        control_cv = cv2.cvtColor(np.array(control_pil), cv2.COLOR_RGB2BGR)
            
            # Create display layout
            display_frame = cv2.resize(frame, (args.resolution, args.resolution))
            output_display = cv2.resize(output_cv, (args.resolution, args.resolution))
            
            if show_preprocessed and control_cv is not None:
                control_display = cv2.resize(control_cv, (args.resolution//2, args.resolution//2))
                # Create 3-panel layout: Input | Control | Output
                top_row = np.hstack([
                    cv2.resize(display_frame, (args.resolution//2, args.resolution//2)), 
                    control_display
                ])
                bottom_row = cv2.resize(output_display, (args.resolution, args.resolution//2))
                combined = np.vstack([top_row, bottom_row])
            else:
                # Simple side-by-side layout
                combined = np.hstack([display_frame, output_display])
            
            display_time = time.time() - display_start
            profile_times['display'].append(display_time)
            
            # Calculate FPS
            end_time = time.time()
            frame_time = end_time - start_time
            fps_counter.append(frame_time)
            profile_times['total'].append(frame_time)
            
            if len(fps_counter) > 30:  # Keep last 30 frames
                fps_counter.pop(0)
            avg_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            
            # Show profiling info every 30 frames
            if frame_count % 30 == 0 and frame_count > 0:
                recent_prep = profile_times['preprocessing'][-30:]
                recent_gen = profile_times['generation'][-30:]
                recent_display = profile_times['display'][-30:]
                recent_total = profile_times['total'][-30:]
                
                print(f"\n📊 Performance Profile (last 30 frames):")
                print(f"  Preprocessing: {sum(recent_prep)/len(recent_prep)*1000:.1f}ms avg")
                print(f"  Generation:    {sum(recent_gen)/len(recent_gen)*1000:.1f}ms avg")
                print(f"  Display:       {sum(recent_display)/len(recent_display)*1000:.1f}ms avg")
                print(f"  Total:         {sum(recent_total)/len(recent_total)*1000:.1f}ms avg")
                print(f"  FPS:           {avg_fps:.1f}")
                
                # Check if tensor processing is being used
                preprocessor = pipeline.preprocessors[0]
                if preprocessor and hasattr(preprocessor, 'process_tensor'):
                    print(f"  ✓ Tensor processing available for {type(preprocessor).__name__}")
                else:
                    print(f"  ⚠️  No tensor processing for {type(preprocessor).__name__}")
            
            # Add info overlay
            current_scale = pipeline.controlnet_scales[0]
            info_text = f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Scale: {current_scale:.2f}"
            
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add preprocessor info
            preprocessor_text = f"Preprocessor: {preprocessor_name}"
            cv2.putText(combined, preprocessor_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add timing info
            if frame_count > 0:
                timing_text = f"Prep: {prep_time*1000:.1f}ms | Gen: {gen_time*1000:.1f}ms"
                cv2.putText(combined, timing_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Add labels
            if show_preprocessed and control_cv is not None:
                cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, f"{preprocessor_name}", (args.resolution//2 + 10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Generated", (10, combined.shape[0] - args.resolution//2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Generated", (args.resolution + 10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'ControlNet StreamDiffusion - {preprocessor_name}', combined)
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current output
                try:
                    timestamp = int(time.time())
                    preprocessor_safe = config.controlnets[0].preprocessor.replace("/", "_").replace("\\", "_")
                    output_path = f"controlnet_{preprocessor_safe}_output_{timestamp}.png"
                    output_image.save(output_path)
                    print(f"💾 Saved output to {output_path}")
                    
                    if show_preprocessed and control_cv is not None:
                        control_path = f"controlnet_{preprocessor_safe}_control_{timestamp}.png"
                        control_pil.save(control_path)
                        print(f"💾 Saved control image to {control_path}")
                        
                except Exception as save_error:
                    print(f"❌ Failed to save: {save_error}")
            elif key == ord('c'):
                # Toggle control image preview
                show_preprocessed = not show_preprocessed
                print(f"🖼️  Control image preview: {'ON' if show_preprocessed else 'OFF'}")
            elif key == ord('+'):
                new_scale = min(2.0, pipeline.controlnet_scales[0] + 0.1)
                pipeline.update_controlnet_scale(0, new_scale)
                print(f"📈 ControlNet scale: {new_scale:.2f}")
            elif key == ord('-'):
                new_scale = max(0.0, pipeline.controlnet_scales[0] - 0.1)
                pipeline.update_controlnet_scale(0, new_scale)
                print(f"📉 ControlNet scale: {new_scale:.2f}")
            elif key == ord('p'):
                # Interactive prompt change
                print(f"\n🎨 Enter new prompt (or press Enter to keep current):")
                try:
                    new_prompt = input(f"Current: {config.prompt}\nNew: ").strip()
                    if new_prompt:
                        pipeline.stream.update_prompt(new_prompt)
                        config.prompt = new_prompt
                        print(f"✓ Updated prompt: {new_prompt}")
                except:
                    print("❌ Failed to update prompt")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 ControlNet demo finished")
        if fps_counter:
            avg_fps = len(fps_counter) / sum(fps_counter)
            print(f"📊 Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main() 