#!/usr/bin/env python3
"""
Multi-ControlNet Webcam Demo for StreamDiffusion

This script demonstrates real-time image generation using webcam input with multiple ControlNet configurations.
It loads a config with multiple ControlNets and applies all preprocessing and conditioning to the webcam feed.
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
    parser = argparse.ArgumentParser(description="Multi-ControlNet Webcam Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent / "configs" / "controlnet_examples" / "multi_controlnet_example.yaml"
    
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to multi-ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str,
                       help="Override base model path from config")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--show-preprocessed", action="store_true",
                       help="Show preprocessed control images")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Camera and output resolution")
    
    args = parser.parse_args()
    
    print("🎨 Starting Multi-ControlNet Webcam Demo")
    
    # Load configuration
    config = load_controlnet_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")
    
    # Validate multi-ControlNet setup
    if len(config.controlnets) < 2:
        print("⚠️  Warning: Config has less than 2 ControlNets. Consider using single ControlNet demo.")
    
    # Override parameters if provided
    if args.model:
        config.model_id = args.model
    if args.prompt:
        config.prompt = args.prompt
    
    # Create ControlNet pipeline
    print("🔄 Creating Multi-ControlNet pipeline...")
    for i, cn in enumerate(config.controlnets):
        print(f"📝 ControlNet {i}: {cn.model_id} ({cn.preprocessor})")
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
    print(f"📏 Resolution: {args.resolution}x{args.resolution}")
    print(f"🎛️  ControlNet Scales: {[cn.conditioning_scale for cn in config.controlnets]}")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current output")
    print("  - Press 'c' to toggle control images preview")
    print("  - Press '1-9' to adjust ControlNet scale for that index")
    print("  - Press '+' to increase all ControlNet scales")
    print("  - Press '-' to decrease all ControlNet scales")
    print("  - Press 'p' to change prompt interactively")
    
    frame_count = 0
    show_preprocessed = args.show_preprocessed
    fps_counter = []
    
    # Get preprocessor names for display
    preprocessor_names = [cn.preprocessor.replace("_", " ").title() for cn in config.controlnets]
    
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
            
            # Create display layout
            display_frame = cv2.resize(frame, (args.resolution, args.resolution))
            output_display = cv2.resize(output_cv, (args.resolution, args.resolution))
            
            if show_preprocessed and len(pipeline.preprocessors) > 0:
                # Get preprocessed images from cache (avoid reprocessing)
                control_images = []
                for i, preprocessor in enumerate(pipeline.preprocessors):
                    if preprocessor is not None:
                        control_pil = pipeline.get_last_processed_image(i)
                        if control_pil is not None:
                            control_cv = cv2.cvtColor(np.array(control_pil), cv2.COLOR_RGB2BGR)
                            control_images.append(control_cv)
                
                if control_images:
                    # Create grid layout: Input | Controls | Output
                    # Resize controls to fit
                    control_size = args.resolution // len(control_images)
                    controls_resized = [cv2.resize(img, (control_size, control_size)) for img in control_images]
                    
                    # Stack controls vertically
                    if len(controls_resized) == 1:
                        controls_column = cv2.resize(controls_resized[0], (args.resolution//2, args.resolution))
                    else:
                        # Stack multiple controls
                        controls_column = np.vstack(controls_resized)
                        if controls_column.shape[0] != args.resolution:
                            controls_column = cv2.resize(controls_column, (args.resolution//2, args.resolution))
                    
                    # Create 3-panel layout
                    input_column = cv2.resize(display_frame, (args.resolution//2, args.resolution))
                    output_column = cv2.resize(output_display, (args.resolution//2, args.resolution))
                    
                    left_panel = np.hstack([input_column, controls_column])
                    combined = np.hstack([left_panel, output_column])
                else:
                    # Fallback to simple layout
                    combined = np.hstack([display_frame, output_display])
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
                
                print(f"\n📊 Multi-ControlNet Performance Profile (last 30 frames):")
                print(f"  Preprocessing: {sum(recent_prep)/len(recent_prep)*1000:.1f}ms avg")
                print(f"  Generation:    {sum(recent_gen)/len(recent_gen)*1000:.1f}ms avg")
                print(f"  Display:       {sum(recent_display)/len(recent_display)*1000:.1f}ms avg")
                print(f"  Total:         {sum(recent_total)/len(recent_total)*1000:.1f}ms avg")
                print(f"  FPS:           {avg_fps:.1f}")
                print(f"  ControlNets:   {len(pipeline.controlnets)}")
                
                # Check if tensor processing is being used
                for i, preprocessor in enumerate(pipeline.preprocessors):
                    if preprocessor and hasattr(preprocessor, 'process_tensor'):
                        print(f"  ✓ Tensor processing available for CN{i}: {type(preprocessor).__name__}")
                    else:
                        print(f"  ⚠️  No tensor processing for CN{i}: {type(preprocessor).__name__}")
            
            # Add info overlay
            current_scales = [f"{scale:.1f}" for scale in pipeline.controlnet_scales]
            info_text = f"Frame: {frame_count} | FPS: {avg_fps:.1f}"
            scales_text = f"Scales: [{', '.join(current_scales)}]"
            
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, scales_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add preprocessor info
            preprocessors_text = f"Preprocessors: {', '.join(preprocessor_names)}"
            cv2.putText(combined, preprocessors_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Add timing info
            if frame_count > 0:
                timing_text = f"Prep: {prep_time*1000:.1f}ms | Gen: {gen_time*1000:.1f}ms"
                cv2.putText(combined, timing_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            cv2.imshow(f'Multi-ControlNet StreamDiffusion', combined)
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current output
                try:
                    timestamp = int(time.time())
                    output_path = f"multi_controlnet_output_{timestamp}.png"
                    output_image.save(output_path)
                    print(f"💾 Saved output to {output_path}")
                        
                except Exception as save_error:
                    print(f"❌ Failed to save: {save_error}")
            elif key == ord('c'):
                # Toggle control images preview
                show_preprocessed = not show_preprocessed
                print(f"🖼️  Control images preview: {'ON' if show_preprocessed else 'OFF'}")
            elif key == ord('+'):
                # Increase all scales
                for i in range(len(pipeline.controlnet_scales)):
                    new_scale = min(2.0, pipeline.controlnet_scales[i] + 0.1)
                    pipeline.update_controlnet_scale(i, new_scale)
                scales_str = [f"{s:.1f}" for s in pipeline.controlnet_scales]
                print(f"📈 All scales: [{', '.join(scales_str)}]")
            elif key == ord('-'):
                # Decrease all scales
                for i in range(len(pipeline.controlnet_scales)):
                    new_scale = max(0.0, pipeline.controlnet_scales[i] - 0.1)
                    pipeline.update_controlnet_scale(i, new_scale)
                scales_str = [f"{s:.1f}" for s in pipeline.controlnet_scales]
                print(f"📉 All scales: [{', '.join(scales_str)}]")
            elif key >= ord('1') and key <= ord('9'):
                # Adjust individual ControlNet scale
                index = key - ord('1')
                if index < len(pipeline.controlnet_scales):
                    print(f"\n🎛️  Adjusting ControlNet {index} ({preprocessor_names[index]}):")
                    print(f"Current scale: {pipeline.controlnet_scales[index]:.2f}")
                    try:
                        new_scale_input = input("Enter new scale (0.0-2.0): ").strip()
                        if new_scale_input:
                            new_scale = float(new_scale_input)
                            new_scale = max(0.0, min(2.0, new_scale))
                            pipeline.update_controlnet_scale(index, new_scale)
                            print(f"✓ Updated ControlNet {index} scale to {new_scale:.2f}")
                    except:
                        print("❌ Invalid input")
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
        print("🏁 Multi-ControlNet demo finished")
        if fps_counter:
            avg_fps = len(fps_counter) / sum(fps_counter)
            print(f"📊 Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main() 