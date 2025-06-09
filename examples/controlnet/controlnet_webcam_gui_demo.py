#!/usr/bin/env python3
"""
ControlNet Webcam GUI Demo for StreamDiffusion

This script demonstrates real-time image generation using webcam input with ControlNet
and a GUI control panel for adjusting parameters in real-time.
Supports multiple ControlNets with independent strength controls.
"""

import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import argparse
from pathlib import Path
import sys
import time
import os
import threading
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import signal

# Import profiler
try:
    from .profiler import start_profiling, stop_profiling
except ImportError:
    # Handle relative import when run as script
    sys.path.append(os.path.dirname(__file__))
    from profiler import start_profiling, stop_profiling


class ControlNetGUI:
    def __init__(self, wrapper, config, args):
        self.wrapper = wrapper
        self.config = config
        self.args = args
        
        # Profiling state
        self.profiling_active = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("ControlNet StreamDiffusion - GUI Demo")
        self.root.geometry("400x800")
        
        # Video processing state
        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # TESTING: Hard-coded frame limit for scientific benchmarking
        self.MAX_FRAMES_FOR_TESTING = 200  # Stop after 200 frames
        print(f"TESTING MODE: Will automatically stop after {self.MAX_FRAMES_FOR_TESTING} frames")
        
        # GUI variables
        self.prompt_var = tk.StringVar(value=config.prompt if hasattr(config, 'prompt') else "")
        self.negative_prompt_var = tk.StringVar(value=getattr(config, 'negative_prompt', ''))
        self.guidance_scale_var = tk.DoubleVar(value=getattr(config, 'guidance_scale', 1.1))
        self.num_steps_var = tk.IntVar(value=getattr(config, 'num_inference_steps', 50))
        self.seed_var = tk.IntVar(value=getattr(config, 'seed', 2))
        
        # Temporal consistency variables
        self.frame_buffer_size_var = tk.IntVar(value=getattr(config, 'frame_buffer_size', 1))
        self.delta_var = tk.DoubleVar(value=getattr(config, 'delta', 1.0))
        self.cfg_type_var = tk.StringVar(value=getattr(config, 'cfg_type', 'self'))
        
        # ControlNet variables - support multiple ControlNets
        self.controlnet_vars = []
        self.controlnet_enabled_vars = []
        
        # Initialize ControlNet variables based on config
        if hasattr(config, 'controlnets') and config.controlnets:
            for i, cn_config in enumerate(config.controlnets):
                scale_var = tk.DoubleVar(value=cn_config.conditioning_scale)
                enabled_var = tk.BooleanVar(value=getattr(cn_config, 'enabled', True))
                self.controlnet_vars.append(scale_var)
                self.controlnet_enabled_vars.append(enabled_var)
        else:
            # Default single ControlNet
            self.controlnet_vars.append(tk.DoubleVar(value=1.0))
            self.controlnet_enabled_vars.append(tk.BooleanVar(value=True))
        
        # Status variables
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.status_var = tk.StringVar(value="Ready")
        
        self.setup_gui()
        
        # Setup signal handlers for clean profiling shutdown
        self.setup_signal_handlers()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main controls tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Main Controls")
        
        # ControlNet tab
        controlnet_frame = ttk.Frame(notebook)
        notebook.add(controlnet_frame, text="ControlNets")
        
        # Advanced tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        
        self.setup_main_controls(main_frame)
        self.setup_controlnet_controls(controlnet_frame)
        self.setup_advanced_controls(advanced_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=2)
        
    def setup_main_controls(self, parent):
        """Setup main parameter controls"""
        # Prompt controls
        prompt_frame = ttk.LabelFrame(parent, text="Prompts", padding=10)
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(prompt_frame, text="Prompt:").pack(anchor=tk.W)
        prompt_entry = tk.Text(prompt_frame, height=3, wrap=tk.WORD)
        prompt_entry.pack(fill=tk.X, pady=2)
        prompt_entry.insert(tk.END, self.prompt_var.get())
        
        def update_prompt(*args):
            self.prompt_var.set(prompt_entry.get(1.0, tk.END).strip())
            self.update_pipeline_params()
        
        prompt_entry.bind('<KeyRelease>', update_prompt)
        
        ttk.Label(prompt_frame, text="Negative Prompt:").pack(anchor=tk.W, pady=(10,0))
        neg_prompt_entry = tk.Text(prompt_frame, height=2, wrap=tk.WORD)
        neg_prompt_entry.pack(fill=tk.X, pady=2)
        neg_prompt_entry.insert(tk.END, self.negative_prompt_var.get())
        
        def update_neg_prompt(*args):
            self.negative_prompt_var.set(neg_prompt_entry.get(1.0, tk.END).strip())
            self.update_pipeline_params()
        
        neg_prompt_entry.bind('<KeyRelease>', update_neg_prompt)
        
        # Generation parameters
        gen_frame = ttk.LabelFrame(parent, text="Generation Parameters", padding=10)
        gen_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Guidance Scale
        ttk.Label(gen_frame, text="Guidance Scale:").pack(anchor=tk.W)
        guidance_frame = ttk.Frame(gen_frame)
        guidance_frame.pack(fill=tk.X, pady=2)
        
        guidance_scale = ttk.Scale(guidance_frame, from_=0.1, to=10.0, 
                                 variable=self.guidance_scale_var, orient=tk.HORIZONTAL,
                                 command=self.update_pipeline_params)
        guidance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        guidance_label = ttk.Label(guidance_frame, text=f"{self.guidance_scale_var.get():.1f}")
        guidance_label.pack(side=tk.RIGHT, padx=(5,0))
        self.guidance_scale_var.trace('w', lambda *args: guidance_label.config(text=f"{self.guidance_scale_var.get():.1f}"))
        
        # Number of steps
        ttk.Label(gen_frame, text="Inference Steps:").pack(anchor=tk.W, pady=(10,0))
        steps_frame = ttk.Frame(gen_frame)
        steps_frame.pack(fill=tk.X, pady=2)
        
        steps_scale = ttk.Scale(steps_frame, from_=1, to=100, 
                              variable=self.num_steps_var, orient=tk.HORIZONTAL,
                              command=self.update_pipeline_params)
        steps_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        steps_label = ttk.Label(steps_frame, text=f"{self.num_steps_var.get()}")
        steps_label.pack(side=tk.RIGHT, padx=(5,0))
        self.num_steps_var.trace('w', lambda *args: steps_label.config(text=f"{self.num_steps_var.get()}"))
        
        # Seed
        ttk.Label(gen_frame, text="Seed:").pack(anchor=tk.W, pady=(10,0))
        seed_frame = ttk.Frame(gen_frame)
        seed_frame.pack(fill=tk.X, pady=2)
        
        seed_scale = ttk.Scale(seed_frame, from_=0, to=999999, 
                              variable=self.seed_var, orient=tk.HORIZONTAL,
                              command=self.update_pipeline_params)
        seed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        seed_label = ttk.Label(seed_frame, text=f"{self.seed_var.get()}")
        seed_label.pack(side=tk.RIGHT, padx=(5,0))
        self.seed_var.trace('w', lambda *args: seed_label.config(text=f"{self.seed_var.get()}"))
        
    def setup_controlnet_controls(self, parent):
        """Setup ControlNet-specific controls"""
        info_frame = ttk.LabelFrame(parent, text="ControlNet Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        if hasattr(self.config, 'controlnets') and self.config.controlnets:
            for i, cn_config in enumerate(self.config.controlnets):
                cn_info_text = f"ControlNet {i+1}: {cn_config.model_id.split('/')[-1]}\nPreprocessor: {cn_config.preprocessor}"
                ttk.Label(info_frame, text=cn_info_text, font=('TkDefaultFont', 8)).pack(anchor=tk.W, pady=2)
        
        # ControlNet strength controls
        controls_frame = ttk.LabelFrame(parent, text="ControlNet Strengths", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.controlnet_frames = []
        
        for i, (scale_var, enabled_var) in enumerate(zip(self.controlnet_vars, self.controlnet_enabled_vars)):
            cn_frame = ttk.Frame(controls_frame)
            cn_frame.pack(fill=tk.X, pady=5)
            
            # ControlNet name and enable checkbox
            header_frame = ttk.Frame(cn_frame)
            header_frame.pack(fill=tk.X, pady=2)
            
            cn_name = f"ControlNet {i+1}"
            if hasattr(self.config, 'controlnets') and i < len(self.config.controlnets):
                preprocessor = self.config.controlnets[i].preprocessor
                cn_name = f"ControlNet {i+1} ({preprocessor})"
            
            enabled_cb = ttk.Checkbutton(header_frame, text=cn_name, variable=enabled_var,
                                       command=lambda idx=i: self.toggle_controlnet(idx))
            enabled_cb.pack(side=tk.LEFT)
            
            # Strength slider
            strength_frame = ttk.Frame(cn_frame)
            strength_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(strength_frame, text="Strength:").pack(side=tk.LEFT)
            
            strength_scale = ttk.Scale(strength_frame, from_=0.0, to=2.0,
                                     variable=scale_var, orient=tk.HORIZONTAL,
                                     command=lambda val, idx=i: self.update_controlnet_strength(idx, val))
            strength_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            strength_label = ttk.Label(strength_frame, text=f"{scale_var.get():.2f}")
            strength_label.pack(side=tk.RIGHT)
            
            # Update label when scale changes
            scale_var.trace('w', lambda *args, lbl=strength_label, var=scale_var: 
                          lbl.config(text=f"{var.get():.2f}"))
            
            self.controlnet_frames.append(cn_frame)
        
        # Global ControlNet controls
        global_frame = ttk.LabelFrame(parent, text="Global ControlNet Controls", padding=10)
        global_frame.pack(fill=tk.X, padx=5, pady=5)
        
        button_frame = ttk.Frame(global_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Enable All", 
                  command=self.enable_all_controlnets).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Disable All", 
                  command=self.disable_all_controlnets).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset Strengths", 
                  command=self.reset_controlnet_strengths).pack(side=tk.LEFT, padx=2)
        
    def setup_advanced_controls(self, parent):
        """Setup advanced controls"""
        # Camera settings
        camera_frame = ttk.LabelFrame(parent, text="Camera Settings", padding=10)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_frame, text=f"Camera Device: {self.args.camera}").pack(anchor=tk.W)
        ttk.Label(camera_frame, text=f"Resolution: {self.args.resolution}x{self.args.resolution}").pack(anchor=tk.W)
        
        # Pipeline info
        pipeline_frame = ttk.LabelFrame(parent, text="Pipeline Information", padding=10)
        pipeline_frame.pack(fill=tk.X, padx=5, pady=5)
        
        pipeline_type = getattr(self.config, 'pipeline_type', 'sd1.5')
        model_id = self.config.model_id if hasattr(self.config, 'model_id') else "Unknown"
        acceleration = getattr(self.config, 'acceleration', 'none')
        
        ttk.Label(pipeline_frame, text=f"Pipeline Type: {pipeline_type}").pack(anchor=tk.W)
        ttk.Label(pipeline_frame, text=f"Model: {model_id.split('/')[-1]}").pack(anchor=tk.W)
        ttk.Label(pipeline_frame, text=f"Acceleration: {acceleration}").pack(anchor=tk.W)
        
        if acceleration.lower() == 'tensorrt':
            ttk.Label(pipeline_frame, text="ℹ️ TensorRT: Frame buffer size locked at config value", 
                     font=('TkDefaultFont', 8), foreground='blue').pack(anchor=tk.W, pady=(5,0))
        
        # Performance settings
        perf_frame = ttk.LabelFrame(parent, text="Performance", padding=10)
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Show FPS", variable=self.show_fps_var).pack(anchor=tk.W)
        
        # Temporal consistency settings
        temporal_frame = ttk.LabelFrame(parent, text="Temporal Consistency", padding=10)
        temporal_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Check if TensorRT is enabled (frame buffer changes not allowed)
        tensorrt_enabled = acceleration.lower() == 'tensorrt'
        
        # Frame Buffer Size
        ttk.Label(temporal_frame, text="Frame Buffer Size:").pack(anchor=tk.W)
        buffer_frame = ttk.Frame(temporal_frame)
        buffer_frame.pack(fill=tk.X, pady=2)
        
        buffer_scale = ttk.Scale(buffer_frame, from_=1, to=10, 
                               variable=self.frame_buffer_size_var, orient=tk.HORIZONTAL,
                               command=self.update_pipeline_params,
                               state="disabled" if tensorrt_enabled else "normal")
        buffer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        buffer_label = ttk.Label(buffer_frame, text=f"{self.frame_buffer_size_var.get()}")
        buffer_label.pack(side=tk.RIGHT, padx=(5,0))
        self.frame_buffer_size_var.trace('w', lambda *args: buffer_label.config(text=f"{self.frame_buffer_size_var.get()}"))
        
        if tensorrt_enabled:
            ttk.Label(temporal_frame, text="⚠️ Frame Buffer Size locked (TensorRT enabled)", 
                     font=('TkDefaultFont', 8), foreground='orange').pack(anchor=tk.W)
        
        # Delta (temporal stability)
        ttk.Label(temporal_frame, text="Delta (Temporal Stability):").pack(anchor=tk.W, pady=(10,0))
        delta_frame = ttk.Frame(temporal_frame)
        delta_frame.pack(fill=tk.X, pady=2)
        
        delta_scale = ttk.Scale(delta_frame, from_=0.1, to=2.0, 
                              variable=self.delta_var, orient=tk.HORIZONTAL,
                              command=self.update_pipeline_params)
        delta_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        delta_label = ttk.Label(delta_frame, text=f"{self.delta_var.get():.2f}")
        delta_label.pack(side=tk.RIGHT, padx=(5,0))
        self.delta_var.trace('w', lambda *args: delta_label.config(text=f"{self.delta_var.get():.2f}"))
        
        # CFG Type dropdown
        ttk.Label(temporal_frame, text="CFG Type:").pack(anchor=tk.W, pady=(10,0))
        cfg_combobox = ttk.Combobox(temporal_frame, textvariable=self.cfg_type_var, 
                                   values=["none", "full", "self", "initialize"], 
                                   state="readonly", width=15)
        cfg_combobox.pack(anchor=tk.W, pady=2)
        cfg_combobox.bind('<<ComboboxSelected>>', self.update_pipeline_params)
        
        # Add help text
        if tensorrt_enabled:
            help_text = ("TensorRT Mode: Frame buffer size fixed at config value\n"
                        "Delta: Lower = more stable, higher = more responsive\n" 
                        "CFG Type: none/initialize = fastest, self/full = higher quality")
        else:
            help_text = ("Frame Buffer: Higher = more temporal consistency, more VRAM\n"
                        "Delta: Lower = more stable, higher = more responsive\n" 
                        "CFG Type: none/initialize = fastest, self/full = higher quality")
        ttk.Label(temporal_frame, text=help_text, font=('TkDefaultFont', 8), 
                 foreground='gray').pack(anchor=tk.W, pady=(5,0))
        
        # Export settings
        export_frame = ttk.LabelFrame(parent, text="Export Options", padding=10)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_control_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(export_frame, text="Save Control Images", 
                       variable=self.save_control_images_var).pack(anchor=tk.W)
        
    def update_pipeline_params(self, *args):
        """Update pipeline parameters"""
        if hasattr(self.wrapper, 'stream'):
            try:
                # Update prompt - this can be done dynamically
                if hasattr(self.wrapper.stream, 'update_prompt'):
                    self.wrapper.stream.update_prompt(self.prompt_var.get())
                
                # For other parameters, we need to re-prepare the pipeline
                # This is expensive but necessary for real-time parameter changes
                if self.running:
                    print("Updating pipeline parameters...")
                    
                    # Check if TensorRT is enabled
                    acceleration = getattr(self.config, 'acceleration', 'none')
                    tensorrt_enabled = acceleration.lower() == 'tensorrt'
                    
                    # Check if we need to recreate the entire pipeline (expensive operations)
                    needs_pipeline_recreation = False
                    
                    if (hasattr(self.wrapper, 'frame_buffer_size') and 
                        self.wrapper.frame_buffer_size != self.frame_buffer_size_var.get()):
                        if tensorrt_enabled:
                            print("ERROR: Cannot change frame buffer size with TensorRT acceleration!")
                            print("TensorRT engines are compiled with fixed batch sizes.")
                            print("Please update the config file and restart the application.")
                            # Reset the GUI value to current wrapper value
                            self.frame_buffer_size_var.set(self.wrapper.frame_buffer_size)
                            return
                        else:
                            needs_pipeline_recreation = True
                    
                    if (hasattr(self.wrapper.stream, 'cfg_type') and 
                        self.wrapper.stream.cfg_type != self.cfg_type_var.get()):
                        needs_pipeline_recreation = True
                    
                    if needs_pipeline_recreation:
                        print("Warning: Frame buffer size or CFG type change requires pipeline recreation (expensive)")
                        print("Consider restarting the application with new settings for optimal performance")
                    
                    # Re-prepare with new parameters (works for most changes)
                    self.wrapper.prepare(
                        prompt=self.prompt_var.get(),
                        negative_prompt=self.negative_prompt_var.get(),
                        num_inference_steps=self.num_steps_var.get(),
                        guidance_scale=self.guidance_scale_var.get(),
                        delta=self.delta_var.get(),
                    )
                    
                    # Update seed in the stream
                    if hasattr(self.wrapper.stream, 'generator'):
                        self.wrapper.stream.generator.manual_seed(int(self.seed_var.get()))
                    elif hasattr(self.wrapper.stream, 'set_seed'):
                        self.wrapper.stream.set_seed(int(self.seed_var.get()))
                
            except Exception as e:
                print(f"Failed to update pipeline params: {e}")
    
    def update_controlnet_strength(self, index, value):
        """Update ControlNet strength"""
        try:
            new_value = float(value)
            if hasattr(self.wrapper, 'update_controlnet_scale'):
                self.wrapper.update_controlnet_scale(index, new_value)
        except Exception as e:
            print(f"Failed to update ControlNet {index} strength: {e}")
    
    def toggle_controlnet(self, index):
        """Toggle ControlNet on/off"""
        try:
            enabled = self.controlnet_enabled_vars[index].get()
            strength = self.controlnet_vars[index].get() if enabled else 0.0
            if hasattr(self.wrapper, 'update_controlnet_scale'):
                self.wrapper.update_controlnet_scale(index, strength)
        except Exception as e:
            print(f"Failed to toggle ControlNet {index}: {e}")
    
    def enable_all_controlnets(self):
        """Enable all ControlNets"""
        for var in self.controlnet_enabled_vars:
            var.set(True)
        for i in range(len(self.controlnet_vars)):
            self.toggle_controlnet(i)
    
    def disable_all_controlnets(self):
        """Disable all ControlNets"""
        for var in self.controlnet_enabled_vars:
            var.set(False)
        for i in range(len(self.controlnet_vars)):
            self.toggle_controlnet(i)
    
    def reset_controlnet_strengths(self):
        """Reset all ControlNet strengths to 1.0"""
        for var in self.controlnet_vars:
            var.set(1.0)
        for i in range(len(self.controlnet_vars)):
            self.update_controlnet_strength(i, 1.0)
    
    def reset_defaults(self):
        """Reset all parameters to defaults"""
        # Reset prompts
        self.prompt_var.set(getattr(self.config, 'prompt', ''))
        self.negative_prompt_var.set(getattr(self.config, 'negative_prompt', ''))
        
        # Reset generation params
        self.guidance_scale_var.set(getattr(self.config, 'guidance_scale', 1.1))
        self.num_steps_var.set(getattr(self.config, 'num_inference_steps', 50))
        self.seed_var.set(getattr(self.config, 'seed', 2))
        
        # Reset temporal consistency variables
        self.frame_buffer_size_var.set(getattr(self.config, 'frame_buffer_size', 1))
        self.delta_var.set(getattr(self.config, 'delta', 1.0))
        self.cfg_type_var.set(getattr(self.config, 'cfg_type', 'self'))
        
        # Reset ControlNet strengths
        self.reset_controlnet_strengths()
        self.enable_all_controlnets()
        
        # Update pipeline
        self.update_pipeline_params()
    
    def toggle_camera(self):
        """Start/stop camera capture"""
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture and processing"""
        try:
            self.cap = cv2.VideoCapture(self.args.camera)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera {self.args.camera}")
                return
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.resolution)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.resolution)
            
            self.running = True
            self.start_button.config(text="Stop Camera")
            self.status_var.set("Running (Profiling Active)")
            
            # Start processing threads
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.display_thread = threading.Thread(target=self.display_results, daemon=True)
            
            self.capture_thread.start()
            self.process_thread.start()
            self.display_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.start_button.config(text="Start Camera")
        self.status_var.set("Stopped")
        
        # Profiling will be stopped by the process_frames thread when it exits
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def capture_frames(self):
        """Capture frames from camera"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame
                frame_resized = cv2.resize(frame, (self.args.resolution, self.args.resolution))
                
                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame_resized)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
            else:
                break
    
    def process_frames(self):
        """Process frames with StreamDiffusion"""
        # Start profiling in the worker thread where ControlNet actually executes
        if not self.profiling_active:
            print("process_frames: Starting profiling session in worker thread...")
            start_profiling()
            self.profiling_active = True
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                start_time = time.time()
                
                # TESTING: Check frame limit
                if self.frame_count >= self.MAX_FRAMES_FOR_TESTING:
                    print(f"TESTING: Reached frame limit ({self.MAX_FRAMES_FOR_TESTING}), stopping...")
                    self.root.after(0, self.stop_camera)
                    break
                
                # Convert frame to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Update control image - THIS IS WHERE CONTROLNET PREPROCESSING HAPPENS
                self.wrapper.update_control_image_efficient(frame_pil)
                
                # Generate image - THIS IS WHERE CONTROLNET DIFFUSION HAPPENS
                output_image = self.wrapper(frame_pil)
                
                # Calculate processing time
                process_time = time.time() - start_time
                
                # Put result in queue
                try:
                    self.result_queue.put_nowait((frame, output_image, process_time))
                except queue.Full:
                    # Skip if queue is full
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
        
        # Stop profiling when processing thread ends
        if self.profiling_active:
            print("process_frames: Stopping profiling session in worker thread...")
            try:
                prof_file, summary_file = stop_profiling()
                self.profiling_active = False
                print(f"process_frames: Profiling results saved:")
                print(f"  - Profile file: {prof_file}")
                print(f"  - Summary file: {summary_file}")
                print(f"  - Run 'snakeviz {prof_file}' to view interactive profile")
            except Exception as e:
                print(f"process_frames: Error saving profiling results: {e}")
                self.profiling_active = False
    
    def display_results(self):
        """Display results and update GUI"""
        while self.running:
            try:
                frame, output_image, process_time = self.result_queue.get(timeout=0.1)
                
                # Update FPS counter
                self.fps_counter.append(process_time)
                self.frame_count += 1
                
                # TESTING: Progress indicator
                if self.frame_count % 10 == 0:  # Log every 10 frames
                    progress = (self.frame_count / self.MAX_FRAMES_FOR_TESTING) * 100
                    print(f"TESTING: Processed {self.frame_count}/{self.MAX_FRAMES_FOR_TESTING} frames ({progress:.1f}%)")
                
                if self.show_fps_var.get() and len(self.fps_counter) > 0:
                    avg_fps = len(self.fps_counter) / sum(self.fps_counter)
                    self.fps_var.set(f"FPS: {avg_fps:.1f} | Frame: {self.frame_count}/{self.MAX_FRAMES_FOR_TESTING}")
                
                # Convert output to display format
                output_array = np.array(output_image)
                output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                
                # Create side-by-side display
                combined = np.hstack([frame, output_bgr])
                
                # Add labels
                cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "Generated", (self.args.resolution + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show result
                cv2.imshow("ControlNet StreamDiffusion - GUI Demo", combined)
                
                # Handle OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.root.after(0, self.stop_camera)
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")
    
    def save_output(self):
        """Save current output"""
        if not self.running:
            messagebox.showwarning("Warning", "Camera is not running")
            return
        
        try:
            # Get current frame from queue if available
            if not self.result_queue.empty():
                frame, output_image, _ = self.result_queue.queue[-1]  # Get latest
                
                timestamp = int(time.time())
                output_path = f"controlnet_gui_output_{timestamp}.png"
                output_image.save(output_path)
                
                if self.save_control_images_var.get():
                    # Save control images if available
                    for i in range(len(self.controlnet_vars)):
                        try:
                            control_image = self.wrapper.get_last_processed_image(i)
                            if control_image:
                                control_path = f"controlnet_gui_control_{i}_{timestamp}.png"
                                control_image.save(control_path)
                        except:
                            pass
                
                messagebox.showinfo("Success", f"Saved output to {output_path}")
            else:
                messagebox.showwarning("Warning", "No output to save")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            print("run: Interrupted by user")
        except Exception as e:
            print(f"run: Unexpected error: {e}")
        finally:
            self.cleanup_with_profiling()
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.stop_camera()
        self.cleanup_with_profiling()
        self.root.destroy()
        # Force exit the entire application
        sys.exit(0)
    
    def cleanup_with_profiling(self):
        """Cleanup resources with profiling support"""
        # Stop processing first to trigger profiling cleanup in worker thread
        self.running = False
        
        # Give the worker thread a moment to stop profiling
        time.sleep(0.5)
        
        # If profiling is still somehow active, try to stop it
        if self.profiling_active:
            print("cleanup_with_profiling: Saving any remaining profiling data...")
            try:
                prof_file, summary_file = stop_profiling()
                self.profiling_active = False
                print(f"cleanup_with_profiling: Profiling results saved:")
                print(f"  - Profile file: {prof_file}")
                print(f"  - Summary file: {summary_file}")
                print(f"  - Run 'snakeviz {prof_file}' to view interactive profile")
            except Exception as e:
                print(f"cleanup_with_profiling: Error saving profiling results: {e}")
        
        # Standard cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Clear queues to prevent hanging
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except:
            pass

    def setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            print("signal_handler: Received shutdown signal, stopping profiling and cleaning up...")
            self.cleanup_with_profiling()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def main():
    # Import heavy modules only when needed
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from utils.wrapper import StreamDiffusionWrapper
    from streamdiffusion.controlnet import load_controlnet_config
    
    parser = argparse.ArgumentParser(description="ControlNet Webcam GUI Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent.parent / "configs" / "controlnet_examples" / "multi_controlnet_example.yaml"
    
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str,
                       help="Override base model path from config")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--resolution", type=int, default=None,
                       help="Camera and output resolution (auto-detects from pipeline type if not specified)")
    
    args = parser.parse_args()
    
    print("Starting ControlNet Webcam GUI Demo")
    
    # Load configuration
    config = load_controlnet_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Detect pipeline type
    pipeline_type = getattr(config, 'pipeline_type', 'sd1.5')
    print(f"Pipeline type: {pipeline_type}")
    
    # Display ControlNet information
    if hasattr(config, 'controlnets') and config.controlnets:
        print(f"Found {len(config.controlnets)} ControlNet(s):")
        for i, cn in enumerate(config.controlnets):
            print(f"  {i+1}. {cn.model_id.split('/')[-1]} (preprocessor: {cn.preprocessor}, strength: {cn.conditioning_scale})")
    else:
        print("Warning: No ControlNets found in configuration")
    
    # Set default resolution based on pipeline type if not specified
    if args.resolution is None:
        if pipeline_type == 'sdxlturbo':
            args.resolution = 1024  # SD-XL Turbo default
        else:
            args.resolution = 512   # SD 1.5 and SD Turbo default
    
    # Override parameters if provided
    model_id = args.model if args.model else config.model_id
    prompt = args.prompt if args.prompt else config.prompt
    
    # Update resolution in config
    config.width = args.resolution
    config.height = args.resolution
    
    # Determine parameters based on pipeline type
    t_index_list = getattr(config, 't_index_list', [0,16])
    if pipeline_type == 'sdturbo':
        cfg_type = getattr(config, 'cfg_type', "none")
        use_lcm_lora = getattr(config, 'use_lcm_lora', False)
        use_tiny_vae = getattr(config, 'use_tiny_vae', True)
    elif pipeline_type == 'sdxlturbo':
        cfg_type = getattr(config, 'cfg_type', "none")
        use_lcm_lora = getattr(config, 'use_lcm_lora', False)
        use_tiny_vae = getattr(config, 'use_tiny_vae', False)
    else:  # sd1.5
        cfg_type = getattr(config, 'cfg_type', 'self')
        use_lcm_lora = getattr(config, 'use_lcm_lora', True)
        use_tiny_vae = getattr(config, 'use_tiny_vae', True)
    
    # Create ControlNet configurations for wrapper - support multiple ControlNets
    controlnet_configs = []
    if hasattr(config, 'controlnets') and config.controlnets:
        for cn_config in config.controlnets:
            controlnet_config = {
                'model_id': cn_config.model_id,
                'preprocessor': cn_config.preprocessor,
                'conditioning_scale': cn_config.conditioning_scale,
                'enabled': getattr(cn_config, 'enabled', True),
                'preprocessor_params': getattr(cn_config, 'preprocessor_params', None),
                'pipeline_type': pipeline_type,
                'control_guidance_start': getattr(cn_config, 'control_guidance_start', 0.0),
                'control_guidance_end': getattr(cn_config, 'control_guidance_end', 1.0),
            }
            controlnet_configs.append(controlnet_config)
    else:
        # Fallback single ControlNet for compatibility
        controlnet_configs = [{
            'model_id': 'lllyasviel/sd-controlnet-depth',
            'preprocessor': 'depth_midas',
            'conditioning_scale': 1.0,
            'enabled': True,
            'preprocessor_params': None,
            'pipeline_type': pipeline_type,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }]
    
    print("Creating StreamDiffusion pipeline with ControlNet(s)...")
    for i, cn_config in enumerate(controlnet_configs):
        print(f"  ControlNet {i+1}: {cn_config['model_id']} (preprocessor: {cn_config['preprocessor']})")
    
    # Create StreamDiffusionWrapper with ControlNet support
    wrapper = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        mode="img2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=getattr(config, 'frame_buffer_size', 1),
        width=args.resolution,
        height=args.resolution,
        warmup=10,
        acceleration=getattr(config, 'acceleration', 'none'),
        do_add_noise=True,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        use_denoising_batch=True,
        cfg_type=getattr(config, 'cfg_type', cfg_type),
        seed=getattr(config, 'seed', 2),
        use_safety_checker=False,
        # ControlNet options - pass all ControlNet configs
        use_controlnet=True,
        controlnet_config=controlnet_configs,  # Pass list of all ControlNet configs
    )
    
    print("Pipeline created successfully")
    
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=getattr(config, 'negative_prompt', ''),
        num_inference_steps=getattr(config, 'num_inference_steps', 50),
        guidance_scale=getattr(config, 'guidance_scale', 1.1 if cfg_type != "none" else 1.0),
        delta=getattr(config, 'delta', 1.0),
    )
    
    print("Pipeline prepared successfully")
    print("Starting GUI...")
    print("")
    print("PROFILING ENABLED:")
    print("- Profiling will start automatically when ControlNet processing begins")
    print("- Profiling captures the actual ControlNet execution in the worker thread")  
    print("- Profiling will stop when you click 'Stop Camera' or exit")
    print("- Results will be saved to 'profiling_results/' directory")
    print("- Use 'snakeviz <filename>.prof' to view interactive profile")
    print("")
    
    # Create and run GUI
    gui = ControlNetGUI(wrapper, config, args)
    gui.run()


if __name__ == "__main__":
    main() 