#!/usr/bin/env python3
"""
Simple profiler for StreamDiffusion ControlNet demos
Shows execution time for ONLY methods in files from changes_summary.txt
"""

import cProfile
import pstats
import io
import time
import os
from pathlib import Path
from datetime import datetime


class StreamDiffusionProfiler:
    """Simple profiler focused only on changed files"""
    
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = None
        self.start_time = None
        self.session_name = None
        
    def start(self, session_name=None):
        """Start profiling session"""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"controlnet_session_{timestamp}"
        
        self.session_name = session_name
        self.start_time = time.time()
        
        print(f"profiler.start: Starting profiling session: {session_name}")
        
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
    def stop(self):
        """Stop profiling and save results"""
        if self.profiler is None:
            print("profiler.stop: No active profiling session")
            return
            
        self.profiler.disable()
        end_time = time.time()
        total_time = end_time - self.start_time
        
        print(f"profiler.stop: Session completed in {total_time:.2f} seconds")
        
        # Generate simple summary
        summary_filename = self.output_dir / f"{self.session_name}_methods.txt"
        self._generate_methods_summary(summary_filename, total_time)
        print(f"profiler.stop: Saved methods summary: {summary_filename}")
        
        return str(summary_filename)
        
    def _generate_methods_summary(self, filename, total_time):
        """Show ONLY methods from changed files with their execution times"""
        
        # Files from changes_summary.txt that were added (A) or modified (M)
        target_files = [
            # Core ControlNet modules
            'base_controlnet_pipeline.py',
            'config.py', 
            'controlnet_pipeline.py',
            'controlnet_sdxlturbo_pipeline.py',
            
            # Preprocessors
            'base.py',
            'canny.py', 
            'depth.py',
            'depth_tensorrt.py',
            'lineart.py',
            'openpose.py',
            'passthrough.py',
            
            # TensorRT ControlNet
            'controlnet_engine.py',
            'controlnet_models.py', 
            'controlnet_wrapper.py',
            'engine_pool.py',
            'model_detection.py',
            
            # Modified files
            'wrapper.py',
            'builder.py',
            'engine.py',
            'models.py',
            'utilities.py',
        ]
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        stats_output = s.getvalue()
        
        # DEBUG: Capture ALL methods first
        all_methods = []
        methods_data = []
        lines = stats_output.split('\n')
        
        for line in lines:
            if not line.strip() or line.startswith('ncalls') or '---' in line:
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                try:
                    ncalls = parts[0]
                    tottime = float(parts[1]) 
                    cumtime = float(parts[3])
                    filename_func = ' '.join(parts[5:])
                    
                    # Extract method name and file
                    if '(' in filename_func and ')' in filename_func:
                        func_name = filename_func.split('(')[-1].replace(')', '')
                        file_part = filename_func.split('(')[0]
                        
                        # Clean up file path
                        file_name = file_part.split('\\')[-1] if '\\' in file_part else file_part.split('/')[-1]
                        
                        # Store ALL methods
                        all_methods.append({
                            'file': file_name,
                            'method': func_name,
                            'calls': ncalls,
                            'own_time': tottime,
                            'total_time': cumtime,
                            'full_path': filename_func
                        })
                        
                        # Check if this is from our target files
                        is_target_file = any(target_file in filename_func for target_file in target_files)
                        
                        if is_target_file:
                            methods_data.append({
                                'file': file_name,
                                'method': func_name,
                                'calls': ncalls,
                                'own_time': tottime,
                                'total_time': cumtime
                            })
                except (ValueError, IndexError):
                    continue
        
        # Sort by total time descending
        methods_data.sort(key=lambda x: x['total_time'], reverse=True)
        all_methods.sort(key=lambda x: x['total_time'], reverse=True)
        
        # Write summary
        with open(filename, 'w') as f:
            f.write("ControlNet Methods Execution Times - DEBUG VERSION\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Total Runtime: {total_time:.2f} seconds\n\n")
            
            f.write("DEBUG: ALL CAPTURED METHODS (First 30):\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Total Time':<12} {'File':<40} {'Method':<30}\n")
            f.write("-" * 90 + "\n")
            for method in all_methods[:30]:
                f.write(f"{method['total_time']:<12.4f} {method['file']:<40} {method['method']:<30}\n")
            f.write(f"\nTotal ALL methods captured: {len(all_methods)}\n\n")
            
            f.write("FILTERED: METHODS FROM YOUR CONTROLNET FILES ONLY:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Total Time':<12} {'Own Time':<10} {'Calls':<8} {'File':<30} {'Method':<40}\n")
            f.write("-" * 110 + "\n")
            
            for method in methods_data:
                f.write(f"{method['total_time']:<12.4f} {method['own_time']:<10.4f} {method['calls']:<8} {method['file']:<30} {method['method']:<40}\n")
            
            f.write(f"\nTotal ControlNet methods found: {len(methods_data)}\n")
            f.write(f"Total ALL methods: {len(all_methods)}\n")
            
            f.write("\nTARGET FILES WE'RE LOOKING FOR:\n")
            for target in target_files:
                f.write(f"- {target}\n")
            
            f.write("\nNOTES:\n")
            f.write("- Total Time = time including calls to other functions\n")
            f.write("- Own Time = time spent only in this method\n")
            f.write("- Focus on methods with high Total Time\n")


# Global profiler instance
_profiler = StreamDiffusionProfiler()

def start_profiling(session_name=None):
    """Start a profiling session"""
    _profiler.start(session_name)

def stop_profiling():
    """Stop profiling and save results"""
    return _profiler.stop()

def get_profiler():
    """Get the global profiler instance"""
    return _profiler 