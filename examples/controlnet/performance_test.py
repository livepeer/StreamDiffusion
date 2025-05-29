#!/usr/bin/env python3
"""
Performance Test for ControlNet Optimizations

This script tests the performance impact of our optimizations to identify real bottlenecks.
"""

import time
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from streamdiffusion.controlnet.preprocessors.passthrough import PassthroughPreprocessor
from streamdiffusion.controlnet.preprocessors.depth import DepthPreprocessor
from streamdiffusion.controlnet.preprocessors.canny import CannyPreprocessor


def create_test_image(size=(512, 512), device='cuda'):
    """Create a test image in different formats"""
    # PIL Image
    pil_image = Image.new('RGB', size, color=(128, 128, 128))
    
    # Numpy array
    np_image = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    
    # Tensor (CPU)
    tensor_cpu = torch.rand(3, size[0], size[1])
    
    # Tensor (GPU)
    tensor_gpu = tensor_cpu.to(device) if torch.cuda.is_available() else tensor_cpu
    
    return pil_image, np_image, tensor_cpu, tensor_gpu


def benchmark_preprocessor(preprocessor_class, test_cases=100, **kwargs):
    """Benchmark a preprocessor with different input types"""
    print(f"\n🧪 Testing {preprocessor_class.__name__}")
    
    # Initialize preprocessor
    preprocessor = preprocessor_class(**kwargs)
    
    # Create test images
    pil_img, np_img, tensor_cpu, tensor_gpu = create_test_image()
    
    results = {}
    
    # Test PIL processing
    print("  Testing PIL processing...")
    start_time = time.time()
    for _ in range(test_cases):
        result = preprocessor.process(pil_img)
    pil_time = (time.time() - start_time) / test_cases
    results['PIL'] = pil_time * 1000  # Convert to ms
    
    # Test tensor processing (if available)
    if hasattr(preprocessor, 'process_tensor'):
        print("  Testing tensor processing...")
        try:
            start_time = time.time()
            for _ in range(test_cases):
                result = preprocessor.process_tensor(tensor_gpu)
            tensor_time = (time.time() - start_time) / test_cases
            results['Tensor'] = tensor_time * 1000  # Convert to ms
            
            speedup = pil_time / tensor_time
            results['Speedup'] = speedup
        except Exception as e:
            print(f"    ⚠️  Tensor processing failed: {e}")
            results['Tensor'] = None
            results['Speedup'] = None
    else:
        print("    ⚠️  No tensor processing available")
        results['Tensor'] = None
        results['Speedup'] = None
    
    return results


def main():
    print("🚀 ControlNet Performance Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, running on CPU")
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
    
    test_cases = 50  # Number of iterations for averaging
    
    # Test different preprocessors
    preprocessors = [
        (PassthroughPreprocessor, {}),
        (CannyPreprocessor, {}),
    ]
    
    # Only test depth if transformers is available
    try:
        depth_test = (DepthPreprocessor, {'model_name': 'Intel/dpt-hybrid-midas'})
        preprocessors.append(depth_test)
    except:
        print("⚠️  Depth preprocessor not available (transformers not installed)")
    
    all_results = {}
    
    for preprocessor_class, kwargs in preprocessors:
        try:
            results = benchmark_preprocessor(preprocessor_class, test_cases, **kwargs)
            all_results[preprocessor_class.__name__] = results
        except Exception as e:
            print(f"❌ Failed to test {preprocessor_class.__name__}: {e}")
    
    # Print summary
    print("\n📊 Performance Summary")
    print("=" * 50)
    print(f"{'Preprocessor':<20} {'PIL (ms)':<10} {'Tensor (ms)':<12} {'Speedup':<8}")
    print("-" * 52)
    
    for name, results in all_results.items():
        pil_time = f"{results['PIL']:.1f}" if results['PIL'] else "N/A"
        tensor_time = f"{results['Tensor']:.1f}" if results['Tensor'] else "N/A"
        speedup = f"{results['Speedup']:.1f}x" if results['Speedup'] else "N/A"
        
        print(f"{name:<20} {pil_time:<10} {tensor_time:<12} {speedup:<8}")
    
    # Test optimization effectiveness
    print("\n🔍 Optimization Analysis")
    print("=" * 50)
    
    total_pil_time = sum(r['PIL'] for r in all_results.values() if r['PIL'])
    total_tensor_time = sum(r['Tensor'] for r in all_results.values() if r['Tensor'])
    
    if total_tensor_time > 0:
        overall_speedup = total_pil_time / total_tensor_time
        print(f"Overall preprocessing speedup: {overall_speedup:.1f}x")
        print(f"Time saved per frame: {total_pil_time - total_tensor_time:.1f}ms")
    else:
        print("⚠️  No tensor processing results available")
    
    # Simulate multi-ControlNet scenario
    print(f"\n🎯 Multi-ControlNet Impact (3 ControlNets)")
    print("-" * 40)
    if total_tensor_time > 0:
        original_time = total_pil_time * 3  # 3 ControlNets without optimization
        optimized_time = total_tensor_time  # Process once with optimization
        multi_speedup = original_time / optimized_time
        print(f"Before optimization: {original_time:.1f}ms per frame")
        print(f"After optimization:  {optimized_time:.1f}ms per frame")
        print(f"Multi-ControlNet speedup: {multi_speedup:.1f}x")
    else:
        print("⚠️  Cannot calculate multi-ControlNet impact")


if __name__ == "__main__":
    main() 