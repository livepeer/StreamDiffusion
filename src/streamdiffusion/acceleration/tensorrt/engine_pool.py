"""ControlNet TensorRT engine pool management"""

import os
import time
import hashlib
from typing import Dict, Optional, Set, Union, Any, List
from pathlib import Path
import torch
from polygraphy import cuda

from .controlnet_engine import ControlNetModelEngine, HybridControlNet
from .controlnet_models import create_controlnet_model
from .builder import EngineBuilder, create_onnx_path
from .model_detection import detect_model_from_diffusers_unet


class ControlNetEnginePool:
    """Manages multiple ControlNet TensorRT engines"""
    
    def __init__(self, engine_dir: str, stream: Optional[cuda.Stream] = None):
        """Initialize ControlNet engine pool"""
        self.engine_dir = Path(engine_dir)
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        
        self.stream = stream if stream is not None else cuda.Stream()
        self.engines: Dict[str, HybridControlNet] = {}
        self.compiled_models: Set[str] = set()
        
        self._discover_existing_engines()
    
    def _discover_existing_engines(self) -> None:
        """Discover existing TensorRT engines in the engine directory"""
        for model_dir in self.engine_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("controlnet_"):
                engine_file = model_dir / "cnet.engine"
                if engine_file.exists():
                    dir_name = model_dir.name
                    if "--batch-" in dir_name:
                        model_part = dir_name.split("--batch-")[0]
                        model_id = model_part.replace("controlnet_", "").replace("_", "/")
                    else:
                        model_id = dir_name.replace("controlnet_", "").replace("_", "/")
                    
                    self.compiled_models.add(model_id)
                    print(f"Discovered existing ControlNet engine: {model_id}")
    
    def get_or_load_engine(self, 
                          model_id: str,
                          pytorch_model: Any,
                          controlnet_type: str = "canny",
                          model_type: str = "sd15",
                          batch_size: int = 1) -> HybridControlNet:
        """Get or load ControlNet engine with TensorRT/PyTorch fallback"""
        cache_key = f"{model_id}--batch-{batch_size}"
        
        if cache_key in self.engines:
            return self.engines[cache_key]
        
        model_engine_dir = self._get_model_engine_dir(model_id, batch_size)
        engine_path = model_engine_dir / "cnet.engine"
        
        if not engine_path.exists():
            print(f"ControlNet engine not found for {model_id}, compiling now...")
            compilation_start = time.time()
            
            try:
                detected_type = detect_model_from_diffusers_unet(pytorch_model)
                model_type = detected_type.lower()
            except Exception as e:
                print(f"Architecture detection failed: {e}, using provided type: {model_type}")
            
            success = self._compile_controlnet(
                pytorch_model, controlnet_type, model_type, str(engine_path), batch_size
            )
            
            compilation_time = time.time() - compilation_start
            
            if success:
                print(f"ControlNet compilation completed in {compilation_time:.2f}s")
                print(f"   Engine saved to: {engine_path}")
            else:
                print(f"ControlNet compilation failed after {compilation_time:.2f}s")
                print(f"   Will use PyTorch fallback for {model_id}")
        
        hybrid_controlnet = HybridControlNet(
            model_id=model_id,
            engine_path=str(engine_path) if engine_path.exists() else None,
            pytorch_model=pytorch_model,
            stream=self.stream
        )
        
        self.engines[cache_key] = hybrid_controlnet
        
        return hybrid_controlnet
    
    def _compile_controlnet(self, 
                           pytorch_model: Any,
                           controlnet_type: str, 
                           model_type: str,
                           engine_path: str,
                           batch_size: int) -> bool:
        """Compile ControlNet to TensorRT"""
        try:
            print(f"Starting ControlNet compilation: {controlnet_type} ({model_type})")
            
            height = 512
            width = 512
            
            if model_type.lower() in ["sdxl", "sdxl-turbo"]:
                embedding_dim = 2048
            elif model_type.lower() in ["sd21", "sd2.1"]:
                embedding_dim = 1024
            else:
                embedding_dim = 768
            
            controlnet_model = create_controlnet_model(
                model_type=model_type,
                controlnet_type=controlnet_type,
                max_batch=batch_size,
                min_batch_size=1,
                embedding_dim=embedding_dim
            )
            
            pytorch_model = pytorch_model.to(torch.device("cuda"), dtype=torch.float16)
            
            engine_dir = Path(engine_path).parent
            onnx_dir = engine_dir / "onnx"
            onnx_dir.mkdir(exist_ok=True)
            
            base_name = "controlnet"
            onnx_path = create_onnx_path(base_name, onnx_dir, opt=False)
            onnx_opt_path = create_onnx_path(base_name, onnx_dir, opt=True)
            
            print(f"ONNX path: {onnx_path}")
            print(f"ONNX optimized path: {onnx_opt_path}")
            print(f"Engine path: {engine_path}")
            
            builder = EngineBuilder(controlnet_model, pytorch_model, device=torch.device("cuda"))
            builder.build(
                str(onnx_path),
                str(onnx_opt_path), 
                str(engine_path),
                opt_batch_size=batch_size,
                opt_image_height=height,
                opt_image_width=width
            )
            
            print(f"Successfully compiled ControlNet engine: {engine_path}")
            return True
            
        except Exception as e:
            print(f"ControlNet compilation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_sample_inputs(self, model_type: str) -> Dict[str, torch.Tensor]:
        """Generate sample inputs for ControlNet compilation"""
        batch_size = 1
        height, width = 512, 512
        
        if model_type.lower() in ["sdxl", "sdxl-turbo"]:
            embedding_dim = 2048
            text_embed_dim = 1280
        elif model_type.lower() in ["sd21", "sd2.1"]:
            embedding_dim = 1024
            text_embed_dim = None
        else:
            embedding_dim = 768
            text_embed_dim = None
        
        inputs = {
            "sample": torch.randn(batch_size, 4, height // 8, width // 8, dtype=torch.float16),
            "timestep": torch.tensor([0.5], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(batch_size, 77, embedding_dim, dtype=torch.float16),
            "controlnet_cond": torch.randn(batch_size, 3, height, width, dtype=torch.float16)
        }
        
        if text_embed_dim:
            inputs["text_embeds"] = torch.randn(batch_size, text_embed_dim, dtype=torch.float16)
            inputs["time_ids"] = torch.randn(batch_size, 6, dtype=torch.float16)
        
        return inputs
    

    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all ControlNets in the pool"""
        status = {
            "total_engines": len(self.engines),
            "compiled_models": len(self.compiled_models),
            "engines": {}
        }
        
        for model_id, engine in self.engines.items():
            engine_status = engine.status.copy()
            status["engines"][model_id] = engine_status
        
        return status
    
    def cleanup(self) -> None:
        """Clean up resources and stop background threads"""
        self.engines.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()

    def _get_model_engine_dir(self, model_id: str, batch_size: int = 1) -> Path:
        """Get the engine directory for a specific ControlNet model"""
        safe_name = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_name = f"controlnet_{safe_name}--batch-{batch_size}"
        
        model_dir = Path(self.engine_dir) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir 