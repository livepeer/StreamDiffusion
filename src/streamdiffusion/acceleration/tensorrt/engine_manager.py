import os
from enum import Enum
from typing import Any, Optional, Dict
from pathlib import Path


class EngineType(Enum):
    """Engine types supported by the TensorRT engine manager."""
    UNET = "unet"
    VAE_ENCODER = "vae_encoder" 
    VAE_DECODER = "vae_decoder"
    CONTROLNET = "controlnet"


class EngineManager:
    """
    Universal TensorRT engine manager using factory pattern.
    
    Consolidates all engine management logic into a single class:
    - Path generation (moves create_prefix from wrapper.py)
    - Compilation (moves compile_* calls from wrapper.py)  
    - Loading (returns appropriate engine objects)
    """
    
    def __init__(self, engine_dir: str):
        """Initialize with engine directory."""
        print(f"EngineManager.__init__: Initializing with engine directory: {engine_dir}")
        self.engine_dir = Path(engine_dir)
        print(f"EngineManager.__init__: Engine directory path object: {self.engine_dir}")
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        print(f"EngineManager.__init__: Engine directory created/verified: {self.engine_dir.exists()}")
        
        # Import the existing compile functions from tensorrt/__init__.py
        from streamdiffusion.acceleration.tensorrt import (
            compile_unet, compile_vae_encoder, compile_vae_decoder
        )
        from streamdiffusion.acceleration.tensorrt.builder import compile_controlnet
        from streamdiffusion.acceleration.tensorrt.runtime_engines.unet_engine import (
            UNet2DConditionModelEngine
        )
        from streamdiffusion.acceleration.tensorrt.runtime_engines.controlnet_engine import (
            ControlNetModelEngine
        )
        
        # Engine configurations - maps each type to its compile function and loader
        self._configs = {
            EngineType.UNET: {
                'filename': 'unet.engine',
                'compile_fn': compile_unet,
                'loader': lambda path, cuda_stream, **kwargs: UNet2DConditionModelEngine(
                    str(path), cuda_stream, use_cuda_graph=True
                )
            },
            EngineType.VAE_ENCODER: {
                'filename': 'vae_encoder.engine', 
                'compile_fn': compile_vae_encoder,
                'loader': lambda path, cuda_stream, **kwargs: str(path)  # Return path for AutoencoderKLEngine
            },
            EngineType.VAE_DECODER: {
                'filename': 'vae_decoder.engine',
                'compile_fn': compile_vae_decoder, 
                'loader': lambda path, cuda_stream, **kwargs: str(path)  # Return path for AutoencoderKLEngine
            },
            EngineType.CONTROLNET: {
                'filename': 'cnet.engine',
                'compile_fn': compile_controlnet,
                'loader': lambda path, cuda_stream, **kwargs: ControlNetModelEngine(
                    str(path), cuda_stream, use_cuda_graph=kwargs.get('use_cuda_graph', False),
                    model_type=kwargs.get('model_type', 'sd15')
                )
            }
        }
    
    def get_engine_path(self, 
                       engine_type: EngineType,
                       model_id_or_path: str,
                       max_batch: int,
                       min_batch_size: int,
                       mode: str,
                       use_lcm_lora: bool,
                       use_tiny_vae: bool,
                       ipadapter_scale: Optional[float] = None,
                       ipadapter_tokens: Optional[int] = None,
                       controlnet_model_id: Optional[str] = None,
                       is_faceid: Optional[bool] = None) -> Path:
        """
        Generate engine path using wrapper.py's current logic.
        
        Moves and consolidates create_prefix() function from wrapper.py lines 995-1014.
        Special handling for ControlNet engines which use model_id-based directories.
        """
        print(f"get_engine_path: Generating path for {engine_type.value}")
        print(f"get_engine_path: Model ID/Path: {model_id_or_path}")
        print(f"get_engine_path: Batch config - max: {max_batch}, min: {min_batch_size}")
        print(f"get_engine_path: Mode: {mode}")
        print(f"get_engine_path: LCM LoRA: {use_lcm_lora}, Tiny VAE: {use_tiny_vae}")
        if ipadapter_scale is not None:
            print(f"get_engine_path: IPAdapter scale: {ipadapter_scale}")
        if ipadapter_tokens is not None:
            print(f"get_engine_path: IPAdapter tokens: {ipadapter_tokens}")
        if controlnet_model_id is not None:
            print(f"get_engine_path: ControlNet model ID: {controlnet_model_id}")
        if is_faceid is not None:
            print(f"get_engine_path: FaceID: {is_faceid}")
        
        filename = self._configs[engine_type]['filename']
        print(f"get_engine_path: Using filename: {filename}")
        
        if engine_type == EngineType.CONTROLNET:
            # ControlNet engines use special model_id-based directory structure
            if controlnet_model_id is None:
                raise ValueError("get_engine_path: controlnet_model_id required for CONTROLNET engines")
            
            # Convert model_id to directory name format (replace "/" with "_")
            model_dir_name = controlnet_model_id.replace("/", "_")
            
            # Use ControlNetEnginePool naming convention: dynamic engines with 384-1024 range
            prefix = f"controlnet_{model_dir_name}--batch-{max_batch}--dyn-384-1024"
            engine_path = self.engine_dir / prefix / filename
            print(f"get_engine_path: ControlNet engine path: {engine_path}")
            return engine_path
        else:
            # Standard engines use the unified prefix format
            # Extract base name (from wrapper.py lines 1002-1003)
            maybe_path = Path(model_id_or_path)
            base_name = maybe_path.stem if maybe_path.exists() else model_id_or_path
            
            # Create prefix (from wrapper.py lines 1005-1013)
            prefix = f"{base_name}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch}--min_batch-{min_batch_size}"
            
            # IP-Adapter differentiation: add type and (optionally) tokens
            # Keep scale out of identity for runtime control, but include a type flag to separate caches
            if is_faceid is True:
                prefix += f"--fid"
            if ipadapter_tokens is not None:
                prefix += f"--tokens{ipadapter_tokens}"
            
            prefix += f"--mode-{mode}"
            
            engine_path = self.engine_dir / prefix / filename
            print(f"get_engine_path: Standard engine path: {engine_path}")
            return engine_path
    
    def _get_embedding_dim_for_model_type(self, model_type: str) -> int:
        """Get embedding dimension based on model type."""
        if model_type.lower() in ["sdxl"]:
            return 2048
        elif model_type.lower() in ["sd21", "sd2.1"]:
            return 1024
        else:  # sd15 and others
            return 768
    
    def _execute_compilation(self, compile_fn, engine_path: Path, model, model_config, batch_size: int, kwargs: Dict) -> None:
        """Execute compilation with common pattern to eliminate duplication."""
        onnx_path = str(engine_path) + ".onnx"
        onnx_opt_path = str(engine_path) + ".opt.onnx"
        engine_build_options = kwargs.get('engine_build_options', {})
        
        print(f"_execute_compilation: Starting compilation with {compile_fn.__name__}")
        print(f"_execute_compilation: ONNX path: {onnx_path}")
        print(f"_execute_compilation: ONNX opt path: {onnx_opt_path}")
        print(f"_execute_compilation: Engine path: {engine_path}")
        print(f"_execute_compilation: Batch size: {batch_size}")
        print(f"_execute_compilation: Build options: {engine_build_options}")
        
        compile_fn(
            model,
            model_config,
            onnx_path,
            onnx_opt_path,
            str(engine_path),
            opt_batch_size=batch_size,
            engine_build_options=engine_build_options
        )
        print(f"_execute_compilation: Compilation completed successfully")
    
    def _prepare_controlnet_models(self, kwargs: Dict):
        """Prepare ControlNet models for compilation."""
        from streamdiffusion.acceleration.tensorrt.models.controlnet_models import create_controlnet_model
        import torch
        
        model_type = kwargs.get('model_type', 'sd15')
        batch_size = kwargs['batch_size']
        embedding_dim = self._get_embedding_dim_for_model_type(model_type)
        
        # Create ControlNet model configuration
        controlnet_model = create_controlnet_model(
            model_type=model_type,
            unet=kwargs.get('unet'),
            model_path=kwargs.get('model_path', ""),
            max_batch=batch_size,
            min_batch_size=1,
            embedding_dim=embedding_dim
        )
        
        # Prepare ControlNet model for compilation
        pytorch_model = kwargs['model'].to(torch.device("cuda"), dtype=torch.float16)
        
        return pytorch_model, controlnet_model
    
    def _get_default_controlnet_build_options(self) -> Dict:
        """Get default engine build options for ControlNet engines."""
        return {
            'opt_image_height': 704,  # Dynamic optimal resolution
            'opt_image_width': 704,
            'build_dynamic_shape': True,
            'min_image_resolution': 384,
            'max_image_resolution': 1024,
            'build_static_batch': False,
        }
    
    def compile_and_load_engine(self, 
                               engine_type: EngineType, 
                               engine_path: Path,
                               **kwargs) -> Any:
        """
        Universal compile and load logic for all engine types.
        
        Moves compilation blocks from wrapper.py lines 1200-1252, 1254-1283, 1285-1313.
        """
        print(f"compile_and_load_engine: Starting compilation for {engine_type.value}")
        print(f"compile_and_load_engine: Engine path: {engine_path}")
        print(f"compile_and_load_engine: Engine exists: {engine_path.exists()}")
        print(f"compile_and_load_engine: Kwargs keys: {list(kwargs.keys())}")
        
        if not engine_path.exists():
            # Get the appropriate compile function for this engine type
            config = self._configs[engine_type]
            compile_fn = config['compile_fn']
            print(f"compile_and_load_engine: Using compile function: {compile_fn.__name__}")
            
            # Ensure parent directory exists
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"compile_and_load_engine: Created parent directory: {engine_path.parent}")
            
            # Handle engine-specific compilation requirements
            if engine_type == EngineType.VAE_DECODER:
                print(f"compile_and_load_engine: VAE_DECODER compilation path")
                # VAE decoder requires modifying forward method during compilation
                stream_vae = kwargs['stream_vae']
                stream_vae.forward = stream_vae.decode
                print(f"compile_and_load_engine: Modified VAE forward method for compilation")
                try:
                    self._execute_compilation(compile_fn, engine_path, kwargs['model'], kwargs['model_config'], kwargs['batch_size'], kwargs)
                finally:
                    # Always clean up the forward attribute
                    delattr(stream_vae, "forward")
                    print(f"compile_and_load_engine: Cleaned up VAE forward method")
            elif engine_type == EngineType.CONTROLNET:
                print(f"compile_and_load_engine: CONTROLNET compilation path")
                # ControlNet requires special model creation and compilation
                model, model_config = self._prepare_controlnet_models(kwargs)
                print(f"compile_and_load_engine: Prepared ControlNet models")
                self._execute_compilation(compile_fn, engine_path, model, model_config, kwargs['batch_size'], kwargs)
            else:
                print(f"compile_and_load_engine: Standard compilation path for {engine_type.value}")
                # Standard compilation for UNet and VAE encoder
                self._execute_compilation(compile_fn, engine_path, kwargs['model'], kwargs['model_config'], kwargs['batch_size'], kwargs)
        else:
            print(f"compile_and_load_engine: Engine already exists, skipping compilation")
        
        # Load and return using the appropriate loader
        print(f"compile_and_load_engine: Loading engine for {engine_type.value}")
        return self.load_engine(engine_type, engine_path, **kwargs)
    
    def load_engine(self, engine_type: EngineType, engine_path: Path, **kwargs: Dict) -> Any:
        """Load engine with type-specific handling."""
        print(f"load_engine: Loading {engine_type.value} engine from {engine_path}")
        config = self._configs[engine_type]
        loader = config['loader']
        print(f"load_engine: Using loader function for {engine_type.value}")
        
        if engine_type == EngineType.UNET:
            print(f"load_engine: UNet-specific loading with metadata")
            # UNet engine needs special handling for metadata and error recovery
            loaded_engine = loader(engine_path, kwargs.get('cuda_stream'))
            self._set_unet_metadata(loaded_engine, kwargs)
            print(f"load_engine: UNet engine loaded and metadata set")
            return loaded_engine
        elif engine_type == EngineType.CONTROLNET:
            print(f"load_engine: ControlNet-specific loading")
            # ControlNet engine needs model_type parameter
            model_type = kwargs.get('model_type', 'sd15')
            use_cuda_graph = kwargs.get('use_cuda_graph', False)
            print(f"load_engine: ControlNet model_type: {model_type}, use_cuda_graph: {use_cuda_graph}")
            loaded_engine = loader(engine_path, kwargs.get('cuda_stream'), 
                         model_type=model_type,
                         use_cuda_graph=use_cuda_graph)
            print(f"load_engine: ControlNet engine loaded successfully")
            return loaded_engine
        else:
            print(f"load_engine: Standard engine loading for {engine_type.value}")
            loaded_engine = loader(engine_path, kwargs.get('cuda_stream'))
            print(f"load_engine: {engine_type.value} engine loaded successfully")
            return loaded_engine
    
    def _set_unet_metadata(self, loaded_engine, kwargs: Dict) -> None:
        """Set metadata on UNet engine for runtime use."""
        setattr(loaded_engine, 'use_control', kwargs.get('use_controlnet_trt', False))
        setattr(loaded_engine, 'use_ipadapter', kwargs.get('use_ipadapter_trt', False))
        
        if kwargs.get('use_controlnet_trt', False):
            setattr(loaded_engine, 'unet_arch', kwargs.get('unet_arch', {}))
            
        if kwargs.get('use_ipadapter_trt', False):
            setattr(loaded_engine, 'ipadapter_arch', kwargs.get('unet_arch', {}))
            # number of IP-attention layers for runtime vector sizing
            if 'num_ip_layers' in kwargs and kwargs['num_ip_layers'] is not None:
                setattr(loaded_engine, 'num_ip_layers', kwargs['num_ip_layers'])
        
    
    def get_or_load_controlnet_engine(self, 
                                    model_id: str,
                                    pytorch_model: Any,
                                    model_type: str = "sd15",
                                    batch_size: int = 1,
                                    cuda_stream = None,
                                    use_cuda_graph: bool = False,
                                    unet = None,
                                    model_path: str = "") -> Any:
        """
        Get or load ControlNet engine, providing unified interface for ControlNet management.
        
        Replaces ControlNetEnginePool.get_or_load_engine functionality.
        """
        # Generate engine path using ControlNet-specific logic
        engine_path = self.get_engine_path(
            EngineType.CONTROLNET,
            model_id_or_path="",  # Not used for ControlNet
            max_batch=batch_size,
            min_batch_size=1,
            mode="",  # Not used for ControlNet
            use_lcm_lora=False,  # Not used for ControlNet
            use_tiny_vae=False,  # Not used for ControlNet
            controlnet_model_id=model_id
        )
        
        # Compile and load ControlNet engine
        return self.compile_and_load_engine(
            EngineType.CONTROLNET,
            engine_path,
            model=pytorch_model,
            model_type=model_type,
            batch_size=batch_size,
            cuda_stream=cuda_stream,
            use_cuda_graph=use_cuda_graph,
            unet=unet,
            model_path=model_path,
            engine_build_options=self._get_default_controlnet_build_options()
        )