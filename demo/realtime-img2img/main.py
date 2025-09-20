from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import time
import mimetypes
import torch
import tempfile
from pathlib import Path
import yaml

from config import config, Args
from util import pil_to_frame, pt_to_frame, bytes_to_pil, bytes_to_pt
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline
from input_control import InputManager, GamepadInput

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120

# Import configuration from separate file to avoid circular imports
from app_config import AVAILABLE_CONTROLNETS, DEFAULT_SETTINGS

# Configure logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for the application"""
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up logger for streamdiffusion modules
    streamdiffusion_logger = logging.getLogger('streamdiffusion')
    streamdiffusion_logger.setLevel(numeric_level)
    
    # Set up logger for this application
    app_logger = logging.getLogger('realtime_img2img')
    app_logger.setLevel(numeric_level)
    
    return app_logger

# Initialize logger
logger = setup_logging(config.log_level)


class App:
    def __init__(self, config: Args):
        self.args = config
        self.pipeline = None  # Pipeline created lazily when needed
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.fps_counter = []
        self.last_fps_update = time.time()
        # Store uploaded ControlNet config separately
        self.uploaded_controlnet_config = None
        self.runtime_controlnet_config = None  # Active runtime config (starts from YAML)
        self.config_needs_reload = False  # Track when pipeline needs recreation
        # Store current resolution for pipeline recreation
        self.new_width = 512
        self.new_height = 512
        # Initialize input manager for controller support
        self.input_manager = InputManager()
        # Initialize input source manager for modular input routing
        from input_sources import InputSourceManager
        self.input_source_manager = InputSourceManager()
        self.init_app()

    def cleanup(self):
        """Cleanup resources when app is shutting down"""
        logger.info("App cleanup: Starting application cleanup...")
        if self.pipeline:
            self._cleanup_pipeline(self.pipeline)
            self.pipeline = None
        if hasattr(self, 'input_source_manager'):
            self.input_source_manager.cleanup()
        self._cleanup_temp_files()
        logger.info("App cleanup: Completed application cleanup")

    def _handle_input_parameter_update(self, parameter_name: str, value: float) -> None:
        """Handle parameter updates from input controls"""
        try:
            if not self.pipeline or not hasattr(self.pipeline, 'stream'):
                logger.warning(f"_handle_input_parameter_update: No pipeline available for parameter {parameter_name}")
                return

            # Map parameter names to pipeline update methods
            if parameter_name == 'guidance_scale':
                self.pipeline.update_stream_params(guidance_scale=value)
            elif parameter_name == 'delta':
                self.pipeline.update_stream_params(delta=value)
            elif parameter_name == 'num_inference_steps':
                self.pipeline.update_stream_params(num_inference_steps=int(value))
            elif parameter_name == 'seed':
                self.pipeline.update_stream_params(seed=int(value))
            elif parameter_name == 'ipadapter_scale':
                self.pipeline.update_stream_params(ipadapter_config={'scale': value})
            elif parameter_name == 'ipadapter_weight_type':
                # For weight type, we need to convert the numeric value to a string
                weight_types = ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
                               "weak input", "weak output", "weak middle", "strong middle", 
                               "style transfer", "composition", "strong style transfer", 
                               "style and composition", "style transfer precise", "composition precise"]
                index = int(value) % len(weight_types)
                self.pipeline.update_ipadapter_weight_type(weight_types[index])
            elif parameter_name.startswith('controlnet_') and parameter_name.endswith('_strength'):
                # Handle ControlNet strength parameters
                import re
                match = re.match(r'controlnet_(\d+)_strength', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Use existing ControlNet strength update logic
                    current_config = self._get_current_controlnet_config()
                    if current_config and index < len(current_config):
                        current_config[index]['conditioning_scale'] = float(value)
                        # Apply the updated config via unified API
                        self.pipeline.update_stream_params(controlnet_config=current_config)
            elif parameter_name.startswith('controlnet_') and '_preprocessor_' in parameter_name:
                # Handle ControlNet preprocessor parameters
                match = re.match(r'controlnet_(\d+)_preprocessor_(.+)', parameter_name)
                if match:
                    controlnet_index = int(match.group(1))
                    param_name = match.group(2)
                    # Use the same approach as the API endpoint
                    current_config = self._get_current_controlnet_config()
                    if current_config and controlnet_index < len(current_config):
                        # Update preprocessor_params for the specified controlnet
                        if 'preprocessor_params' not in current_config[controlnet_index]:
                            current_config[controlnet_index]['preprocessor_params'] = {}
                        current_config[controlnet_index]['preprocessor_params'][param_name] = value
                        self.pipeline.update_stream_params(controlnet_config=current_config)
            elif parameter_name.startswith('prompt_weight_'):
                # Handle prompt blending weights
                match = re.match(r'prompt_weight_(\d+)', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Get current prompt list from unified state and update specific weight
                    state = self.pipeline.stream.get_stream_state()
                    current_prompts = state.get('prompt_list', [])
                    if current_prompts and index < len(current_prompts):
                        updated_prompts = list(current_prompts)
                        updated_prompts[index] = (updated_prompts[index][0], float(value))
                        self.pipeline.update_stream_params(prompt_list=updated_prompts)
            elif parameter_name.startswith('seed_weight_'):
                # Handle seed blending weights  
                match = re.match(r'seed_weight_(\d+)', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Get current seed list from unified state and update specific weight
                    state = self.pipeline.stream.get_stream_state()
                    current_seeds = state.get('seed_list', [])
                    if current_seeds and index < len(current_seeds):
                        updated_seeds = list(current_seeds)
                        updated_seeds[index] = (updated_seeds[index][0], float(value))
                        self.pipeline.update_stream_params(seed_list=updated_seeds)
            else:
                logger.warning(f"_handle_input_parameter_update: Unknown parameter {parameter_name}")

        except Exception as e:
            logger.exception(f"_handle_input_parameter_update: Failed to update {parameter_name}: {e}")


    


    def _get_controlnet_pipeline(self):
        """Get the ControlNet pipeline from the main pipeline structure"""
        if not self.pipeline:
            return None
            
        stream = self.pipeline.stream
        
        # Module-aware: module installs expose preprocessors on stream
        if hasattr(stream, 'preprocessors'):
            return stream
            
        # Check if stream has nested stream (IPAdapter wrapper)
        if hasattr(stream, 'stream') and hasattr(stream.stream, 'preprocessors'):
            return stream.stream
            
        # New module path on stream
        if hasattr(stream, '_controlnet_module'):
            return stream._controlnet_module
        return None

    def _get_current_config(self, config_type: str):
        """Get the current configuration state from the pipeline using public API"""
        if not self.pipeline or not self.pipeline.stream:
            logging.warning(f"_get_current_config: No pipeline or stream wrapper for {config_type}")
            return []
        
        try:
            # Use the public get_stream_state API from the wrapper
            stream_state = self.pipeline.stream.get_stream_state()
            config_key = f"{config_type}_config"
            config = stream_state.get(config_key, [])
            return config
        except Exception as e:
            logging.warning(f"_get_current_config: Failed to get {config_type} config via get_stream_state: {e}")
            return []

    def _get_current_controlnet_config(self):
        """Get the current ControlNet configuration state from the pipeline using public API"""
        return self._get_current_config("controlnet")

    def _get_current_hook_config(self, hook_type: str):
        """Get the current hook configuration state from the pipeline using public API"""
        return self._get_current_config(hook_type)

    def init_app(self):
        # Enhanced CORS for API-only development mode
        if self.args.api_only:
            # More permissive CORS for development
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # Include common Vite dev ports
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        else:
            # Standard CORS for production
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Set up input manager callback for parameter updates
        self.input_manager.set_parameter_update_callback(self._handle_input_parameter_update)

        # Register route modules
        self._register_routes()
    
    def _register_routes(self):
        """Register all route modules with dependency injection"""
        from routes import parameters, controlnet, ipadapter, inference, pipeline_hooks, websocket, input_sources
        from routes.common.dependencies import get_app_instance as shared_get_app_instance, get_pipeline_class as shared_get_pipeline_class, get_default_settings as shared_get_default_settings, get_available_controlnets as shared_get_available_controlnets
        
        # Create dependency overrides to inject app instance and other dependencies
        def get_app_instance():
            return self
            
        def get_pipeline_class():
            return Pipeline
            
        def get_default_settings():
            return DEFAULT_SETTINGS
            
        def get_available_controlnets():
            return AVAILABLE_CONTROLNETS
        
        # Include routers and set up dependency overrides on the main app
        for router_module in [parameters, controlnet, ipadapter, inference, pipeline_hooks, websocket, input_sources]:
            # Include the router
            self.app.include_router(router_module.router)
        
        # Set up dependency overrides on the main app (not individual routers)
        self.app.dependency_overrides[shared_get_app_instance] = get_app_instance
        self.app.dependency_overrides[shared_get_pipeline_class] = get_pipeline_class
        self.app.dependency_overrides[shared_get_default_settings] = get_default_settings
        self.app.dependency_overrides[shared_get_available_controlnets] = get_available_controlnets
        
        # Set up static files if not in API-only mode
        if not self.args.api_only:
            self.app.mount("/", StaticFiles(directory="frontend/public", html=True), name="public")

    def _normalize_prompt_config(self, config_data):
        """
        Normalize prompt configuration to always return a list format.
        Priority: prompt_blending.prompt_list > prompt_blending (direct list) > prompt (converted to single-item list) > default
        """
        if not config_data:
            return None
            
        # Check for explicit prompt_blending first (highest priority)
        if 'prompt_blending' in config_data:
            prompt_blending = config_data['prompt_blending']
            
            # Handle nested structure: prompt_blending.prompt_list
            if isinstance(prompt_blending, dict) and 'prompt_list' in prompt_blending:
                prompt_list = prompt_blending['prompt_list']
                if isinstance(prompt_list, list) and len(prompt_list) > 0:
                    return prompt_list
            
            # Handle flat structure: prompt_blending as direct list
            elif isinstance(prompt_blending, list) and len(prompt_blending) > 0:
                return prompt_blending
        
        # Check for direct prompt_list key  
        if 'prompt_list' in config_data:
            prompt_list = config_data['prompt_list']
            if isinstance(prompt_list, list) and len(prompt_list) > 0:
                return prompt_list
        
        # Check for simple prompt key and convert to list format
        if 'prompt' in config_data:
            prompt = config_data['prompt']
            if prompt and isinstance(prompt, str):
                return [(prompt, 1.0)]  # Convert to list format with weight 1.0
        
        return None

    def _normalize_seed_config(self, config_data):
        """
        Normalize seed configuration to always return a list format.
        Priority: seed_blending.seed_list > seed_blending (direct list) > seed (converted to single-item list) > default
        """
        if not config_data:
            return None
            
        # Check for explicit seed_blending first (highest priority)
        if 'seed_blending' in config_data:
            seed_blending = config_data['seed_blending']
            
            # Handle nested structure: seed_blending.seed_list
            if isinstance(seed_blending, dict) and 'seed_list' in seed_blending:
                seed_list = seed_blending['seed_list']
                if isinstance(seed_list, list) and len(seed_list) > 0:
                    return seed_list
            
            # Handle flat structure: seed_blending as direct list
            elif isinstance(seed_blending, list) and len(seed_blending) > 0:
                return seed_blending
        
        # Check for direct seed_list key  
        if 'seed_list' in config_data:
            seed_list = config_data['seed_list']
            if isinstance(seed_list, list) and len(seed_list) > 0:
                return seed_list
        
        # Check for simple seed key and convert to list format
        if 'seed' in config_data:
            seed = config_data['seed']
            if seed is not None and isinstance(seed, (int, float)):
                return [(int(seed), 1.0)]  # Convert to list format with weight 1.0
        
        return None

    def _create_default_pipeline(self):
        """Create the default pipeline (standard mode)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        
        return Pipeline(self.args, device, torch_dtype, width=self.new_width, height=self.new_height)

    def _create_pipeline_with_config(self, controlnet_config_path=None):
        """Create a new pipeline with optional ControlNet configuration"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        
        # Use uploaded config if available
        if self.uploaded_controlnet_config:
            # Create a temporary config file for the pipeline to use
            import tempfile
            import yaml
            import os
            import atexit
            from config import Args
            
            # Create temp file with delete=False to control cleanup timing
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            temp_path = temp_file.name
            
            try:
                # Create enhanced config that includes runtime parameters from args
                enhanced_config = dict(self.uploaded_controlnet_config)
                
                # Ensure critical args parameters are included in the config
                # These are needed for proper wrapper creation
                if 'acceleration' not in enhanced_config:
                    enhanced_config['acceleration'] = self.args.acceleration
                if 'engine_dir' not in enhanced_config:
                    enhanced_config['engine_dir'] = self.args.engine_dir
                if 'use_safety_checker' not in enhanced_config:
                    enhanced_config['use_safety_checker'] = self.args.safety_checker
                if 'use_tiny_vae' not in enhanced_config and self.args.taesd:
                    enhanced_config['use_tiny_vae'] = self.args.taesd
                
                # Include resolution if not already specified
                if 'width' not in enhanced_config:
                    enhanced_config['width'] = self.new_width
                if 'height' not in enhanced_config:
                    enhanced_config['height'] = self.new_height
                
                # Force output_type to "pt" for optimal tensor performance
                enhanced_config['output_type'] = 'pt'
                
                # Write enhanced config to temp file
                yaml.dump(enhanced_config, temp_file)
                temp_file.close()
                
                # Create new Args object with updated controlnet_config (NamedTuple is immutable)
                args_dict = self.args._asdict()
                args_dict['controlnet_config'] = temp_path
                modified_args = Args(**args_dict)
                
                # Load config style images into InputSourceManager before creating pipeline
                self._load_config_style_images()
                
                # Create pipeline
                pipeline = Pipeline(modified_args, device, torch_dtype, width=self.new_width, height=self.new_height)
                
                # Store temp file path for cleanup later
                if not hasattr(self, '_temp_config_files'):
                    self._temp_config_files = []
                self._temp_config_files.append(temp_path)
                
                # Register cleanup on exit
                atexit.register(lambda: self._cleanup_temp_files())
                
                return pipeline
                
            except Exception as e:
                # Clean up temp file if pipeline creation fails
                temp_file.close()
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e
        
        return Pipeline(self.args, device, torch_dtype, width=self.new_width, height=self.new_height)

    def _load_config_style_images(self):
        """Load style images from config into InputSourceManager"""
        if not self.uploaded_controlnet_config:
            return
            
        try:
            # Load IPAdapter style images from config
            ipadapters = self.uploaded_controlnet_config.get('ipadapters', [])
            if ipadapters:
                first_ipadapter = ipadapters[0]
                style_image_path = first_ipadapter.get('style_image')
                if style_image_path:
                    # Use the config file path as base for relative paths
                    base_config_path = getattr(self.args, 'controlnet_config', None)
                    self.input_source_manager.load_config_style_image(style_image_path, base_config_path)
        except Exception as e:
            logging.exception(f"_load_config_style_images: Error loading config style images: {e}")

    def _cleanup_temp_files(self):
        """Clean up any temporary config files"""
        if hasattr(self, '_temp_config_files'):
            import os
            for temp_path in self._temp_config_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
            self._temp_config_files.clear()

    def _get_controlnet_info(self):
        """Get ControlNet information from uploaded config or active pipeline"""
        controlnet_info = {
            "enabled": False,
            "controlnets": []
        }
        
        raw_controlnets = []
        if self.pipeline and hasattr(self.pipeline, 'use_config') and self.pipeline.use_config:
            if self.pipeline.config and 'controlnets' in self.pipeline.config:
                controlnet_info["enabled"] = True
                raw_controlnets = self.pipeline.config['controlnets']
        elif self.uploaded_controlnet_config and 'controlnets' in self.uploaded_controlnet_config:
            controlnet_info["enabled"] = True  
            raw_controlnets = self.uploaded_controlnet_config['controlnets']
        
        # Map conditioning_scale to strength for frontend compatibility
        processed_controlnets = []
        for i, controlnet in enumerate(raw_controlnets):
            processed = dict(controlnet)
            processed['index'] = i
            processed['name'] = controlnet.get('model_id', '')
            processed['strength'] = controlnet.get('conditioning_scale', 0.0)
            processed_controlnets.append(processed)
        
        controlnet_info["controlnets"] = processed_controlnets
        return controlnet_info

    def _get_ipadapter_info(self):
        """Get IPAdapter information from uploaded config or active pipeline"""
        ipadapter_info = {
            "enabled": False,
            "has_style_image": False
        }
        
        # Get config values
        if self.uploaded_controlnet_config:
            if self.uploaded_controlnet_config.get('use_ipadapter', False):
                ipadapter_info["enabled"] = True
            
            ipadapters = self.uploaded_controlnet_config.get('ipadapters', [])
            if ipadapters:
                first = ipadapters[0]
                ipadapter_info["scale"] = first.get('scale', 1.0)
                ipadapter_info["weight_type"] = first.get('weight_type', 'linear')
                if first.get('style_image'):
                    ipadapter_info["has_style_image"] = True
                    ipadapter_info["style_image_path"] = first['style_image']
        
        if self.pipeline and hasattr(self.pipeline, 'has_ipadapter'):
            ipadapter_info["enabled"] = self.pipeline.has_ipadapter
        
        # Check if IPAdapter has a style image from InputSourceManager
        if hasattr(self, 'input_source_manager'):
            ipadapter_source_info = self.input_source_manager.get_source_info('ipadapter')
            if ipadapter_source_info.get('has_data', False):
                ipadapter_info["has_style_image"] = True
            
        return ipadapter_info

    def _get_hook_info(self, hook_type: str):
        """Get hook information for a specific hook type"""
        hook_info = {
            "enabled": False,
            "processors": []
        }
        
        if self.pipeline and hasattr(self.pipeline, 'stream'):
            # Use the proper method to get current hook configuration
            hooks_config = self._get_current_hook_config(hook_type)
            if hooks_config:
                hook_info["enabled"] = True
                
                # Process raw processors to add frontend-expected fields
                processed_processors = []
                for index, processor in enumerate(hooks_config):
                    if isinstance(processor, dict):
                        processed_processor = {
                            "index": index,
                            "name": processor.get("type", "unknown"),  # Map type to name
                            "type": processor.get("type", "unknown"),
                            "enabled": processor.get("enabled", False),
                            "order": processor.get("order", index + 1),
                            "params": processor.get("params", {})
                        }
                        processed_processors.append(processed_processor)
                
                hook_info["processors"] = processed_processors
        elif self.uploaded_controlnet_config and hook_type in self.uploaded_controlnet_config:
            # Fallback to config when no pipeline
            config = self.uploaded_controlnet_config[hook_type]
            if isinstance(config, dict):
                hook_info["enabled"] = config.get("enabled", False)
                raw_processors = config.get("processors", [])
                
                # Process raw processors to add frontend-expected fields
                processed_processors = []
                for index, processor in enumerate(raw_processors):
                    if isinstance(processor, dict):
                        processed_processor = {
                            "index": index,
                            "name": processor.get("type", "unknown"),  # Map type to name
                            "type": processor.get("type", "unknown"),
                            "enabled": processor.get("enabled", False),
                            "order": processor.get("order", index + 1),
                            "params": processor.get("params", {})
                        }
                        processed_processors.append(processed_processor)
                
                hook_info["processors"] = processed_processors
        
        return hook_info

    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate and return aspect ratio as a string"""
        import math
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        ratio_gcd = gcd(width, height)
        return f"{width//ratio_gcd}:{height//ratio_gcd}"

    def _cleanup_pipeline(self, pipeline):
        """Properly cleanup a pipeline and free VRAM"""
        if pipeline is None:
            return
        
        try:
            if hasattr(pipeline, 'cleanup'):
                pipeline.cleanup()
            del pipeline
            torch.cuda.empty_cache()
        except Exception as e:
            logging.exception(f"_cleanup_pipeline: Error during cleanup: {e}")


app = App(config).app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )