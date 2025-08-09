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

def load_controlnet_registry():
    """Load ControlNet registry from YAML config file"""
    try:
        registry_path = Path(__file__).parent / "controlnet_registry.yaml"
        with open(registry_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract the available_controlnets section
        return config_data.get('available_controlnets', {})
    except Exception as e:
        logging.error(f"load_controlnet_registry: Failed to load ControlNet registry: {e}")
        # Fallback to empty registry
        return {}

def load_default_settings():
    """Load default settings from YAML config file"""
    try:
        registry_path = Path(__file__).parent / "controlnet_registry.yaml"
        with open(registry_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data.get('defaults', {})
    except Exception as e:
        logging.error(f"load_default_settings: Failed to load default settings: {e}")
        # Fallback to hardcoded defaults
        return {
            'guidance_scale': 1.1,
            'delta': 0.7,
            'num_inference_steps': 50,
            'seed': 2,
            't_index_list': [35, 45],
            'ipadapter_scale': 1.0,
            'normalize_prompt_weights': True,
            'normalize_seed_weights': True,
            'prompt': "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        }

# Load ControlNet registry from config file
AVAILABLE_CONTROLNETS = load_controlnet_registry()
DEFAULT_SETTINGS = load_default_settings()

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
        # Store uploaded style image persistently
        self.uploaded_style_image = None
        # Initialize input manager for controller support
        self.input_manager = InputManager()
        self.init_app()

    def cleanup(self):
        """Cleanup resources when app is shutting down"""
        logger.info("App cleanup: Starting application cleanup...")
        if self.pipeline:
            self._cleanup_pipeline(self.pipeline)
            self.pipeline = None
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
                    # Get current prompt list and update specific weight
                    current_prompts = self.pipeline.get_current_prompts()
                    if current_prompts and index < len(current_prompts):
                        # Create updated prompt list with new weight
                        updated_prompts = current_prompts.copy()
                        updated_prompts[index] = (updated_prompts[index][0], value)
                        # Update prompt list with new weights
                        self.pipeline.update_prompt_weights([weight for _, weight in updated_prompts])
            elif parameter_name.startswith('seed_weight_'):
                # Handle seed blending weights  
                match = re.match(r'seed_weight_(\d+)', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Get current seed list and update specific weight
                    current_seeds = self.pipeline.get_current_seeds()
                    if current_seeds and index < len(current_seeds):
                        # Create updated seed list with new weight
                        updated_seeds = current_seeds.copy()
                        updated_seeds[index] = (updated_seeds[index][0], value)
                        # Update seed list with new weights
                        self.pipeline.update_seed_weights([weight for _, weight in updated_seeds])
            else:
                logger.warning(f"_handle_input_parameter_update: Unknown parameter {parameter_name}")

            logger.info(f"_handle_input_parameter_update: Updated {parameter_name} to {value}")
        except Exception as e:
            logger.error(f"_handle_input_parameter_update: Failed to update {parameter_name}: {e}")


    


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

    def _get_current_controlnet_config(self):
        """Get the current ControlNet configuration state from the pipeline"""
        cn_pipeline = self._get_controlnet_pipeline()
        if not cn_pipeline or not hasattr(cn_pipeline, 'controlnets'):
            return []
        
        current_config = []
        for i, controlnet in enumerate(cn_pipeline.controlnets):
            model_id = getattr(controlnet, 'model_id', f'controlnet_{i}')
            scale = cn_pipeline.controlnet_scales[i] if hasattr(cn_pipeline, 'controlnet_scales') and i < len(cn_pipeline.controlnet_scales) else 1.0
            
            config = {
                'model_id': model_id,
                'conditioning_scale': scale,
                'preprocessor': getattr(cn_pipeline.preprocessors[i], '__class__.__name__', '').replace('Preprocessor', '').lower() if cn_pipeline.preprocessors[i] else None,
                'enabled': True,
                'preprocessor_params': getattr(cn_pipeline.preprocessors[i], 'params', {}) if cn_pipeline.preprocessors[i] else {}
            }

            current_config.append(config)
        return current_config

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

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    if data is None:
                        break
                    if data["status"] == "next_frame":
                        params = await self.conn_manager.receive_json(user_id)
                        params = Pipeline.InputParams(**params)
                        params = SimpleNamespace(**params.dict())
                        
                        # Check if we need image data based on pipeline
                        need_image = True
                        if self.pipeline and hasattr(self.pipeline, 'pipeline_mode'):
                            # Need image for img2img OR for txt2img with ControlNets
                            has_controlnets = self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config
                            need_image = self.pipeline.pipeline_mode == "img2img" or has_controlnets
                        elif self.uploaded_controlnet_config and 'mode' in self.uploaded_controlnet_config:
                            # Need image for img2img OR for txt2img with ControlNets
                            has_controlnets = 'controlnets' in self.uploaded_controlnet_config
                            need_image = self.uploaded_controlnet_config['mode'] == "img2img" or has_controlnets
                        
                        if need_image:
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            
                            # Always use direct bytes-to-tensor conversion for efficiency
                            params.image = bytes_to_pt(image_data)
                        else:
                            params.image = None
                        
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                # Create pipeline if it doesn't exist yet
                if self.pipeline is None:
                    if self.uploaded_controlnet_config:
                        logger.info("stream: Creating pipeline with ControlNet config...")
                        self.pipeline = self._create_pipeline_with_config()
                    else:
                        logger.info("stream: Creating default pipeline...")
                        self.pipeline = self._create_default_pipeline()
                    logger.info("stream: Pipeline created successfully")
                    try:
                        acc = getattr(self.args, 'acceleration', None)
                        logger.debug(f"stream: acceleration={acc}, use_config={getattr(self.pipeline, 'use_config', False)}")
                        stream_obj = getattr(self.pipeline, 'stream', None)
                        unet_obj = getattr(stream_obj, 'unet', None)
                        is_trt = unet_obj is not None and hasattr(unet_obj, 'engine') and hasattr(unet_obj, 'stream')
                        logger.debug(f"stream: unet_is_trt={is_trt}, has_ipadapter={getattr(self.pipeline, 'has_ipadapter', False)}")
                        if is_trt:
                            logger.debug(f"stream: unet.use_ipadapter={getattr(unet_obj, 'use_ipadapter', None)}, num_ip_layers={getattr(unet_obj, 'num_ip_layers', None)}")
                        if hasattr(stream_obj, 'ipadapter_scale'):
                            try:
                                scale_val = getattr(stream_obj, 'ipadapter_scale')
                                if hasattr(scale_val, 'shape'):
                                    logger.debug(f"stream: ipadapter_scale tensor shape={tuple(scale_val.shape)}")
                                else:
                                    logger.debug(f"stream: ipadapter_scale scalar={scale_val}")
                            except Exception:
                                pass
                        logger.debug(f"stream: ipadapter_weight_type={getattr(stream_obj, 'ipadapter_weight_type', None)}")
                    except Exception:
                        logger.exception("stream: failed to log pipeline state after creation")
                
                # Recreate pipeline if config changed (but not resolution - that's handled separately)
                elif self.config_needs_reload or (self.uploaded_controlnet_config and not (self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config)) or (self.uploaded_controlnet_config and not self.pipeline.use_config):
                    if self.config_needs_reload:
                        logger.info("stream: Recreating pipeline with new ControlNet config...")
                    else:
                        logger.info("stream: Upgrading to ControlNet pipeline...")
                    
                    # Properly cleanup the old pipeline before creating new one
                    old_pipeline = self.pipeline
                    self.pipeline = None
                    
                    if old_pipeline:
                        self._cleanup_pipeline(old_pipeline)
                        old_pipeline = None
                    
                    # Create new pipeline
                    if self.uploaded_controlnet_config:
                        self.pipeline = self._create_pipeline_with_config()
                    else:
                        self.pipeline = self._create_default_pipeline()
                    
                    self.config_needs_reload = False  # Reset the flag
                    logger.info("stream: Pipeline recreated successfully")

                async def generate():
                    while True:
                        frame_start_time = time.time()
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        
                        try:
                            try:
                                stream_obj = getattr(self.pipeline, 'stream', None)
                                unet_obj = getattr(stream_obj, 'unet', None)
                                is_trt = unet_obj is not None and hasattr(unet_obj, 'engine') and hasattr(unet_obj, 'stream')
                                logger.debug(f"generate: calling predict; acceleration={getattr(self.args, 'acceleration', None)}, is_trt={is_trt}, mode={getattr(self.pipeline, 'pipeline_mode', None)}, has_ipadapter={getattr(self.pipeline, 'has_ipadapter', False)}, has_controlnet={(self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config) if getattr(self.pipeline, 'use_config', False) else False}")
                                img = getattr(params, 'image', None)
                                if isinstance(img, torch.Tensor):
                                    logger.debug(f"generate: params.image tensor shape={tuple(img.shape)}, dtype={img.dtype}")
                                else:
                                    logger.debug(f"generate: params.image type={type(img).__name__}")
                                if is_trt:
                                    logger.debug(f"generate: unet.use_ipadapter={getattr(unet_obj, 'use_ipadapter', None)}, num_ip_layers={getattr(unet_obj, 'num_ip_layers', None)}")
                                    try:
                                        base_scale = getattr(stream_obj, 'ipadapter_scale', None)
                                        if base_scale is not None:
                                            if hasattr(base_scale, 'shape'):
                                                logger.debug(f"generate: base ipadapter_scale shape={tuple(base_scale.shape)}")
                                            else:
                                                logger.debug(f"generate: base ipadapter_scale scalar={base_scale}")
                                        logger.debug(f"generate: ipadapter_weight_type={getattr(stream_obj, 'ipadapter_weight_type', None)}")
                                    except Exception:
                                        pass
                            except Exception:
                                logger.exception("generate: pre-predict logging failed")

                            image = self.pipeline.predict(params)
                            if image is None:
                                logger.error("generate: predict returned None image; skipping frame")
                                continue
                            
                            # Use appropriate frame conversion based on output type
                            if self.pipeline.output_type == "pt":
                                frame = pt_to_frame(image)
                            else:
                                frame = pil_to_frame(image)
                        except Exception as e:
                            logger.exception(f"generate: predict failed with exception: {e}")
                            continue
                        
                        # Update FPS counter
                        frame_time = time.time() - frame_start_time
                        self.fps_counter.append(frame_time)
                        if len(self.fps_counter) > 30:  # Keep last 30 frames
                            self.fps_counter.pop(0)
                        
                        yield frame
                        if self.args.debug:
                            logger.debug(f"Time taken: {time.time() - frame_start_time}")
                        
                        # Add delay for testing - 1 frame per second
                        # await asyncio.sleep(1.0)

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            # Use Pipeline class directly for schema info (doesn't require instance)
            info_schema = Pipeline.Info.schema()
            info = Pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = Pipeline.InputParams.schema()
            
            # Add ControlNet information 
            controlnet_info = self._get_controlnet_info()
            
            # Add IPAdapter information
            ipadapter_info = self._get_ipadapter_info()
            
            # Include config prompt if available, otherwise use default
            config_prompt = None
            if self.uploaded_controlnet_config and 'prompt' in self.uploaded_controlnet_config:
                config_prompt = self.uploaded_controlnet_config['prompt']
            elif not config_prompt:
                config_prompt = DEFAULT_SETTINGS.get('prompt')
            
            # Get current t_index_list from pipeline or config
            current_t_index_list = None
            if self.pipeline and hasattr(self.pipeline.stream, 't_list'):
                current_t_index_list = self.pipeline.stream.t_list
            elif self.uploaded_controlnet_config and 't_index_list' in self.uploaded_controlnet_config:
                current_t_index_list = self.uploaded_controlnet_config['t_index_list']
            else:
                # Default values
                current_t_index_list = DEFAULT_SETTINGS.get('t_index_list', [35, 45])
            
            # Get current acceleration setting
            current_acceleration = self.args.acceleration
            
            # Get current resolution
            current_resolution = f"{self.new_width}x{self.new_height}"
            # Add aspect ratio for display
            aspect_ratio = self._calculate_aspect_ratio(self.new_width, self.new_height)
            if aspect_ratio:
                current_resolution += f" ({aspect_ratio})"
            if self.uploaded_controlnet_config and 'acceleration' in self.uploaded_controlnet_config:
                current_acceleration = self.uploaded_controlnet_config['acceleration']
            
            # Get current streaming parameters (default values or from pipeline if available)
            current_guidance_scale = DEFAULT_SETTINGS.get('guidance_scale', 1.1)
            current_delta = DEFAULT_SETTINGS.get('delta', 0.7)
            current_num_inference_steps = DEFAULT_SETTINGS.get('num_inference_steps', 50)
            current_seed = DEFAULT_SETTINGS.get('seed', 2)
            
            if self.pipeline:
                current_guidance_scale = getattr(self.pipeline.stream, 'guidance_scale', DEFAULT_SETTINGS.get('guidance_scale', 1.1))
                current_delta = getattr(self.pipeline.stream, 'delta', DEFAULT_SETTINGS.get('delta', 0.7))
                current_num_inference_steps = getattr(self.pipeline.stream, 'num_inference_steps', DEFAULT_SETTINGS.get('num_inference_steps', 50))
                # Get seed from generator if available
                if hasattr(self.pipeline.stream, 'generator') and self.pipeline.stream.generator is not None:
                    # We can't directly get seed from generator, but we'll use the configured value
                    current_seed = getattr(self.pipeline.stream, 'current_seed', DEFAULT_SETTINGS.get('seed', 2))
            elif self.uploaded_controlnet_config:
                current_guidance_scale = self.uploaded_controlnet_config.get('guidance_scale', DEFAULT_SETTINGS.get('guidance_scale', 1.1))
                current_delta = self.uploaded_controlnet_config.get('delta', DEFAULT_SETTINGS.get('delta', 0.7))
                current_num_inference_steps = self.uploaded_controlnet_config.get('num_inference_steps', DEFAULT_SETTINGS.get('num_inference_steps', 50))
                current_seed = self.uploaded_controlnet_config.get('seed', DEFAULT_SETTINGS.get('seed', 2))
            
            # Get prompt and seed blending configuration from uploaded config or pipeline
            prompt_blending_config = None
            seed_blending_config = None
            
            # First try to get from current pipeline if available
            if self.pipeline:
                try:
                    current_prompts = self.pipeline.stream.get_current_prompts()
                    if current_prompts and len(current_prompts) > 0:
                        prompt_blending_config = current_prompts
                except:
                    pass
                    
                try:
                    current_seeds = self.pipeline.stream.get_current_seeds()
                    if current_seeds and len(current_seeds) > 0:
                        seed_blending_config = current_seeds
                except:
                    pass
            
            # If not available from pipeline, get from uploaded config and normalize
            if not prompt_blending_config:
                prompt_blending_config = self._normalize_prompt_config(self.uploaded_controlnet_config)
            
            if not seed_blending_config:
                seed_blending_config = self._normalize_seed_config(self.uploaded_controlnet_config)
            
            # Get current normalize weights settings
            normalize_prompt_weights = True  # default
            normalize_seed_weights = True    # default
            
            if self.pipeline:
                normalize_prompt_weights = self.pipeline.stream.get_normalize_prompt_weights()
                normalize_seed_weights = self.pipeline.stream.get_normalize_seed_weights()
            elif self.uploaded_controlnet_config:
                normalize_prompt_weights = self.uploaded_controlnet_config.get('normalize_weights', True)
                normalize_seed_weights = self.uploaded_controlnet_config.get('normalize_weights', True)
            
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                    "controlnet": controlnet_info,
                    "ipadapter": ipadapter_info,
                    "config_prompt": config_prompt,
                    "t_index_list": current_t_index_list,
                    "acceleration": current_acceleration,
                    "guidance_scale": current_guidance_scale,
                    "delta": current_delta,
                    "num_inference_steps": current_num_inference_steps,
                    "seed": current_seed,
                    "current_resolution": current_resolution,
                    "prompt_blending": prompt_blending_config,
                    "seed_blending": seed_blending_config,
                    "normalize_prompt_weights": normalize_prompt_weights,
                    "normalize_seed_weights": normalize_seed_weights,
                }
            )

        @self.app.post("/api/controlnet/upload-config")
        async def upload_controlnet_config(file: UploadFile = File(...)):
            """Upload and load a new ControlNet YAML configuration"""
            try:
                if not file.filename.endswith(('.yaml', '.yml')):
                    raise HTTPException(status_code=400, detail="File must be a YAML file")
                
                # Save uploaded file temporarily
                content = await file.read()
                
                # Parse YAML content
                try:
                    config_data = yaml.safe_load(content.decode('utf-8'))
                except yaml.YAMLError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
                
                # YAML is source of truth - completely replace any runtime modifications
                self.uploaded_controlnet_config = config_data
                self.runtime_controlnet_config = None  # Clear any runtime additions
                self.config_needs_reload = True  # Mark that pipeline needs recreation
                
                logger.info(f"upload_controlnet_config: YAML uploaded - resetting ControlNet configuration to source of truth")
                
                # Log IPAdapter configuration for debugging
    
                
                # Get config prompt if available
                config_prompt = config_data.get('prompt', None)
                
                # Get t_index_list from config if available
                t_index_list = config_data.get('t_index_list', DEFAULT_SETTINGS.get('t_index_list', [35, 45]))
                
                # Get acceleration from config if available
                config_acceleration = config_data.get('acceleration', self.args.acceleration)
                
                # Get width and height from config if available
                config_width = config_data.get('width', None)
                config_height = config_data.get('height', None)
                
                # Update resolution if width/height are specified in config
                if config_width is not None and config_height is not None:
                    try:
                        # Validate resolution
                        if config_width % 64 != 0 or config_height % 64 != 0:
                            raise HTTPException(status_code=400, detail="Resolution must be multiples of 64")
                        
                        if not (384 <= config_width <= 1024) or not (384 <= config_height <= 1024):
                            raise HTTPException(status_code=400, detail="Resolution must be between 384 and 1024")
                        
                        # Update the resolution
                        self.new_width = config_width
                        self.new_height = config_height
                        logger.info(f"upload_controlnet_config: Updated resolution to {config_width}x{config_height}")
                    except Exception as e:
                        logging.error(f"upload_controlnet_config: Failed to update resolution: {e}")
                        # Don't fail the upload, just log the error
                
                # Normalize prompt and seed configurations for frontend
                normalized_prompt_blending = self._normalize_prompt_config(config_data)
                normalized_seed_blending = self._normalize_seed_config(config_data)
                
                # Debug logging
                logger.debug(f"upload_controlnet_config: Raw prompt_blending in config: {config_data.get('prompt_blending', 'NOT FOUND')}")
                logger.debug(f"upload_controlnet_config: Raw seed_blending in config: {config_data.get('seed_blending', 'NOT FOUND')}")
                logger.debug(f"upload_controlnet_config: Normalized prompt blending: {normalized_prompt_blending}")
                logger.debug(f"upload_controlnet_config: Normalized seed blending: {normalized_seed_blending}")
                
                # Get other streaming parameters from config
                config_guidance_scale = config_data.get('guidance_scale', 1.1)
                config_delta = config_data.get('delta', 0.7)
                config_num_inference_steps = config_data.get('num_inference_steps', 50)
                config_seed = config_data.get('seed', 2)
                
                # Get normalization settings
                config_normalize_weights = config_data.get('normalize_weights', True)
                
                # Calculate current resolution string for frontend
                current_resolution = f"{self.new_width}x{self.new_height}"
                aspect_ratio = self._calculate_aspect_ratio(self.new_width, self.new_height)
                if aspect_ratio:
                    current_resolution += f" ({aspect_ratio})"
                
                # Get updated IPAdapter info for response
                response_ipadapter_info = self._get_ipadapter_info()

                
                return JSONResponse({
                    "status": "success",
                    "message": "ControlNet configuration uploaded successfully",
                    "controls_updated": True,  # Flag for frontend to update controls
                    "controlnet": self._get_controlnet_info(),
                    "ipadapter": response_ipadapter_info,  # Include updated IPAdapter info
                    "config_prompt": config_prompt,
                    "t_index_list": t_index_list,
                    "acceleration": config_acceleration,
                    "guidance_scale": config_guidance_scale,
                    "delta": config_delta,
                    "num_inference_steps": config_num_inference_steps,
                    "seed": config_seed,
                    "prompt_blending": normalized_prompt_blending,
                    "seed_blending": normalized_seed_blending,
                    "current_resolution": current_resolution,  # Include updated resolution
                    "normalize_prompt_weights": config_normalize_weights,
                    "normalize_seed_weights": config_normalize_weights,
                })
                
            except Exception as e:
                logging.error(f"upload_controlnet_config: Failed to upload config: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload configuration: {str(e)}")

        @self.app.get("/api/controlnet/info")
        async def get_controlnet_info():
            """Get current ControlNet configuration info"""
            return JSONResponse({"controlnet": self._get_controlnet_info()})

        @self.app.get("/api/blending/current")
        async def get_current_blending_config():
            """Get current prompt and seed blending configurations"""
            try:
                # Get normalized configurations (same logic as settings endpoint)
                prompt_blending_config = None
                seed_blending_config = None
                
                # First try to get from current pipeline if available
                if self.pipeline:
                    try:
                        current_prompts = self.pipeline.stream.get_current_prompts()
                        if current_prompts and len(current_prompts) > 0:
                            prompt_blending_config = current_prompts
                    except Exception:
                        pass
                        
                    try:
                        current_seeds = self.pipeline.stream.get_current_seeds()
                        if current_seeds and len(current_seeds) > 0:
                            seed_blending_config = current_seeds
                    except:
                        pass
                
                # If not available from pipeline, get from uploaded config and normalize
                if not prompt_blending_config:
                    prompt_blending_config = self._normalize_prompt_config(self.uploaded_controlnet_config)
                
                if not seed_blending_config:
                    seed_blending_config = self._normalize_seed_config(self.uploaded_controlnet_config)
                
                # Get normalization settings
                normalize_prompt_weights = True
                normalize_seed_weights = True
                
                if self.pipeline:
                    normalize_prompt_weights = self.pipeline.stream.get_normalize_prompt_weights()
                    normalize_seed_weights = self.pipeline.stream.get_normalize_seed_weights()
                elif self.uploaded_controlnet_config:
                    normalize_prompt_weights = self.uploaded_controlnet_config.get('normalize_weights', True)
                    normalize_seed_weights = self.uploaded_controlnet_config.get('normalize_weights', True)
                
                return JSONResponse({
                    "prompt_blending": prompt_blending_config,
                    "seed_blending": seed_blending_config,
                    "normalize_prompt_weights": normalize_prompt_weights,
                    "normalize_seed_weights": normalize_seed_weights,
                    "has_config": self.uploaded_controlnet_config is not None,
                    "pipeline_active": self.pipeline is not None
                })
                
            except Exception as e:
                logging.error(f"get_current_blending_config: Failed to get blending config: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get blending config: {str(e)}")

        @self.app.post("/api/controlnet/update-strength")
        async def update_controlnet_strength(request: Request):
            """Update ControlNet strength in real-time"""
            try:
                data = await request.json()
                controlnet_index = data.get("index")
                strength = data.get("strength")
                
                if controlnet_index is None or strength is None:
                    raise HTTPException(status_code=400, detail="Missing index or strength parameter")
                
                # Check if ControlNet is enabled using config system
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Check if we're using config mode and have controlnets configured
                controlnet_enabled = (self.pipeline.use_config and 
                                    self.pipeline.config and 
                                    'controlnets' in self.pipeline.config)
                
                if not controlnet_enabled:
                    raise HTTPException(status_code=400, detail="ControlNet is not enabled")
                
                # Update ControlNet strength using consolidated API
                current_config = self._get_current_controlnet_config()
                logger.info(f"update_controlnet_strength: Current config: {current_config}")
                
                if controlnet_index >= len(current_config):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                # Update only the conditioning_scale for the specified controlnet
                old_strength = current_config[controlnet_index]['conditioning_scale']
                current_config[controlnet_index]['conditioning_scale'] = float(strength)
                logger.info(f"update_controlnet_strength: Updating ControlNet {controlnet_index} strength from {old_strength} to {strength}")
                logger.info(f"update_controlnet_strength: Sending config: {current_config}")
                
                self.pipeline.update_stream_params(controlnet_config=current_config)
                logger.info(f"update_controlnet_strength: update_stream_params call completed")
                    
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated ControlNet {controlnet_index} strength to {strength}"
                })
                
            except Exception as e:
                logging.error(f"update_controlnet_strength: Failed to update strength: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update strength: {str(e)}")

        @self.app.get("/api/controlnet/available")
        async def get_available_controlnets():
            """Get list of available ControlNets that can be added"""
            try:
                # Detect current model architecture to filter appropriate ControlNets
                model_type = "sd15"  # Default fallback
                
                if self.pipeline and hasattr(self.pipeline, 'config') and self.pipeline.config:
                    # Try to determine model type from config
                    model_id = self.pipeline.config.get('model_id', '')
                    if 'sdxl' in model_id.lower() or 'xl' in model_id.lower():
                        model_type = "sdxl"
                
                available = AVAILABLE_CONTROLNETS.get(model_type, [])
                
                # Filter out already active ControlNets
                current_controlnets = []
                # Check runtime config first, then fall back to uploaded config
                if self.runtime_controlnet_config and 'controlnets' in self.runtime_controlnet_config:
                    current_controlnets = [cn.get('model_id', '') for cn in self.runtime_controlnet_config['controlnets']]
                elif self.uploaded_controlnet_config and 'controlnets' in self.uploaded_controlnet_config:
                    current_controlnets = [cn.get('model_id', '') for cn in self.uploaded_controlnet_config['controlnets']]
                
                filtered_available = []
                for cn in available:
                    if cn['model_id'] not in current_controlnets:
                        filtered_available.append(cn)
                
                return JSONResponse({
                    "status": "success",
                    "available_controlnets": filtered_available,
                    "model_type": model_type
                })
                
            except Exception as e:
                logging.error(f"get_available_controlnets: Failed to get available ControlNets: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get available ControlNets: {str(e)}")

        @self.app.post("/api/controlnet/add")
        async def add_controlnet(request: Request):
            """Add a ControlNet from the predefined list"""
            try:
                data = await request.json()
                controlnet_id = data.get("controlnet_id")
                conditioning_scale = data.get("conditioning_scale", None)
                
                if not controlnet_id:
                    raise HTTPException(status_code=400, detail="Missing controlnet_id parameter")
                
                # Find the ControlNet definition
                controlnet_def = None
                for model_type, controlnets in AVAILABLE_CONTROLNETS.items():
                    for cn in controlnets:
                        if cn['id'] == controlnet_id:
                            controlnet_def = cn
                            break
                    if controlnet_def:
                        break
                
                if not controlnet_def:
                    raise HTTPException(status_code=400, detail=f"ControlNet {controlnet_id} not found in registry")
                
                # Use provided scale or default
                if conditioning_scale is None:
                    conditioning_scale = controlnet_def['default_scale']
                
                # Initialize runtime config from YAML if not already done
                if self.runtime_controlnet_config is None:
                    if self.uploaded_controlnet_config:
                        # Copy from YAML (deep copy to avoid modifying original)
                        import copy
                        self.runtime_controlnet_config = copy.deepcopy(self.uploaded_controlnet_config)
                    else:
                        # Create minimal config if no YAML exists
                        self.runtime_controlnet_config = {'controlnets': []}
                
                # Ensure controlnets key exists in runtime config
                if 'controlnets' not in self.runtime_controlnet_config:
                    self.runtime_controlnet_config['controlnets'] = []
                
                # Create new ControlNet entry
                new_controlnet = {
                    'model_id': controlnet_def['model_id'],
                    'conditioning_scale': conditioning_scale,
                    'preprocessor': controlnet_def['default_preprocessor'],
                    'preprocessor_params': controlnet_def.get('preprocessor_params', {}),
                    'enabled': True
                }
                
                # Add to runtime config (not YAML)
                self.runtime_controlnet_config['controlnets'].append(new_controlnet)
                
                # Update pipeline using consolidated API
                try:
                    current_config = self._get_current_controlnet_config()
                    current_config.append(new_controlnet)
                    self.pipeline.update_stream_params(controlnet_config=current_config)
                    logger.info(f"add_controlnet: Successfully added ControlNet using consolidated API")
                except Exception as e:
                    logger.error(f"add_controlnet: Failed to add ControlNet: {e}")
                    # Mark for reload as fallback
                    self.config_needs_reload = True
                
                logger.info(f"add_controlnet: Added {controlnet_def['name']} with scale {conditioning_scale}")
                
                # Return updated ControlNet info immediately
                updated_info = self._get_controlnet_info()
                added_index = len(self.runtime_controlnet_config['controlnets']) - 1
                
                return JSONResponse({
                    "status": "success", 
                    "message": f"Added {controlnet_def['name']}",
                    "controlnet_index": added_index,
                    "controlnet_info": updated_info
                })
                
            except Exception as e:
                logging.error(f"add_controlnet: Failed to add ControlNet: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add ControlNet: {str(e)}")

        @self.app.get("/api/controlnet/status")
        async def get_controlnet_status():
            """Get the status of ControlNet configuration"""
            try:
                controlnet_pipeline = self._get_controlnet_pipeline()
                
                if not controlnet_pipeline:
                    return JSONResponse({
                        "status": "no_pipeline",
                        "message": "No ControlNet pipeline available",
                        "controlnet_count": 0
                    })
                
                current_config = self._get_current_controlnet_config()
                
                return JSONResponse({
                    "status": "ready",
                    "controlnet_count": len(current_config),
                    "message": f"{len(current_config)} ControlNet(s) configured" if current_config else "No ControlNets configured"
                })
                
            except Exception as e:
                logger.error(f"get_controlnet_status: Failed to get status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

        @self.app.post("/api/controlnet/remove")
        async def remove_controlnet(request: Request):
            """Remove a ControlNet by index"""
            try:
                data = await request.json()
                index = data.get("index")
                
                if index is None:
                    raise HTTPException(status_code=400, detail="Missing index parameter")
                
                # Initialize runtime config from YAML if not already done
                if self.runtime_controlnet_config is None:
                    if self.uploaded_controlnet_config:
                        # Copy from YAML (deep copy to avoid modifying original)
                        import copy
                        self.runtime_controlnet_config = copy.deepcopy(self.uploaded_controlnet_config)
                    else:
                        raise HTTPException(status_code=400, detail="No ControlNet configuration found")
                
                if 'controlnets' not in self.runtime_controlnet_config:
                    raise HTTPException(status_code=400, detail="No ControlNet configuration found")
                
                controlnets = self.runtime_controlnet_config['controlnets']
                
                if index < 0 or index >= len(controlnets):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
                
                removed_controlnet = controlnets.pop(index)
                
                # Update pipeline using consolidated API
                try:
                    current_config = self._get_current_controlnet_config()
                    if index >= len(current_config):
                        raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
                    
                    # Remove the controlnet at the specified index
                    current_config.pop(index)
                    self.pipeline.update_stream_params(controlnet_config=current_config)
                    logger.info(f"remove_controlnet: Successfully removed ControlNet using consolidated API")
                except Exception as e:
                    logger.error(f"remove_controlnet: Failed to remove ControlNet: {e}")
                    # Mark for reload as fallback
                    self.config_needs_reload = True
                
                logger.info(f"remove_controlnet: Removed ControlNet at index {index}: {removed_controlnet.get('model_id', 'unknown')}")
                
                # Return updated ControlNet info immediately
                updated_info = self._get_controlnet_info()
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Removed ControlNet at index {index}",
                    "controlnet_info": updated_info
                })
                
            except Exception as e:
                logging.error(f"remove_controlnet: Failed to remove ControlNet: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to remove ControlNet: {str(e)}")

        @self.app.post("/api/ipadapter/upload-style-image")
        async def upload_style_image(file: UploadFile = File(...)):
            """Upload a style image for IPAdapter"""
            try:
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Read file content
                content = await file.read()
                
                # Save temporarily and load as PIL Image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    # Load and validate image
                    from PIL import Image
                    style_image = Image.open(tmp_path).convert("RGB")
                    
                    # Store the uploaded style image persistently FIRST
                    self.uploaded_style_image = style_image
                    print(f"upload_style_image: Stored style image with size: {style_image.size}")
                    
                    # If pipeline exists and has IPAdapter, update it immediately
                    pipeline_updated = False
                    if self.pipeline and getattr(self.pipeline, 'has_ipadapter', False):
                        print("upload_style_image: Applying to existing pipeline")
                        success = self.pipeline.update_ipadapter_style_image(style_image)
                        if success:
                            pipeline_updated = True
                            print("upload_style_image: Successfully applied to existing pipeline")
                            
                            # Force prompt re-encoding to apply new style image embeddings
                            try:
                                current_prompts = self.pipeline.stream.get_current_prompts()
                                if current_prompts:
                                    print("upload_style_image: Forcing prompt re-encoding to apply new style image")
                                    self.pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                                    print("upload_style_image: Prompt re-encoding completed")
                            except Exception as e:
                                print(f"upload_style_image: Failed to force prompt re-encoding: {e}")
                        else:
                            print("upload_style_image: Failed to apply to existing pipeline")
                    elif self.pipeline:
                        print(f"upload_style_image: Pipeline exists but has_ipadapter={getattr(self.pipeline, 'has_ipadapter', False)}")
                    else:
                        print("upload_style_image: No pipeline exists yet")
                    
                    # Return success
                    message = "Style image uploaded successfully"
                    if pipeline_updated:
                        message += " and applied to active pipeline"
                    else:
                        message += " and will be applied when pipeline starts"
                    
                    return JSONResponse({
                        "status": "success",
                        "message": message
                    })
                    
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to upload style image: {str(e)}")

        @self.app.get("/api/ipadapter/uploaded-style-image")
        async def get_uploaded_style_image():
            """Get the currently uploaded style image"""
            try:
                if not self.uploaded_style_image:
                    raise HTTPException(status_code=404, detail="No style image uploaded")
                
                # Convert PIL image to bytes for streaming
                import io
                img_buffer = io.BytesIO()
                self.uploaded_style_image.save(img_buffer, format='JPEG', quality=95)
                img_buffer.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(img_buffer.read()),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve style image: {str(e)}")

        @self.app.get("/api/default-image")
        async def get_default_image():
            """Get the default image (input.png)"""
            try:
                import os
                default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
                
                if not os.path.exists(default_image_path):
                    raise HTTPException(status_code=404, detail="Default image not found")
                
                # Read and return the default image file
                with open(default_image_path, "rb") as image_file:
                    image_content = image_file.read()
                
                return Response(content=image_content, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
                
            except Exception as e:
                logging.error(f"get_default_image: Failed to retrieve default image: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve default image: {str(e)}")

        @self.app.post("/api/ipadapter/update-scale")
        async def update_ipadapter_scale(request: Request):
            """Update IPAdapter scale/strength in real-time"""
            try:
                data = await request.json()
                scale = data.get("scale")
                
                if scale is None:
                    raise HTTPException(status_code=400, detail="Missing scale parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Check if we're using config mode and have ipadapters configured
                ipadapter_enabled = (self.pipeline.use_config and 
                                    self.pipeline.config and 
                                    'ipadapters' in self.pipeline.config)
                
                if not ipadapter_enabled:
                    raise HTTPException(status_code=400, detail="IPAdapter is not enabled")
                
                # Update IPAdapter scale in the pipeline
                success = self.pipeline.update_ipadapter_scale(float(scale))
                
                if success:
                    return JSONResponse({
                        "status": "success",
                        "message": f"Updated IPAdapter scale to {scale}"
                    })
                else:
                    raise HTTPException(status_code=500, detail="Failed to update scale in pipeline")
                
            except Exception as e:
                logging.error(f"update_ipadapter_scale: Failed to update scale: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update scale: {str(e)}")

        @self.app.post("/api/ipadapter/update-weight-type")
        async def update_ipadapter_weight_type(request: Request):
            """Update IPAdapter weight type in real-time"""
            try:
                data = await request.json()
                weight_type = data.get("weight_type")
                
                if weight_type is None:
                    raise HTTPException(status_code=400, detail="Missing weight_type parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Check if we're using config mode and have ipadapters configured
                ipadapter_enabled = (self.pipeline.use_config and 
                                    self.pipeline.config and 
                                    'ipadapters' in self.pipeline.config)
                
                if not ipadapter_enabled:
                    raise HTTPException(status_code=400, detail="IPAdapter is not enabled")
                
                # Update IPAdapter weight type in the pipeline
                success = self.pipeline.update_ipadapter_weight_type(weight_type)
                
                if success:
                    return JSONResponse({
                        "status": "success",
                        "message": f"Updated IPAdapter weight type to {weight_type}"
                    })
                else:
                    raise HTTPException(status_code=500, detail="Failed to update weight type in pipeline")
                
            except Exception as e:
                logging.error(f"update_ipadapter_weight_type: Failed to update weight type: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update weight type: {str(e)}")

        @self.app.post("/api/params")
        async def update_params(request: Request):
            """Update multiple streaming parameters in a single unified call"""
            try:
                data = await request.json()
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Extract and validate parameters
                params = {}
                updated_params = []
                
                # Handle t_index_list
                if "t_index_list" in data:
                    t_index_list = data["t_index_list"]
                    if not isinstance(t_index_list, list) or not all(isinstance(x, int) for x in t_index_list):
                        raise HTTPException(status_code=400, detail="t_index_list must be a list of integers")
                    params["t_index_list"] = t_index_list
                    updated_params.append("t_index_list")
                
                # Handle guidance_scale
                if "guidance_scale" in data:
                    params["guidance_scale"] = float(data["guidance_scale"])
                    updated_params.append("guidance_scale")
                
                # Handle delta
                if "delta" in data:
                    params["delta"] = float(data["delta"])
                    updated_params.append("delta")
                
                # Handle num_inference_steps
                if "num_inference_steps" in data:
                    params["num_inference_steps"] = int(data["num_inference_steps"])
                    updated_params.append("num_inference_steps")
                
                # Handle seed
                if "seed" in data:
                    params["seed"] = int(data["seed"])
                    updated_params.append("seed")
                
                # Handle resolution (special case - triggers pipeline recreation)
                if "resolution" in data:
                    resolution = data["resolution"]
                    if isinstance(resolution, dict) and "width" in resolution and "height" in resolution:
                        width, height = int(resolution["width"]), int(resolution["height"])
                        self._update_resolution(width, height)
                        updated_params.append("resolution")
                    elif isinstance(resolution, str):
                        # Handle string format like "512x768 (2:3)"
                        resolution_part = resolution.split(' ')[0]  # Get "512x768" part
                        try:
                            width, height = map(int, resolution_part.split('x'))
                            self._update_resolution(width, height)
                            updated_params.append("resolution")
                        except ValueError:
                            raise HTTPException(status_code=400, detail="Invalid resolution format")
                    else:
                        raise HTTPException(status_code=400, detail="Resolution must be {width: int, height: int} or 'widthxheight' string")
                
                # Handle normalization settings
                if "normalize_prompt_weights" in data:
                    params["normalize_prompt_weights"] = bool(data["normalize_prompt_weights"])
                    updated_params.append("normalize_prompt_weights")
                
                if "normalize_seed_weights" in data:
                    params["normalize_seed_weights"] = bool(data["normalize_seed_weights"])
                    updated_params.append("normalize_seed_weights")
                
                if not params and "resolution" not in data:
                    raise HTTPException(status_code=400, detail="No valid parameters provided")
                
                # Update parameters using unified API (excluding resolution which was handled above)
                if params:
                    self.pipeline.update_stream_params(**params)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated parameters: {', '.join(updated_params)}",
                    "updated": updated_params
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"update_params: Failed to update parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update parameters: {str(e)}")

        # Individual parameter update endpoints for input controls
        @self.app.post("/api/update-guidance-scale")
        async def update_guidance_scale(request: Request):
            """Update guidance scale parameter"""
            try:
                data = await request.json()
                guidance_scale = float(data.get("guidance_scale", 1.0))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                self.pipeline.update_stream_params(guidance_scale=guidance_scale)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated guidance_scale to {guidance_scale}",
                    "guidance_scale": guidance_scale
                })
                
            except Exception as e:
                logging.error(f"update_guidance_scale: Failed to update guidance scale: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update guidance scale: {str(e)}")

        @self.app.post("/api/update-delta")
        async def update_delta(request: Request):
            """Update delta parameter"""
            try:
                data = await request.json()
                delta = float(data.get("delta", 0.7))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                self.pipeline.update_stream_params(delta=delta)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated delta to {delta}",
                    "delta": delta
                })
                
            except Exception as e:
                logging.error(f"update_delta: Failed to update delta: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update delta: {str(e)}")

        @self.app.post("/api/update-num-inference-steps")
        async def update_num_inference_steps(request: Request):
            """Update number of inference steps parameter"""
            try:
                data = await request.json()
                num_inference_steps = int(data.get("num_inference_steps", 50))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                self.pipeline.update_stream_params(num_inference_steps=num_inference_steps)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated num_inference_steps to {num_inference_steps}",
                    "num_inference_steps": num_inference_steps
                })
                
            except Exception as e:
                logging.error(f"update_num_inference_steps: Failed to update num_inference_steps: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update num_inference_steps: {str(e)}")

        @self.app.post("/api/update-seed")
        async def update_seed(request: Request):
            """Update seed parameter"""
            try:
                data = await request.json()
                seed = int(data.get("seed", 2))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                self.pipeline.update_stream_params(seed=seed)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated seed to {seed}",
                    "seed": seed
                })
                
            except Exception as e:
                logging.error(f"update_seed: Failed to update seed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update seed: {str(e)}")

        @self.app.post("/api/blending")
        async def update_blending(request: Request):
            """Update prompt and/or seed blending configuration in real-time"""
            try:
                data = await request.json()
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                params = {}
                updated_types = []
                
                # Handle prompt blending
                if "prompt_list" in data:
                    prompt_list = data["prompt_list"]
                    interpolation_method = data.get("prompt_interpolation_method", "slerp")
                    
                    if not isinstance(prompt_list, list):
                        raise HTTPException(status_code=400, detail="prompt_list must be a list")
                    
                    # Validate and convert format
                    prompt_tuples = []
                    for item in prompt_list:
                        if isinstance(item, list) and len(item) == 2:
                            prompt_tuples.append((str(item[0]), float(item[1])))
                        elif isinstance(item, dict) and "prompt" in item and "weight" in item:
                            prompt_tuples.append((str(item["prompt"]), float(item["weight"])))
                        else:
                            raise HTTPException(status_code=400, detail="Each prompt item must be [prompt, weight] or {prompt: str, weight: float}")
                    
                    params["prompt_list"] = prompt_tuples
                    params["prompt_interpolation_method"] = interpolation_method
                    updated_types.append("prompt blending")
                
                # Handle seed blending
                if "seed_list" in data:
                    seed_list = data["seed_list"]
                    interpolation_method = data.get("seed_interpolation_method", "linear")
                    
                    if not isinstance(seed_list, list):
                        raise HTTPException(status_code=400, detail="seed_list must be a list")
                    
                    # Validate and convert format
                    seed_tuples = []
                    for item in seed_list:
                        if isinstance(item, list) and len(item) == 2:
                            seed_tuples.append((int(item[0]), float(item[1])))
                        elif isinstance(item, dict) and "seed" in item and "weight" in item:
                            seed_tuples.append((int(item["seed"]), float(item["weight"])))
                        else:
                            raise HTTPException(status_code=400, detail="Each seed item must be [seed, weight] or {seed: int, weight: float}")
                    
                    params["seed_list"] = seed_tuples
                    params["seed_interpolation_method"] = interpolation_method
                    updated_types.append("seed blending")
                
                if not params:
                    raise HTTPException(status_code=400, detail="No blending parameters provided")
                
                # Update blending using unified API
                self.pipeline.update_stream_params(**params)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated {' and '.join(updated_types)}",
                    "updated": updated_types
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"update_blending: Failed to update blending: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update blending: {str(e)}")

        @self.app.get("/api/fps")
        async def get_fps():
            """Get current FPS"""
            if len(self.fps_counter) > 0:
                avg_frame_time = sum(self.fps_counter) / len(self.fps_counter)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                fps = 0
            
            return JSONResponse({"fps": round(fps, 1)})

        @self.app.get("/api/preprocessors/info")
        async def get_preprocessors_info():
            """Get preprocessor information using metadata from preprocessor classes"""
            try:
                from src.streamdiffusion.preprocessing.processors import list_preprocessors, get_preprocessor
                
                available_preprocessors = list_preprocessors()
                preprocessors_info = {}
                
                for preprocessor_name in available_preprocessors:
                    try:
                        preprocessor_class = get_preprocessor(preprocessor_name).__class__
                        
                        # Get comprehensive metadata from class
                        metadata = preprocessor_class.get_preprocessor_metadata()
                        
                        # Use metadata directly, with the preprocessor name as key
                        preprocessors_info[preprocessor_name] = metadata
                        
                    except Exception as e:
                        logger.warning(f"get_preprocessors_info: Could not extract info for {preprocessor_name}: {e}")
                        # Fallback to basic info if metadata method fails
                        preprocessors_info[preprocessor_name] = {
                            "display_name": preprocessor_name.replace("_", " ").title(),
                            "description": f"Preprocessor for {preprocessor_name}",
                            "parameters": {},
                            "use_cases": []
                        }
                        continue
                
                return JSONResponse({
                    "preprocessors": preprocessors_info,
                    "available": available_preprocessors
                })
                
            except Exception as e:
                logger.error(f"get_preprocessors_info: Error loading preprocessor info: {e}")
                return JSONResponse({
                    "preprocessors": {},
                    "available": [],
                    "error": "Could not load preprocessor information"
                })

        @self.app.post("/api/preprocessors/switch")
        async def switch_preprocessor(request: Request):
            """Switch preprocessor for a specific ControlNet"""
            try:
                data = await request.json()
                controlnet_index = data.get("controlnet_index", 0)
                new_preprocessor = data.get("preprocessor")
                preprocessor_params = data.get("preprocessor_params", {})
                
                logger.info(f"switch_preprocessor: Switching ControlNet {controlnet_index} to {new_preprocessor}")
                
                if not new_preprocessor:
                    raise HTTPException(status_code=400, detail="Missing preprocessor parameter")
                
                # Get ControlNet pipeline using helper
                cn_pipeline = self._get_controlnet_pipeline()
                if not cn_pipeline:
                    raise HTTPException(status_code=400, detail="ControlNet pipeline not found")
                
                if controlnet_index >= len(cn_pipeline.preprocessors):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                # Create new preprocessor instance
                from src.streamdiffusion.preprocessing.processors import get_preprocessor
                new_preprocessor_instance = get_preprocessor(new_preprocessor)

                # Resolve stream object and preprocessor list regardless of module or stream facade
                stream_obj = getattr(cn_pipeline, '_stream', None)
                if stream_obj is None:
                    stream_obj = getattr(self.pipeline, 'stream', None)
                if stream_obj is None:
                    raise HTTPException(status_code=500, detail="Pipeline stream not available")

                preproc_list = getattr(cn_pipeline, 'preprocessors', None)
                if preproc_list is None:
                    preproc_list = getattr(stream_obj, 'preprocessors', None)
                if preproc_list is None:
                    raise HTTPException(status_code=500, detail="ControlNet preprocessors not available")

                # Set system parameters
                system_params = {
                    'device': stream_obj.device,
                    'dtype': stream_obj.dtype,
                    'image_width': stream_obj.width,
                    'image_height': stream_obj.height,
                }
                system_params.update(preprocessor_params)
                new_preprocessor_instance.params.update(system_params)

                # Set pipeline reference for feedback preprocessor
                if hasattr(new_preprocessor_instance, 'set_pipeline_ref'):
                    new_preprocessor_instance.set_pipeline_ref(stream_obj)

                # Replace the preprocessor
                old_preprocessor = preproc_list[controlnet_index]
                preproc_list[controlnet_index] = new_preprocessor_instance
                
                logger.info(f"switch_preprocessor: Successfully switched ControlNet {controlnet_index} from {type(old_preprocessor).__name__ if old_preprocessor else 'None'} to {type(new_preprocessor_instance).__name__}")
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Successfully switched to {new_preprocessor} preprocessor",
                    "controlnet_index": controlnet_index,
                    "preprocessor": new_preprocessor,
                    "parameters": preprocessor_params
                })
                    
            except Exception as e:
                logger.error(f"switch_preprocessor: Failed to switch preprocessor: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to switch preprocessor: {str(e)}")
        
        @self.app.post("/api/preprocessors/update-params")
        async def update_preprocessor_params(request: Request):
            """Update preprocessor parameters for a specific ControlNet"""
            try:
                data = await request.json()
                controlnet_index = data.get("controlnet_index", 0)
                preprocessor_params = data.get("preprocessor_params", {})
                

                
                if not preprocessor_params:
                    raise HTTPException(status_code=400, detail="Missing preprocessor_params parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Fast path: update module preprocessor directly when available
                cn_pipeline = self._get_controlnet_pipeline()
                preproc_list = getattr(cn_pipeline, 'preprocessors', None)
                if preproc_list is None:
                    raise HTTPException(status_code=400, detail="ControlNet preprocessors not available")

                if controlnet_index >= len(preproc_list):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range (max: {len(preproc_list)-1})")

                target_preproc = preproc_list[controlnet_index]
                if target_preproc is None:
                    raise HTTPException(status_code=400, detail="ControlNet preprocessor is not set")

                # Merge params: update both the params map and setattr when attribute exists
                if hasattr(target_preproc, 'params') and isinstance(target_preproc.params, dict):
                    target_preproc.params.update(preprocessor_params)
                for name, value in preprocessor_params.items():
                    if hasattr(target_preproc, name):
                        setattr(target_preproc, name, value)
                
                return JSONResponse({
                    "status": "success",
                    "message": "Successfully updated preprocessor parameters",
                    "controlnet_index": controlnet_index,
                    "updated_parameters": preprocessor_params
                })
                    
            except Exception as e:
                logger.error(f"update_preprocessor_params: Failed to update parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update preprocessor parameters: {str(e)}")

        @self.app.post("/api/blending/update-prompt-weight")
        async def update_prompt_weight(request: Request):
            """Update a specific prompt weight in the current blending configuration"""
            try:
                data = await request.json()
                index = data.get('index')
                weight = data.get('weight')
                
                if index is None or weight is None:
                    raise HTTPException(status_code=400, detail="Missing index or weight parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Get current prompt blending configuration using the same logic as the blending/current endpoint
                current_prompts = None
                try:
                    current_prompts = self.pipeline.stream.get_current_prompts()
                except Exception:
                    pass
                
                # If not available from pipeline, get from uploaded config and normalize
                if not current_prompts:
                    current_prompts = self._normalize_prompt_config(self.uploaded_controlnet_config)
                    
                if current_prompts and index < len(current_prompts):
                    # Create updated prompt list with new weight
                    updated_prompts = list(current_prompts)  # Make a copy
                    updated_prompts[index] = (updated_prompts[index][0], float(weight))
                    
                    # Use the same update method as the main blending endpoint
                    params = {
                        "prompt_list": updated_prompts,
                        "prompt_interpolation_method": "slerp"  # Default method
                    }
                    
                    # Apply the update using the working method
                    result = self.pipeline.update_stream_params(**params)
                    

                    return JSONResponse({
                        "status": "success",
                        "message": f"Successfully updated prompt {index} weight",
                        "index": index,
                        "weight": weight
                    })
                else:
                    raise HTTPException(status_code=400, detail=f"Prompt index {index} out of range or no prompts available")
                    
            except Exception as e:
                logger.error(f"update_prompt_weight: Failed to update prompt weight: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update prompt weight: {str(e)}")

        @self.app.post("/api/blending/update-seed-weight") 
        async def update_seed_weight(request: Request):
            """Update a specific seed weight in the current blending configuration"""
            try:
                data = await request.json()
                index = data.get('index')
                weight = data.get('weight')
                
                if index is None or weight is None:
                    raise HTTPException(status_code=400, detail="Missing index or weight parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Get current seed blending configuration using the same logic as the blending/current endpoint
                current_seeds = None
                try:
                    current_seeds = self.pipeline.stream.get_current_seeds()
                except Exception:
                    pass
                
                # If not available from pipeline, get from uploaded config and normalize
                if not current_seeds:
                    current_seeds = self._normalize_seed_config(self.uploaded_controlnet_config)
                    
                if current_seeds and index < len(current_seeds):
                    # Create updated seed list with new weight
                    updated_seeds = list(current_seeds)  # Make a copy
                    updated_seeds[index] = (updated_seeds[index][0], float(weight))
                    
                    # Use the same update method as the main blending endpoint
                    params = {
                        "seed_list": updated_seeds,
                        "seed_interpolation_method": "linear"  # Default method
                    }
                    
                    # Apply the update using the working method
                    result = self.pipeline.update_stream_params(**params)
                    

                    return JSONResponse({
                        "status": "success",
                        "message": f"Successfully updated seed {index} weight",
                        "index": index,
                        "weight": weight
                    })
                else:
                    raise HTTPException(status_code=400, detail=f"Seed index {index} out of range or no seeds available")
                    
            except Exception as e:
                logger.error(f"update_seed_weight: Failed to update seed weight: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update seed weight: {str(e)}")

        @self.app.get("/api/preprocessors/current-params/{controlnet_index}")
        async def get_current_preprocessor_params(controlnet_index: int):
            """Get current parameter values for a specific ControlNet preprocessor"""
            try:
                # Get ControlNet pipeline using helper
                cn_pipeline = self._get_controlnet_pipeline()
                if not cn_pipeline:
                    raise HTTPException(status_code=400, detail="ControlNet pipeline not found")
                
                # Module-aware: allow accessing module's preprocessors list
                preprocessors = getattr(cn_pipeline, 'preprocessors', None)
                if preprocessors is None:
                    raise HTTPException(status_code=400, detail="ControlNet preprocessors not available")
                if controlnet_index >= len(preprocessors):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                current_preprocessor = preprocessors[controlnet_index]
                if not current_preprocessor:
                    return JSONResponse({
                        "preprocessor": None,
                        "parameters": {}
                    })
                
                # Get user-configurable parameters metadata
                metadata = current_preprocessor.__class__.get_preprocessor_metadata()
                user_param_meta = metadata.get("parameters", {})
                
                # Extract current values, using defaults if not set
                current_values = {}
                for param_name, param_meta in user_param_meta.items():
                    if hasattr(current_preprocessor, 'params') and param_name in current_preprocessor.params:
                        current_values[param_name] = current_preprocessor.params[param_name]
                    else:
                        current_values[param_name] = param_meta.get("default")
                
                return JSONResponse({
                    "preprocessor": current_preprocessor.__class__.__name__.replace("Preprocessor", "").lower(),
                    "parameters": current_values
                })
                    
            except Exception as e:
                logger.error(f"get_current_preprocessor_params: Failed to get current parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get current preprocessor parameters: {str(e)}")

        # Only mount static files if not in API-only mode
        if not self.args.api_only:
            if not os.path.exists("public"):
                os.makedirs("public")
            self.app.mount(
                "/", StaticFiles(directory="./frontend/public", html=True), name="public"
            )
        else:
            # In API-only mode, add a simple root endpoint for health check
            @self.app.get("/")
            async def api_root():
                return JSONResponse({
                    "message": "StreamDiffusion API Server", 
                    "mode": "api-only",
                    "frontend": "Run separately with 'npm run dev' in ./frontend/"
                })

        # Input control management endpoints
        @self.app.post("/api/input-control/add")
        async def add_input_control(request: Request):
            """Add a new input control"""
            try:
                data = await request.json()
                
                input_id = data.get("input_id")
                input_type = data.get("input_type")
                parameter_name = data.get("parameter_name")
                min_value = data.get("min_value", 0.0)
                max_value = data.get("max_value", 1.0)
                
                if not all([input_id, input_type, parameter_name]):
                    raise HTTPException(status_code=400, detail="Missing required parameters: input_id, input_type, parameter_name")
                
                # Handle different input types
                if input_type == "gamepad":
                    # Backend gamepad control
                    gamepad_index = data.get("gamepad_index", 0)
                    axis_index = data.get("axis_index", 0)
                    deadzone = data.get("deadzone", 0.1)
                    
                    gamepad_control = GamepadInput(
                        parameter_name=parameter_name,
                        min_value=min_value,
                        max_value=max_value,
                        gamepad_index=gamepad_index,
                        axis_index=axis_index,
                        deadzone=deadzone
                    )
                    
                    self.input_manager.add_input(input_id, gamepad_control)
                    logger.info(f"add_input_control: Added gamepad control for parameter {parameter_name}")
                    
                elif input_type == "microphone":
                    # Frontend-based control
                    raise HTTPException(status_code=400, detail="Microphone inputs are managed in the frontend")
                elif input_type == "hand_tracking":
                    # Frontend-based control
                    raise HTTPException(status_code=400, detail="Hand tracking inputs are managed in the frontend")
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported input type: {input_type}")
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Added {input_type} input control for {parameter_name}"
                })
                
            except Exception as e:
                logging.error(f"add_input_control: Failed to add input control: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add input control: {str(e)}")

        @self.app.post("/api/input-control/start/{input_id}")
        async def start_input_control(input_id: str):
            """Start a specific input control"""
            try:
                await self.input_manager.start_input(input_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Started input control {input_id}"
                })
            except Exception as e:
                logging.error(f"start_input_control: Failed to start input control {input_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start input control: {str(e)}")

        @self.app.post("/api/input-control/stop/{input_id}")
        async def stop_input_control(input_id: str):
            """Stop a specific input control"""
            try:
                await self.input_manager.stop_input(input_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Stopped input control {input_id}"
                })
            except Exception as e:
                logging.error(f"stop_input_control: Failed to stop input control {input_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to stop input control: {str(e)}")

        @self.app.delete("/api/input-control/{input_id}")
        async def remove_input_control(input_id: str):
            """Remove an input control"""
            try:
                self.input_manager.remove_input(input_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Removed input control {input_id}"
                })
            except Exception as e:
                logging.error(f"remove_input_control: Failed to remove input control {input_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to remove input control: {str(e)}")

        @self.app.get("/api/input-control/status")
        async def get_input_control_status():
            """Get status of all input controls"""
            try:
                status = self.input_manager.get_input_status()
                return JSONResponse({
                    "status": "success",
                    "input_controls": status
                })
            except Exception as e:
                logging.error(f"get_input_control_status: Failed to get input control status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get input control status: {str(e)}")

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
                    normalized = []
                    for item in prompt_list:
                        if isinstance(item, list) and len(item) == 2:
                            normalized.append([str(item[0]), float(item[1])])
                        elif isinstance(item, tuple) and len(item) == 2:
                            normalized.append([str(item[0]), float(item[1])])
                    if normalized:
                        return normalized
                        
            # Handle direct list format: prompt_blending: [["text", weight], ...]
            elif isinstance(prompt_blending, list) and len(prompt_blending) > 0:
                normalized = []
                for item in prompt_blending:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                if normalized:
                    return normalized
        
        # Fall back to single prompt, convert to list format
        if 'prompt' in config_data:
            prompt = config_data['prompt']
            if isinstance(prompt, str) and prompt.strip():
                return [[prompt, 1.0]]  # Convert single prompt to list with weight 1.0
            elif isinstance(prompt, list) and len(prompt) > 0:
                # Handle case where prompt is already a list (but not in prompt_blending key)
                normalized = []
                for item in prompt:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, str):
                        normalized.append([item, 1.0])
                if normalized:
                    return normalized
        
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
                    normalized = []
                    for item in seed_list:
                        if isinstance(item, list) and len(item) == 2:
                            normalized.append([int(item[0]), float(item[1])])
                        elif isinstance(item, tuple) and len(item) == 2:
                            normalized.append([int(item[0]), float(item[1])])
                    if normalized:
                        return normalized
                        
            # Handle direct list format: seed_blending: [[seed, weight], ...]
            elif isinstance(seed_blending, list) and len(seed_blending) > 0:
                normalized = []
                for item in seed_blending:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                if normalized:
                    return normalized
        
        # Fall back to single seed, convert to list format
        if 'seed' in config_data:
            seed = config_data['seed']
            if isinstance(seed, int):
                return [[seed, 1.0]]  # Convert single seed to list with weight 1.0
            elif isinstance(seed, list) and len(seed) > 0:
                # Handle case where seed is already a list (but not in seed_blending key)
                normalized = []
                for item in seed:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                    elif isinstance(item, int):
                        normalized.append([item, 1.0])
                if normalized:
                    return normalized
        
        return None

    def _create_default_pipeline(self):
        """Create the default pipeline (standard mode)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        pipeline = Pipeline(self.args, device, torch_dtype, width=self.new_width, height=self.new_height)
        
        # Initialize with default prompt blending (single prompt with weight 1.0)
        default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        pipeline.stream.update_prompt([(default_prompt, 1.0)], prompt_interpolation_method="slerp")
        
        return pipeline

    def _create_pipeline_with_config(self, controlnet_config_path=None):
        """Create a new pipeline with optional ControlNet configuration"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        
        # Use runtime config if available (includes YAML + runtime additions), otherwise fallback to uploaded config
        if controlnet_config_path:
            new_args = self.args._replace(controlnet_config=controlnet_config_path)
        elif self.runtime_controlnet_config:
            # Use runtime config (includes YAML + runtime additions/removals)
            temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.runtime_controlnet_config, temp_config_path, default_flow_style=False)
            temp_config_path.close()
            
            # Merge config values into args, respecting config overrides
            config_acceleration = self.runtime_controlnet_config.get('acceleration', self.args.acceleration)
            new_args = self.args._replace(
                controlnet_config=temp_config_path.name,
                acceleration=config_acceleration
            )
        elif self.uploaded_controlnet_config:
            # Fallback to original YAML config if no runtime modifications exist
            temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.uploaded_controlnet_config, temp_config_path, default_flow_style=False)
            temp_config_path.close()
            
            # Merge YAML config values into args, respecting config overrides
            config_acceleration = self.uploaded_controlnet_config.get('acceleration', self.args.acceleration)
            new_args = self.args._replace(
                controlnet_config=temp_config_path.name,
                acceleration=config_acceleration
            )
        else:
            new_args = self.args
        
        new_pipeline = Pipeline(new_args, device, torch_dtype, width=self.new_width, height=self.new_height)
        
        # Initialize prompt blending from config (use runtime config if available)
        config_for_prompts = self.runtime_controlnet_config if self.runtime_controlnet_config else self.uploaded_controlnet_config
        normalized_prompt_config = self._normalize_prompt_config(config_for_prompts)
        if normalized_prompt_config:
            # Convert to tuple format and set up prompt blending
            prompt_tuples = [(item[0], item[1]) for item in normalized_prompt_config]
            new_pipeline.stream.update_prompt(prompt_tuples, prompt_interpolation_method="slerp")
        else:
            # Fallback to default single prompt
            default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
            new_pipeline.stream.update_prompt([(default_prompt, 1.0)], prompt_interpolation_method="slerp")
        
        # Apply style image (uploaded or default) if pipeline has IPAdapter
        has_ipadapter = getattr(new_pipeline, 'has_ipadapter', False)
        print(f"_create_pipeline_with_config: Pipeline has_ipadapter: {has_ipadapter}")
        
        if has_ipadapter:
            style_image = None
            style_source = ""
            
            if self.uploaded_style_image:
                style_image = self.uploaded_style_image
                style_source = "uploaded"
                print("_create_pipeline_with_config: Using uploaded style image")
            else:
                # Try to load default style image
                print("_create_pipeline_with_config: No uploaded style image, trying to load default")
                style_image = self._load_default_style_image()
                if style_image:
                    style_source = "default"
                    print("_create_pipeline_with_config: Default style image loaded successfully")
                else:
                    print("_create_pipeline_with_config: Failed to load default style image")
            
            if style_image:
                print(f"_create_pipeline_with_config: Applying {style_source} style image to new pipeline")
                success = new_pipeline.update_ipadapter_style_image(style_image)
                if success:
                    print(f"_create_pipeline_with_config: {style_source.capitalize()} style image applied successfully")
                    
                    # Force prompt re-encoding to apply style image embeddings
                    try:
                        current_prompts = new_pipeline.stream.get_current_prompts()
                        if current_prompts:
                            print("_create_pipeline_with_config: Forcing prompt re-encoding to apply style image")
                            new_pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                            print("_create_pipeline_with_config: Prompt re-encoding completed")
                    except Exception as e:
                        print(f"_create_pipeline_with_config: Failed to force prompt re-encoding: {e}")
                else:
                    print(f"_create_pipeline_with_config: Failed to apply {style_source} style image")
            else:
                print("_create_pipeline_with_config: No style image available (neither uploaded nor default)")
        else:
            print("_create_pipeline_with_config: Pipeline does not have IPAdapter enabled")
        
        # Clean up temp file if created
        if self.uploaded_controlnet_config and not controlnet_config_path:
            try:
                os.unlink(new_args.controlnet_config)
            except:
                pass
        
        return new_pipeline

    def _get_controlnet_info(self):
        """Get ControlNet information from uploaded config or active pipeline"""
        controlnet_info = {
            "enabled": False,
            "config_loaded": False,
            "controlnets": []
        }
        
        # Check runtime config first (includes YAML + runtime additions/removals)
        if self.runtime_controlnet_config:
            controlnet_info["enabled"] = True
            controlnet_info["config_loaded"] = True
            if 'controlnets' in self.runtime_controlnet_config:
                for i, cn_config in enumerate(self.runtime_controlnet_config['controlnets']):
                    controlnet_info["controlnets"].append({
                        "index": i,
                        "name": cn_config['model_id'].split('/')[-1],
                        "preprocessor": cn_config['preprocessor'],
                        "strength": cn_config['conditioning_scale']
                    })
        # Fall back to uploaded YAML config if no runtime config exists
        elif self.uploaded_controlnet_config:
            controlnet_info["enabled"] = True
            controlnet_info["config_loaded"] = True
            if 'controlnets' in self.uploaded_controlnet_config:
                for i, cn_config in enumerate(self.uploaded_controlnet_config['controlnets']):
                    controlnet_info["controlnets"].append({
                        "index": i,
                        "name": cn_config['model_id'].split('/')[-1],
                        "preprocessor": cn_config['preprocessor'],
                        "strength": cn_config['conditioning_scale']
                    })
        # Otherwise check active pipeline
        elif self.pipeline and self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config:
            controlnet_info["enabled"] = True
            controlnet_info["config_loaded"] = True
            if 'controlnets' in self.pipeline.config:
                for i, cn_config in enumerate(self.pipeline.config['controlnets']):
                    controlnet_info["controlnets"].append({
                        "index": i,
                        "name": cn_config['model_id'].split('/')[-1],
                        "preprocessor": cn_config['preprocessor'],
                        "strength": cn_config['conditioning_scale']
                    })
        
        return controlnet_info

    def _load_default_style_image(self):
        """Load the default style image for IPAdapter"""
        try:
            import os
            from PIL import Image
            
            default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
            
            if os.path.exists(default_image_path):
                print(f"_load_default_style_image: Loading default style image (input.png) from {default_image_path}")
                return Image.open(default_image_path).convert("RGB")
            else:
                print(f"_load_default_style_image: Default style image not found at {default_image_path}")
                return None
                
        except Exception as e:
            print(f"_load_default_style_image: Failed to load default style image: {e}")
            return None

    def _get_ipadapter_info(self):
        """Get IPAdapter information from uploaded config or active pipeline"""
        ipadapter_info = {
            "enabled": False,
            "config_loaded": False,
            "scale": 1.0,
            "model_path": None,
            "style_image_set": False,
            "style_image_path": None
        }
        
        # Check uploaded config first
        if self.uploaded_controlnet_config:
            if 'ipadapters' in self.uploaded_controlnet_config and len(self.uploaded_controlnet_config['ipadapters']) > 0:
                ipadapter_info["enabled"] = True
                ipadapter_info["config_loaded"] = True
                
                # Get info from first IPAdapter config
                first_ipadapter = self.uploaded_controlnet_config['ipadapters'][0]
                ipadapter_info["scale"] = first_ipadapter.get('scale', DEFAULT_SETTINGS.get('ipadapter_scale', 1.0))
                ipadapter_info["model_path"] = first_ipadapter.get('ipadapter_model_path')
                
                # Check for style image - prioritize uploaded style image over config style image over default
                if self.uploaded_style_image:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = "/api/ipadapter/uploaded-style-image"  # URL to fetch uploaded image
                elif 'style_image' in first_ipadapter:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = first_ipadapter['style_image']
                else:
                    # Check if default image exists
                    import os
                    default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
                    if os.path.exists(default_image_path):
                        ipadapter_info["style_image_set"] = True
                        ipadapter_info["style_image_path"] = "/api/default-image"
                    
        # Otherwise check active pipeline
        elif self.pipeline and self.pipeline.use_config and self.pipeline.config and 'ipadapters' in self.pipeline.config:
            if len(self.pipeline.config['ipadapters']) > 0:
                ipadapter_info["enabled"] = True
                ipadapter_info["config_loaded"] = True
                
                # Get info from first IPAdapter config
                first_ipadapter = self.pipeline.config['ipadapters'][0]
                ipadapter_info["scale"] = first_ipadapter.get('scale', DEFAULT_SETTINGS.get('ipadapter_scale', 1.0))
                ipadapter_info["model_path"] = first_ipadapter.get('ipadapter_model_path')
                
                # Check for style image - prioritize uploaded style image over config style image over default
                if self.uploaded_style_image:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = "/api/ipadapter/uploaded-style-image"  # URL to fetch uploaded image
                elif 'style_image' in first_ipadapter:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = first_ipadapter['style_image']
                else:
                    # Check if default image exists
                    import os
                    default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
                    if os.path.exists(default_image_path):
                        ipadapter_info["style_image_set"] = True
                        ipadapter_info["style_image_path"] = "/api/default-image"
                    
            # Try to get current scale from active pipeline if available
            try:
                if hasattr(self.pipeline, 'get_ipadapter_info'):
                    pipeline_info = self.pipeline.get_ipadapter_info()
                    if pipeline_info.get("enabled"):
                        ipadapter_info["scale"] = pipeline_info.get("scale", ipadapter_info["scale"])
            except:
                pass
        
        return ipadapter_info

    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate and return aspect ratio as a string"""
        import math
        
        # Find GCD to simplify the ratio
        gcd = math.gcd(width, height)
        simplified_width = width // gcd
        simplified_height = height // gcd
        
        return f"{simplified_width}:{simplified_height}"

    def _cleanup_pipeline(self, pipeline):
        """Properly cleanup a pipeline and free VRAM using StreamDiffusion's built-in cleanup"""
        if pipeline is None:
            return
            
        try:
            logger.info("Starting pipeline cleanup...")
            
            # Use StreamDiffusion's built-in cleanup method which properly handles:
            # - TensorRT engine cleanup
            # - ControlNet engine cleanup  
            # - Multiple garbage collection cycles
            # - CUDA cache clearing
            # - Memory tracking
            if hasattr(pipeline, 'stream') and pipeline.stream and hasattr(pipeline.stream, 'cleanup_gpu_memory'):
                pipeline.stream.cleanup_gpu_memory()
                logger.info("Pipeline cleanup completed using StreamDiffusion cleanup")
            else:
                # Fallback cleanup if the method doesn't exist
                logger.warning("StreamDiffusion cleanup method not found, using fallback cleanup")
                if hasattr(pipeline, 'stream') and pipeline.stream:
                    del pipeline.stream
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
            # Still try to clear CUDA cache even if cleanup fails
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _update_resolution(self, width: int, height: int) -> None:
        """Create a new pipeline with the specified resolution and replace the old one."""
        logger.info(f"Creating new pipeline with resolution {width}x{height}")
        
        # Store current pipeline state before cleanup
        current_prompt = getattr(self.pipeline, 'prompt', '') if self.pipeline else ''
        current_negative_prompt = getattr(self.pipeline, 'negative_prompt', '') if self.pipeline else ''
        current_guidance_scale = getattr(self.pipeline, 'guidance_scale', 1.2) if self.pipeline else 1.2
        current_num_inference_steps = getattr(self.pipeline, 'num_inference_steps', 50) if self.pipeline else 50
        
        # Store reference to old pipeline for cleanup
        old_pipeline = self.pipeline
        
        # Clear current pipeline reference before cleanup to prevent any access during cleanup
        self.pipeline = None
        
        # Cleanup old pipeline and free VRAM
        if old_pipeline:
            self._cleanup_pipeline(old_pipeline)
            old_pipeline = None
        
        # Update current resolution 
        self.new_width = width
        self.new_height = height
        
        # Create new pipeline with new resolution
        try:
            if self.uploaded_controlnet_config:
                new_pipeline = self._create_pipeline_with_config()
            else:
                new_pipeline = self._create_default_pipeline()
            
            # Apply style image (uploaded or default) if pipeline has IPAdapter
            has_ipadapter = getattr(new_pipeline, 'has_ipadapter', False)
            print(f"_update_resolution: Pipeline has_ipadapter: {has_ipadapter}")
            
            if has_ipadapter:
                style_image = None
                style_source = ""
                
                if self.uploaded_style_image:
                    style_image = self.uploaded_style_image
                    style_source = "uploaded"
                    print("_update_resolution: Using uploaded style image")
                else:
                    # Try to load default style image
                    print("_update_resolution: No uploaded style image, trying to load default")
                    style_image = self._load_default_style_image()
                    if style_image:
                        style_source = "default"
                        print("_update_resolution: Default style image loaded successfully")
                    else:
                        print("_update_resolution: Failed to load default style image")
                
                if style_image:
                    print(f"_update_resolution: Applying {style_source} style image to new pipeline")
                    success = new_pipeline.update_ipadapter_style_image(style_image)
                    if success:
                        print(f"_update_resolution: {style_source.capitalize()} style image applied successfully")
                        
                        # Force prompt re-encoding to apply style image embeddings
                        try:
                            current_prompts = new_pipeline.stream.get_current_prompts()
                            if current_prompts:
                                print("_update_resolution: Forcing prompt re-encoding to apply style image")
                                new_pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                                print("_update_resolution: Prompt re-encoding completed")
                        except Exception as e:
                            print(f"_update_resolution: Failed to force prompt re-encoding: {e}")
                    else:
                        print(f"_update_resolution: Failed to apply {style_source} style image")
                else:
                    print("_update_resolution: No style image available (neither uploaded nor default)")
            else:
                print("_update_resolution: Pipeline does not have IPAdapter enabled")
            
            # Set the new pipeline
            self.pipeline = new_pipeline
            
            # Restore pipeline state
            if current_prompt:
                self.pipeline.stream.prepare(
                    prompt=current_prompt,
                    negative_prompt=current_negative_prompt,
                    guidance_scale=current_guidance_scale,
                    num_inference_steps=current_num_inference_steps
                )
                # Also update the pipeline's stored values
                self.pipeline.prompt = current_prompt
                self.pipeline.negative_prompt = current_negative_prompt
                self.pipeline.guidance_scale = current_guidance_scale
                self.pipeline.num_inference_steps = current_num_inference_steps
                self.pipeline.last_prompt = current_prompt
            
            logger.info(f"Pipeline updated successfully to {width}x{height}")
            
        except Exception as e:
            logger.error(f"Failed to create new pipeline: {e}")
            # Make sure we don't leave the system in a broken state
            self.pipeline = None
            raise

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
