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
from util import pil_to_frame, bytes_to_pil
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120

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
        self.config_needs_reload = False  # Track when pipeline needs recreation
        # Store current resolution for pipeline recreation
        self.new_width = 512
        self.new_height = 512
        # Store uploaded style image persistently
        self.uploaded_style_image = None
        self.init_app()

    def cleanup(self):
        """Cleanup resources when app is shutting down"""
        logger.info("App cleanup: Starting application cleanup...")
        if self.pipeline:
            self._cleanup_pipeline(self.pipeline)
            self.pipeline = None
        logger.info("App cleanup: Completed application cleanup")


    


    def _get_controlnet_pipeline(self):
        """Get the ControlNet pipeline from the main pipeline structure"""
        if not self.pipeline:
            return None
            
        stream = self.pipeline.stream
        
        # Check if stream is ControlNet pipeline directly
        if hasattr(stream, 'preprocessors'):
            return stream
            
        # Check if stream has nested stream (IPAdapter wrapper)
        if hasattr(stream, 'stream') and hasattr(stream.stream, 'preprocessors'):
            return stream.stream
            
        return None

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
                            params.image = bytes_to_pil(image_data)
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
                            image = self.pipeline.predict(params)
                            if image is None:
                                continue
                            frame = pil_to_frame(image)
                        except Exception as e:
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
            
            # Include config prompt if available
            config_prompt = None
            if self.uploaded_controlnet_config and 'prompt' in self.uploaded_controlnet_config:
                config_prompt = self.uploaded_controlnet_config['prompt']
            
            # Get current t_index_list from pipeline or config
            current_t_index_list = None
            if self.pipeline and hasattr(self.pipeline.stream, 't_list'):
                current_t_index_list = self.pipeline.stream.t_list
            elif self.uploaded_controlnet_config and 't_index_list' in self.uploaded_controlnet_config:
                current_t_index_list = self.uploaded_controlnet_config['t_index_list']
            else:
                # Default values
                current_t_index_list = [35, 45]
            
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
            current_guidance_scale = 1.1
            current_delta = 0.7
            current_num_inference_steps = 50
            current_seed = 2
            
            if self.pipeline:
                current_guidance_scale = getattr(self.pipeline.stream, 'guidance_scale', 1.1)
                current_delta = getattr(self.pipeline.stream, 'delta', 0.7)
                current_num_inference_steps = getattr(self.pipeline.stream, 'num_inference_steps', 50)
                # Get seed from generator if available
                if hasattr(self.pipeline.stream, 'generator') and self.pipeline.stream.generator is not None:
                    # We can't directly get seed from generator, but we'll use the configured value
                    current_seed = getattr(self.pipeline.stream, 'current_seed', 2)
            elif self.uploaded_controlnet_config:
                current_guidance_scale = self.uploaded_controlnet_config.get('guidance_scale', 1.1)
                current_delta = self.uploaded_controlnet_config.get('delta', 0.7)
                current_num_inference_steps = self.uploaded_controlnet_config.get('num_inference_steps', 50)
                current_seed = self.uploaded_controlnet_config.get('seed', 2)
            
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
                
                # Store config and mark for reload
                self.uploaded_controlnet_config = config_data
                self.config_needs_reload = True  # Mark that pipeline needs recreation
                
                # Log IPAdapter configuration for debugging
                if 'ipadapters' in config_data:
                    print(f"upload_controlnet_config: Found IPAdapter configuration: {config_data['ipadapters']}")
                
                # Get config prompt if available
                config_prompt = config_data.get('prompt', None)
                
                # Get t_index_list from config if available
                t_index_list = config_data.get('t_index_list', [35, 45])
                
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
                print(f"upload_controlnet_config: Returning IPAdapter info to frontend: {response_ipadapter_info}")
                
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
                        logger.debug(f"get_current_blending_config: Retrieved current prompts from pipeline: {current_prompts}")
                        if current_prompts and len(current_prompts) > 0:
                            prompt_blending_config = current_prompts
                            logger.debug(f"get_current_blending_config: Using pipeline prompts: {prompt_blending_config}")
                    except Exception as e:
                        logger.debug(f"get_current_blending_config: Error getting current prompts: {e}")
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
                    current_normalize = self.pipeline.stream.get_normalize_weights()
                    normalize_prompt_weights = current_normalize
                    normalize_seed_weights = current_normalize
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
                
                # Update ControlNet strength in the pipeline
                if hasattr(self.pipeline.stream, 'update_controlnet_scale'):
                    self.pipeline.stream.update_controlnet_scale(controlnet_index, float(strength))
                    
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated ControlNet {controlnet_index} strength to {strength}"
                })
                
            except Exception as e:
                logging.error(f"update_controlnet_strength: Failed to update strength: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update strength: {str(e)}")

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

        @self.app.post("/api/update-t-index-list")
        async def update_t_index_list(request: Request):
            """Update t_index_list values in real-time"""
            try:
                data = await request.json()
                t_index_list = data.get("t_index_list")
                
                if t_index_list is None:
                    raise HTTPException(status_code=400, detail="Missing t_index_list parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Validate that the list contains integers
                if not all(isinstance(x, int) for x in t_index_list):
                    raise HTTPException(status_code=400, detail="All t_index values must be integers")
                
                # Update t_index_list in the pipeline
                self.pipeline.stream.update_stream_params(t_index_list=t_index_list)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated t_index_list to {t_index_list}"
                })
                
            except Exception as e:
                logging.error(f"update_t_index_list: Failed to update t_index_list: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update t_index_list: {str(e)}")

        @self.app.post("/api/update-guidance-scale")
        async def update_guidance_scale(request: Request):
            """Update guidance_scale value in real-time"""
            try:
                data = await request.json()
                guidance_scale = data.get("guidance_scale")
                
                if guidance_scale is None:
                    raise HTTPException(status_code=400, detail="Missing guidance_scale parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Update guidance_scale in the pipeline
                self.pipeline.stream.update_stream_params(guidance_scale=float(guidance_scale))
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated guidance_scale to {guidance_scale}"
                })
                
            except Exception as e:
                logging.error(f"update_guidance_scale: Failed to update guidance_scale: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update guidance_scale: {str(e)}")

        @self.app.post("/api/update-delta")
        async def update_delta(request: Request):
            """Update delta value in real-time"""
            try:
                data = await request.json()
                delta = data.get("delta")
                
                if delta is None:
                    raise HTTPException(status_code=400, detail="Missing delta parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Update delta in the pipeline
                self.pipeline.stream.update_stream_params(delta=float(delta))
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated delta to {delta}"
                })
                
            except Exception as e:
                logging.error(f"update_delta: Failed to update delta: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update delta: {str(e)}")

        @self.app.post("/api/update-num-inference-steps")
        async def update_num_inference_steps(request: Request):
            """Update num_inference_steps value in real-time"""
            try:
                data = await request.json()
                num_inference_steps = data.get("num_inference_steps")
                
                if num_inference_steps is None:
                    raise HTTPException(status_code=400, detail="Missing num_inference_steps parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Update num_inference_steps in the pipeline
                self.pipeline.stream.update_stream_params(num_inference_steps=int(num_inference_steps))
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated num_inference_steps to {num_inference_steps}"
                })
                
            except Exception as e:
                logging.error(f"update_num_inference_steps: Failed to update num_inference_steps: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update num_inference_steps: {str(e)}")

        @self.app.post("/api/update-seed")
        async def update_seed(request: Request):
            """Update seed value in real-time"""
            try:
                data = await request.json()
                seed = data.get("seed")
                
                if seed is None:
                    raise HTTPException(status_code=400, detail="Missing seed parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Update seed in the pipeline
                self.pipeline.stream.update_stream_params(seed=int(seed))
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated seed to {seed}"
                })
                
            except Exception as e:
                logging.error(f"update_seed: Failed to update seed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update seed: {str(e)}")

        @self.app.post("/api/update-resolution")
        async def update_resolution(request: Request):
            """Update resolution (width x height) by creating a new pipeline"""
            try:
                data = await request.json()
                resolution_str = data.get('resolution')
                
                if not resolution_str:
                    raise HTTPException(status_code=400, detail="Resolution parameter is required")
                
                # Parse resolution string (e.g., "512x768 (2:3)" -> width=512, height=768)
                resolution_part = resolution_str.split(' ')[0]  # Get "512x768" part
                try:
                    width, height = map(int, resolution_part.split('x'))
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid resolution format. Use 'widthxheight' (e.g., '512x768')")
                
                # Validate resolution
                if width % 64 != 0 or height % 64 != 0:
                    raise HTTPException(status_code=400, detail="Resolution must be multiples of 64")
                
                if not (384 <= width <= 1024) or not (384 <= height <= 1024):
                    raise HTTPException(status_code=400, detail="Resolution must be between 384 and 1024")
                
                # Check if resolution actually changed
                if width == self.new_width and height == self.new_height:
                    raise HTTPException(status_code=400, detail="Resolution unchanged")
                
                logger.info(f"API: Updating resolution from {self.new_width}x{self.new_height} to {width}x{height}")
                
                # Create new pipeline with new resolution
                try:
                    self._update_resolution(width, height)
                    
                    logger.info(f"API: Resolution update successful: {width}x{height}")
                    return JSONResponse({
                        "status": "success",
                        "message": f"Resolution updated to {width}x{height}",
                    })
                    
                except Exception as update_error:
                    logger.error(f"API: Resolution update failed: {update_error}")
                    raise HTTPException(status_code=500, detail=f"Failed to update resolution: {update_error}")

            except Exception as e:
                logger.error(f"API: Resolution update error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update resolution: {e}")

        @self.app.post("/api/update-normalize-prompt-weights")
        async def update_normalize_prompt_weights(request: Request):
            """Update normalize weights flag for prompt blending in real-time"""
            try:
                data = await request.json()
                normalize = data.get("normalize")
                
                if normalize is None:
                    raise HTTPException(status_code=400, detail="Missing normalize parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Update normalize weights setting for prompt blending
                # For now, use the existing single flag (this can be enhanced later with separate flags)
                self.pipeline.stream.set_normalize_weights(bool(normalize))
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated prompt weight normalization to {normalize}"
                })
                
            except Exception as e:
                logging.error(f"update_normalize_prompt_weights: Failed to update normalize prompt weights: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update normalize prompt weights: {str(e)}")

        @self.app.post("/api/update-normalize-seed-weights")
        async def update_normalize_seed_weights(request: Request):
            """Update normalize weights flag for seed blending in real-time"""
            try:
                data = await request.json()
                normalize = data.get("normalize")
                
                if normalize is None:
                    raise HTTPException(status_code=400, detail="Missing normalize parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Update normalize weights setting for seed blending
                # For now, use the existing single flag (this can be enhanced later with separate flags)
                self.pipeline.stream.set_normalize_weights(bool(normalize))
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated seed weight normalization to {normalize}"
                })
                
            except Exception as e:
                logging.error(f"update_normalize_seed_weights: Failed to update normalize seed weights: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update normalize seed weights: {str(e)}")

        @self.app.post("/api/prompt-blending/update")
        async def update_prompt_blending(request: Request):
            """Update prompt blending configuration in real-time"""
            try:
                data = await request.json()
                prompt_list = data.get("prompt_list")
                interpolation_method = data.get("interpolation_method", "slerp")
                
                logger.debug(f"update_prompt_blending: Received request with {len(prompt_list) if prompt_list else 0} prompts")
                logger.debug(f"update_prompt_blending: prompt_list = {prompt_list}")
                logger.debug(f"update_prompt_blending: interpolation_method = {interpolation_method}")
                
                if prompt_list is None:
                    raise HTTPException(status_code=400, detail="Missing prompt_list parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Validate prompt_list structure
                if not isinstance(prompt_list, list):
                    raise HTTPException(status_code=400, detail="prompt_list must be a list")
                
                for item in prompt_list:
                    if not isinstance(item, list) or len(item) != 2:
                        raise HTTPException(status_code=400, detail="Each prompt_list item must be [prompt, weight]")
                    if not isinstance(item[0], str) or not isinstance(item[1], (int, float)):
                        raise HTTPException(status_code=400, detail="Each prompt_list item must be [string, number]")
                
                # Convert list format [[prompt, weight], ...] to tuple format [(prompt, weight), ...]
                prompt_tuples = [(item[0], item[1]) for item in prompt_list]
                
                logger.debug(f"update_prompt_blending: Calling pipeline.stream.update_prompt with {len(prompt_tuples)} prompts")
                
                # Update prompt blending using the unified public interface
                self.pipeline.stream.update_prompt(
                    prompt_tuples,  # Pass as first positional argument
                    prompt_interpolation_method=interpolation_method
                )
                
                logger.debug(f"update_prompt_blending: Successfully updated prompt blending")
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated prompt blending with {len(prompt_list)} prompts"
                })
                
            except Exception as e:
                logger.error(f"update_prompt_blending: Error: {e}")
                logging.error(f"update_prompt_blending: Failed to update prompt blending: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update prompt blending: {str(e)}")

        @self.app.post("/api/seed-blending/update")
        async def update_seed_blending(request: Request):
            """Update seed blending configuration in real-time"""
            try:
                data = await request.json()
                seed_list = data.get("seed_list")
                seed_interpolation_method = data.get("seed_interpolation_method", "linear")
                
                if seed_list is None:
                    raise HTTPException(status_code=400, detail="Missing seed_list parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Validate seed_list structure
                if not isinstance(seed_list, list):
                    raise HTTPException(status_code=400, detail="seed_list must be a list")
                
                for item in seed_list:
                    if not isinstance(item, list) or len(item) != 2:
                        raise HTTPException(status_code=400, detail="Each seed_list item must be [seed, weight]")
                    if not isinstance(item[0], int) or not isinstance(item[1], (int, float)):
                        raise HTTPException(status_code=400, detail="Each seed_list item must be [int, number]")
                
                # Convert list format [[seed, weight], ...] to tuple format [(seed, weight), ...]
                seed_tuples = [(item[0], item[1]) for item in seed_list]
                
                # Update seed blending using the public interface
                self.pipeline.stream.update_seed_blending(
                    seed_list=seed_tuples,
                    interpolation_method=seed_interpolation_method
                )
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated seed blending with {len(seed_list)} seeds"
                })
                
            except Exception as e:
                logging.error(f"update_seed_blending: Failed to update seed blending: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update seed blending: {str(e)}")

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
                
                # Set system parameters
                system_params = {
                    'device': cn_pipeline.device,
                    'dtype': cn_pipeline.dtype,
                    'image_width': cn_pipeline.stream.width,
                    'image_height': cn_pipeline.stream.height,
                }
                system_params.update(preprocessor_params)
                new_preprocessor_instance.params.update(system_params)
                
                # Set pipeline reference for feedback preprocessor
                if hasattr(new_preprocessor_instance, 'set_pipeline_ref'):
                    new_preprocessor_instance.set_pipeline_ref(cn_pipeline.stream)
                
                # Replace the preprocessor
                old_preprocessor = cn_pipeline.preprocessors[controlnet_index]
                cn_pipeline.preprocessors[controlnet_index] = new_preprocessor_instance
                
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
                
                logger.info(f"update_preprocessor_params: Updating ControlNet {controlnet_index} params: {preprocessor_params}")
                
                if not preprocessor_params:
                    raise HTTPException(status_code=400, detail="Missing preprocessor_params parameter")
                
                # Get ControlNet pipeline using helper
                cn_pipeline = self._get_controlnet_pipeline()
                if not cn_pipeline:
                    raise HTTPException(status_code=400, detail="ControlNet pipeline not found")
                
                if controlnet_index >= len(cn_pipeline.preprocessors):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                current_preprocessor = cn_pipeline.preprocessors[controlnet_index]
                if not current_preprocessor:
                    raise HTTPException(status_code=400, detail=f"No preprocessor found at index {controlnet_index}")
                
                # Use all provided parameters
                user_params = preprocessor_params
                
                # Update preprocessor parameters
                current_preprocessor.params.update(user_params)
                
                # Update direct attributes if they exist
                for param_name, param_value in user_params.items():
                    if hasattr(current_preprocessor, param_name):
                        setattr(current_preprocessor, param_name, param_value)
                
                logger.info(f"update_preprocessor_params: Successfully updated ControlNet {controlnet_index} with params: {user_params}")
                
                return JSONResponse({
                    "status": "success",
                    "message": "Successfully updated preprocessor parameters",
                    "controlnet_index": controlnet_index,
                    "updated_parameters": user_params
                })
                    
            except Exception as e:
                logger.error(f"update_preprocessor_params: Failed to update parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update preprocessor parameters: {str(e)}")

        @self.app.get("/api/preprocessors/current-params/{controlnet_index}")
        async def get_current_preprocessor_params(controlnet_index: int):
            """Get current parameter values for a specific ControlNet preprocessor"""
            try:
                # Get ControlNet pipeline using helper
                cn_pipeline = self._get_controlnet_pipeline()
                if not cn_pipeline:
                    raise HTTPException(status_code=400, detail="ControlNet pipeline not found")
                
                if controlnet_index >= len(cn_pipeline.preprocessors):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                current_preprocessor = cn_pipeline.preprocessors[controlnet_index]
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
                    if param_name in current_preprocessor.params:
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
        
        # Use uploaded config if available, otherwise use original args
        if controlnet_config_path:
            new_args = self.args._replace(controlnet_config=controlnet_config_path)
        elif self.uploaded_controlnet_config:
            # Create temporary file from stored config
            temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.uploaded_controlnet_config, temp_config_path, default_flow_style=False)
            temp_config_path.close()
            
            # Merge YAML config values into args, respecting config overrides
            # This ensures that acceleration settings from YAML config override command line args
            config_acceleration = self.uploaded_controlnet_config.get('acceleration', self.args.acceleration)
            new_args = self.args._replace(
                controlnet_config=temp_config_path.name,
                acceleration=config_acceleration
            )
        else:
            new_args = self.args
        
        new_pipeline = Pipeline(new_args, device, torch_dtype, width=self.new_width, height=self.new_height)
        
        # Initialize prompt blending from config
        normalized_prompt_config = self._normalize_prompt_config(self.uploaded_controlnet_config)
        if normalized_prompt_config:
            # Convert to tuple format and set up prompt blending
            prompt_tuples = [(item[0], item[1]) for item in normalized_prompt_config]
            new_pipeline.stream.update_prompt(prompt_tuples, prompt_interpolation_method="slerp")
        else:
            # Fallback to default single prompt
            default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
            new_pipeline.stream.update_prompt([(default_prompt, 1.0)], prompt_interpolation_method="slerp")
        
        # Apply style image (uploaded or default) if pipeline has IPAdapter
        if getattr(new_pipeline, 'has_ipadapter', False):
            style_image = None
            style_source = ""
            
            if self.uploaded_style_image:
                style_image = self.uploaded_style_image
                style_source = "uploaded"
            else:
                # Try to load default style image
                style_image = self._load_default_style_image()
                if style_image:
                    style_source = "default"
            
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
        
        # Check uploaded config first
        if self.uploaded_controlnet_config:
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
                return Image.open(default_image_path)
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
                ipadapter_info["scale"] = first_ipadapter.get('scale', 1.0)
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
                ipadapter_info["scale"] = first_ipadapter.get('scale', 1.0)
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
            if getattr(new_pipeline, 'has_ipadapter', False):
                style_image = None
                style_source = ""
                
                if self.uploaded_style_image:
                    style_image = self.uploaded_style_image
                    style_source = "uploaded"
                else:
                    # Try to load default style image
                    style_image = self._load_default_style_image()
                    if style_image:
                        style_source = "default"
                
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
