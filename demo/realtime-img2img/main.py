from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File
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
# logging.basicConfig(level=logging.DEBUG)


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
        self.init_app()

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
                        print("stream: Creating pipeline with ControlNet config...")
                        self.pipeline = self._create_pipeline_with_config()
                    else:
                        print("stream: Creating default pipeline...")
                        self.pipeline = self._create_default_pipeline()
                    print("stream: Pipeline created successfully")
                
                # Recreate pipeline if config changed (but not resolution - that's handled separately)
                elif (self.config_needs_reload or 
                      (self.uploaded_controlnet_config and not (self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config))):
                    
                    if self.config_needs_reload:
                        print("stream: Recreating pipeline with new ControlNet config...")
                    else:
                        print("stream: Upgrading to ControlNet pipeline...")
                    
                    if self.uploaded_controlnet_config:
                        self.pipeline = self._create_pipeline_with_config()
                    else:
                        self.pipeline = self._create_default_pipeline()
                    
                    self.config_needs_reload = False  # Reset the flag
                    print("stream: Pipeline recreated successfully")

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
                            print(f"Time taken: {time.time() - frame_start_time}")
                        
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
                current_normalize = self.pipeline.stream.get_normalize_weights()
                normalize_prompt_weights = current_normalize
                normalize_seed_weights = current_normalize
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
                        
                        if not (512 <= config_width <= 1024) or not (512 <= config_height <= 1024):
                            raise HTTPException(status_code=400, detail="Resolution must be between 512 and 1024")
                        
                        # Update the resolution
                        self.new_width = config_width
                        self.new_height = config_height
                        print(f"upload_controlnet_config: Updated resolution to {config_width}x{config_height}")
                    except Exception as e:
                        logging.error(f"upload_controlnet_config: Failed to update resolution: {e}")
                        # Don't fail the upload, just log the error
                
                # Normalize prompt and seed configurations for frontend
                normalized_prompt_blending = self._normalize_prompt_config(config_data)
                normalized_seed_blending = self._normalize_seed_config(config_data)
                
                # Debug logging
                print(f"upload_controlnet_config: Raw prompt_blending in config: {config_data.get('prompt_blending', 'NOT FOUND')}")
                print(f"upload_controlnet_config: Raw seed_blending in config: {config_data.get('seed_blending', 'NOT FOUND')}")
                print(f"upload_controlnet_config: Normalized prompt blending: {normalized_prompt_blending}")
                print(f"upload_controlnet_config: Normalized seed blending: {normalized_seed_blending}")
                
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
                
                return JSONResponse({
                    "status": "success",
                    "message": "ControlNet configuration uploaded successfully",
                    "controls_updated": True,  # Flag for frontend to update controls
                    "controlnet": self._get_controlnet_info(),
                    "config_prompt": config_prompt,
                    "t_index_list": t_index_list,
                    "acceleration": config_acceleration,
                    "guidance_scale": config_guidance_scale,
                    "delta": config_delta,
                    "num_inference_steps": config_num_inference_steps,
                    "seed": config_seed,
                    "prompt_blending": normalized_prompt_blending,
                    "seed_blending": normalized_seed_blending,
                    "normalize_prompt_weights": config_normalize_weights,
                    "normalize_seed_weights": config_normalize_weights,
                    "current_resolution": current_resolution,  # Include updated resolution
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
                        print(f"get_current_blending_config: Retrieved current prompts from pipeline: {current_prompts}")
                        if current_prompts and len(current_prompts) > 0:
                            prompt_blending_config = current_prompts
                            print(f"get_current_blending_config: Using pipeline prompts: {prompt_blending_config}")
                    except Exception as e:
                        print(f"get_current_blending_config: Error getting current prompts: {e}")
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
                    return JSONResponse({"success": False, "detail": "Resolution parameter is required"}, status_code=400)
                
                # Parse resolution string (e.g., "512x768 (2:3)" -> width=512, height=768)
                resolution_part = resolution_str.split(' ')[0]  # Get "512x768" part
                try:
                    width, height = map(int, resolution_part.split('x'))
                except ValueError:
                    return JSONResponse({"success": False, "detail": "Invalid resolution format. Use 'widthxheight' (e.g., '512x768')"}, status_code=400)
                
                # Validate resolution
                if width % 64 != 0 or height % 64 != 0:
                    return JSONResponse({"success": False, "detail": "Resolution must be multiples of 64"}, status_code=400)
                
                if not (512 <= width <= 1024) or not (512 <= height <= 1024):
                    return JSONResponse({"success": False, "detail": "Resolution must be between 512 and 1024"}, status_code=400)
                
                # Check if resolution actually changed
                if width == self.new_width and height == self.new_height:
                    return JSONResponse({"success": True, "detail": "Resolution unchanged"})
                
                print(f"API: Updating resolution from {self.new_width}x{self.new_height} to {width}x{height}")
                
                # Create new pipeline with new resolution
                try:
                    self._update_resolution(width, height)
                    
                    print(f"API: Resolution update successful: {width}x{height}")
                    return JSONResponse({
                        "success": True,
                        "detail": f"Resolution updated to {width}x{height}",
                        "method": "pipeline_recreation"
                    })
                    
                except Exception as update_error:
                    print(f"API: Resolution update failed: {update_error}")
                    return JSONResponse({
                        "success": False,
                        "detail": f"Failed to update resolution: {update_error}",
                        "method": "failed"
                    }, status_code=500)
                
            except Exception as e:
                print(f"API: Resolution update error: {e}")
                return JSONResponse({
                    "success": False,
                    "detail": f"Failed to update resolution: {e}"
                }, status_code=500)

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
                
                print(f"update_prompt_blending: Received request with {len(prompt_list) if prompt_list else 0} prompts")
                print(f"update_prompt_blending: prompt_list = {prompt_list}")
                print(f"update_prompt_blending: interpolation_method = {interpolation_method}")
                
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
                
                print(f"update_prompt_blending: Calling pipeline.stream.update_prompt with {len(prompt_tuples)} prompts")
                
                # Update prompt blending using the unified public interface
                self.pipeline.stream.update_prompt(
                    prompt_tuples,  # Pass as first positional argument
                    interpolation_method=interpolation_method
                )
                
                print(f"update_prompt_blending: Successfully updated prompt blending")
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated prompt blending with {len(prompt_list)} prompts"
                })
                
            except Exception as e:
                print(f"update_prompt_blending: Error: {e}")
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
            """Get comprehensive preprocessor information and templates"""
            
            # Define preprocessor information with parameters, descriptions, and examples
            preprocessors_info = {
                "canny": {
                    "name": "Canny Edge Detection",
                    "description": "Detects edges in the input image using the Canny edge detection algorithm. Good for line art and architectural images.",
                    "requirements": ["OpenCV"],
                    "parameters": {
                        "low_threshold": {
                            "type": "int",
                            "default": 100,
                            "range": [50, 150],
                            "description": "Lower threshold for edge detection. Lower values detect more edges."
                        },
                        "high_threshold": {
                            "type": "int", 
                            "default": 200,
                            "range": [150, 300],
                            "description": "Upper threshold for edge detection. Higher values are more selective."
                        }
                    },
                    "example_config": {
                        "model_id": "lllyasviel/control_v11p_sd15_canny",
                        "conditioning_scale": 1.0,
                        "preprocessor": "canny",
                        "preprocessor_params": {
                            "low_threshold": 100,
                            "high_threshold": 200
                        },
                        "enabled": True
                    },
                    "use_cases": ["Line art", "Architecture", "Technical drawings", "Clean edge detection"]
                },
                
                "depth": {
                    "name": "Depth Estimation",
                    "description": "Estimates depth from the input image using MiDaS. Good for adding depth-based control to generation.",
                    "requirements": ["PyTorch", "Transformers"],
                    "parameters": {
                        "image_resolution": {
                            "type": "int",
                            "default": 512,
                            "range": [256, 1024],
                            "description": "Output image resolution"
                        }
                    },
                    "example_config": {
                        "model_id": "lllyasviel/control_v11f1p_sd15_depth",
                        "conditioning_scale": 0.8,
                        "preprocessor": "depth",
                        "preprocessor_params": {
                            "image_resolution": 512
                        },
                        "enabled": True
                    },
                    "use_cases": ["3D-aware generation", "Depth preservation", "Scene understanding"]
                },
                
                "depth_tensorrt": {
                    "name": "Depth Estimation (TensorRT)",
                    "description": "Fast TensorRT-optimized depth estimation using Depth Anything model. Significantly faster than standard depth estimation.",
                    "requirements": ["TensorRT", "Polygraphy", "Pre-built TensorRT engine"],
                    "parameters": {
                        "engine_path": {
                            "type": "string",
                            "default": "path/to/depth_anything.engine",
                            "description": "Path to the TensorRT engine file for Depth Anything model"
                        },
                        "detect_resolution": {
                            "type": "int", 
                            "default": 518,
                            "range": [256, 1024],
                            "description": "Resolution for depth detection (should match engine input size)"
                        },
                        "image_resolution": {
                            "type": "int",
                            "default": 512, 
                            "range": [256, 1024],
                            "description": "Final output image resolution"
                        }
                    },
                    "example_config": {
                        "model_id": "lllyasviel/control_v11f1p_sd15_depth",
                        "conditioning_scale": 0.8,
                        "preprocessor": "depth_tensorrt",
                        "preprocessor_params": {
                            "engine_path": "C:\\_dev\\comfy\\ComfyUI\\models\\tensorrt\\depth-anything\\v2_depth_anything_v2_vits-fp16.engine",
                            "detect_resolution": 518,
                            "image_resolution": 512
                        },
                        "enabled": True
                    },
                    "use_cases": ["High-performance depth estimation", "Real-time applications", "Production deployments"],
                    "setup_notes": "Requires building TensorRT engine from Depth Anything ONNX model"
                },
                
                "pose_tensorrt": {
                    "name": "Pose Detection (TensorRT)",
                    "description": "Fast TensorRT-optimized pose detection using YOLO-NAS Pose model. Detects human pose keypoints.",
                    "requirements": ["TensorRT", "Polygraphy", "Pre-built YOLO-NAS Pose TensorRT engine"],
                    "parameters": {
                        "engine_path": {
                            "type": "string",
                            "default": "path/to/yolo_nas_pose.engine",
                            "description": "Path to the TensorRT engine file for YOLO-NAS Pose model"
                        },
                        "detect_resolution": {
                            "type": "int",
                            "default": 640,
                            "range": [320, 1280], 
                            "description": "Resolution for pose detection (should match engine input size)"
                        },
                        "image_resolution": {
                            "type": "int",
                            "default": 512,
                            "range": [256, 1024],
                            "description": "Final output image resolution"
                        }
                    },
                    "example_config": {
                        "model_id": "thibaud/controlnet-sd21-openpose-diffusers",
                        "conditioning_scale": 0.5,
                        "preprocessor": "pose_tensorrt", 
                        "preprocessor_params": {
                            "engine_path": "C:\\_dev\\comfy\\ComfyUI\\models\\tensorrt\\yolo-nas-pose\\yolo_nas_pose_l_0.8-fp16.engine",
                            "detect_resolution": 640,
                            "image_resolution": 512
                        },
                        "enabled": True
                    },
                    "use_cases": ["Human pose control", "Character animation", "Pose-guided generation"],
                    "setup_notes": "Requires building TensorRT engine from YOLO-NAS Pose ONNX model"
                },
                
                "openpose": {
                    "name": "OpenPose",
                    "description": "Human pose estimation using OpenPose. Detects body keypoints and skeleton structure.",
                    "requirements": ["OpenPose library or compatible implementation"],
                    "parameters": {
                        "image_resolution": {
                            "type": "int",
                            "default": 512,
                            "range": [256, 1024],
                            "description": "Output image resolution"
                        }
                    },
                    "example_config": {
                        "model_id": "lllyasviel/control_v11p_sd15_openpose",
                        "conditioning_scale": 0.8,
                        "preprocessor": "openpose",
                        "preprocessor_params": {
                            "image_resolution": 512
                        },
                        "enabled": True
                    },
                    "use_cases": ["Human pose control", "Dance movements", "Character poses"]
                },
                
                "lineart": {
                    "name": "Line Art Detection",
                    "description": "Detects line art and sketches from input images. Good for converting photos to line drawings.",
                    "requirements": ["PyTorch", "Transformers"],
                    "parameters": {
                        "image_resolution": {
                            "type": "int",
                            "default": 512,
                            "range": [256, 1024],
                            "description": "Output image resolution"
                        }
                    },
                    "example_config": {
                        "model_id": "lllyasviel/control_v11p_sd15_lineart",
                        "conditioning_scale": 0.8,
                        "preprocessor": "lineart",
                        "preprocessor_params": {
                            "image_resolution": 512
                        },
                        "enabled": True
                    },
                    "use_cases": ["Sketch to image", "Line art generation", "Clean line extraction"]
                },
                
                "passthrough": {
                    "name": "Passthrough",
                    "description": "Passes the input image through with minimal processing. Used for tile ControlNet or when you want to use the input image directly.",
                    "requirements": ["None"],
                    "parameters": {
                        "image_resolution": {
                            "type": "int",
                            "default": 512,
                            "range": [256, 1024],
                            "description": "Output image resolution (input will be resized to this)"
                        }
                    },
                    "example_config": {
                        "model_id": "lllyasviel/control_v11f1e_sd15_tile",
                        "conditioning_scale": 0.2,
                        "preprocessor": "passthrough",
                        "preprocessor_params": {
                            "image_resolution": 512
                        },
                        "enabled": True
                    },
                    "use_cases": ["Tile ControlNet", "Image-to-image with structure preservation", "Upscaling with control"]
                }
            }
            
            # Template for creating full configuration
            full_config_template = {
                "model_id": "C:\\_dev\\comfy\\ComfyUI\\models\\checkpoints\\your-model.safetensors",
                "t_index_list": [32, 45],
                "width": 512,
                "height": 512,
                "device": "cuda",
                "dtype": "float16",
                "prompt": "your prompt here",
                "negative_prompt": "blurry, low quality",
                "guidance_scale": 1.1,
                "num_inference_steps": 50,
                "use_denoising_batch": True,
                "delta": 0.7,
                "frame_buffer_size": 1,
                "use_lcm_lora": True,
                "use_tiny_vae": True,
                "acceleration": "xformers",
                "cfg_type": "self",
                "seed": 42,
                "controlnets": [
                    "// Add your ControlNet configurations here using the examples above"
                ]
            }
            
            return JSONResponse({
                "preprocessors": preprocessors_info,
                "template": full_config_template,
                "common_model_ids": {
                    "canny": ["lllyasviel/control_v11p_sd15_canny", "lllyasviel/control_v11p_sd15_canny"],
                    "depth": ["lllyasviel/control_v11f1p_sd15_depth", "thibaud/controlnet-sd21-depth-diffusers"],
                    "openpose": ["lllyasviel/control_v11p_sd15_openpose", "thibaud/controlnet-sd21-openpose-diffusers"],
                    "lineart": ["lllyasviel/control_v11p_sd15_lineart", "lllyasviel/control_v11p_sd15s2_lineart_anime"],
                    "tile": ["lllyasviel/control_v11f1e_sd15_tile"]
                },
                "setup_guides": {
                    "tensorrt_engines": "TensorRT engines must be built from ONNX models. Place them in models/tensorrt/ directory.",
                    "model_downloads": "ControlNet models will be automatically downloaded from HuggingFace on first use.",
                    "performance_tips": "Use TensorRT preprocessors for real-time performance. Standard preprocessors are fine for non-realtime use."
                }
            })

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
        pipeline.stream.update_prompt([(default_prompt, 1.0)], interpolation_method="slerp")
        
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
            new_pipeline.stream.update_prompt(prompt_tuples, interpolation_method="slerp")
        else:
            # Fallback to default single prompt
            default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
            new_pipeline.stream.update_prompt([(default_prompt, 1.0)], interpolation_method="slerp")
        
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

    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate and return aspect ratio as a string"""
        import math
        
        # Find GCD to simplify the ratio
        gcd = math.gcd(width, height)
        simplified_width = width // gcd
        simplified_height = height // gcd
        
        return f"{simplified_width}:{simplified_height}"

    def _update_resolution(self, width: int, height: int) -> None:
        """Create a new pipeline with the specified resolution and replace the old one."""
        print(f"Creating new pipeline with resolution {width}x{height}")
        
        # Store current pipeline state
        current_prompt = getattr(self.pipeline, 'prompt', '')
        current_negative_prompt = getattr(self.pipeline, 'negative_prompt', '')
        current_guidance_scale = getattr(self.pipeline, 'guidance_scale', 1.2)
        current_num_inference_steps = getattr(self.pipeline, 'num_inference_steps', 50)
        
        # Update current resolution BEFORE creating new pipeline
        self.new_width = width
        self.new_height = height
        
        # Create new pipeline with new resolution
        if self.uploaded_controlnet_config:
            new_pipeline = self._create_pipeline_with_config()
        else:
            new_pipeline = self._create_default_pipeline()
        
        # Replace old pipeline
        old_pipeline = self.pipeline
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
        
        # Clean up old pipeline
        if old_pipeline:
            try:
                # Clear any references to free memory
                del old_pipeline
            except:
                pass
        
        print(f"Pipeline updated successfully to {width}x{height}")


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
