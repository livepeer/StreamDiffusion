"""
Inference and system status endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import uuid
import markdown2

from .common.api_utils import handle_api_request, create_success_response, handle_api_error
from .common.dependencies import get_app_instance, get_pipeline_class, get_default_settings

router = APIRouter(prefix="/api", tags=["inference"])

@router.get("/queue")
async def get_queue_size(app_instance=Depends(get_app_instance)):
    """Get current queue size"""
    queue_size = app_instance.conn_manager.get_user_count()
    return JSONResponse({"queue_size": queue_size})

@router.get("/stream/{user_id}")
async def stream(user_id: uuid.UUID, request: Request, app_instance=Depends(get_app_instance)):
    """Main streaming endpoint for inference"""
    try:
        # Create pipeline if it doesn't exist yet
        if app_instance.pipeline is None:
            app_instance.app_state.pipeline_lifecycle = "starting"
            logging.info("stream: Creating pipeline using AppState as single source of truth...")
            app_instance.pipeline = app_instance._create_pipeline()
            app_instance.app_state.pipeline_lifecycle = "running"
            logging.info("stream: Pipeline created successfully")
        
        # Recreate pipeline if config changed (but not resolution - that's handled separately)
        elif app_instance.app_state.config_needs_reload or (app_instance.app_state.uploaded_config and not (app_instance.pipeline.use_config and app_instance.pipeline.config and 'controlnets' in app_instance.pipeline.config)) or (app_instance.app_state.uploaded_config and not app_instance.pipeline.use_config):
            if app_instance.app_state.config_needs_reload:
                logging.info("stream: Recreating pipeline with new ControlNet config...")
            else:
                logging.info("stream: Upgrading to ControlNet pipeline...")
            
            app_instance.app_state.pipeline_lifecycle = "restarting"
            
            # Properly cleanup the old pipeline before creating new one
            old_pipeline = app_instance.pipeline
            app_instance.pipeline = None
            
            if old_pipeline:
                app_instance._cleanup_pipeline(old_pipeline)
                old_pipeline = None
            
            # Create new pipeline
            app_instance.pipeline = app_instance._create_pipeline()
            
            app_instance.app_state.config_needs_reload = False
            app_instance.app_state.pipeline_lifecycle = "running"
            logging.info("stream: Pipeline recreated successfully")

        # Resolution changes are now handled immediately in _update_resolution()
        # No need to check for resolution mismatches here

        # Check for acceleration changes (requires pipeline recreation)
        acceleration_changed = False
        if hasattr(app_instance, 'new_acceleration') and app_instance.new_acceleration != app_instance.args.acceleration:
            logging.info(f"stream: Acceleration change detected: {app_instance.args.acceleration} -> {app_instance.new_acceleration}")
            
            # Create new Args object with updated acceleration (NamedTuple is immutable)
            from config import Args
            args_dict = app_instance.args._asdict()
            args_dict['acceleration'] = app_instance.new_acceleration
            app_instance.args = Args(**args_dict)
            delattr(app_instance, 'new_acceleration')
            
            # Recreate pipeline with new acceleration
            old_pipeline = app_instance.pipeline
            app_instance.pipeline = None
            if old_pipeline:
                app_instance._cleanup_pipeline(old_pipeline)
            
            app_instance.pipeline = app_instance._create_pipeline()
            acceleration_changed = True
            logging.info(f"stream: Pipeline recreated with new acceleration")

        # IPAdapter style images are now handled dynamically in pipeline.predict()
        # No static application needed here

        # Generate and stream frames
        if app_instance.pipeline is None:
            raise HTTPException(status_code=500, detail="Failed to create pipeline")

        # HTTP streaming works by using the latest parameters from the WebSocket connection manager
        # This allows the HTTP stream to show generated images while WebSocket handles input data

        # Generate and stream frames using pipeline.predict() in a loop (like original)
        try:
            async def generate_frames():
                try:
                    while True:
                        # Request a new frame from the frontend via WebSocket
                        await app_instance.conn_manager.send_json(user_id, {"status": "send_frame"})
                        
                        # Get the latest parameters from the WebSocket connection manager
                        # This consumes data from the queue after requesting a new frame
                        # Get latest data from the queue (blocks until new data arrives)
                        params = await app_instance.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        
                        # Attach InputSourceManager to params for modular input routing
                        if hasattr(app_instance, 'input_source_manager'):
                            params.input_manager = app_instance.input_source_manager
                        
                        # Generate frame using pipeline.predict()
                        image = app_instance.pipeline.predict(params)
                        if image is None:
                            logging.error("stream: predict returned None image; skipping frame")
                            continue
                            
                        # Update FPS counter
                        import time
                        current_time = time.time()
                        if hasattr(app_instance, 'last_frame_time'):
                            frame_time = current_time - app_instance.last_frame_time
                            app_instance.fps_counter.append(frame_time)
                            if len(app_instance.fps_counter) > 30:  # Keep last 30 frames
                                app_instance.fps_counter.pop(0)
                        app_instance.last_frame_time = current_time
                        
                        # Convert image to frame format for streaming
                        # Use appropriate frame conversion based on output type
                        if app_instance.pipeline.output_type == "pt":
                            from util import pt_to_frame
                            frame = pt_to_frame(image)
                        else:
                            from util import pil_to_frame
                            frame = pil_to_frame(image)
                        yield frame
                        
                except Exception as e:
                    logging.exception(f"stream: Error in frame generation: {e}")

            return StreamingResponse(
                generate_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
            
        except Exception as e:
            raise e
            
    except Exception as e:
        logging.exception(f"stream: Error in streaming endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@router.get("/state")
async def get_app_state(app_instance=Depends(get_app_instance), pipeline_class=Depends(get_pipeline_class), default_settings=Depends(get_default_settings)):
    """Get complete application state - replaces /api/settings with centralized state management"""
    try:
        # Update app_state with current dynamic values - SINGLE SOURCE OF TRUTH
        app_instance.app_state.pipeline_active = app_instance.pipeline is not None
        
        # Update FPS from fps_counter
        if len(app_instance.fps_counter) > 0:
            avg_frame_time = sum(app_instance.fps_counter) / len(app_instance.fps_counter)
            app_instance.app_state.fps = round(1.0 / avg_frame_time if avg_frame_time > 0 else 0, 1)
        else:
            app_instance.app_state.fps = 0
            
        # Update queue size
        app_instance.app_state.queue_size = app_instance.conn_manager.get_user_count()
        
        # Update pipeline parameters schema
        app_instance.app_state.pipeline_params = pipeline_class.InputParams.schema()
        
        # Update page content
        if app_instance.pipeline and hasattr(app_instance.pipeline, 'info'):
            info = app_instance.pipeline.info
        else:
            info = pipeline_class.Info()
        
        if info.page_content:
            app_instance.app_state.page_content = markdown2.markdown(info.page_content)
        
        # Get complete state
        state_data = app_instance.app_state.get_complete_state()
        
        # Add additional fields expected by frontend for backward compatibility
        state_data.update({
            "info": pipeline_class.Info.schema(),
            "input_params": app_instance.app_state.pipeline_params,
            "max_queue_size": app_instance.args.max_queue_size,
            "acceleration": app_instance.args.acceleration,
            # Add config prompt for backward compatibility
            "config_prompt": app_instance.app_state.uploaded_config.get('prompt') if app_instance.app_state.uploaded_config else None,
            # Add resolution in expected format
            "resolution": f"{app_instance.app_state.current_resolution['width']}x{app_instance.app_state.current_resolution['height']}",
        })
        
        return JSONResponse(state_data)
        
    except Exception as e:
        logging.error(f"get_app_state: Error getting application state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get application state: {str(e)}")

@router.get("/settings")
async def settings(app_instance=Depends(get_app_instance), pipeline_class=Depends(get_pipeline_class), default_settings=Depends(get_default_settings)):
    """Get pipeline settings and configuration info"""
    # Use Pipeline class directly for schema info (doesn't require instance)
    info_schema = pipeline_class.Info.schema()
    
    # Get info from pipeline instance if available to get correct input_mode
    if app_instance.pipeline and hasattr(app_instance.pipeline, 'info'):
        info = app_instance.pipeline.info
    else:
        info = pipeline_class.Info()
    
    page_content = ""
    if info.page_content:
        page_content = markdown2.markdown(info.page_content)

    input_params = pipeline_class.InputParams.schema()
    
    # Add ControlNet information - SINGLE SOURCE OF TRUTH
    controlnet_info = app_instance.app_state.controlnet_info
    
    # Add IPAdapter information - SINGLE SOURCE OF TRUTH
    ipadapter_info = app_instance.app_state.ipadapter_info
    
    # Include config prompt if available, otherwise use default
    config_prompt = None
    if app_instance.app_state.uploaded_config and 'prompt' in app_instance.app_state.uploaded_config:
        config_prompt = app_instance.app_state.uploaded_config['prompt']
    elif not config_prompt:
        config_prompt = default_settings.get('prompt')
    
    # Get current t_index_list from pipeline or config
    current_t_index_list = None
    if app_instance.pipeline and hasattr(app_instance.pipeline.stream, 't_list'):
        current_t_index_list = app_instance.pipeline.stream.t_list
    elif app_instance.app_state.uploaded_config and 't_index_list' in app_instance.app_state.uploaded_config:
        current_t_index_list = app_instance.app_state.uploaded_config['t_index_list']
    else:
        # Default values
        current_t_index_list = default_settings.get('t_index_list', [35, 45])
    
    # Get current acceleration setting
    current_acceleration = app_instance.args.acceleration
    
    # Get current resolution
    current_resolution = f"{app_instance.app_state.current_resolution['width']}x{app_instance.app_state.current_resolution['height']}"
    # Add aspect ratio for display
    aspect_ratio = app_instance._calculate_aspect_ratio(app_instance.app_state.current_resolution['width'], app_instance.app_state.current_resolution['height'])
    if aspect_ratio:
        current_resolution += f" ({aspect_ratio})"
    if app_instance.app_state.uploaded_config and 'acceleration' in app_instance.app_state.uploaded_config:
        current_acceleration = app_instance.app_state.uploaded_config['acceleration']
    
    # Get current streaming parameters (default values or from pipeline if available)
    current_guidance_scale = default_settings.get('guidance_scale', 1.1)
    current_delta = default_settings.get('delta', 0.7)
    current_num_inference_steps = default_settings.get('num_inference_steps', 50)
    current_seed = default_settings.get('seed', 2)
    
    if app_instance.pipeline and hasattr(app_instance.pipeline, 'stream'):
        try:
            state = app_instance.pipeline.stream.get_stream_state()
            current_guidance_scale = state.get('guidance_scale', current_guidance_scale)
            current_delta = state.get('delta', current_delta)
            current_num_inference_steps = state.get('num_inference_steps', current_num_inference_steps)
            current_seed = state.get('seed', current_seed)
        except Exception as e:
            logging.warning(f"settings: Failed to get current stream parameters: {e}")
    
    # Get negative prompt if available
    current_negative_prompt = default_settings.get('negative_prompt', '')
    if app_instance.app_state.uploaded_config and 'negative_prompt' in app_instance.app_state.uploaded_config:
        current_negative_prompt = app_instance.app_state.uploaded_config['negative_prompt']
    elif app_instance.pipeline and hasattr(app_instance.pipeline, 'stream'):
        try:
            state = app_instance.pipeline.stream.get_stream_state()
            current_negative_prompt = state.get('negative_prompt', current_negative_prompt)
        except Exception:
            pass
    
    # Get prompt and seed blending configuration - SINGLE SOURCE OF TRUTH
    prompt_blending_config = app_instance.app_state.prompt_blending
    seed_blending_config = app_instance.app_state.seed_blending
    
    # Get normalization settings - SINGLE SOURCE OF TRUTH
    normalize_prompt_weights = app_instance.app_state.normalize_prompt_weights
    normalize_seed_weights = app_instance.app_state.normalize_seed_weights
    
    # Get current skip_diffusion setting - SINGLE SOURCE OF TRUTH
    current_skip_diffusion = app_instance.app_state.skip_diffusion
    
    # Determine current model id for UI badge - SINGLE SOURCE OF TRUTH
    model_id_for_ui = app_instance.app_state.model_id
    
    # Check if pipeline is active
    pipeline_active = app_instance.pipeline is not None
    
    # Build config_values for other parameters that frontend may expect
    config_values = {}
    if app_instance.app_state.uploaded_config:
        for key in [
            'use_taesd',
            'cfg_type', 
            'safety_checker',
        ]:
            if key in app_instance.app_state.uploaded_config:
                config_values[key] = app_instance.app_state.uploaded_config[key]

    response_data = {
        "info": info_schema,
        "input_params": input_params,
        "max_queue_size": app_instance.args.max_queue_size,
        "page_content": page_content if info.page_content else "",
        "pipeline_active": pipeline_active,
        "controlnet": controlnet_info,
        "ipadapter": ipadapter_info,
        "config_prompt": config_prompt,
        "negative_prompt": current_negative_prompt,
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
        "skip_diffusion": current_skip_diffusion,
        "model_id": model_id_for_ui,
        "config_values": config_values,
    }
    
    return JSONResponse(response_data)

@router.get("/fps")
async def get_fps(app_instance=Depends(get_app_instance)):
    """Get current FPS"""
    if len(app_instance.fps_counter) > 0:
        avg_frame_time = sum(app_instance.fps_counter) / len(app_instance.fps_counter)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    else:
        fps = 0
    
    return JSONResponse({"fps": round(fps, 1)})



