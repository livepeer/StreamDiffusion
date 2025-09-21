"""
ControlNet-related endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
import logging
import yaml
import tempfile
from pathlib import Path
import copy

from .common.api_utils import handle_api_request, create_success_response, handle_api_error, validate_feature_enabled, validate_config_mode
from .common.dependencies import get_app_instance, get_available_controlnets

router = APIRouter(prefix="/api", tags=["controlnet"])

def _ensure_runtime_controlnet_config(app_instance):
    """Ensure runtime controlnet config is initialized from uploaded config or create minimal config"""
    if app_instance.app_state.runtime_config is None:
        if app_instance.app_state.uploaded_config:
            # Copy from YAML (deep copy to avoid modifying original)
            app_instance.app_state.runtime_config = copy.deepcopy(app_instance.app_state.uploaded_config)
        else:
            # Create minimal config if no YAML exists
            app_instance.app_state.runtime_config = {'controlnets': []}
    
    # Ensure controlnets key exists in runtime config
    if 'controlnets' not in app_instance.app_state.runtime_config:
        app_instance.app_state.runtime_config['controlnets'] = []


@router.post("/controlnet/upload-config")
async def upload_controlnet_config(file: UploadFile = File(...), app_instance=Depends(get_app_instance)):
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
        app_instance.app_state.uploaded_config = config_data
        app_instance.app_state.runtime_config = None
        app_instance.app_state.config_needs_reload = True
        
        # SINGLE SOURCE OF TRUTH: Populate AppState from config
        app_instance.app_state.populate_from_config(config_data)
        
        # RESET ALL INPUT SOURCES TO DEFAULTS WHEN NEW CONFIG IS UPLOADED
        if hasattr(app_instance, 'input_source_manager'):
            try:
                app_instance.input_source_manager.reset_to_defaults()
                logging.info("upload_controlnet_config: Reset all input sources to defaults")
            except Exception as e:
                logging.exception(f"upload_controlnet_config: Failed to reset input sources: {e}")
        
        # FORCE DESTROY ACTIVE PIPELINE TO MAKE CONFIG THE SOURCE OF TRUTH
        if app_instance.pipeline:
            logging.info("upload_controlnet_config: Destroying active pipeline to force config as source of truth")
            app_instance.app_state.pipeline_lifecycle = "stopping"
            old_pipeline = app_instance.pipeline
            app_instance.pipeline = None
            app_instance._cleanup_pipeline(old_pipeline)
            app_instance.app_state.pipeline_lifecycle = "stopped"
        
        # Get config prompt if available
        config_prompt = config_data.get('prompt', None)
        # Get negative prompt if available
        config_negative_prompt = config_data.get('negative_prompt', None)
        
        # Get t_index_list from config if available
        from app_config import DEFAULT_SETTINGS
        t_index_list = config_data.get('t_index_list', DEFAULT_SETTINGS.get('t_index_list', [35, 45]))
        
        # Get acceleration from config if available
        config_acceleration = config_data.get('acceleration', app_instance.args.acceleration)
        
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
                
                app_instance.app_state.current_resolution = {
                    "width": int(config_width),
                    "height": int(config_height)
                }
                
                logging.info(f"upload_controlnet_config: Updated resolution from config to {config_width}x{config_height}")
                
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Invalid width/height values in config")
        
        # Build current resolution string
        current_resolution = None
        if config_width and config_height:
            current_resolution = f"{config_width}x{config_height}"
            # Add aspect ratio for display
            aspect_ratio = app_instance._calculate_aspect_ratio(config_width, config_height)
            if aspect_ratio:
                current_resolution += f" ({aspect_ratio})"
        
        # Build config_values for other parameters that frontend may expect
        config_values = {}
        for key in [
            'use_taesd',
            'cfg_type', 
            'safety_checker',
        ]:
            if key in config_data:
                config_values[key] = config_data[key]

        # Response with comprehensive data as expected by frontend - SINGLE SOURCE OF TRUTH
        response_data = {
            "status": "success",
            "message": "ControlNet configuration uploaded successfully",
            "filename": file.filename,
            "controls_updated": True,  # Flag for frontend to update controls
            "controlnet": app_instance.app_state.controlnet_info,
            "ipadapter": app_instance.app_state.ipadapter_info,
            "config_prompt": config_prompt,
            "negative_prompt": app_instance.app_state.negative_prompt,
            "model_id": app_instance.app_state.model_id,
            "t_index_list": app_instance.app_state.t_index_list,
            "acceleration": config_acceleration,
            "guidance_scale": app_instance.app_state.guidance_scale,
            "delta": app_instance.app_state.delta,
            "num_inference_steps": app_instance.app_state.num_inference_steps,
            "seed": app_instance.app_state.seed,
            "prompt_blending": app_instance.app_state.prompt_blending,
            "seed_blending": app_instance.app_state.seed_blending,
            "current_resolution": current_resolution,  # Include updated resolution
            "normalize_prompt_weights": app_instance.app_state.normalize_prompt_weights,
            "normalize_seed_weights": app_instance.app_state.normalize_seed_weights,
            "skip_diffusion": app_instance.app_state.skip_diffusion,
            "config_values": config_values,
            # Include pipeline hooks info - SINGLE SOURCE OF TRUTH
            "image_preprocessing": app_instance.app_state.pipeline_hooks["image_preprocessing"],
            "image_postprocessing": app_instance.app_state.pipeline_hooks["image_postprocessing"],
            "latent_preprocessing": app_instance.app_state.pipeline_hooks["latent_preprocessing"],
            "latent_postprocessing": app_instance.app_state.pipeline_hooks["latent_postprocessing"],
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logging.exception(f"upload_controlnet_config: Failed to upload configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload configuration: {str(e)}")

@router.get("/controlnet/info")
async def get_controlnet_info(app_instance=Depends(get_app_instance)):
    """Get current ControlNet configuration info - SINGLE SOURCE OF TRUTH"""
    return JSONResponse({"controlnet": app_instance.app_state.controlnet_info})

@router.get("/blending/current")
async def get_current_blending_config(app_instance=Depends(get_app_instance)):
    """Get current prompt and seed blending configurations"""
    try:
        if app_instance.pipeline and hasattr(app_instance.pipeline, 'stream') and hasattr(app_instance.pipeline.stream, 'get_stream_state'):
            state = app_instance.pipeline.stream.get_stream_state(include_caches=False)
            return JSONResponse({
                "prompt_blending": state.get("prompt_list", []),
                "seed_blending": state.get("seed_list", []),
                "normalize_prompt_weights": state.get("normalize_prompt_weights", True),
                "normalize_seed_weights": state.get("normalize_seed_weights", True),
                "has_config": app_instance.app_state.uploaded_config is not None,
                "pipeline_active": True
            })

        # Fallback to AppState when pipeline not initialized - SINGLE SOURCE OF TRUTH
        return JSONResponse({
            "prompt_blending": app_instance.app_state.prompt_blending,
            "seed_blending": app_instance.app_state.seed_blending,
            "normalize_prompt_weights": app_instance.app_state.normalize_prompt_weights,
            "normalize_seed_weights": app_instance.app_state.normalize_seed_weights,
            "has_config": app_instance.app_state.uploaded_config is not None,
            "pipeline_active": False
        })
        
    except Exception as e:
        raise handle_api_error(e, "get_current_blending_config")

@router.post("/controlnet/update-strength")
async def update_controlnet_strength(request: Request, app_instance=Depends(get_app_instance)):
    """Update ControlNet strength in real-time"""
    try:
        data = await request.json()
        controlnet_index = data.get("index")
        strength = data.get("strength")
        
        if controlnet_index is None or strength is None:
            raise HTTPException(status_code=400, detail="Missing index or strength parameter")
        
        # Update ControlNet strength in AppState - SINGLE SOURCE OF TRUTH
        app_instance.app_state.update_controlnet_strength(controlnet_index, float(strength))
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                # Convert AppState to pipeline format for update
                controlnet_config = []
                for cn in app_instance.app_state.controlnet_info["controlnets"]:
                    config_entry = dict(cn)
                    config_entry['conditioning_scale'] = cn['strength']  # Map strength back to conditioning_scale
                    controlnet_config.append(config_entry)
                app_instance.pipeline.update_stream_params(controlnet_config=controlnet_config)
            except Exception as e:
                logging.exception(f"update_controlnet_strength: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
            
        return create_success_response(f"Updated ControlNet {controlnet_index} strength to {strength}")
        
    except Exception as e:
        raise handle_api_error(e, "update_controlnet_strength")

@router.get("/controlnet/available")
async def get_available_controlnets_endpoint(app_instance=Depends(get_app_instance), available_controlnets=Depends(get_available_controlnets)):
    """Get list of available ControlNets that can be added"""
    try:
        # Debug the dependency injection
        
        # Detect current model architecture to filter appropriate ControlNets
        model_type = "sd15"  # Default fallback
        
        # Try to determine model type from pipeline config or uploaded config
        if app_instance.pipeline and hasattr(app_instance.pipeline, 'config') and app_instance.pipeline.config:
            model_id = app_instance.pipeline.config.get('model_id', '')
            if 'sdxl' in model_id.lower() or 'xl' in model_id.lower():
                model_type = "sdxl"
        elif app_instance.app_state.uploaded_config:
            # If no pipeline yet, try to get model type from uploaded config
            model_id = app_instance.app_state.uploaded_config.get('model_id_or_path', '')
            if 'sdxl' in model_id.lower() or 'xl' in model_id.lower():
                model_type = "sdxl"
        
        # Handle case where available_controlnets dependency returns None
        if available_controlnets is None:
            logging.warning("get_available_controlnets: available_controlnets dependency returned None")
            available = []
        else:
            available = available_controlnets.get(model_type, [])
        
        # Filter out already active ControlNets
        current_controlnets = []
        # Check runtime config first, then fall back to uploaded config
        if app_instance.app_state.runtime_config and 'controlnets' in app_instance.app_state.runtime_config:
            current_controlnets = [cn.get('model_id', '') for cn in app_instance.app_state.runtime_config['controlnets']]
        elif app_instance.app_state.uploaded_config and 'controlnets' in app_instance.app_state.uploaded_config:
            current_controlnets = [cn.get('model_id', '') for cn in app_instance.app_state.uploaded_config['controlnets']]
        
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
        raise handle_api_error(e, "get_available_controlnets_endpoint")

@router.post("/controlnet/add")
async def add_controlnet(request: Request, app_instance=Depends(get_app_instance), available_controlnets=Depends(get_available_controlnets)):
    """Add a ControlNet from the predefined list"""
    try:
        data = await request.json()
        controlnet_id = data.get("controlnet_id")
        conditioning_scale = data.get("conditioning_scale", None)
        
        if not controlnet_id:
            raise HTTPException(status_code=400, detail="Missing controlnet_id parameter")
        
        # Find the ControlNet definition
        controlnet_def = None
        for model_type, controlnets in available_controlnets.items():
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
        _ensure_runtime_controlnet_config(app_instance)
        
        # Create new ControlNet entry
        new_controlnet = {
            'model_id': controlnet_def['model_id'],
            'conditioning_scale': conditioning_scale,
            'preprocessor': controlnet_def['default_preprocessor'],
            'preprocessor_params': controlnet_def.get('preprocessor_params', {}),
            'enabled': True
        }
        
        # Add to runtime config (not YAML)
        app_instance.app_state.runtime_config['controlnets'].append(new_controlnet)
        
        # Add to AppState - SINGLE SOURCE OF TRUTH
        app_instance.app_state.add_controlnet(new_controlnet)
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                # Convert AppState to pipeline format for update
                controlnet_config = []
                for cn in app_instance.app_state.controlnet_info["controlnets"]:
                    config_entry = dict(cn)
                    config_entry['conditioning_scale'] = cn['strength']  # Map strength back to conditioning_scale
                    controlnet_config.append(config_entry)
                app_instance.pipeline.update_stream_params(controlnet_config=controlnet_config)
            except Exception as e:
                logging.exception(f"add_controlnet: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
        
        # Return updated ControlNet info immediately - SINGLE SOURCE OF TRUTH
        added_index = len(app_instance.app_state.runtime_config['controlnets']) - 1
        
        return JSONResponse({
            "status": "success", 
            "message": f"Added {controlnet_def['name']}",
            "controlnet_index": added_index,
            "controlnet_info": app_instance.app_state.controlnet_info
        })
        
    except Exception as e:
        raise handle_api_error(e, "add_controlnet")

@router.get("/controlnet/status")
async def get_controlnet_status(app_instance=Depends(get_app_instance)):
    """Get the status of ControlNet configuration"""
    try:
        controlnet_pipeline = app_instance._get_controlnet_pipeline()
        
        if not controlnet_pipeline:
            return JSONResponse({
                "status": "no_pipeline",
                "message": "No ControlNet pipeline available",
                "controlnet_count": 0
            })
        
        # Use AppState - SINGLE SOURCE OF TRUTH
        controlnet_count = len(app_instance.app_state.controlnet_info["controlnets"])
        
        return JSONResponse({
            "status": "ready",
            "controlnet_count": controlnet_count,
            "message": f"{controlnet_count} ControlNet(s) configured" if controlnet_count > 0 else "No ControlNets configured"
        })
        
    except Exception as e:
        raise handle_api_error(e, "get_controlnet_status")

@router.post("/controlnet/remove")
async def remove_controlnet(request: Request, app_instance=Depends(get_app_instance)):
    """Remove a ControlNet by index"""
    try:
        data = await request.json()
        index = data.get("index")
        
        if index is None:
            raise HTTPException(status_code=400, detail="Missing index parameter")
        
        # Initialize runtime config from YAML if not already done
        _ensure_runtime_controlnet_config(app_instance)
        
        if 'controlnets' not in app_instance.app_state.runtime_config:
            raise HTTPException(status_code=400, detail="No ControlNet configuration found")
        
        controlnets = app_instance.app_state.runtime_config['controlnets']
        
        if index < 0 or index >= len(controlnets):
            raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
        
        removed_controlnet = controlnets.pop(index)
        
        # Remove from AppState - SINGLE SOURCE OF TRUTH
        app_instance.app_state.remove_controlnet(index)
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                # Convert AppState to pipeline format for update
                controlnet_config = []
                for cn in app_instance.app_state.controlnet_info["controlnets"]:
                    config_entry = dict(cn)
                    config_entry['conditioning_scale'] = cn['strength']  # Map strength back to conditioning_scale
                    controlnet_config.append(config_entry)
                app_instance.pipeline.update_stream_params(controlnet_config=controlnet_config)
            except Exception as e:
                logging.exception(f"remove_controlnet: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
        
        # Return updated ControlNet info immediately - SINGLE SOURCE OF TRUTH
        return create_success_response(f"Removed ControlNet at index {index}", controlnet_info=app_instance.app_state.controlnet_info)
        
    except Exception as e:
        raise handle_api_error(e, "remove_controlnet")

# Preprocessor endpoints (closely related to ControlNet)
@router.get("/preprocessors/info")
async def get_preprocessors_info(app_instance=Depends(get_app_instance)):
    """Get preprocessor information using metadata from preprocessor classes"""
    try:
        # Use the same processor registry as pipeline hooks
        from streamdiffusion.preprocessing.processors import list_preprocessors, get_preprocessor_class
        
        available_processors = list_preprocessors()
        processors_info = {}
        
        for processor_name in available_processors:
            try:
                processor_class = get_preprocessor_class(processor_name)
                if hasattr(processor_class, 'get_preprocessor_metadata'):
                    metadata = processor_class.get_preprocessor_metadata()
                    processors_info[processor_name] = {
                        "name": metadata.get("name", processor_name),
                        "description": metadata.get("description", ""),
                        "parameters": metadata.get("parameters", {})
                    }
                else:
                    processors_info[processor_name] = {
                        "name": processor_name,
                        "description": f"{processor_name} processor",
                        "parameters": {}
                    }
            except Exception as e:
                logging.warning(f"get_preprocessors_info: Failed to load metadata for {processor_name}: {e}")
                processors_info[processor_name] = {
                    "name": processor_name,
                    "description": f"{processor_name} processor",
                    "parameters": {}
                }
        
        return JSONResponse({
            "status": "success",
            "available": list(processors_info.keys()),
            "preprocessors": processors_info
        })
        
    except Exception as e:
        raise handle_api_error(e, "get_preprocessors_info")

@router.post("/preprocessors/switch")
async def switch_preprocessor(request: Request, app_instance=Depends(get_app_instance)):
    """Switch preprocessor for a specific ControlNet"""
    try:
        data = await request.json()
        # Support both parameter naming conventions for compatibility
        controlnet_index = data.get("controlnet_index") or data.get("processor_index")
        preprocessor_name = data.get("preprocessor") or data.get("processor")
        
        if controlnet_index is None or not preprocessor_name:
            raise HTTPException(status_code=400, detail="Missing controlnet_index/processor_index or preprocessor/processor parameter")
        
        # Validate AppState has ControlNet configuration (pipeline not required)
        if not app_instance.app_state.controlnet_info["enabled"] or not app_instance.app_state.controlnet_info["controlnets"]:
            raise HTTPException(status_code=400, detail="No ControlNet configuration available. Please upload a config first.")
        
        # Update AppState - SINGLE SOURCE OF TRUTH (works before pipeline creation)
        if controlnet_index >= len(app_instance.app_state.controlnet_info["controlnets"]):
            raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
        
        # Update the preprocessor in AppState
        controlnet = app_instance.app_state.controlnet_info["controlnets"][controlnet_index]
        old_preprocessor = controlnet.get('preprocessor', 'unknown')
        controlnet['preprocessor'] = preprocessor_name
        controlnet['preprocessor_params'] = {}  # Reset parameters when switching
        
        # Update runtime config to keep in sync
        if app_instance.app_state.runtime_config and 'controlnets' in app_instance.app_state.runtime_config:
            if controlnet_index < len(app_instance.app_state.runtime_config['controlnets']):
                app_instance.app_state.runtime_config['controlnets'][controlnet_index]['preprocessor'] = preprocessor_name
                app_instance.app_state.runtime_config['controlnets'][controlnet_index]['preprocessor_params'] = {}
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                controlnet_config = []
                for cn in app_instance.app_state.controlnet_info["controlnets"]:
                    config_entry = dict(cn)
                    config_entry['conditioning_scale'] = cn['strength']
                    controlnet_config.append(config_entry)
                app_instance.pipeline.update_stream_params(controlnet_config=controlnet_config)
            except Exception as e:
                logging.exception(f"switch_preprocessor: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
        
        return create_success_response(f"Switched ControlNet {controlnet_index} preprocessor to {preprocessor_name}")
        
    except Exception as e:
        raise handle_api_error(e, "switch_preprocessor")

@router.post("/preprocessors/update-params")
async def update_preprocessor_params(request: Request, app_instance=Depends(get_app_instance)):
    """Update preprocessor parameters for a specific ControlNet"""
    try:
        # Parse JSON directly
        try:
            data = await request.json()
        except Exception as json_error:
            logging.error(f"update_preprocessor_params: JSON parsing failed: {json_error}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {json_error}")
            
        controlnet_index = data.get("controlnet_index")
        params = data.get("params", {})
        
        if controlnet_index is None:
            logging.error(f"update_preprocessor_params: Missing controlnet_index parameter")
            raise HTTPException(status_code=400, detail="Missing controlnet_index parameter")
        
        # Validate AppState has ControlNet configuration (pipeline not required)
        if not app_instance.app_state.controlnet_info["enabled"] or not app_instance.app_state.controlnet_info["controlnets"]:
            logging.error(f"update_preprocessor_params: No ControlNet configuration available in AppState")
            raise HTTPException(status_code=400, detail="No ControlNet configuration available. Please upload a config first.")
        
        # Update AppState - SINGLE SOURCE OF TRUTH (works before pipeline creation)
        if controlnet_index >= len(app_instance.app_state.controlnet_info["controlnets"]):
            logging.error(f"update_preprocessor_params: ControlNet index {controlnet_index} out of range (max: {len(app_instance.app_state.controlnet_info['controlnets'])-1})")
            raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
        
        # Update preprocessor parameters in AppState
        controlnet = app_instance.app_state.controlnet_info["controlnets"][controlnet_index]
        if 'preprocessor_params' not in controlnet:
            controlnet['preprocessor_params'] = {}
        controlnet['preprocessor_params'].update(params)
        
        # Update runtime config to keep in sync
        if app_instance.app_state.runtime_config and 'controlnets' in app_instance.app_state.runtime_config:
            if controlnet_index < len(app_instance.app_state.runtime_config['controlnets']):
                if 'preprocessor_params' not in app_instance.app_state.runtime_config['controlnets'][controlnet_index]:
                    app_instance.app_state.runtime_config['controlnets'][controlnet_index]['preprocessor_params'] = {}
                app_instance.app_state.runtime_config['controlnets'][controlnet_index]['preprocessor_params'].update(params)
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                controlnet_config = []
                for cn in app_instance.app_state.controlnet_info["controlnets"]:
                    config_entry = dict(cn)
                    config_entry['conditioning_scale'] = cn['strength']
                    controlnet_config.append(config_entry)
                app_instance.pipeline.update_stream_params(controlnet_config=controlnet_config)
            except Exception as e:
                logging.exception(f"update_preprocessor_params: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
        logging.debug(f"update_preprocessor_params: Updated ControlNet {controlnet_index} preprocessor params: {params}")
        
        return create_success_response(f"Updated ControlNet {controlnet_index} preprocessor parameters", updated_params=params)
        
    except Exception as e:
        logging.exception(f"update_preprocessor_params: Exception occurred: {str(e)}")
        raise handle_api_error(e, "update_preprocessor_params")

@router.get("/preprocessors/current-params/{controlnet_index}")
async def get_current_preprocessor_params(controlnet_index: int, app_instance=Depends(get_app_instance)):
    """Get current parameter values for a specific ControlNet preprocessor"""
    try:
        # First try to get from uploaded config if no pipeline
        if not app_instance.pipeline and app_instance.app_state.uploaded_config:
            controlnets = app_instance.app_state.uploaded_config.get('controlnets', [])
            if controlnet_index < len(controlnets):
                controlnet = controlnets[controlnet_index]
                return JSONResponse({
                    "status": "success",
                    "controlnet_index": controlnet_index,
                    "preprocessor": controlnet.get('preprocessor', 'unknown'),
                    "parameters": controlnet.get('preprocessor_params', {}),
                    "note": "From uploaded config"
                })
        
        # Return empty/default response if no config available
        if not app_instance.pipeline:
            return JSONResponse({
                "status": "success",
                "controlnet_index": controlnet_index,
                "preprocessor": "unknown",
                "parameters": {},
                "note": "Pipeline not initialized - no config available"
            })
        
        # Use AppState - SINGLE SOURCE OF TRUTH
        if controlnet_index >= len(app_instance.app_state.controlnet_info["controlnets"]):
            return JSONResponse({
                "status": "success",
                "controlnet_index": controlnet_index,
                "preprocessor": "unknown",
                "parameters": {},
                "note": f"ControlNet index {controlnet_index} out of range"
            })
        
        controlnet = app_instance.app_state.controlnet_info["controlnets"][controlnet_index]
        preprocessor = controlnet.get('preprocessor', 'unknown')
        preprocessor_params = controlnet.get('preprocessor_params', {})
        
        return JSONResponse({
            "status": "success",
            "controlnet_index": controlnet_index,
            "preprocessor": preprocessor,
            "parameters": preprocessor_params
        })
        
    except Exception as e:
        raise handle_api_error(e, "get_current_preprocessor_params")

