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

from .common.api_utils import handle_api_request, create_success_response, handle_api_error, validate_pipeline, validate_feature_enabled, validate_config_mode
from .common.dependencies import get_app_instance, get_available_controlnets

router = APIRouter(prefix="/api", tags=["controlnet"])

def _ensure_runtime_controlnet_config(app_instance):
    """Ensure runtime controlnet config is initialized from uploaded config or create minimal config"""
    if app_instance.runtime_controlnet_config is None:
        if app_instance.uploaded_controlnet_config:
            # Copy from YAML (deep copy to avoid modifying original)
            app_instance.runtime_controlnet_config = copy.deepcopy(app_instance.uploaded_controlnet_config)
        else:
            # Create minimal config if no YAML exists
            app_instance.runtime_controlnet_config = {'controlnets': []}
    
    # Ensure controlnets key exists in runtime config
    if 'controlnets' not in app_instance.runtime_controlnet_config:
        app_instance.runtime_controlnet_config['controlnets'] = []

def _update_pipeline_controlnet_config(app_instance, operation_name: str):
    """Update pipeline with current controlnet config, with fallback to reload"""
    try:
        current_config = app_instance._get_current_controlnet_config()
        app_instance.pipeline.update_stream_params(controlnet_config=current_config)
        logging.info(f"{operation_name}: Successfully updated ControlNet using consolidated API")
    except Exception as e:
        logging.exception(f"{operation_name}: Failed to update ControlNet: {e}")
        # Mark for reload as fallback
        app_instance.config_needs_reload = True

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
        app_instance.uploaded_controlnet_config = config_data
        app_instance.runtime_controlnet_config = None  # Clear any runtime additions
        app_instance.config_needs_reload = True  # Mark that pipeline needs recreation
        
        # FORCE DESTROY ACTIVE PIPELINE TO MAKE CONFIG THE SOURCE OF TRUTH
        if app_instance.pipeline:
            logging.info("upload_controlnet_config: Destroying active pipeline to force config as source of truth")
            old_pipeline = app_instance.pipeline
            app_instance.pipeline = None
            app_instance._cleanup_pipeline(old_pipeline)
        
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
                
                app_instance.new_width = int(config_width)
                app_instance.new_height = int(config_height)
                logging.info(f"upload_controlnet_config: Updated resolution from config to {config_width}x{config_height}")
                
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Invalid width/height values in config")
        
        # Store acceleration if different
        if config_acceleration != app_instance.args.acceleration:
            app_instance.new_acceleration = config_acceleration
        
        # Normalize blending configurations using existing methods
        normalized_prompt_blending = app_instance._normalize_prompt_config(config_data)
        normalized_seed_blending = app_instance._normalize_seed_config(config_data)
        
        # Get normalization weights from config
        config_normalize_weights = config_data.get('normalize_weights', True)
        
        # Get guidance scale, delta, num_inference_steps, seed, and skip_diffusion from config
        config_guidance_scale = config_data.get('guidance_scale', None)
        config_delta = config_data.get('delta', None)
        config_num_inference_steps = config_data.get('num_inference_steps', None)
        config_seed = config_data.get('seed', None)
        config_skip_diffusion = config_data.get('skip_diffusion', None)
        
        # Build current resolution string
        current_resolution = None
        if config_width and config_height:
            current_resolution = f"{config_width}x{config_height}"
            # Add aspect ratio for display
            aspect_ratio = app_instance._calculate_aspect_ratio(config_width, config_height)
            if aspect_ratio:
                current_resolution += f" ({aspect_ratio})"
        
        # Get IPAdapter info for response
        response_ipadapter_info = app_instance._get_ipadapter_info()
        
        # Build config_values for other parameters that frontend may expect
        config_values = {}
        for key in [
            'use_taesd',
            'cfg_type', 
            'safety_checker',
        ]:
            if key in config_data:
                config_values[key] = config_data[key]

        # Response with comprehensive data as expected by frontend
        response_data = {
            "status": "success",
            "message": "ControlNet configuration uploaded successfully",
            "filename": file.filename,
            "controls_updated": True,  # Flag for frontend to update controls
            "controlnet": app_instance._get_controlnet_info(),
            "ipadapter": response_ipadapter_info,  # Include updated IPAdapter info
            "config_prompt": config_prompt,
            "negative_prompt": config_negative_prompt,
            "model_id": config_data.get('model_id_or_path', ''),
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
            "skip_diffusion": config_skip_diffusion,
            "config_values": config_values,
            # Include pipeline hooks info
            "image_preprocessing": app_instance._get_hook_info("image_preprocessing"),
            "image_postprocessing": app_instance._get_hook_info("image_postprocessing"),
            "latent_preprocessing": app_instance._get_hook_info("latent_preprocessing"),
            "latent_postprocessing": app_instance._get_hook_info("latent_postprocessing"),
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logging.exception(f"upload_controlnet_config: Failed to upload configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload configuration: {str(e)}")

@router.get("/controlnet/info")
async def get_controlnet_info(app_instance=Depends(get_app_instance)):
    """Get current ControlNet configuration info"""
    return JSONResponse({"controlnet": app_instance._get_controlnet_info()})

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
                "has_config": app_instance.uploaded_controlnet_config is not None,
                "pipeline_active": True
            })

        # Fallback to uploaded config normalization when pipeline not initialized
        prompt_blending_config = app_instance._normalize_prompt_config(app_instance.uploaded_controlnet_config)
        seed_blending_config = app_instance._normalize_seed_config(app_instance.uploaded_controlnet_config)
        normalize_weights = app_instance.uploaded_controlnet_config.get('normalize_weights', True) if app_instance.uploaded_controlnet_config else True
        return JSONResponse({
            "prompt_blending": prompt_blending_config,
            "seed_blending": seed_blending_config,
            "normalize_prompt_weights": normalize_weights,
            "normalize_seed_weights": normalize_weights,
            "has_config": app_instance.uploaded_controlnet_config is not None,
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
        
        validate_pipeline(app_instance.pipeline, "update_controlnet_strength")
        
        # Check if we're using config mode and have controlnets configured
        validate_config_mode(app_instance.pipeline, "controlnets")
        
        # Update ControlNet strength using consolidated API
        current_config = app_instance._get_current_controlnet_config()
        if controlnet_index >= len(current_config):
            raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
        
        # Update only the conditioning_scale for the specified controlnet
        old_strength = current_config[controlnet_index]['conditioning_scale']
        current_config[controlnet_index]['conditioning_scale'] = float(strength)
        
        app_instance.pipeline.update_stream_params(controlnet_config=current_config)
            
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
        elif app_instance.uploaded_controlnet_config:
            # If no pipeline yet, try to get model type from uploaded config
            model_id = app_instance.uploaded_controlnet_config.get('model_id_or_path', '')
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
        if app_instance.runtime_controlnet_config and 'controlnets' in app_instance.runtime_controlnet_config:
            current_controlnets = [cn.get('model_id', '') for cn in app_instance.runtime_controlnet_config['controlnets']]
        elif app_instance.uploaded_controlnet_config and 'controlnets' in app_instance.uploaded_controlnet_config:
            current_controlnets = [cn.get('model_id', '') for cn in app_instance.uploaded_controlnet_config['controlnets']]
        
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
        app_instance.runtime_controlnet_config['controlnets'].append(new_controlnet)
        
        # Update pipeline using consolidated API
        current_config = app_instance._get_current_controlnet_config()
        current_config.append(new_controlnet)
        _update_pipeline_controlnet_config(app_instance, "add_controlnet")
        
        
        # Return updated ControlNet info immediately
        updated_info = app_instance._get_controlnet_info()
        added_index = len(app_instance.runtime_controlnet_config['controlnets']) - 1
        
        return JSONResponse({
            "status": "success", 
            "message": f"Added {controlnet_def['name']}",
            "controlnet_index": added_index,
            "controlnet_info": updated_info
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
        
        current_config = app_instance._get_current_controlnet_config()
        
        return JSONResponse({
            "status": "ready",
            "controlnet_count": len(current_config),
            "message": f"{len(current_config)} ControlNet(s) configured" if current_config else "No ControlNets configured"
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
        
        if 'controlnets' not in app_instance.runtime_controlnet_config:
            raise HTTPException(status_code=400, detail="No ControlNet configuration found")
        
        controlnets = app_instance.runtime_controlnet_config['controlnets']
        
        if index < 0 or index >= len(controlnets):
            raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
        
        removed_controlnet = controlnets.pop(index)
        
        # Update pipeline using consolidated API
        current_config = app_instance._get_current_controlnet_config()
        if index >= len(current_config):
            raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
        
        # Remove the controlnet at the specified index
        current_config.pop(index)
        _update_pipeline_controlnet_config(app_instance, "remove_controlnet")
        
        
        # Return updated ControlNet info immediately
        updated_info = app_instance._get_controlnet_info()
        
        return create_success_response(f"Removed ControlNet at index {index}", controlnet_info=updated_info)
        
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
        controlnet_index = data.get("controlnet_index")
        preprocessor_name = data.get("preprocessor")
        
        if controlnet_index is None or not preprocessor_name:
            raise HTTPException(status_code=400, detail="Missing controlnet_index or preprocessor parameter")
        
        validate_pipeline(app_instance.pipeline, "switch_preprocessor")
        validate_config_mode(app_instance.pipeline, "controlnets")
        
        # Get current ControlNet configuration
        current_config = app_instance._get_current_controlnet_config()
        
        if controlnet_index >= len(current_config):
            raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
        
        # Update the preprocessor
        old_preprocessor = current_config[controlnet_index].get('preprocessor', 'unknown')
        current_config[controlnet_index]['preprocessor'] = preprocessor_name
        
        # Reset preprocessor parameters to defaults when switching
        current_config[controlnet_index]['preprocessor_params'] = {}
        
        # Update pipeline
        app_instance.pipeline.update_stream_params(controlnet_config=current_config)
        
        # Update runtime config to keep in sync
        if app_instance.runtime_controlnet_config and 'controlnets' in app_instance.runtime_controlnet_config:
            if controlnet_index < len(app_instance.runtime_controlnet_config['controlnets']):
                app_instance.runtime_controlnet_config['controlnets'][controlnet_index]['preprocessor'] = preprocessor_name
                app_instance.runtime_controlnet_config['controlnets'][controlnet_index]['preprocessor_params'] = {}
        
        
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
        
        validate_pipeline(app_instance.pipeline, "update_preprocessor_params")
        validate_config_mode(app_instance.pipeline, "controlnets")
        
        # Get current ControlNet configuration
        current_config = app_instance._get_current_controlnet_config()
        
        if controlnet_index >= len(current_config):
            logging.error(f"update_preprocessor_params: ControlNet index {controlnet_index} out of range (max: {len(current_config)-1})")
            raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
        
        # Update preprocessor parameters
        if 'preprocessor_params' not in current_config[controlnet_index]:
            current_config[controlnet_index]['preprocessor_params'] = {}
        
        current_config[controlnet_index]['preprocessor_params'].update(params)
        
        # Update pipeline
        app_instance.pipeline.update_stream_params(controlnet_config=current_config)
        
        # Update runtime config to keep in sync
        if app_instance.runtime_controlnet_config and 'controlnets' in app_instance.runtime_controlnet_config:
            if controlnet_index < len(app_instance.runtime_controlnet_config['controlnets']):
                if 'preprocessor_params' not in app_instance.runtime_controlnet_config['controlnets'][controlnet_index]:
                    app_instance.runtime_controlnet_config['controlnets'][controlnet_index]['preprocessor_params'] = {}
                app_instance.runtime_controlnet_config['controlnets'][controlnet_index]['preprocessor_params'].update(params)
        
        logging.info(f"update_preprocessor_params: Updated ControlNet {controlnet_index} preprocessor params: {params}")
        
        return create_success_response(f"Updated ControlNet {controlnet_index} preprocessor parameters", updated_params=params)
        
    except Exception as e:
        logging.exception(f"update_preprocessor_params: Exception occurred: {str(e)}")
        raise handle_api_error(e, "update_preprocessor_params")

@router.get("/preprocessors/current-params/{controlnet_index}")
async def get_current_preprocessor_params(controlnet_index: int, app_instance=Depends(get_app_instance)):
    """Get current parameter values for a specific ControlNet preprocessor"""
    try:
        # First try to get from uploaded config if no pipeline
        if not app_instance.pipeline and app_instance.uploaded_controlnet_config:
            controlnets = app_instance.uploaded_controlnet_config.get('controlnets', [])
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
        
        validate_config_mode(app_instance.pipeline, "controlnets")
        
        # Get current ControlNet configuration
        current_config = app_instance._get_current_controlnet_config()
        
        if controlnet_index >= len(current_config):
            return JSONResponse({
                "status": "success",
                "controlnet_index": controlnet_index,
                "preprocessor": "unknown",
                "parameters": {},
                "note": f"ControlNet index {controlnet_index} out of range"
            })
        
        controlnet = current_config[controlnet_index]
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

