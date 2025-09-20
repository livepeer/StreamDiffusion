
"""
Pipeline hooks endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from .common.api_utils import handle_api_request, create_success_response, handle_api_error, validate_pipeline
from .common.dependencies import get_app_instance

router = APIRouter(prefix="/api", tags=["pipeline-hooks"])

def _update_pipeline_hook_config(app_instance, hook_type: str, current_hooks: list, operation_name: str):
    """Update pipeline with current hook config"""
    update_kwargs = {f"{hook_type}_config": current_hooks}
    app_instance.pipeline.update_stream_params(**update_kwargs)
    logging.info(f"{operation_name}: Successfully updated {hook_type} config")

# Pipeline Hooks API Endpoints
@router.get("/pipeline-hooks/info-config")
async def get_pipeline_hooks_info_config(app_instance=Depends(get_app_instance)):
    """Get pipeline hooks configuration info"""
    try:
        hooks_info = {
            "image_preprocessing": app_instance._get_hook_info("image_preprocessing"),
            "image_postprocessing": app_instance._get_hook_info("image_postprocessing"),
            "latent_preprocessing": app_instance._get_hook_info("latent_preprocessing"),
            "latent_postprocessing": app_instance._get_hook_info("latent_postprocessing")
        }
        return JSONResponse(hooks_info)
    except Exception as e:
        raise handle_api_error(e, "get_pipeline_hooks_info_config")

# Individual hook type endpoints that frontend expects
@router.get("/pipeline-hooks/image_preprocessing/info-config")
async def get_image_preprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get image preprocessing hook configuration info"""
    try:
        hook_info = app_instance._get_hook_info("image_preprocessing")
        return JSONResponse({"image_preprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"image_preprocessing": None})

@router.get("/pipeline-hooks/image_postprocessing/info-config")
async def get_image_postprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get image postprocessing hook configuration info"""
    try:
        hook_info = app_instance._get_hook_info("image_postprocessing")
        return JSONResponse({"image_postprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"image_postprocessing": None})

@router.get("/pipeline-hooks/latent_preprocessing/info-config")
async def get_latent_preprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get latent preprocessing hook configuration info"""
    try:
        hook_info = app_instance._get_hook_info("latent_preprocessing")
        return JSONResponse({"latent_preprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"latent_preprocessing": None})

@router.get("/pipeline-hooks/latent_postprocessing/info-config")
async def get_latent_postprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get latent postprocessing hook configuration info"""
    try:
        hook_info = app_instance._get_hook_info("latent_postprocessing")
        return JSONResponse({"latent_postprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"latent_postprocessing": None})

@router.get("/pipeline-hooks/{hook_type}/info")
async def get_hook_processors_info(hook_type: str, app_instance=Depends(get_app_instance)):
    """Get available processors for a specific hook type"""
    try:
        if hook_type not in ["image_preprocessing", "image_postprocessing", "latent_preprocessing", "latent_postprocessing"]:
            raise HTTPException(status_code=400, detail=f"Invalid hook type: {hook_type}")
        
        # Use the same processor registry as ControlNet
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
                logging.warning(f"get_hook_processors_info: Failed to load metadata for {processor_name}: {e}")
                processors_info[processor_name] = {
                    "name": processor_name,
                    "description": f"{processor_name} processor",
                    "parameters": {}
                }
        
        return JSONResponse({
            "status": "success",
            "hook_type": hook_type,
            "available": list(processors_info.keys()),
            "preprocessors": processors_info
        })
        
    except Exception as e:
        raise handle_api_error(e, "get_hook_processors_info")

@router.post("/pipeline-hooks/{hook_type}/add")
async def add_hook_processor(hook_type: str, request: Request, app_instance=Depends(get_app_instance)):
    """Add a new processor to a hook"""
    try:
        data = await request.json()
        processor_type = data.get("processor_type")
        processor_params = data.get("processor_params", {})
        
        if not processor_type:
            raise HTTPException(status_code=400, detail="Missing processor_type parameter")
        
        validate_pipeline(app_instance.pipeline, "add_hook_processor")
        
        if hook_type not in ["image_preprocessing", "image_postprocessing", "latent_preprocessing", "latent_postprocessing"]:
            raise HTTPException(status_code=400, detail=f"Invalid hook type: {hook_type}")
        
        logging.info(f"add_hook_processor: Adding {processor_type} to {hook_type}")
        
        # Create processor config
        new_processor = {
            "type": processor_type,
            "params": processor_params,
            "enabled": True
        }
        
        # Use proper hook configuration access pattern (same as ControlNet)
        current_hooks = app_instance._get_current_hook_config(hook_type)
        current_hooks.append(new_processor)
        
        # Update using the standard parameter update mechanism
        _update_pipeline_hook_config(app_instance, hook_type, current_hooks, "add_hook_processor")
        
        logging.info(f"add_hook_processor: Successfully added {processor_type} to {hook_type}")
        
        return create_success_response(f"Added {processor_type} processor to {hook_type}")
        
    except Exception as e:
        raise handle_api_error(e, "add_hook_processor")

@router.delete("/pipeline-hooks/{hook_type}/remove/{processor_index}")
async def remove_hook_processor(hook_type: str, processor_index: int, app_instance=Depends(get_app_instance)):
    """Remove a processor from a hook"""
    try:
        validate_pipeline(app_instance.pipeline, "remove_hook_processor")
        
        logging.info(f"remove_hook_processor: Removing processor {processor_index} from {hook_type}")
        
        # Use proper hook configuration access pattern (same as ControlNet)
        current_hooks = app_instance._get_current_hook_config(hook_type)
        
        if processor_index >= len(current_hooks):
            raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
        
        removed_processor = current_hooks.pop(processor_index)
        
        # Update using the standard parameter update mechanism
        _update_pipeline_hook_config(app_instance, hook_type, current_hooks, "remove_hook_processor")
        
        logging.info(f"remove_hook_processor: Successfully removed processor {processor_index} ({removed_processor.get('type', 'unknown')}) from {hook_type}")
        
        return create_success_response(f"Removed processor {processor_index} from {hook_type}")
        
    except Exception as e:
        raise handle_api_error(e, "remove_hook_processor")

@router.post("/pipeline-hooks/{hook_type}/toggle")
async def toggle_hook_processor(hook_type: str, request: Request, app_instance=Depends(get_app_instance)):
    """Toggle a processor enabled/disabled"""
    try:
        data = await request.json()
        processor_index = data.get("processor_index")
        enabled = data.get("enabled")
        
        if processor_index is None or enabled is None:
            raise HTTPException(status_code=400, detail="Missing processor_index or enabled parameter")
        
        validate_pipeline(app_instance.pipeline, "toggle_hook_processor")
        
        logging.info(f"toggle_hook_processor: Toggling processor {processor_index} in {hook_type} to {'enabled' if enabled else 'disabled'}")
        
        # Use proper hook configuration access pattern (same as ControlNet)
        current_hooks = app_instance._get_current_hook_config(hook_type)
        
        if processor_index >= len(current_hooks):
            raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
        
        current_hooks[processor_index]['enabled'] = bool(enabled)
        
        # Update using the standard parameter update mechanism
        _update_pipeline_hook_config(app_instance, hook_type, current_hooks, "toggle_hook_processor")
        
        logging.info(f"toggle_hook_processor: Successfully toggled processor {processor_index} in {hook_type}")
        
        return create_success_response(f"Processor {processor_index} in {hook_type} {'enabled' if enabled else 'disabled'}")
        
    except Exception as e:
        raise handle_api_error(e, "toggle_hook_processor")

@router.post("/pipeline-hooks/{hook_type}/switch")
async def switch_hook_processor(hook_type: str, request: Request, app_instance=Depends(get_app_instance)):
    """Switch a processor to a different type"""
    try:
        data = await request.json()
        processor_index = data.get("processor_index")
        # Support both parameter naming conventions for compatibility
        new_processor_type = data.get("processor_type") or data.get("processor")
        
        if processor_index is None or not new_processor_type:
            raise HTTPException(status_code=400, detail="Missing processor_index or processor_type/processor parameter")
        
        # Handle config-only mode when no pipeline is active
        if not app_instance.pipeline:
            if not app_instance.uploaded_controlnet_config:
                raise HTTPException(status_code=400, detail="No pipeline active and no uploaded config available")
            
            logging.info(f"switch_hook_processor: Updating config for {hook_type} processor {processor_index} to {new_processor_type}")
            
            # Update the uploaded config directly
            hook_config = app_instance.uploaded_controlnet_config.get(hook_type, {"enabled": False, "processors": []})
            if processor_index >= len(hook_config.get("processors", [])):
                raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
            
            # Update processor type in config
            hook_config["processors"][processor_index]["type"] = new_processor_type
            hook_config["processors"][processor_index]["params"] = {}
            app_instance.uploaded_controlnet_config[hook_type] = hook_config
            
        else:
            validate_pipeline(app_instance.pipeline, "switch_hook_processor")
            
            logging.info(f"switch_hook_processor: Switching processor {processor_index} in {hook_type} to {new_processor_type}")
            
            # Use proper hook configuration access pattern (same as ControlNet)
            current_hooks = app_instance._get_current_hook_config(hook_type)
            
            if processor_index >= len(current_hooks):
                raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
            
            # Update the processor type and reset params
            current_hooks[processor_index]['type'] = new_processor_type
            current_hooks[processor_index]['params'] = {}
            
            # Update using the standard parameter update mechanism
            _update_pipeline_hook_config(app_instance, hook_type, current_hooks, "switch_hook_processor")
        
        logging.info(f"switch_hook_processor: Successfully switched processor {processor_index} in {hook_type} to {new_processor_type}")
        
        return create_success_response(f"Switched processor {processor_index} in {hook_type} to {new_processor_type}")
        
    except Exception as e:
        raise handle_api_error(e, "switch_hook_processor")

@router.post("/pipeline-hooks/{hook_type}/update-params")
async def update_hook_processor_params(hook_type: str, request: Request, app_instance=Depends(get_app_instance)):
    """Update parameters for a specific processor"""
    try:
        logging.info(f"update_hook_processor_params: ===== STARTING {hook_type} REQUEST =====")
        data = await request.json()
        logging.info(f"update_hook_processor_params: Received data: {data}")
        
        processor_index = data.get("processor_index")
        processor_params = data.get("processor_params", {})
        logging.info(f"update_hook_processor_params: processor_index={processor_index}, processor_params={processor_params}")
        
        if processor_index is None:
            logging.error(f"update_hook_processor_params: Missing processor_index parameter")
            raise HTTPException(status_code=400, detail="Missing processor_index parameter")
        
        validate_pipeline(app_instance.pipeline, "update_hook_processor_params")
        
        logging.info(f"update_hook_processor_params: Updating params for processor {processor_index} in {hook_type}")
        
        # Use proper hook configuration access pattern (same as ControlNet)
        current_hooks = app_instance._get_current_hook_config(hook_type)
        logging.info(f"update_hook_processor_params: Current hooks config: {current_hooks}")
        
        if not current_hooks:
            logging.error(f"update_hook_processor_params: Hook type {hook_type} not found or empty")
            raise HTTPException(status_code=400, detail=f"No processors configured for {hook_type}. Add a processor first using the 'Add {hook_type.replace('_', ' ').title()} Processor' button.")
            
        if processor_index >= len(current_hooks):
            logging.error(f"update_hook_processor_params: Processor index {processor_index} out of range for {hook_type} (max: {len(current_hooks)-1})")
            raise HTTPException(status_code=400, detail=f"Processor index {processor_index} not found. Only {len(current_hooks)} processors are configured for {hook_type}.")
        
        # Update the processor parameters
        logging.info(f"update_hook_processor_params: Current processor config: {current_hooks[processor_index]}")
        
        # Handle 'enabled' field separately as it's a top-level processor field, not a parameter
        if 'enabled' in processor_params:
            enabled_value = processor_params.pop('enabled')  # Remove from params dict
            current_hooks[processor_index]['enabled'] = bool(enabled_value)
            logging.info(f"update_hook_processor_params: Updated enabled field to: {enabled_value}")
        
        # Update remaining parameters in the params field
        if processor_params:  # Only update if there are remaining params
            current_hooks[processor_index]['params'].update(processor_params)
        
        logging.info(f"update_hook_processor_params: Updated processor config: {current_hooks[processor_index]}")
        
        # Update using the standard parameter update mechanism
        update_kwargs = {f"{hook_type}_config": current_hooks}
        logging.info(f"update_hook_processor_params: Calling update_stream_params with: {update_kwargs}")
        app_instance.pipeline.update_stream_params(**update_kwargs)
        logging.info(f"update_hook_processor_params: update_stream_params completed successfully")
        
        logging.info(f"update_hook_processor_params: Successfully updated params for processor {processor_index} in {hook_type}")
        
        return create_success_response(f"Updated parameters for processor {processor_index} in {hook_type}", updated_params=processor_params)
        
    except Exception as e:
        logging.exception(f"update_hook_processor_params: Exception occurred: {str(e)}")
        logging.error(f"update_hook_processor_params: Exception type: {type(e).__name__}")
        raise handle_api_error(e, "update_hook_processor_params")

@router.get("/pipeline-hooks/{hook_type}/current-params/{processor_index}")
async def get_current_hook_processor_params(hook_type: str, processor_index: int, app_instance=Depends(get_app_instance)):
    """Get current parameters for a specific processor"""
    try:
        # First try to get from uploaded config if no pipeline
        if not app_instance.pipeline and app_instance.uploaded_controlnet_config:
            hook_config = app_instance.uploaded_controlnet_config.get(hook_type, {})
            processors = hook_config.get("processors", [])
            if processor_index < len(processors):
                processor = processors[processor_index]
                return JSONResponse({
                    "status": "success",
                    "hook_type": hook_type,
                    "processor_index": processor_index,
                    "processor_type": processor.get('type', 'unknown'),
                    "parameters": processor.get('params', {}),
                    "enabled": processor.get('enabled', True),
                    "note": "From uploaded config"
                })
        
        # Return empty if no config available
        if not app_instance.pipeline:
            return JSONResponse({
                "status": "success",
                "hook_type": hook_type,
                "processor_index": processor_index,
                "processor_type": "unknown",
                "parameters": {},
                "enabled": False,
                "note": "Pipeline not initialized - no config available"
            })
        
        validate_pipeline(app_instance.pipeline, "get_current_hook_processor_params")
        
        # Use proper hook configuration access pattern (same as ControlNet)
        current_hooks = app_instance._get_current_hook_config(hook_type)
        
        if processor_index >= len(current_hooks):
            raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
        
        processor = current_hooks[processor_index]
        
        return JSONResponse({
            "status": "success",
            "hook_type": hook_type,
            "processor_index": processor_index,
            "processor_type": processor.get('type', 'unknown'),
            "parameters": processor.get('params', {}),
            "enabled": processor.get('enabled', True)
        })
        
    except Exception as e:
        raise handle_api_error(e, "get_current_hook_processor_params")