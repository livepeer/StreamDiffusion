
"""
Pipeline hooks endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from .common.api_utils import handle_api_request, create_success_response, handle_api_error
from .common.dependencies import get_app_instance

router = APIRouter(prefix="/api", tags=["pipeline-hooks"])


# Pipeline Hooks API Endpoints
@router.get("/pipeline-hooks/info-config")
async def get_pipeline_hooks_info_config(app_instance=Depends(get_app_instance)):
    """Get pipeline hooks configuration info"""
    try:
        # SINGLE SOURCE OF TRUTH - Return hooks info from AppState
        hooks_info = {
            "image_preprocessing": app_instance.app_state.pipeline_hooks["image_preprocessing"],
            "image_postprocessing": app_instance.app_state.pipeline_hooks["image_postprocessing"],
            "latent_preprocessing": app_instance.app_state.pipeline_hooks["latent_preprocessing"],
            "latent_postprocessing": app_instance.app_state.pipeline_hooks["latent_postprocessing"]
        }
        return JSONResponse(hooks_info)
    except Exception as e:
        raise handle_api_error(e, "get_pipeline_hooks_info_config")

# Individual hook type endpoints that frontend expects
@router.get("/pipeline-hooks/image_preprocessing/info-config")
async def get_image_preprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get image preprocessing hook configuration info - SINGLE SOURCE OF TRUTH"""
    try:
        hook_info = app_instance.app_state.pipeline_hooks["image_preprocessing"]
        return JSONResponse({"image_preprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"image_preprocessing": None})

@router.get("/pipeline-hooks/image_postprocessing/info-config")
async def get_image_postprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get image postprocessing hook configuration info - SINGLE SOURCE OF TRUTH"""
    try:
        hook_info = app_instance.app_state.pipeline_hooks["image_postprocessing"]
        return JSONResponse({"image_postprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"image_postprocessing": None})

@router.get("/pipeline-hooks/latent_preprocessing/info-config")
async def get_latent_preprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get latent preprocessing hook configuration info - SINGLE SOURCE OF TRUTH"""
    try:
        hook_info = app_instance.app_state.pipeline_hooks["latent_preprocessing"]
        return JSONResponse({"latent_preprocessing": hook_info})
    except Exception as e:
        return JSONResponse({"latent_preprocessing": None})

@router.get("/pipeline-hooks/latent_postprocessing/info-config")
async def get_latent_postprocessing_info_config(app_instance=Depends(get_app_instance)):
    """Get latent postprocessing hook configuration info - SINGLE SOURCE OF TRUTH"""
    try:
        hook_info = app_instance.app_state.pipeline_hooks["latent_postprocessing"]
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
        
        # No pipeline validation needed - AppState updates work before pipeline creation
        
        if hook_type not in ["image_preprocessing", "image_postprocessing", "latent_preprocessing", "latent_postprocessing"]:
            raise HTTPException(status_code=400, detail=f"Invalid hook type: {hook_type}")
        
        logging.debug(f"add_hook_processor: Adding {processor_type} to {hook_type}")
        
        # Create processor config
        new_processor = {
            "type": processor_type,
            "params": processor_params,
            "enabled": True
        }
        
        # Add to AppState - SINGLE SOURCE OF TRUTH
        app_instance.app_state.add_hook_processor(hook_type, new_processor)
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                hook_config = []
                for processor in app_instance.app_state.pipeline_hooks[hook_type]["processors"]:
                    config_entry = {
                        "type": processor["type"],
                        "params": processor["params"],
                        "enabled": processor["enabled"]
                    }
                    hook_config.append(config_entry)
                update_kwargs = {f"{hook_type}_config": hook_config}
                app_instance.pipeline.update_stream_params(**update_kwargs)
            except Exception as e:
                logging.exception(f"add_hook_processor: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
        logging.info(f"add_hook_processor: Successfully added {processor_type} to {hook_type}")
        
        return create_success_response(f"Added {processor_type} processor to {hook_type}")
        
    except Exception as e:
        raise handle_api_error(e, "add_hook_processor")

@router.delete("/pipeline-hooks/{hook_type}/remove/{processor_index}")
async def remove_hook_processor(hook_type: str, processor_index: int, app_instance=Depends(get_app_instance)):
    """Remove a processor from a hook"""
    try:
        # No pipeline validation needed - AppState updates work before pipeline creation
        
        logging.debug(f"remove_hook_processor: Removing processor {processor_index} from {hook_type}")
        
        # Remove from AppState - SINGLE SOURCE OF TRUTH
        app_instance.app_state.remove_hook_processor(hook_type, processor_index)
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                hook_config = []
                for processor in app_instance.app_state.pipeline_hooks[hook_type]["processors"]:
                    config_entry = {
                        "type": processor["type"],
                        "params": processor["params"],
                        "enabled": processor["enabled"]
                    }
                    hook_config.append(config_entry)
                update_kwargs = {f"{hook_type}_config": hook_config}
                app_instance.pipeline.update_stream_params(**update_kwargs)
            except Exception as e:
                logging.exception(f"remove_hook_processor: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
        logging.info(f"remove_hook_processor: Successfully removed processor {processor_index} from {hook_type}")
        
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
        
        # No pipeline validation needed - AppState updates work before pipeline creation
        
        logging.debug(f"toggle_hook_processor: Toggling processor {processor_index} in {hook_type} to {'enabled' if enabled else 'disabled'}")
        
        # Update AppState - SINGLE SOURCE OF TRUTH
        app_instance.app_state.update_hook_processor(hook_type, processor_index, {"enabled": bool(enabled)})
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                hook_config = []
                for processor in app_instance.app_state.pipeline_hooks[hook_type]["processors"]:
                    config_entry = {
                        "type": processor["type"],
                        "params": processor["params"],
                        "enabled": processor["enabled"]
                    }
                    hook_config.append(config_entry)
                update_kwargs = {f"{hook_type}_config": hook_config}
                app_instance.pipeline.update_stream_params(**update_kwargs)
            except Exception as e:
                logging.exception(f"toggle_hook_processor: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
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
            if not app_instance.app_state.uploaded_config:
                raise HTTPException(status_code=400, detail="No pipeline active and no uploaded config available")
            
            logging.info(f"switch_hook_processor: Updating config for {hook_type} processor {processor_index} to {new_processor_type}")
            
            # Update the uploaded config directly
            hook_config = app_instance.app_state.uploaded_config.get(hook_type, {"enabled": False, "processors": []})
            if processor_index >= len(hook_config.get("processors", [])):
                raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
            
            # Update processor type in config
            hook_config["processors"][processor_index]["type"] = new_processor_type
            hook_config["processors"][processor_index]["params"] = {}
            app_instance.app_state.uploaded_config[hook_type] = hook_config
            
        else:
            # No pipeline validation needed - AppState updates work before pipeline creation
            
            logging.debug(f"switch_hook_processor: Switching processor {processor_index} in {hook_type} to {new_processor_type}")
            
            # Update AppState - SINGLE SOURCE OF TRUTH
            processors = app_instance.app_state.pipeline_hooks[hook_type]["processors"]
            
            if processor_index >= len(processors):
                raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
            
            # Update the processor type and reset params in AppState
            app_instance.app_state.update_hook_processor(hook_type, processor_index, {
                "type": new_processor_type,
                "name": new_processor_type,
                "params": {}
            })
            
            # Update pipeline if active
            if app_instance.pipeline:
                try:
                    hook_config = []
                    for processor in app_instance.app_state.pipeline_hooks[hook_type]["processors"]:
                        config_entry = {
                            "type": processor["type"],
                            "params": processor["params"],
                            "enabled": processor["enabled"]
                        }
                        hook_config.append(config_entry)
                    update_kwargs = {f"{hook_type}_config": hook_config}
                    app_instance.pipeline.update_stream_params(**update_kwargs)
                except Exception as e:
                    logging.exception(f"switch_hook_processor: Failed to update pipeline: {e}")
                    # Mark for reload as fallback
                    app_instance.app_state.config_needs_reload = True
        
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
        
        # No pipeline validation needed - AppState updates work before pipeline creation
        
        logging.debug(f"update_hook_processor_params: Updating params for processor {processor_index} in {hook_type}")
        
        # Check if processors exist in AppState
        processors = app_instance.app_state.pipeline_hooks[hook_type]["processors"]
        if not processors:
            logging.error(f"update_hook_processor_params: Hook type {hook_type} not found or empty")
            raise HTTPException(status_code=400, detail=f"No processors configured for {hook_type}. Add a processor first using the 'Add {hook_type.replace('_', ' ').title()} Processor' button.")
            
        if processor_index >= len(processors):
            logging.error(f"update_hook_processor_params: Processor index {processor_index} out of range for {hook_type} (max: {len(processors)-1})")
            raise HTTPException(status_code=400, detail=f"Processor index {processor_index} not found. Only {len(processors)} processors are configured for {hook_type}.")
        
        # Update the processor parameters in AppState - SINGLE SOURCE OF TRUTH
        logging.info(f"update_hook_processor_params: Current processor config: {processors[processor_index]}")
        
        # Handle 'enabled' field separately as it's a top-level processor field, not a parameter
        updates = {}
        if 'enabled' in processor_params:
            enabled_value = processor_params.pop('enabled')  # Remove from params dict
            updates['enabled'] = bool(enabled_value)
            logging.info(f"update_hook_processor_params: Updated enabled field to: {enabled_value}")
        
        # Update remaining parameters in the params field
        if processor_params:  # Only update if there are remaining params
            current_params = processors[processor_index].get('params', {})
            current_params.update(processor_params)
            updates['params'] = current_params
        
        # Apply updates to AppState
        app_instance.app_state.update_hook_processor(hook_type, processor_index, updates)
        
        # Update pipeline if active
        if app_instance.pipeline:
            try:
                hook_config = []
                for processor in app_instance.app_state.pipeline_hooks[hook_type]["processors"]:
                    config_entry = {
                        "type": processor["type"],
                        "params": processor["params"],
                        "enabled": processor["enabled"]
                    }
                    hook_config.append(config_entry)
                update_kwargs = {f"{hook_type}_config": hook_config}
                logging.info(f"update_hook_processor_params: Calling update_stream_params with: {update_kwargs}")
                app_instance.pipeline.update_stream_params(**update_kwargs)
                logging.info(f"update_hook_processor_params: update_stream_params completed successfully")
            except Exception as e:
                logging.exception(f"update_hook_processor_params: Failed to update pipeline: {e}")
                # Mark for reload as fallback
                app_instance.app_state.config_needs_reload = True
        
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
        if not app_instance.pipeline and app_instance.app_state.uploaded_config:
            hook_config = app_instance.app_state.uploaded_config.get(hook_type, {})
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
        
        # Use AppState - SINGLE SOURCE OF TRUTH
        processors = app_instance.app_state.pipeline_hooks[hook_type]["processors"]
        
        if processor_index >= len(processors):
            raise HTTPException(status_code=400, detail=f"Invalid processor index {processor_index} for {hook_type}")
        
        processor = processors[processor_index]
        
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