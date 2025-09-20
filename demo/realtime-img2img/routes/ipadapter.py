"""
IPAdapter-related endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
import logging
import os

from .common.api_utils import handle_api_request, create_success_response, handle_api_error, validate_pipeline, validate_feature_enabled, validate_config_mode
from .common.dependencies import get_app_instance

router = APIRouter(prefix="/api", tags=["ipadapter"])

# Legacy upload endpoint removed - use /api/input-sources/upload-image/ipadapter instead

# Legacy get uploaded image endpoint removed - use InputSourceManager instead

@router.get("/default-image")
async def get_default_image():
    """Get the default image (input.png)"""
    try:
        default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "images", "inputs", "input.png")
        
        if not os.path.exists(default_image_path):
            raise HTTPException(status_code=404, detail="Default image not found")
        
        # Read and return the default image file
        with open(default_image_path, "rb") as image_file:
            image_content = image_file.read()
        
        return Response(content=image_content, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
        
    except Exception as e:
        raise handle_api_error(e, "get_default_image")

@router.post("/ipadapter/update-scale")
async def update_ipadapter_scale(request: Request, app_instance=Depends(get_app_instance)):
    """Update IPAdapter scale/strength in real-time"""
    try:
        data = await handle_api_request(request, "update_ipadapter_scale", ["scale"])
        scale = data.get("scale")
        
        validate_pipeline(app_instance.pipeline, "update_ipadapter_scale")
        validate_config_mode(app_instance.pipeline, "ipadapters")
        
        # Update IPAdapter scale in the pipeline
        success = app_instance.pipeline.update_ipadapter_scale(float(scale))
        
        if success:
            return create_success_response(f"Updated IPAdapter scale to {scale}")
        else:
            raise HTTPException(status_code=500, detail="Failed to update scale in pipeline")
        
    except Exception as e:
        raise handle_api_error(e, "update_ipadapter_scale")

@router.post("/ipadapter/update-weight-type")
async def update_ipadapter_weight_type(request: Request, app_instance=Depends(get_app_instance)):
    """Update IPAdapter weight type in real-time"""
    try:
        data = await handle_api_request(request, "update_ipadapter_weight_type", ["weight_type"])
        weight_type = data.get("weight_type")
        
        validate_pipeline(app_instance.pipeline, "update_ipadapter_weight_type")
        validate_config_mode(app_instance.pipeline, "ipadapters")
        
        # Update IPAdapter weight type in the pipeline
        success = app_instance.pipeline.update_ipadapter_weight_type(weight_type)
        
        if success:
            return create_success_response(f"Updated IPAdapter weight type to {weight_type}")
        else:
            raise HTTPException(status_code=500, detail="Failed to update weight type in pipeline")
        
    except Exception as e:
        raise handle_api_error(e, "update_ipadapter_weight_type")

@router.post("/ipadapter/update-enabled")
async def update_ipadapter_enabled(request: Request, app_instance=Depends(get_app_instance)):
    """Enable or disable IPAdapter in real-time"""
    try:
        data = await handle_api_request(request, "update_ipadapter_enabled", ["enabled"])
        enabled = data.get("enabled")
        
        validate_pipeline(app_instance.pipeline, "update_ipadapter_enabled")
        validate_config_mode(app_instance.pipeline, "ipadapters")
        
        # Update IPAdapter enabled state in the pipeline
        app_instance.pipeline.stream.update_stream_params(
            ipadapter_config={'enabled': bool(enabled)}
        )
        
        return create_success_response(f"IPAdapter {'enabled' if enabled else 'disabled'} successfully")
        
    except Exception as e:
        raise handle_api_error(e, "update_ipadapter_enabled")

