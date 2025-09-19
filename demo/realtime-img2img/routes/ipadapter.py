"""
IPAdapter-related endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends, UploadFile, File, Response
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import tempfile
import os
import io

from .common.api_utils import handle_api_request, create_success_response, handle_api_error, validate_pipeline, validate_feature_enabled, validate_config_mode
from .common.dependencies import get_app_instance

router = APIRouter(prefix="/api", tags=["ipadapter"])

@router.post("/ipadapter/upload-style-image")
async def upload_style_image(file: UploadFile = File(...), app_instance=Depends(get_app_instance)):
    """Upload a style image for IPAdapter"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Load and validate image
            from PIL import Image
            style_image = Image.open(tmp_path).convert("RGB")

            # Store the uploaded style image persistently FIRST
            app_instance.uploaded_style_image = style_image
            logging.info(f"upload_style_image: Stored style image with size: {style_image.size}")

            # If pipeline exists and has IPAdapter, update it immediately
            pipeline_updated = False
            if app_instance.pipeline and getattr(app_instance.pipeline, 'has_ipadapter', False):
                logging.info("upload_style_image: Applying to existing pipeline")
                success = app_instance.pipeline.update_ipadapter_style_image(style_image)
                if success:
                    pipeline_updated = True
                    logging.info("upload_style_image: Successfully applied to existing pipeline")

                    # Force prompt re-encoding to apply new style image embeddings
                    try:
                        state = app_instance.pipeline.stream.get_stream_state()
                        current_prompts = state.get('prompt_list', [])
                        if current_prompts:
                            logging.info("upload_style_image: Forcing prompt re-encoding to apply new style image")
                            app_instance.pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                            logging.info("upload_style_image: Prompt re-encoding completed")
                    except Exception as e:
                        logging.exception(f"upload_style_image: Failed to force prompt re-encoding: {e}")
                else:
                    logging.error("upload_style_image: Failed to apply to existing pipeline")
            elif app_instance.pipeline:
                logging.info(f"upload_style_image: Pipeline exists but has_ipadapter={getattr(app_instance.pipeline, 'has_ipadapter', False)}")
            else:
                logging.info("upload_style_image: No pipeline exists yet")

            # Return success
            message = "Style image uploaded successfully"
            if pipeline_updated:
                message += " and applied to active pipeline"
            else:
                message += " and will be applied when pipeline starts"
            
            return create_success_response(message)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_error(e, "upload_style_image")

@router.get("/ipadapter/uploaded-style-image")
async def get_uploaded_style_image(app_instance=Depends(get_app_instance)):
    """Get the currently uploaded style image"""
    try:
        if not app_instance.uploaded_style_image:
            raise HTTPException(status_code=404, detail="No style image uploaded")
        
        # Convert PIL image to bytes for streaming
        img_buffer = io.BytesIO()
        app_instance.uploaded_style_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_buffer.read()),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        raise handle_api_error(e, "get_uploaded_style_image")

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

