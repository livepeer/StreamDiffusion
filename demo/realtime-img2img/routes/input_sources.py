"""
Input Source Management API endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from typing import Optional, Any, Dict
import uuid
from PIL import Image
import io

from .common.api_utils import handle_api_request, create_success_response, handle_api_error
from .common.dependencies import get_app_instance
from input_sources import InputSource, InputSourceType, InputSourceManager
from utils.video_utils import validate_video_file, is_supported_video_format

router = APIRouter(prefix="/api/input-sources", tags=["input-sources"])

logger = logging.getLogger("input_sources_api")


def _get_input_source_manager(app_instance) -> InputSourceManager:
    """Get or create the input source manager for the app instance."""
    if not hasattr(app_instance, 'input_source_manager'):
        app_instance.input_source_manager = InputSourceManager()
    return app_instance.input_source_manager


@router.post("/set")
async def set_input_source(request: Request, app_instance=Depends(get_app_instance)):
    """
    Set input source for a component.
    
    Body: {
        component: str,  # 'controlnet', 'ipadapter', 'base'
        index?: int,     # Required for controlnet
        source_type: str, # 'webcam', 'uploaded_image', 'uploaded_video'
        source_data?: any # Path, image data, etc.
    }
    """
    try:
        data = await handle_api_request(request, "set_input_source", 
                                      required_params=['component', 'source_type'],
                                      pipeline_required=False)
        
        component = data['component']
        source_type_str = data['source_type']
        index = data.get('index')
        source_data = data.get('source_data')
        
        # Validate component
        if component not in ['controlnet', 'ipadapter', 'base']:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        # Validate source type
        try:
            source_type = InputSourceType(source_type_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source type: {source_type_str}")
        
        # Validate index for controlnet
        if component == 'controlnet' and index is None:
            raise HTTPException(status_code=400, detail="Index is required for ControlNet components")
        
        # Get input source manager
        manager = _get_input_source_manager(app_instance)
        
        # Create input source
        input_source = InputSource(source_type, source_data)
        
        # Set the source
        manager.set_source(component, input_source, index)
        
        logger.info(f"set_input_source: Set {component} input source to {source_type_str}")
        
        return create_success_response({
            'message': f'Input source set for {component}',
            'component': component,
            'source_type': source_type_str,
            'index': index
        })
        
    except Exception as e:
        return handle_api_error(e, "set_input_source")


@router.post("/upload-image/{component}")
async def upload_component_image(
    component: str, 
    file: UploadFile = File(...),
    index: Optional[int] = None,
    app_instance=Depends(get_app_instance)
):
    """
    Upload image for specific component.
    
    Args:
        component: Component name ('controlnet', 'ipadapter', 'base')
        file: Image file to upload
        index: Index for ControlNet (required for controlnet component)
    """
    try:
        # Validate component
        if component not in ['controlnet', 'ipadapter', 'base']:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        # Validate index for controlnet
        if component == 'controlnet' and index is None:
            raise HTTPException(status_code=400, detail="Index is required for ControlNet components")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix if file.filename else '.jpg'
        filename = f"{component}_{index}_{file_id}{file_extension}" if index is not None else f"{component}_{file_id}{file_extension}"
        
        # Save file
        uploads_dir = Path("uploads/images")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        file_path = uploads_dir / filename
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Create PIL Image for input source
        try:
            image = Image.open(io.BytesIO(content))
            image = image.convert('RGB')  # Ensure RGB format
        except Exception as e:
            # Clean up file if image processing fails
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Get input source manager and set source
        manager = _get_input_source_manager(app_instance)
        input_source = InputSource(InputSourceType.UPLOADED_IMAGE, image)
        manager.set_source(component, input_source, index)
        
        logger.info(f"upload_component_image: Uploaded image for {component} (index: {index})")
        
        return create_success_response({
            'message': f'Image uploaded for {component}',
            'component': component,
            'index': index,
            'filename': filename,
            'file_path': str(file_path)
        })
        
    except Exception as e:
        return handle_api_error(e, "upload_component_image")


@router.post("/upload-video/{component}")
async def upload_component_video(
    component: str,
    file: UploadFile = File(...),
    index: Optional[int] = None,
    app_instance=Depends(get_app_instance)
):
    """
    Upload video for specific component.
    
    Args:
        component: Component name ('controlnet', 'ipadapter', 'base')
        file: Video file to upload
        index: Index for ControlNet (required for controlnet component)
    """
    try:
        # Validate component
        if component not in ['controlnet', 'ipadapter', 'base']:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        # Validate index for controlnet
        if component == 'controlnet' and index is None:
            raise HTTPException(status_code=400, detail="Index is required for ControlNet components")
        
        # Validate file type
        if not file.filename or not is_supported_video_format(file.filename):
            raise HTTPException(status_code=400, detail="File must be a supported video format")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{component}_{index}_{file_id}{file_extension}" if index is not None else f"{component}_{file_id}{file_extension}"
        
        # Save file
        uploads_dir = Path("uploads/videos")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        file_path = uploads_dir / filename
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Validate video file
        is_valid, error_msg = validate_video_file(str(file_path))
        if not is_valid:
            # Clean up file if validation fails
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid video file: {error_msg}")
        
        # Get input source manager and set source
        manager = _get_input_source_manager(app_instance)
        input_source = InputSource(InputSourceType.UPLOADED_VIDEO, str(file_path))
        manager.set_source(component, input_source, index)
        
        logger.info(f"upload_component_video: Uploaded video for {component} (index: {index})")
        
        return create_success_response({
            'message': f'Video uploaded for {component}',
            'component': component,
            'index': index,
            'filename': filename,
            'file_path': str(file_path)
        })
        
    except Exception as e:
        return handle_api_error(e, "upload_component_video")


@router.get("/info/{component}")
async def get_component_source_info(
    component: str,
    index: Optional[int] = None,
    app_instance=Depends(get_app_instance)
):
    """
    Get information about a component's input source.
    
    Args:
        component: Component name ('controlnet', 'ipadapter', 'base')
        index: Index for ControlNet (required for controlnet component)
    """
    try:
        # Validate component
        if component not in ['controlnet', 'ipadapter', 'base']:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        # Validate index for controlnet
        if component == 'controlnet' and index is None:
            raise HTTPException(status_code=400, detail="Index is required for ControlNet components")
        
        # Get input source manager
        manager = _get_input_source_manager(app_instance)
        
        # Get source info
        source_info = manager.get_source_info(component, index)
        
        return create_success_response({
            'component': component,
            'index': index,
            'source_info': source_info
        })
        
    except Exception as e:
        return handle_api_error(e, "get_component_source_info")


@router.get("/list")
async def list_all_source_info(app_instance=Depends(get_app_instance)):
    """Get information about all configured input sources."""
    try:
        # Get input source manager
        manager = _get_input_source_manager(app_instance)
        
        # Collect all source information
        all_sources = {
            'base': manager.get_source_info('base'),
            'ipadapter': manager.get_source_info('ipadapter'),
            'controlnets': {}
        }
        
        # Get all controlnet sources
        for index, source in manager.sources['controlnet'].items():
            all_sources['controlnets'][index] = manager.get_source_info('controlnet', index)
        
        return create_success_response({
            'sources': all_sources
        })
        
    except Exception as e:
        return handle_api_error(e, "list_all_source_info")


@router.post("/reset/{component}")
async def reset_component_source(
    component: str,
    index: Optional[int] = None,
    app_instance=Depends(get_app_instance)
):
    """
    Reset a component's input source to webcam (default).
    
    Args:
        component: Component name ('controlnet', 'ipadapter', 'base')
        index: Index for ControlNet (required for controlnet component)
    """
    try:
        # Validate component
        if component not in ['controlnet', 'ipadapter', 'base']:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        # Validate index for controlnet
        if component == 'controlnet' and index is None:
            raise HTTPException(status_code=400, detail="Index is required for ControlNet components")
        
        # Get input source manager
        manager = _get_input_source_manager(app_instance)
        
        # Create webcam input source
        webcam_source = InputSource(InputSourceType.WEBCAM)
        manager.set_source(component, webcam_source, index)
        
        logger.info(f"reset_component_source: Reset {component} to webcam (index: {index})")
        
        return create_success_response({
            'message': f'Input source reset to webcam for {component}',
            'component': component,
            'index': index,
            'source_type': 'webcam'
        })
        
    except Exception as e:
        return handle_api_error(e, "reset_component_source")


@router.post("/reset-all")
async def reset_all_input_sources(app_instance=Depends(get_app_instance)):
    """
    Reset all input sources to their default states.
    This is typically called when a new config is uploaded.
    """
    try:
        # Get input source manager
        manager = _get_input_source_manager(app_instance)
        
        # Reset all sources to defaults
        manager.reset_to_defaults()
        
        logger.info("reset_all_input_sources: Reset all input sources to defaults")
        
        return create_success_response({
            'message': 'All input sources reset to defaults',
            'defaults': {
                'base': 'webcam',
                'ipadapter': 'uploaded_image (default image)',
                'controlnet': 'fallback to base pipeline'
            }
        })
        
    except Exception as e:
        return handle_api_error(e, "reset_all_input_sources")
