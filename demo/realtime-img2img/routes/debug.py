"""
Debug mode API endpoints for realtime-img2img
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from .common.dependencies import get_app_instance

router = APIRouter(prefix="/api/debug", tags=["debug"])

class DebugResponse(BaseModel):
    success: bool
    message: str
    debug_mode: bool
    debug_pending_frame: bool = False

@router.post("/enable", response_model=DebugResponse)
async def enable_debug_mode(app_instance=Depends(get_app_instance)):
    """Enable debug mode - pauses automatic frame processing"""
    try:
        app_instance.app_state.debug_mode = True
        app_instance.app_state.debug_pending_frame = False
        
        logging.info("enable_debug_mode: Debug mode enabled")
        
        return DebugResponse(
            success=True,
            message="Debug mode enabled. Frame processing is now paused.",
            debug_mode=True,
            debug_pending_frame=False
        )
    except Exception as e:
        logging.exception(f"enable_debug_mode: Failed to enable debug mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable debug mode: {str(e)}")

@router.post("/disable", response_model=DebugResponse)
async def disable_debug_mode(app_instance=Depends(get_app_instance)):
    """Disable debug mode - resumes automatic frame processing"""
    try:
        app_instance.app_state.debug_mode = False
        app_instance.app_state.debug_pending_frame = False
        
        logging.info("disable_debug_mode: Debug mode disabled")
        
        return DebugResponse(
            success=True,
            message="Debug mode disabled. Automatic frame processing resumed.",
            debug_mode=False,
            debug_pending_frame=False
        )
    except Exception as e:
        logging.exception(f"disable_debug_mode: Failed to disable debug mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable debug mode: {str(e)}")

@router.post("/step", response_model=DebugResponse)
async def step_frame(app_instance=Depends(get_app_instance)):
    """Process exactly one frame when in debug mode"""
    try:
        if not app_instance.app_state.debug_mode:
            raise HTTPException(status_code=400, detail="Debug mode is not enabled")
        
        # Set pending frame flag to allow one frame to be processed
        app_instance.app_state.debug_pending_frame = True
        
        logging.info("step_frame: Frame step requested")
        
        return DebugResponse(
            success=True,
            message="Frame step requested. Next frame will be processed.",
            debug_mode=True,
            debug_pending_frame=True
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"step_frame: Failed to step frame: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to step frame: {str(e)}")

@router.get("/status", response_model=DebugResponse)
async def get_debug_status(app_instance=Depends(get_app_instance)):
    """Get current debug mode status"""
    try:
        return DebugResponse(
            success=True,
            message="Debug status retrieved",
            debug_mode=app_instance.app_state.debug_mode,
            debug_pending_frame=app_instance.app_state.debug_pending_frame
        )
    except Exception as e:
        logging.exception(f"get_debug_status: Failed to get debug status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get debug status: {str(e)}")
