"""
Utility functions for API endpoints to reduce code duplication
"""

import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional


async def handle_api_request(
    request: Request,
    operation_name: str,
    required_params: list = None,
    pipeline_required: bool = True
) -> Dict[str, Any]:
    """
    Standard request handler for API endpoints
    
    Args:
        request: FastAPI request object
        operation_name: Name of the operation for logging
        required_params: List of required parameter names
        pipeline_required: Whether an active pipeline is required
        
    Returns:
        Parsed JSON data from request
        
    Raises:
        HTTPException: For validation errors
    """
    try:
        data = await request.json()
        
        # Check required parameters
        if required_params:
            missing_params = [param for param in required_params if param not in data]
            if missing_params:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required parameters: {', '.join(missing_params)}"
                )
        
        return data
        
    except Exception as e:
        logging.error(f"{operation_name}: Failed to parse request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")


def create_success_response(message: str, **extra_data) -> JSONResponse:
    """
    Create a standardized success response
    
    Args:
        message: Success message
        **extra_data: Additional data to include in response
        
    Returns:
        JSONResponse with standardized format
    """
    response_data = {
        "status": "success",
        "message": message
    }
    response_data.update(extra_data)
    return JSONResponse(response_data)


def handle_api_error(error: Exception, operation_name: str, status_code: int = 500) -> HTTPException:
    """
    Standard error handler for API endpoints
    
    Args:
        error: The caught exception
        operation_name: Name of the operation for logging
        status_code: HTTP status code to return
        
    Returns:
        HTTPException with standardized error message
    """
    logging.error(f"{operation_name}: Failed: {error}")
    return HTTPException(
        status_code=status_code, 
        detail=f"Failed to {operation_name.lower()}: {str(error)}"
    )


def validate_pipeline(pipeline: Any, operation_name: str) -> None:
    """
    Validate that pipeline exists and is initialized
    
    Args:
        pipeline: Pipeline object to validate
        operation_name: Name of the operation for error messages
        
    Raises:
        HTTPException: If pipeline is not valid
    """
    if not pipeline:
        raise HTTPException(
            status_code=400, 
            detail="Pipeline is not initialized"
        )


def validate_feature_enabled(pipeline: Any, feature_name: str, feature_check_attr: str) -> None:
    """
    Validate that a specific feature is enabled in the pipeline
    
    Args:
        pipeline: Pipeline object
        feature_name: Human-readable feature name (e.g., "ControlNet", "IPAdapter")
        feature_check_attr: Attribute name to check (e.g., "has_controlnet", "has_ipadapter")
        
    Raises:
        HTTPException: If feature is not enabled
    """
    if not getattr(pipeline, feature_check_attr, False):
        raise HTTPException(
            status_code=400, 
            detail=f"{feature_name} is not enabled"
        )


def validate_config_mode(pipeline: Any, config_check: Optional[str] = None) -> None:
    """
    Validate that pipeline is using config mode
    
    Args:
        pipeline: Pipeline object
        config_check: Optional specific config key to check for existence
        
    Raises:
        HTTPException: If not in config mode or config key missing
    """
    if not (pipeline.use_config and pipeline.config):
        raise HTTPException(
            status_code=400, 
            detail="Pipeline is not using configuration mode"
        )
    
    if config_check and config_check not in pipeline.config:
        raise HTTPException(
            status_code=400, 
            detail=f"Configuration missing required section: {config_check}"
        ) 