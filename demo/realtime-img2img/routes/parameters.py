"""
Parameter update endpoints for realtime-img2img
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from .common.api_utils import handle_api_request, create_success_response, handle_api_error, validate_pipeline
from .common.dependencies import get_app_instance

router = APIRouter(prefix="/api", tags=["parameters"])

@router.post("/params")
async def update_params(request: Request, app_instance=Depends(get_app_instance)):
    """Update multiple streaming parameters in a single unified call"""
    try:
        data = await request.json()
        
        # Allow updating resolution even when pipeline is not initialized.
        # We save the new values so they take effect on the next stream start.
        if "resolution" in data and not app_instance.pipeline:
            resolution = data["resolution"]
            if isinstance(resolution, dict) and "width" in resolution and "height" in resolution:
                width, height = int(resolution["width"]), int(resolution["height"])
                app_instance.new_width = width
                app_instance.new_height = height
            elif isinstance(resolution, str):
                # Handle string format like "512x768 (2:3)" or "512x768"
                resolution_part = resolution.split(' ')[0]
                try:
                    width, height = map(int, resolution_part.split('x'))
                    app_instance.new_width = width
                    app_instance.new_height = height
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid resolution format")
            else:
                raise HTTPException(status_code=400, detail="Resolution must be {width: int, height: int} or 'widthxheight' string")
            return JSONResponse({
                "status": "success",
                "message": f"Updated resolution to {app_instance.new_width}x{app_instance.new_height} (will apply on next stream start)"
            })

        if not app_instance.pipeline:
            raise HTTPException(status_code=400, detail="Pipeline is not initialized")

        # Update parameters that exist in the data
        params = {}
        
        if "guidance_scale" in data:
            params["guidance_scale"] = float(data["guidance_scale"])
        if "delta" in data:
            params["delta"] = float(data["delta"])
        if "num_inference_steps" in data:
            params["num_inference_steps"] = int(data["num_inference_steps"])
        if "seed" in data:
            params["seed"] = int(data["seed"])
        if "t_index_list" in data:
            t_index_list = data["t_index_list"]
            if isinstance(t_index_list, list) and all(isinstance(x, int) for x in t_index_list):
                params["t_index_list"] = t_index_list
            else:
                raise HTTPException(status_code=400, detail="t_index_list must be a list of integers")

        if params:
            app_instance.pipeline.update_stream_params(**params)
            return JSONResponse({
                "status": "success",
                "message": f"Updated parameters: {list(params.keys())}",
                "updated_params": params
            })
        else:
            return JSONResponse({
                "status": "success",
                "message": "No valid parameters provided to update"
            })
        
    except Exception as e:
        logging.exception(f"update_params: Failed to update parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update parameters: {str(e)}")

async def _update_single_parameter(
    request: Request, 
    app_instance, 
    parameter_name: str, 
    value_converter: callable,
    operation_name: str
):
    """Generic function to update a single parameter"""
    try:
        data = await handle_api_request(request, operation_name, [parameter_name])
        validate_pipeline(app_instance.pipeline, operation_name)
        
        value = value_converter(data[parameter_name])
        app_instance.pipeline.update_stream_params(**{parameter_name: value})
        
        return create_success_response(f"Updated {parameter_name} to {value}", **{parameter_name: value})
        
    except Exception as e:
        raise handle_api_error(e, operation_name)

@router.post("/update-guidance-scale")
async def update_guidance_scale(request: Request, app_instance=Depends(get_app_instance)):
    """Update guidance scale parameter"""
    return await _update_single_parameter(
        request, app_instance, "guidance_scale", float, "update_guidance_scale"
    )

@router.post("/update-delta")
async def update_delta(request: Request, app_instance=Depends(get_app_instance)):
    """Update delta parameter"""
    return await _update_single_parameter(
        request, app_instance, "delta", float, "update_delta"
    )

@router.post("/update-num-inference-steps")
async def update_num_inference_steps(request: Request, app_instance=Depends(get_app_instance)):
    """Update number of inference steps parameter"""
    return await _update_single_parameter(
        request, app_instance, "num_inference_steps", int, "update_num_inference_steps"
    )

@router.post("/update-seed")
async def update_seed(request: Request, app_instance=Depends(get_app_instance)):
    """Update seed parameter"""
    return await _update_single_parameter(
        request, app_instance, "seed", int, "update_seed"
    )

@router.post("/blending")
async def update_blending(request: Request, app_instance=Depends(get_app_instance)):
    """Update prompt and/or seed blending configuration in real-time"""
    try:
        data = await request.json()
        
        validate_pipeline(app_instance.pipeline, "update_blending")
        
        params = {}
        updated_types = []
        
        # Handle prompt blending
        if "prompt_list" in data:
            prompt_list = data["prompt_list"]
            interpolation_method = data.get("prompt_interpolation_method", "slerp")
            
            if not isinstance(prompt_list, list):
                raise HTTPException(status_code=400, detail="prompt_list must be a list")
            
            # Validate and convert format
            prompt_tuples = []
            for item in prompt_list:
                if isinstance(item, list) and len(item) == 2:
                    prompt_tuples.append((str(item[0]), float(item[1])))
                elif isinstance(item, dict) and "prompt" in item and "weight" in item:
                    prompt_tuples.append((str(item["prompt"]), float(item["weight"])))
                else:
                    raise HTTPException(status_code=400, detail="Each prompt item must be [prompt, weight] or {prompt: str, weight: float}")
            
            params["prompt_list"] = prompt_tuples
            params["prompt_interpolation_method"] = interpolation_method
            updated_types.append("prompt")

        # Handle seed blending
        if "seed_list" in data:
            seed_list = data["seed_list"]
            interpolation_method = data.get("seed_interpolation_method", "linear")
            
            if not isinstance(seed_list, list):
                raise HTTPException(status_code=400, detail="seed_list must be a list")
            
            # Validate and convert format
            seed_tuples = []
            for item in seed_list:
                if isinstance(item, list) and len(item) == 2:
                    seed_tuples.append((int(item[0]), float(item[1])))
                elif isinstance(item, dict) and "seed" in item and "weight" in item:
                    seed_tuples.append((int(item["seed"]), float(item["weight"])))
                else:
                    raise HTTPException(status_code=400, detail="Each seed item must be [seed, weight] or {seed: int, weight: float}")
            
            params["seed_list"] = seed_tuples
            params["seed_interpolation_method"] = interpolation_method
            updated_types.append("seed")

        if not params:
            raise HTTPException(status_code=400, detail="No valid blending parameters provided")

        # Apply the update
        result = app_instance.pipeline.update_stream_params(**params)
        
        return create_success_response(f"Updated {' and '.join(updated_types)} blending", updated_types=updated_types)
        
    except Exception as e:
        raise handle_api_error(e, "update_blending")

@router.post("/blending/update-prompt-weight")
async def update_prompt_weight(request: Request, app_instance=Depends(get_app_instance)):
    """Update a specific prompt weight in the current blending configuration"""
    try:
        data = await request.json()
        index = data.get('index')
        weight = data.get('weight')
        
        if index is None or weight is None:
            raise HTTPException(status_code=400, detail="Missing index or weight parameter")
        
        validate_pipeline(app_instance.pipeline, "update_prompt_weight")
        
        # Get current prompt blending configuration via unified getter, fallback to uploaded config
        state = app_instance.pipeline.stream.get_stream_state()
        current_prompts = state.get('prompt_list') or app_instance._normalize_prompt_config(app_instance.uploaded_controlnet_config)
            
        if current_prompts and index < len(current_prompts):
            # Create updated prompt list with new weight
            updated_prompts = list(current_prompts)  # Make a copy
            updated_prompts[index] = (updated_prompts[index][0], float(weight))
            
            # Use the same update method as the main blending endpoint
            params = {
                "prompt_list": updated_prompts,
                "prompt_interpolation_method": "slerp"  # Default method
            }
            
            # Apply the update using the working method
            result = app_instance.pipeline.update_stream_params(**params)
            
            return create_success_response(f"Updated prompt weight {index} to {weight}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid prompt index {index}")
        
    except Exception as e:
        raise handle_api_error(e, "update_prompt_weight")

@router.post("/blending/update-seed-weight") 
async def update_seed_weight(request: Request, app_instance=Depends(get_app_instance)):
    """Update a specific seed weight in the current blending configuration"""
    try:
        data = await request.json()
        index = data.get('index')
        weight = data.get('weight')
        
        if index is None or weight is None:
            raise HTTPException(status_code=400, detail="Missing index or weight parameter")
        
        validate_pipeline(app_instance.pipeline, "update_seed_weight")
        
        # Get current seed blending configuration via unified getter, fallback to uploaded config
        state = app_instance.pipeline.stream.get_stream_state()
        current_seeds = state.get('seed_list') or app_instance._normalize_seed_config(app_instance.uploaded_controlnet_config)
            
        if current_seeds and index < len(current_seeds):
            # Create updated seed list with new weight
            updated_seeds = list(current_seeds)  # Make a copy
            updated_seeds[index] = (updated_seeds[index][0], float(weight))
            
            # Use the same update method as the main blending endpoint
            params = {
                "seed_list": updated_seeds,
                "seed_interpolation_method": "linear"  # Default method
            }
            
            # Apply the update using the working method
            result = app_instance.pipeline.update_stream_params(**params)
            
            return create_success_response(f"Updated seed weight {index} to {weight}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid seed index {index}")
        
    except Exception as e:
        raise handle_api_error(e, "update_seed_weight")

