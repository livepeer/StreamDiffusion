"""
WebSocket endpoints for realtime-img2img
"""
from fastapi import APIRouter, WebSocket, HTTPException, Depends
import logging
import uuid
import time
from types import SimpleNamespace

from util import bytes_to_pt
from connection_manager import ServerFullException
from .common.dependencies import get_app_instance, get_pipeline_class

router = APIRouter(prefix="/api", tags=["websocket"])

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket, app_instance=Depends(get_app_instance), pipeline_class=Depends(get_pipeline_class)):
    """Main WebSocket endpoint for real-time communication"""
    try:
        await app_instance.conn_manager.connect(
            user_id, websocket, app_instance.args.max_queue_size
        )
        await handle_websocket_data(user_id, app_instance, pipeline_class)
    except ServerFullException as e:
        logging.exception(f"websocket_endpoint: Server Full: {e}")
    finally:
        await app_instance.conn_manager.disconnect(user_id)
        logging.info(f"websocket_endpoint: User disconnected: {user_id}")

async def handle_websocket_data(user_id: uuid.UUID, app_instance, pipeline_class):
    """Handle WebSocket data flow for a specific user"""
    if not app_instance.conn_manager.check_user(user_id):
        return HTTPException(status_code=404, detail="User not found")
    last_time = time.time()
    try:
        while True:
            if (
                app_instance.args.timeout > 0
                and time.time() - last_time > app_instance.args.timeout
            ):
                await app_instance.conn_manager.send_json(
                    user_id,
                    {
                        "status": "timeout",
                        "message": "Your session has ended",
                    },
                )
                await app_instance.conn_manager.disconnect(user_id)
                return
            data = await app_instance.conn_manager.receive_json(user_id)
            if data is None:
                break
            if data["status"] == "next_frame":
                params = await app_instance.conn_manager.receive_json(user_id)
                params = pipeline_class.InputParams(**params)
                params = SimpleNamespace(**params.dict())
                
                # Check if we need image data based on pipeline
                need_image = True
                if app_instance.pipeline and hasattr(app_instance.pipeline, 'pipeline_mode'):
                    # Need image for img2img OR for txt2img with ControlNets
                    has_controlnets = app_instance.pipeline.use_config and app_instance.pipeline.config and 'controlnets' in app_instance.pipeline.config
                    need_image = app_instance.pipeline.pipeline_mode == "img2img" or has_controlnets
                elif app_instance.uploaded_controlnet_config and 'mode' in app_instance.uploaded_controlnet_config:
                    # Need image for img2img OR for txt2img with ControlNets
                    has_controlnets = 'controlnets' in app_instance.uploaded_controlnet_config
                    need_image = app_instance.uploaded_controlnet_config['mode'] == "img2img" or has_controlnets
                
                if need_image:
                    image_data = await app_instance.conn_manager.receive_bytes(user_id)
                    if len(image_data) == 0:
                        await app_instance.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        continue
                    
                    # Always use direct bytes-to-tensor conversion for efficiency
                    params.image = bytes_to_pt(image_data)
                else:
                    params.image = None
                
                await app_instance.conn_manager.update_data(user_id, params)

    except Exception as e:
        logging.exception(f"handle_websocket_data: Websocket Error: {e}, {user_id} ")
        await app_instance.conn_manager.disconnect(user_id)

