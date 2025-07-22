from typing import Dict, Union
from uuid import UUID
import asyncio
from fastapi import WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect
import logging
from types import SimpleNamespace

Connections = Dict[UUID, Dict[str, Union[WebSocket, asyncio.Queue]]]


class ServerFullException(Exception):
    """Exception raised when the server is full."""

    pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Connections = {}

    async def connect(
        self, user_id: UUID, websocket: WebSocket, max_queue_size: int = 0
    ):
        await websocket.accept()
        user_count = self.get_user_count()
        print(f"User count: {user_count}")
        if max_queue_size > 0 and user_count >= max_queue_size:
            print("Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            raise ServerFullException("Server is full")
        print(f"New user connected: {user_id}")
        self.active_connections[user_id] = {
            "websocket": websocket,
            "queue": asyncio.Queue(),
        }
        await websocket.send_json(
            {"status": "connected", "message": "Connected"},
        )
        await websocket.send_json({"status": "wait"})
        await websocket.send_json({"status": "send_frame"})

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.active_connections

    async def update_data(self, user_id: UUID, new_data: SimpleNamespace):
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            await queue.put(new_data)

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            try:
                return await queue.get()
            except asyncio.QueueEmpty:
                return None

    def delete_user(self, user_id: UUID):
        user_session = self.active_connections.pop(user_id, None)
        if user_session:
            queue = user_session["queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_websocket(self, user_id: UUID) -> WebSocket:
        user_session = self.active_connections.get(user_id)
        if user_session:
            websocket = user_session["websocket"]
            if websocket.client_state == WebSocketState.CONNECTED:
                return user_session["websocket"]
        return None

    async def disconnect(self, user_id: UUID):
        """Gracefully disconnect a user"""
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                # Try to send a final message before closing
                try:
                    await websocket.send_json({"status": "disconnecting"})
                except:
                    pass  # Ignore if can't send (connection already broken)
                
                # Close the websocket gracefully
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close(code=1000, reason="Normal closure")
                except:
                    pass  # Ignore if connection already closed
        except Exception as e:
            logging.error(f"Error during websocket disconnect: {e}")
        finally:
            # Always clean up the user session
            self.delete_user(user_id)

    async def send_json(self, user_id: UUID, data: Dict):
        try:
            websocket = self.get_websocket(user_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except ConnectionResetError:
            # Connection was reset by client
            logging.debug(f"Connection reset while sending to {user_id}")
            self.delete_user(user_id)
        except Exception as e:
            # Only log as error if it's not a common disconnect scenario
            if "1005" not in str(e) and "no status received" not in str(e):
                logging.error(f"Error: Send json: {e}")
            else:
                logging.debug(f"WebSocket closed during send to {user_id}: {e}")
            # Clean up user on send failure
            self.delete_user(user_id)

    async def receive_json(self, user_id: UUID) -> Dict:
        try:
            websocket = self.get_websocket(user_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                return await websocket.receive_json()
        except ConnectionResetError:
            logging.debug(f"Connection reset while receiving from {user_id}")
            self.delete_user(user_id)
        except Exception as e:
            if "1005" not in str(e) and "no status received" not in str(e):
                logging.error(f"Error: Receive json: {e}")
            else:
                logging.debug(f"WebSocket closed during receive from {user_id}: {e}")
            self.delete_user(user_id)
        return None

    async def receive_bytes(self, user_id: UUID) -> bytes:
        try:
            websocket = self.get_websocket(user_id)
            if websocket and websocket.client_state == WebSocketState.CONNECTED:
                return await websocket.receive_bytes()
        except ConnectionResetError:
            logging.debug(f"Connection reset while receiving bytes from {user_id}")
            self.delete_user(user_id)
        except Exception as e:
            if "1005" not in str(e) and "no status received" not in str(e):
                logging.error(f"Error: Receive bytes: {e}")
            else:
                logging.debug(f"WebSocket closed during receive bytes from {user_id}: {e}")
            self.delete_user(user_id)
        return None
