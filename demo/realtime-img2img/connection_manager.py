from typing import Dict, Union, List
from uuid import UUID
import asyncio
from fastapi import WebSocket
from starlette.websockets import WebSocketState
import logging
from types import SimpleNamespace
from collections import deque

Connections = Dict[UUID, Dict[str, Union[WebSocket, asyncio.Queue, deque]]]


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
            "frame_buffer": deque(),
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

    def add_frame_to_buffer(self, user_id: UUID, frame, max_buffer_size: int):
        """Add a frame to the user's frame buffer, maintaining the buffer size."""
        user_session = self.active_connections.get(user_id)
        if user_session:
            frame_buffer = user_session["frame_buffer"]
            frame_buffer.append(frame)
            # Keep only the last max_buffer_size frames
            while len(frame_buffer) > max_buffer_size:
                frame_buffer.popleft()

    def get_frame_buffer(self, user_id: UUID) -> List:
        """Get the current frame buffer for a user."""
        user_session = self.active_connections.get(user_id)
        if user_session:
            return list(user_session["frame_buffer"])
        return []

    def is_frame_buffer_ready(self, user_id: UUID, required_size: int) -> bool:
        """Check if the frame buffer has enough frames for batching."""
        user_session = self.active_connections.get(user_id)
        if user_session:
            return len(user_session["frame_buffer"]) >= required_size
        return False

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
        websocket = self.get_websocket(user_id)
        if websocket:
            await websocket.close()
        self.delete_user(user_id)

    async def send_json(self, user_id: UUID, data: Dict):
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                await websocket.send_json(data)
        except Exception as e:
            logging.error(f"Error: Send json: {e}")

    async def receive_json(self, user_id: UUID) -> Dict:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_json()
        except Exception as e:
            logging.error(f"Error: Receive json: {e}")

    async def receive_bytes(self, user_id: UUID) -> bytes:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_bytes()
        except Exception as e:
            logging.error(f"Error: Receive bytes: {e}")
