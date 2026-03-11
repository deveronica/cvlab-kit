"""WebSocket API endpoints for real-time communication.

Provides bidirectional WebSocket connections alongside existing SSE.
Supports subscription-based event filtering and message types.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections with subscription support.

    Features:
    - Connection tracking with subscription channels
    - Heartbeat (ping/pong) support
    - Message broadcasting with channel filtering
    - Graceful disconnection handling
    """

    def __init__(self):
        # WebSocket -> set of subscribed channels
        self.active_connections: Dict[WebSocket, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, initial_channels: Set[str] | None = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[websocket] = initial_channels or {"all"}
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe a connection to a channel."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections[websocket].add(channel)
                logger.debug(f"Subscribed to channel: {channel}")

    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe a connection from a channel."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections[websocket].discard(channel)
                logger.debug(f"Unsubscribed from channel: {channel}")

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any], channel: str = "all"):
        """Broadcast a message to all connections subscribed to the channel.

        Args:
            message: Message to broadcast
            channel: Target channel. Use "all" for all connections.
        """
        if not self.active_connections:
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat() + "Z"

        disconnected: Set[WebSocket] = set()

        # Capture a snapshot of current connections to iterate safely
        async with self._lock:
            connections = list(self.active_connections.items())

        for ws, channels in connections:
            # Check if connection is subscribed to this channel or "all"
            if channel == "all" or "all" in channels or channel in channels:
                # Requirement: Check connection state before sending to avoid ASGI errors
                from starlette.websockets import WebSocketState
                if ws.client_state != WebSocketState.CONNECTED:
                    disconnected.add(ws)
                    continue

                try:
                    await asyncio.wait_for(ws.send_json(message), timeout=1.0)
                except (asyncio.TimeoutError, RuntimeError, WebSocketDisconnect):
                    # Quietly handle typical disconnection errors
                    disconnected.add(ws)
                except Exception as e:
                    # Only log unexpected errors
                    if "Unexpected ASGI message" not in str(e):
                        logger.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.add(ws)

        # Remove disconnected clients
        if disconnected:
            for ws in disconnected:
                await self.disconnect(ws)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about active connections."""
        return {
            "total_connections": len(self.active_connections),
            "connections": [
                {"channels": list(channels)}
                for channels in self.active_connections.values()
            ]
        }


# Singleton connection manager
ws_manager = ConnectionManager()


def get_ws_manager() -> ConnectionManager:
    """Get the singleton WebSocket connection manager."""
    return ws_manager


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication.

    Message Protocol:
    - subscribe: {"type": "subscribe", "channel": "queue"}
    - unsubscribe: {"type": "unsubscribe", "channel": "queue"}
    - ping: {"type": "ping"} -> responds with {"type": "pong"}

    Available Channels:
    - all: All events (default)
    - queue: Queue status updates
    - devices: Device status updates
    - runs: Run completion updates
    - nodes: Node graph sync updates (for Builder)
    """
    await ws_manager.connect(websocket)

    # Send connection confirmation
    await ws_manager.send_personal(websocket, {
        "type": "connection",
        "message": "Connected to CVLab-Kit Web Helper",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    try:
        while True:
            try:
                # Wait for incoming messages with timeout for heartbeat
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second heartbeat interval
                )

                try:
                    message = json.loads(data)
                    await handle_message(websocket, message)
                except json.JSONDecodeError:
                    await ws_manager.send_personal(websocket, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })

            except asyncio.TimeoutError:
                # Send heartbeat ping
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break  # Connection lost

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


async def handle_message(websocket: WebSocket, message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    msg_type = message.get("type", "")

    if msg_type == "ping":
        await ws_manager.send_personal(websocket, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    elif msg_type == "subscribe":
        channel = message.get("channel", "all")
        await ws_manager.subscribe(websocket, channel)
        await ws_manager.send_personal(websocket, {
            "type": "subscribed",
            "channel": channel
        })

    elif msg_type == "unsubscribe":
        channel = message.get("channel")
        if channel:
            await ws_manager.unsubscribe(websocket, channel)
            await ws_manager.send_personal(websocket, {
                "type": "unsubscribed",
                "channel": channel
            })

    elif msg_type == "status":
        # Return connection status
        await ws_manager.send_personal(websocket, {
            "type": "status",
            "data": ws_manager.get_connection_info()
        })

    else:
        await ws_manager.send_personal(websocket, {
            "type": "error",
            "message": f"Unknown message type: {msg_type}"
        })
