"""Server-Sent Events management for real-time updates.

Supports Dual-stack: SSE + WebSocket for backward compatibility.
WebSocket connections receive events with channel-based filtering.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Set

from fastapi import Request

logger = logging.getLogger(__name__)


class EventManager:
    """Manages Server-Sent Events connections and broadcasts.

    Dual-stack support: Broadcasts to both SSE and WebSocket clients.
    WebSocket integration is optional and gracefully degrades if unavailable.
    """

    def __init__(self):
        self.connections: Set[asyncio.Queue] = set()
        self._running = False
        self._ws_manager = None  # Lazy loaded to avoid circular imports

    def _get_ws_manager(self):
        """Lazy load WebSocket manager to avoid circular imports."""
        if self._ws_manager is None:
            try:
                from web_helper.backend.api.websocket import get_ws_manager
                self._ws_manager = get_ws_manager()
            except ImportError:
                logger.debug("WebSocket module not available, SSE-only mode")
                self._ws_manager = False  # Mark as unavailable
        return self._ws_manager if self._ws_manager else None

    async def connect(self, request: Request) -> asyncio.Queue:
        """Add a new SSE connection."""
        queue = asyncio.Queue()
        self.connections.add(queue)
        logger.info(f"New SSE connection. Total: {len(self.connections)}")

        # Send initial connection event
        await self.broadcast(
            {
                "type": "connection",
                "message": "Connected to CVLab-Kit Web Helper",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

        return queue

    async def disconnect(self, queue: asyncio.Queue):
        """Remove an SSE connection."""
        if queue in self.connections:
            self.connections.remove(queue)
            logger.info(f"SSE connection removed. Total: {len(self.connections)}")

    async def broadcast(self, data: Dict[str, Any], ws_channel: str = "all"):
        """Broadcast data to all connected clients (SSE + WebSocket).

        Args:
            data: Message data to broadcast
            ws_channel: WebSocket channel for filtering (default: "all")
        """
        # Broadcast to SSE clients
        await self._broadcast_sse(data)

        # Broadcast to WebSocket clients
        await self._broadcast_ws(data, ws_channel)

    async def _broadcast_sse(self, data: Dict[str, Any]):
        """Broadcast to SSE clients only."""
        if not self.connections:
            return

        message = json.dumps(data)
        disconnected = set()

        # Create a copy to avoid "Set changed size during iteration" error
        for queue in list(self.connections):
            try:
                await asyncio.wait_for(queue.put(message), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("SSE queue timeout, marking for removal")
                disconnected.add(queue)
            except Exception as e:
                logger.error(f"Error broadcasting to SSE client: {e}")
                disconnected.add(queue)

        # Remove disconnected clients
        for queue in disconnected:
            await self.disconnect(queue)

    async def _broadcast_ws(self, data: Dict[str, Any], channel: str = "all"):
        """Broadcast to WebSocket clients only."""
        ws_manager = self._get_ws_manager()
        if ws_manager:
            try:
                await ws_manager.broadcast(data, channel)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket clients: {e}")

    async def send_device_update(self, device_data: Dict[str, Any]):
        """Send device status update."""
        await self.broadcast(
            {
                "type": "device_update",
                "data": device_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ws_channel="devices"
        )

    async def send_queue_update(self, queue_data: Dict[str, Any]):
        """Send queue status update."""
        await self.broadcast(
            {
                "type": "queue_update",
                "data": queue_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ws_channel="queue"
        )

    async def send_run_update(self, run_data: Dict[str, Any]):
        """Send experiment run update."""
        await self.broadcast(
            {
                "type": "run_update",
                "data": run_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ws_channel="runs"
        )

    async def send_node_update(self, node_data: Dict[str, Any]):
        """Send node graph update for Builder view."""
        await self.broadcast(
            {
                "type": "node_update",
                "data": node_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ws_channel="nodes"
        )

    def get_connection_stats(self) -> Dict[str, int]:
        """Get connection statistics for both SSE and WebSocket."""
        ws_manager = self._get_ws_manager()
        return {
            "sse_connections": len(self.connections),
            "ws_connections": ws_manager.get_connection_count() if ws_manager else 0,
        }


# Global event manager instance
event_manager = EventManager()
