"""Server-Sent Events management for real-time updates."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Set

from fastapi import Request

logger = logging.getLogger(__name__)


class EventManager:
    """Manages Server-Sent Events connections and broadcasts."""

    def __init__(self):
        self.connections: Set[asyncio.Queue] = set()
        self._running = False

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

    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
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

    async def send_device_update(self, device_data: Dict[str, Any]):
        """Send device status update."""
        await self.broadcast(
            {
                "type": "device_update",
                "data": device_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    async def send_queue_update(self, queue_data: Dict[str, Any]):
        """Send queue status update."""
        await self.broadcast(
            {
                "type": "queue_update",
                "data": queue_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    async def send_run_update(self, run_data: Dict[str, Any]):
        """Send experiment run update."""
        await self.broadcast(
            {
                "type": "run_update",
                "data": run_data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )


# Global event manager instance
event_manager = EventManager()
