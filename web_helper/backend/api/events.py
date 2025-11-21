"""Server-Sent Events API endpoints for real-time updates."""

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..services.event_manager import event_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events")


@router.get("/stream")
async def event_stream(request: Request):
    """Server-Sent Events endpoint for real-time updates."""

    async def generate():
        queue = await event_manager.connect(request)

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # Wait for new events with timeout
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat every 30 seconds
                    yield f"data: {{'type': 'heartbeat', 'timestamp': '{asyncio.get_event_loop().time()}'}}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled")
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
        finally:
            await event_manager.disconnect(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
