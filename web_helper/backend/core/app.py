"""FastAPI application factory and lifespan management."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from web_helper.backend.api import create_router
from web_helper.backend.models import init_database
from web_helper.backend.services import file_monitor
from web_helper.backend.services.indexer import log_indexer
from web_helper.backend.services.queue_file_monitor import queue_file_monitor
from web_helper.backend.services.queue_indexer import (
    index_queue_experiments_startup,
    periodic_indexer,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logging.info("🚀 CVLab-Kit Web Helper starting...")

    # Initialize database
    init_database()

    # Create required directories
    Path("web_helper").mkdir(parents=True, exist_ok=True)

    # Start file monitoring
    logging.info("📂 Starting file monitors...")
    await file_monitor.start()
    logging.info("📋 Starting queue file monitor...")
    await queue_file_monitor.start()
    logging.info("✅ All monitors started")

    # Perform initial indexing of existing logs
    logging.info("🔍 Starting initial log indexing...")
    try:
        stats = await log_indexer.scan_and_index()
        logging.info(f"📊 Initial indexing complete: {stats}")
    except Exception as e:
        logging.error(f"❌ Initial indexing failed: {e}")

    # Perform initial indexing of queue experiments
    logging.info("🔍 Starting queue experiment indexing...")
    try:
        index_queue_experiments_startup()
        logging.info("📊 Queue experiment indexing complete")
    except Exception as e:
        logging.error(f"❌ Queue experiment indexing failed: {e}")

    # Start periodic background tasks for queue integrity
    logging.info("⏰ Starting periodic indexing tasks...")
    try:
        await periodic_indexer.start(
            status_check_interval=60,  # Check status every 1 minute
            full_reindex_interval=600,  # Full reindex every 10 minutes
        )
    except Exception as e:
        logging.error(f"❌ Failed to start periodic indexer: {e}")

    yield

    # Stop periodic indexing tasks
    logging.info("⏸️  Stopping periodic indexing tasks...")
    try:
        await periodic_indexer.stop()
    except Exception as e:
        logging.error(f"❌ Failed to stop periodic indexer: {e}")

    # Stop file monitoring
    await file_monitor.stop()
    await queue_file_monitor.stop()
    logging.info("👋 CVLab-Kit Web Helper stopping...")


def create_app(dev_mode: bool = None):
    """Create FastAPI application with modular structure.

    Args:
        dev_mode: If True, run in development mode (redirect to Vite).
                 If None, read from CVLABKIT_DEV_MODE env var.

    Returns:
        Configured FastAPI application instance.
    """
    # Determine dev_mode from environment variable if not explicitly passed
    if dev_mode is None:
        dev_mode = os.environ.get("CVLABKIT_DEV_MODE", "false").lower() == "true"

    app = FastAPI(
        title="CVLab-Kit Web Helper",
        description="Experiment management and monitoring for CVLab-Kit",
        version="1.0.0",
        lifespan=lifespan,
        servers=[
            {"url": "http://localhost:8000", "description": "Local development"},
            {"url": "http://127.0.0.1:8000", "description": "Local (127.0.0.1)"},
            {
                "url": "{protocol}://{host}:{port}",
                "description": "Custom server",
                "variables": {
                    "protocol": {
                        "default": "http",
                        "enum": ["http", "https"],
                        "description": "Protocol",
                    },
                    "host": {
                        "default": "localhost",
                        "description": "Server host or IP address",
                    },
                    "port": {"default": "8000", "description": "Server port"},
                },
            },
        ],
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(create_router())

    if dev_mode:
        # In development mode, redirect root to Vite's HMR server
        @app.get("/")
        async def redirect_to_frontend_dev():
            return RedirectResponse(url="http://localhost:5173")

        # No static files are served by FastAPI in dev_mode
    else:
        # In production mode, serve frontend static files
        frontend_dist = Path("web_helper/frontend/dist")
        if frontend_dist.exists():
            # Mount static assets (CSS, JS, images, etc.)
            app.mount(
                "/assets",
                StaticFiles(directory=str(frontend_dist / "assets")),
                name="assets",
            )

            # Serve index.html for all non-API routes (SPA fallback)
            @app.get("/{path:path}")
            async def serve_spa(request: Request, path: str = ""):
                # If path is an API route, don't handle it here
                if (
                    path.startswith("api/")
                    or path.startswith("docs")
                    or path.startswith("openapi.json")
                ):
                    return {"detail": "Not Found"}

                # Serve index.html for all SPA routes
                return FileResponse(str(frontend_dist / "index.html"))

            # Explicitly serve root
            @app.get("/")
            async def serve_root():
                return FileResponse(str(frontend_dist / "index.html"))
        else:
            @app.get("/")
            async def root_fallback():
                return {
                    "message": "CVLab-Kit Web Helper API",
                    "frontend": "Build frontend with: cd web_helper/frontend && npm run build",
                    "api_docs": "/docs",
                }

    return app
