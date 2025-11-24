#!/usr/bin/env python3
"""CVLab-Kit Web Helper - Application Entry Point.

Streamlined entry point for CVLab-Kit experiment management.
All implementation logic is in web_helper/ modules.
"""

import argparse
import os
import socket
import sys

from web_helper.backend.core.server import run_development, run_production
from web_helper.backend.daemon.process_manager import (
    ProcessManager,
    show_status,
    stop_all,
)
from web_helper.middleend.cli import run_worker


def create_parser():
    """Create argument parser for CLI.

    Returns:
        ArgumentParser with all command-line options
    """
    parser = argparse.ArgumentParser(
        description="CVLab-Kit Web Helper - Experiment Management Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local development (backend + frontend + middleend)
  uv run app.py --dev

  # Production mode (backend only, static serving)
  uv run app.py

  # Distributed execution
  uv run app.py --server-only              # Central server
  uv run app.py --client-only --url <URL>  # Remote worker

  # Daemon mode (SSH-independent)
  uv run app.py --dev --daemon             # Start all as daemons
  uv run app.py --status                   # Check daemon status
  uv run app.py --stop                     # Stop all daemons
        """,
    )

    # Execution mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dev",
        action="store_true",
        help="Development mode (backend + frontend + middleend)",
    )
    mode_group.add_argument(
        "--server-only",
        action="store_true",
        help="Server-only mode (no local middleend)",
    )
    mode_group.add_argument(
        "--client-only",
        action="store_true",
        help="Client-only mode (middleend worker)",
    )

    # Daemon management
    daemon_group = parser.add_mutually_exclusive_group()
    daemon_group.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon process (SSH-independent)",
    )
    daemon_group.add_argument(
        "--status",
        action="store_true",
        help="Show status of daemon processes",
    )
    daemon_group.add_argument(
        "--stop",
        action="store_true",
        help="Stop all daemon processes",
    )

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Security
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CVLABKIT_API_KEY"),
        help="API key for authentication (or set CVLABKIT_API_KEY env var)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=os.environ.get("CVLABKIT_SSL_CERTFILE"),
        help="SSL certificate file path for HTTPS",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=os.environ.get("CVLABKIT_SSL_KEYFILE"),
        help="SSL key file path for HTTPS",
    )

    # Middleend configuration
    parser.add_argument(
        "--full",
        action="store_true",
        help="Enable full worker mode (job execution + log sync). Default: heartbeat-only",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL for client mode (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--client-host-id",
        default=None,
        help="Custom host identifier for client (default: hostname)",
    )
    parser.add_argument(
        "--client-interval",
        type=int,
        default=10,
        help="Heartbeat interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Job polling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Maximum concurrent jobs for client mode (default: 1)",
    )

    return parser


def main():
    """Main entry point for CVLab-Kit Web Helper."""
    parser = create_parser()
    args = parser.parse_args()

    # Daemon management commands
    if args.status:
        show_status()
        return
    if args.stop:
        stop_all()
        return

    # Client-only mode: run middleend worker
    if args.client_only:
        run_worker(args, daemon=args.daemon)
        return

    # Server modes: development or production
    if args.dev:
        run_development(args, daemon=args.daemon)
    else:
        # Production mode
        run_production(args, daemon=args.daemon)


if __name__ == "__main__":
    main()
