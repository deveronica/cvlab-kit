"""Middleend CLI - Worker execution entry points."""

import asyncio
import logging
import socket
import sys

from web_helper.middleend.heartbeat import ClientAgent


def run_worker(args, daemon=False):
    """Run middleend worker (heartbeat + job execution + log sync).

    Args:
        args: Parsed command-line arguments
        daemon: If True, run as daemon process
    """
    if daemon:
        # Daemon mode: start process in background
        from web_helper.backend.daemon.process_manager import ProcessManager
        pm = ProcessManager()
        pm.start_middleend_daemon(args)
        return

    # Normal mode: run in foreground
    try:
        from web_helper.middleend.device_agent import DeviceAgent
    except ImportError as e:
        print(f"❌ Failed to import DeviceAgent: {e}")
        print("   Make sure all dependencies are installed: uv sync")
        sys.exit(1)

    api_key = getattr(args, "api_key", None)
    connect_timeout = getattr(args, "connect_timeout", 10.0)
    request_timeout = getattr(args, "request_timeout", 30.0)
    max_jobs = getattr(args, "max_jobs", 1)

    agent = DeviceAgent(
        server_url=args.url,
        host_id=args.client_host_id or socket.gethostname(),
        heartbeat_interval=args.client_interval,
        poll_interval=args.poll_interval,
        api_key=api_key,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        max_jobs=max_jobs,
    )

    print("💓 Starting full device agent...")
    print(f"   Server URL: {args.url}")
    print(f"   Host ID: {agent.host_id}")
    print(f"   Heartbeat interval: {args.client_interval}s")
    print(f"   Poll interval: {args.poll_interval}s")
    print(f"   Connect timeout: {connect_timeout}s")
    print(f"   Request timeout: {request_timeout}s")
    print(f"   Max concurrent jobs: {max_jobs}")
    print(f"   Workspace: {agent.workspace}")
    if api_key:
        print("   🔐 API key: configured")

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\n👋 Device agent stopped")
        asyncio.run(agent.stop())


def run_heartbeat_only(url: str, interval: int, host_id: str = None):
    """Run simple client agent (heartbeat-only) for development mode.

    This is used in development mode when backend runs locally and wants
    to monitor its own GPU stats.

    Args:
        url: Backend URL
        interval: Heartbeat interval in seconds
        host_id: Custom host identifier (default: hostname)
    """
    agent = ClientAgent(
        web_helper_url=url,
        heartbeat_interval=interval,
        host_id=host_id,
    )

    print("💓 Starting heartbeat-only agent...")
    print(f"   Server URL: {url}")
    print(f"   Heartbeat interval: {interval}s")
    print(f"   Host ID: {agent.host_id}")

    try:
        agent.start()
    except KeyboardInterrupt:
        agent.stop()


def run_local_worker(
    url: str,
    interval: int = 10,
    poll_interval: int = 5,
    host_id: str = None,
    api_key: str = None,
    max_jobs: int = 1,
):
    """Run full local worker (unified execution path).

    This creates a local DeviceAgent that executes jobs and syncs logs,
    enabling a unified execution path where local experiments use the
    same flow as remote ones (like Minecraft singleplayer).

    Args:
        url: Backend URL (usually localhost)
        interval: Heartbeat interval in seconds
        poll_interval: Job polling interval in seconds
        host_id: Custom host identifier (default: hostname)
        api_key: API key for authentication
        max_jobs: Maximum concurrent jobs
    """
    try:
        from web_helper.middleend.device_agent import DeviceAgent
    except ImportError as e:
        print(f"❌ Failed to import DeviceAgent: {e}")
        print("   Falling back to heartbeat-only mode")
        run_heartbeat_only(url, interval, host_id)
        return

    agent = DeviceAgent(
        server_url=url,
        host_id=host_id or socket.gethostname(),
        heartbeat_interval=interval,
        poll_interval=poll_interval,
        api_key=api_key,
        max_jobs=max_jobs,
    )

    print("🚀 Starting local worker (unified execution path)...")
    print(f"   Server URL: {url}")
    print(f"   Host ID: {agent.host_id}")
    print(f"   Heartbeat interval: {interval}s")
    print(f"   Poll interval: {poll_interval}s")
    print(f"   Max concurrent jobs: {max_jobs}")
    print(f"   Workspace: {agent.workspace}")

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        asyncio.run(agent.stop())
