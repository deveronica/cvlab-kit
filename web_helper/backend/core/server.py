"""Server runners for development and production modes."""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn

from web_helper.backend.core.app import create_app
from web_helper.backend.models import init_database
from web_helper.backend.services import file_monitor
from web_helper.backend.services.indexer import log_indexer


def run_development(args, daemon=False):
    """Run development servers (backend + frontend + optional middleend).

    Args:
        args: Parsed command-line arguments
        daemon: If True, run as daemon process (SSH-independent)
    """
    if daemon:
        # Daemon mode: start processes in background
        from web_helper.backend.daemon.process_manager import ProcessManager

        pm = ProcessManager()
        pm.start_development_daemon(args)
        return

    # Normal mode: run in foreground
    run_development_servers(args)


def run_production(args, daemon=False):
    """Run production server (backend with static serving).

    Args:
        args: Parsed command-line arguments
        daemon: If True, run as daemon process (SSH-independent)
    """
    if daemon:
        # Daemon mode: start process in background
        from web_helper.backend.daemon.process_manager import ProcessManager

        pm = ProcessManager()
        pm.start_backend_daemon(args, dev_mode=False)
        return

    # Normal mode: run in foreground
    run_production_server(args, run_client=not args.server_only)


def run_development_servers(args):
    """Runs backend and frontend development servers concurrently.

    This function is called when running in normal (non-daemon) mode.
    It starts:
    1. Backend server (uvicorn with auto-reload)
    2. Optional middleend agent (for local GPU monitoring)
    3. Frontend dev server (Vite HMR)
    """
    backend_process = None
    frontend_process = None
    client_thread = None

    # Manually run startup events from lifespan
    print("🚀 CVLab-Kit Web Helper starting (Dev Mode)...")
    init_database()
    Path("web_helper").mkdir(parents=True, exist_ok=True)

    async def run_startup_tasks():
        await file_monitor.start()
        print("🔍 Starting initial log indexing...")
        try:
            stats = await log_indexer.scan_and_index()
            print(f"📊 Initial indexing complete: {stats}")
        except Exception as e:
            print(f"❌ Initial indexing failed: {e}")

    # Run startup tasks in a separate thread to not block the main thread
    startup_thread = threading.Thread(
        target=lambda: asyncio.run(run_startup_tasks()), daemon=True
    )
    startup_thread.start()

    # Use the current python interpreter to run uvicorn, ensuring it's from the correct venv
    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "web_helper.backend.core.app:create_app",
        "--factory",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--reload",
        "--reload-dir",
        "web_helper",
        "--reload-dir",
        "config",
        "--reload-dir",
        "logs",
    ]
    frontend_dir = Path("web_helper/frontend")

    try:
        print("🚀 Starting development servers (backend + frontend + middleend)...")

        # 1. Start backend server
        print("[1/3] Starting backend server...")
        backend_env = os.environ.copy()
        backend_env["CVLABKIT_DEV_MODE"] = "true"
        # Pass API key to backend via environment variable
        api_key = getattr(args, "api_key", None)
        if api_key:
            backend_env["CVLABKIT_API_KEY"] = api_key
            print("🔐 API key authentication enabled")
        backend_process = subprocess.Popen(
            backend_command, preexec_fn=os.setsid, env=backend_env
        )

        # Wait for backend to be ready
        print("⏳ Waiting for backend to be ready...")
        backend_url = f"http://{args.host}:{args.port}/api/projects/"
        for attempt in range(30):  # Try for up to 30 seconds
            try:
                import urllib.request

                urllib.request.urlopen(backend_url, timeout=1)
                print("✅ Backend ready!")
                break
            except Exception:
                time.sleep(1)
        else:
            print("⚠️  Backend took longer than expected to start, proceeding anyway...")

        # 2. Start middleend agent in development mode (unless disabled)
        if not getattr(args, "server_only", False):
            server_url = f"http://{args.host}:{args.port}"
            full_mode = getattr(args, "full", False)

            if full_mode:
                # Full worker mode: job execution + log sync (unified execution path)
                print("[2/3] Starting full local worker (unified execution path)...")
                api_key = getattr(args, "api_key", None)
                poll_interval = getattr(args, "poll_interval", 5)
                max_jobs = getattr(args, "max_jobs", 1)
                client_thread = threading.Thread(
                    target=run_local_worker_thread,
                    args=(server_url, args.client_interval, poll_interval,
                          args.client_host_id, api_key, max_jobs),
                    daemon=True,
                )
                client_thread.start()
                print(
                    f"🚀 Local worker started (heartbeat: {args.client_interval}s, "
                    f"poll: {poll_interval}s, max jobs: {max_jobs})"
                )
            else:
                # Heartbeat-only mode: just monitoring
                print("[2/3] Starting middleend agent (local heartbeat)...")
                client_thread = threading.Thread(
                    target=run_local_heartbeat,
                    args=(server_url, args.client_interval, args.client_host_id),
                    daemon=True,
                )
                client_thread.start()
                print(
                    f"💓 Middleend agent started (heartbeat interval: {args.client_interval}s)"
                )
        else:
            print("[2/3] Middleend agent disabled (--server-only)")

        # 3. Start frontend server
        if not frontend_dir.is_dir():
            raise FileNotFoundError(
                f"Frontend directory not found. Searched at: {frontend_dir.resolve()}"
            )
        print(f"[3/3] Starting frontend server: npm run dev in {frontend_dir}")
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"], cwd=str(frontend_dir), preexec_fn=os.setsid
        )

        # Wait for the frontend process to terminate (e.g., user hits Ctrl+C)
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n🛑 User interrupted. Shutting down servers...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("--- Shutting down ---")
        # Terminate frontend process group
        if frontend_process and frontend_process.poll() is None:
            print("Terminating frontend server...")
            try:
                os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process already terminated
        # Terminate backend process group
        if backend_process and backend_process.poll() is None:
            print("Terminating backend server...")
            try:
                os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process already terminated
        # Client thread will terminate automatically (daemon=True)
        print("👋 Servers shut down.")


def run_local_heartbeat(url: str, interval: int, host_id: str = None):
    """Run simple heartbeat agent for local development.

    This sends heartbeat to the local backend server so GPU stats
    are visible in the web UI during development.

    Args:
        url: Backend URL
        interval: Heartbeat interval in seconds
        host_id: Custom host identifier
    """
    from web_helper.middleend.cli import run_heartbeat_only

    run_heartbeat_only(url, interval, host_id)


def run_local_worker_thread(
    url: str,
    interval: int = 10,
    poll_interval: int = 5,
    host_id: str = None,
    api_key: str = None,
    max_jobs: int = 1,
):
    """Run full local worker in a thread (unified execution path).

    This creates a local DeviceAgent that executes jobs and syncs logs,
    enabling a unified execution path where local experiments use the
    same flow as remote ones.

    Args:
        url: Backend URL (usually localhost)
        interval: Heartbeat interval in seconds
        poll_interval: Job polling interval in seconds
        host_id: Custom host identifier
        api_key: API key for authentication
        max_jobs: Maximum concurrent jobs
    """
    from web_helper.middleend.cli import run_local_worker

    run_local_worker(url, interval, poll_interval, host_id, api_key, max_jobs)


def run_production_server(args, run_client: bool = False):
    """Runs the FastAPI application in production mode.

    Args:
        args: Parsed command-line arguments
        run_client: If True, also start a local heartbeat agent
    """
    api_key = getattr(args, "api_key", None)
    app = create_app(dev_mode=False, api_key=api_key)

    # Determine protocol for display URLs
    ssl_certfile = getattr(args, "ssl_certfile", None)
    ssl_keyfile = getattr(args, "ssl_keyfile", None)
    protocol = "https" if (ssl_certfile and ssl_keyfile) else "http"

    print("🚀 Starting CVLab-Kit Web Helper (Production Mode)")
    print(f"📊 Web interface: {protocol}://{args.host}:{args.port}")
    print(f"🔧 API docs: {protocol}://{args.host}:{args.port}/docs")

    client_thread = None
    if run_client:
        server_url = f"http://{args.host}:{args.port}"
        full_mode = getattr(args, "full", False)

        if full_mode:
            # Full worker mode: job execution + log sync (unified execution path)
            print("🚀 Middleend agent: Enabled (full worker - unified execution path)")
            api_key = getattr(args, "api_key", None)
            poll_interval = getattr(args, "poll_interval", 5)
            max_jobs = getattr(args, "max_jobs", 1)
            client_thread = threading.Thread(
                target=run_local_worker_thread,
                args=(server_url, args.client_interval, poll_interval,
                      args.client_host_id, api_key, max_jobs),
                daemon=True,
            )
            client_thread.start()
        else:
            # Heartbeat-only mode: just monitoring
            print("💓 Middleend agent: Enabled (local heartbeat)")
            client_thread = threading.Thread(
                target=run_local_heartbeat,
                args=(server_url, args.client_interval, args.client_host_id),
                daemon=True,
            )
            client_thread.start()
    else:
        print("💓 Middleend agent: Disabled")

    if ssl_certfile and ssl_keyfile:
        print(f"🔒 HTTPS enabled (cert: {ssl_certfile})")

    print("🛑 Press Ctrl+C to stop\n")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            reload=args.reload,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )
    except KeyboardInterrupt:
        print("\n👋 CVLab-Kit Web Helper stopped")
