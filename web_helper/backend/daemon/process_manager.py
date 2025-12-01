"""Daemon process manager for SSH-independent execution."""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class ProcessManager:
    """Manages daemon processes (backend, frontend, middleend).

    This allows processes to survive SSH session disconnects by:
    - Starting processes with start_new_session=True
    - Storing PID in JSON file (no DB dependency)
    - Managing process lifecycle independently
    """

    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Use JSON file instead of database for portability
        # Store in web_helper/state/ for easy Docker volume mounting
        self.state_dir = Path("web_helper/state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "daemon_state.json"

    def _load_state(self) -> dict:
        """Load daemon state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_state(self, state: dict):
        """Save daemon state to JSON file."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _update_process_state(self, process_type: str, data: dict):
        """Update state for a specific process type."""
        state = self._load_state()
        state[process_type] = data
        self._save_state(state)

    def _get_process_state(self, process_type: str) -> dict | None:
        """Get state for a specific process type."""
        state = self._load_state()
        return state.get(process_type)

    def _remove_process_state(self, process_type: str):
        """Remove state for a specific process type."""
        state = self._load_state()
        if process_type in state:
            del state[process_type]
            self._save_state(state)

    def start_backend_daemon(self, args, dev_mode=False):
        """Start backend server as daemon.

        Args:
            args: Parsed command-line arguments
            dev_mode: If True, run in development mode with auto-reload

        Returns:
            Process PID
        """
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "web_helper.backend.core.app:create_app",
            "--factory",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]

        if dev_mode:
            cmd.extend(["--reload", "--reload-dir", "web_helper"])

        # Environment
        env = os.environ.copy()
        if dev_mode:
            env["CVLABKIT_DEV_MODE"] = "true"

        # Start daemon process
        log_file = self.log_dir / "backend.log"
        err_file = self.log_dir / "backend.err"

        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, "w"),
            stderr=open(err_file, "w"),
            start_new_session=True,  # SSH-independent
            env=env,
        )

        # Save to state file
        self._update_process_state(
            "backend",
            {
                "pid": process.pid,
                "status": "running",
                "host": args.host,
                "port": args.port,
                "started_at": datetime.utcnow().isoformat(),
                "config": {
                    "dev_mode": dev_mode,
                    "host": args.host,
                    "port": args.port,
                    "log_level": getattr(args, "log_level", "info"),
                },
                "log_file": str(log_file),
                "err_file": str(err_file),
            },
        )

        print(f"✅ Backend daemon started (PID: {process.pid})")
        print(f"📊 URL: http://{args.host}:{args.port}")
        print(f"📋 Logs: {log_file}")
        print(f"❌ Errors: {err_file}")

        return process.pid

    def start_frontend_daemon(self, args):
        """Start frontend dev server as daemon.

        Args:
            args: Parsed command-line arguments

        Returns:
            Process PID
        """
        frontend_dir = Path("web_helper/frontend")
        if not frontend_dir.is_dir():
            raise FileNotFoundError(f"Frontend directory not found: {frontend_dir}")

        # Start daemon process
        log_file = self.log_dir / "frontend.log"
        err_file = self.log_dir / "frontend.err"

        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            stdout=open(log_file, "w"),
            stderr=open(err_file, "w"),
            start_new_session=True,  # SSH-independent
        )

        # Save to state file
        self._update_process_state(
            "frontend",
            {
                "pid": process.pid,
                "status": "running",
                "host": "localhost",
                "port": 5173,
                "started_at": datetime.utcnow().isoformat(),
                "config": {"dev_mode": True},
                "log_file": str(log_file),
                "err_file": str(err_file),
            },
        )

        print(f"✅ Frontend daemon started (PID: {process.pid})")
        print(f"📊 URL: http://localhost:5173")
        print(f"📋 Logs: {log_file}")
        print(f"❌ Errors: {err_file}")

        return process.pid

    def start_middleend_daemon(self, args):
        """Start middleend worker as daemon.

        Args:
            args: Parsed command-line arguments

        Returns:
            Process PID
        """
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "web_helper.middleend.device_agent",
            "--server",
            args.url,
            "--host-id",
            args.client_host_id or socket.gethostname(),
            "--heartbeat-interval",
            str(args.client_interval),
            "--poll-interval",
            str(args.poll_interval),
        ]

        # Add optional arguments if present
        if getattr(args, "api_key", None):
            cmd.extend(["--api-key", args.api_key])
        if getattr(args, "connect_timeout", None):
            cmd.extend(["--connect-timeout", str(args.connect_timeout)])
        if getattr(args, "request_timeout", None):
            cmd.extend(["--request-timeout", str(args.request_timeout)])
        if getattr(args, "max_jobs", None):
            cmd.extend(["--max-jobs", str(args.max_jobs)])

        # Start daemon process
        log_file = self.log_dir / "middleend.log"
        err_file = self.log_dir / "middleend.err"

        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, "w"),
            stderr=open(err_file, "w"),
            start_new_session=True,  # SSH-independent
        )

        # Save to state file
        self._update_process_state(
            "middleend",
            {
                "pid": process.pid,
                "status": "running",
                "host": args.client_host_id or socket.gethostname(),
                "server_url": args.url,
                "started_at": datetime.utcnow().isoformat(),
                "config": {
                    "url": args.url,
                    "client_host_id": args.client_host_id,
                    "client_interval": args.client_interval,
                    "poll_interval": args.poll_interval,
                },
                "log_file": str(log_file),
                "err_file": str(err_file),
            },
        )

        print(f"✅ Middleend daemon started (PID: {process.pid})")
        print(f"📊 Server: {args.url}")
        print(f"📋 Logs: {log_file}")
        print(f"❌ Errors: {err_file}")

        return process.pid

    def start_development_daemon(self, args):
        """Start all development daemons (backend + frontend + middleend).

        Args:
            args: Parsed command-line arguments
        """
        print("🚀 Starting development daemons...")

        # Start backend
        self.start_backend_daemon(args, dev_mode=True)

        # Start frontend
        self.start_frontend_daemon(args)

        # Start local middleend (unless disabled)
        if not getattr(args, "server_only", False):
            # Create args for local middleend
            import argparse

            middleend_args = argparse.Namespace(
                url=f"http://{args.host}:{args.port}",
                client_host_id=args.client_host_id,
                client_interval=args.client_interval,
                poll_interval=args.poll_interval,
                api_key=getattr(args, "api_key", None),
                connect_timeout=getattr(args, "connect_timeout", None),
                request_timeout=getattr(args, "request_timeout", None),
                max_jobs=getattr(args, "max_jobs", None),
            )
            self.start_middleend_daemon(middleend_args)

        print("\n✨ All development daemons started successfully!")
        print("📊 Check status: uv run app.py --status")
        print("🛑 Stop all: uv run app.py --stop")

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def stop_process(self, process_type: str):
        """Stop a daemon process by type.

        Args:
            process_type: "backend", "frontend", or "middleend"
        """
        state = self._get_process_state(process_type)

        if not state or state.get("status") != "running":
            print(f"⚠️  No running {process_type} found")
            return

        pid = state.get("pid")
        if not pid:
            print(f"⚠️  No PID found for {process_type}")
            self._remove_process_state(process_type)
            return

        try:
            os.kill(pid, signal.SIGTERM)
            print(f"✅ {process_type} stopped (PID: {pid})")
        except ProcessLookupError:
            print(f"⚠️  {process_type} process not found (PID: {pid})")
        except Exception as e:
            logging.error(f"Failed to stop {process_type}: {e}")
            print(f"❌ Failed to stop {process_type}: {e}")

        # Update state
        state["status"] = "stopped"
        state["stopped_at"] = datetime.utcnow().isoformat()
        self._update_process_state(process_type, state)

    def stop_all(self):
        """Stop all running daemon processes."""
        print("🛑 Stopping all daemon processes...")

        for ptype in ["backend", "frontend", "middleend"]:
            self.stop_process(ptype)

        print("👋 All daemons stopped")

    def show_status(self):
        """Show status of all daemon processes."""
        state = self._load_state()

        if not state:
            print("No daemon processes registered")
            return

        running_count = 0
        print("Daemon processes:\n")

        for ptype, pstate in state.items():
            pid = pstate.get("pid")
            status = pstate.get("status", "unknown")

            # Check if process is actually running
            if pid and status == "running":
                if self._is_process_running(pid):
                    icon = "🟢"
                    actual_status = "running"
                    running_count += 1
                else:
                    icon = "🔴"
                    actual_status = "dead (not responding)"
            else:
                icon = "⚪"
                actual_status = status

            print(f"  {icon} {ptype} (PID: {pid}) - {actual_status}")
            if pstate.get("started_at"):
                print(f"     Started: {pstate['started_at']}")
            if pstate.get("host") and pstate.get("port"):
                print(f"     URL: http://{pstate['host']}:{pstate['port']}")
            elif pstate.get("server_url"):
                print(f"     Server: {pstate['server_url']}")
            if pstate.get("log_file"):
                print(f"     Log: {pstate['log_file']}")
            print()

        if running_count == 0:
            print("No processes currently running")


# Convenience functions for module-level imports
def show_status():
    """Show status of all daemon processes."""
    pm = ProcessManager()
    pm.show_status()


def stop_all():
    """Stop all running daemon processes."""
    pm = ProcessManager()
    pm.stop_all()
