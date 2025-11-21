"""Daemon process manager for SSH-independent execution."""

import logging
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from web_helper.backend.daemon.models import ProcessState
from web_helper.backend.models.database import get_session


class ProcessManager:
    """Manages daemon processes (backend, frontend, middleend).

    This allows processes to survive SSH session disconnects by:
    - Starting processes with start_new_session=True
    - Storing PID in database
    - Managing process lifecycle independently
    """

    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

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

        # Save to database
        with get_session() as session:
            state = ProcessState(
                process_type="backend",
                pid=process.pid,
                status="running",
                host=args.host,
                port=args.port,
                started_at=datetime.utcnow(),
                config={
                    "dev_mode": dev_mode,
                    "host": args.host,
                    "port": args.port,
                    "log_level": getattr(args, "log_level", "info"),
                },
                log_file=str(log_file),
            )
            session.add(state)
            session.commit()

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

        # Save to database
        with get_session() as session:
            state = ProcessState(
                process_type="frontend",
                pid=process.pid,
                status="running",
                host="localhost",
                port=5173,  # Vite default
                started_at=datetime.utcnow(),
                config={"dev_mode": True},
                log_file=str(log_file),
            )
            session.add(state)
            session.commit()

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
            "web_helper.middleend.cli",
            "--url",
            args.url,
            "--client-host-id",
            args.client_host_id or socket.gethostname(),
            "--client-interval",
            str(args.client_interval),
            "--poll-interval",
            str(args.poll_interval),
        ]

        # Start daemon process
        log_file = self.log_dir / "middleend.log"
        err_file = self.log_dir / "middleend.err"

        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, "w"),
            stderr=open(err_file, "w"),
            start_new_session=True,  # SSH-independent
        )

        # Save to database
        with get_session() as session:
            state = ProcessState(
                process_type="middleend",
                pid=process.pid,
                status="running",
                host=args.client_host_id or socket.gethostname(),
                port=None,
                started_at=datetime.utcnow(),
                config={
                    "url": args.url,
                    "client_host_id": args.client_host_id,
                    "client_interval": args.client_interval,
                    "poll_interval": args.poll_interval,
                },
                log_file=str(log_file),
            )
            session.add(state)
            session.commit()

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
            )
            self.start_middleend_daemon(middleend_args)

        print("\n✨ All development daemons started successfully!")
        print("📊 Check status: uv run app.py --status")
        print("🛑 Stop all: uv run app.py --stop")

    def stop_process(self, process_type):
        """Stop a daemon process by type.

        Args:
            process_type: "backend", "frontend", or "middleend"
        """
        with get_session() as session:
            state = (
                session.query(ProcessState)
                .filter_by(process_type=process_type, status="running")
                .first()
            )

            if not state:
                print(f"⚠️  No running {process_type} found")
                return

            try:
                os.kill(state.pid, signal.SIGTERM)
                state.status = "stopped"
                state.stopped_at = datetime.utcnow()
                session.commit()
                print(f"✅ {process_type} stopped (PID: {state.pid})")
            except ProcessLookupError:
                # Process already dead
                state.status = "stopped"
                state.stopped_at = datetime.utcnow()
                session.commit()
                print(f"⚠️  {process_type} process not found (PID: {state.pid})")
            except Exception as e:
                logging.error(f"Failed to stop {process_type}: {e}")
                print(f"❌ Failed to stop {process_type}: {e}")

    def stop_all(self):
        """Stop all running daemon processes."""
        print("🛑 Stopping all daemon processes...")

        for ptype in ["backend", "frontend", "middleend"]:
            self.stop_process(ptype)

        print("👋 All daemons stopped")

    def show_status(self):
        """Show status of all daemon processes."""
        with get_session() as session:
            states = session.query(ProcessState).filter_by(status="running").all()

            if not states:
                print("No running processes")
                return

            print("Running processes:\n")
            for state in states:
                print(f"  📦 {state.process_type} (PID: {state.pid})")
                print(f"     Started: {state.started_at}")
                if state.host and state.port:
                    print(f"     URL: http://{state.host}:{state.port}")
                if state.log_file:
                    print(f"     Log: {state.log_file}")
                print()


# Convenience functions for module-level imports
def show_status():
    """Show status of all daemon processes."""
    pm = ProcessManager()
    pm.show_status()


def stop_all():
    """Stop all running daemon processes."""
    pm = ProcessManager()
    pm.stop_all()
