import os
import asyncio
import sys
import glob
import json
from datetime import datetime
from sqlalchemy.orm import Session
import signal

from . import crud, schemas, models
from .database import (
    PROJECT_ROOT,
    WEB_ROOT,
    LAUNCH_CONFIG_DIR,
    TERMINAL_LOG_DIR,
    STATIC_DIR,
    TEMPLATES_DIR
)

# --- Metadata Helper Functions ---

def _get_meta_path(config_path):
    """Gets the metadata file path for a given config file path."""
    return os.path.splitext(config_path)[0] + '.meta.json'

def write_metadata(config_path, data):
    """Writes data to a metadata file."""
    full_meta_path = os.path.join(PROJECT_ROOT, _get_meta_path(config_path))
    # Ensure datetime objects are converted to ISO 8601 strings
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
            
    with open(full_meta_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_metadata(config_path):
    """Reads data from a metadata file."""
    full_meta_path = os.path.join(PROJECT_ROOT, _get_meta_path(config_path))
    if not os.path.exists(full_meta_path):
        return {}
    with open(full_meta_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

# --- Core Service Functions ---

def setup_directories():
    os.makedirs(WEB_ROOT, exist_ok=True)
    os.makedirs(LAUNCH_CONFIG_DIR, exist_ok=True)
    os.makedirs(TERMINAL_LOG_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

def sync_database_with_filesystem(db: Session):
    """
    Synchronizes the database with the state of the filesystem using metadata files.
    Deletes all records and rebuilds them from the launch configs and metadata on disk.
    """
    print("--- Syncing database with filesystem... ---")
    
    db.query(models.ExperimentGroup).delete()
    db.commit()

    config_files = glob.glob(f"{LAUNCH_CONFIG_DIR}/**/*.yaml", recursive=True)
    
    for config_path in config_files:
        rel_config_path = os.path.relpath(config_path, PROJECT_ROOT)
        print(f"Processing config file: {rel_config_path}")
        try:
            metadata = read_metadata(rel_config_path)
            file_mod_time = datetime.utcfromtimestamp(os.path.getmtime(config_path))

            base_name = os.path.splitext(os.path.basename(rel_config_path))[0]
            run_name = '_'.join(base_name.split('_')[:-1]) if '_' in base_name else base_name

            log_file_path = os.path.join(TERMINAL_LOG_DIR, f"group_{base_name}.log")
            rel_log_path = os.path.relpath(log_file_path, PROJECT_ROOT) if os.path.exists(log_file_path) else None

            created_at = datetime.fromisoformat(metadata['created_at']) if metadata.get('created_at') else file_mod_time
            finished_at = datetime.fromisoformat(metadata['finished_at']) if metadata.get('finished_at') else None
            status = metadata.get('status', 'unknown')
            pid = metadata.get('pid')

            group_schema = schemas.ExperimentGroupCreate(
                name=run_name,
                launch_config_path=rel_config_path,
                terminal_log_path=rel_log_path,
                created_at=created_at
            )
            group = crud.create_experiment_group(db, group_schema)
            
            group.status = status
            group.finished_at = finished_at
            group.pid = pid
            db.commit()

        except Exception as e:
            print(f"Error processing config file {rel_config_path}: {e}")

    print("--- Sync complete. ---")


async def run_experiment_process(db: Session, group_id: int, websocket):
    group = crud.get_experiment_group(db, group_id)
    if not group or not group.terminal_log_path:
        raise FileNotFoundError("Terminal log path is not set for the group.")

    python_executable = sys.executable
    main_py_path = os.path.join(PROJECT_ROOT, 'main.py')
    full_launch_config_path = os.path.join(PROJECT_ROOT, group.launch_config_path)
    terminal_log_full_path = os.path.join(PROJECT_ROOT, group.terminal_log_path)
    
    command = f'"{python_executable}" -u "{main_py_path}" --config "{full_launch_config_path}" --fast'
    
    process = None
    try:
        with open(terminal_log_full_path, 'w') as log_file:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=log_file,
                stderr=log_file,
                cwd=PROJECT_ROOT
            )
            
            crud.update_group_pid(db, group_id, process.pid)
            db.refresh(group)
            write_metadata(group.launch_config_path, {"pid": group.pid, "status": "running", "created_at": group.created_at})
            await websocket.send_text(f"✅ Process launched (PID: {process.pid}). View logs in History tab.")
            print(f"Experiment Group '{group.name}' launched (PID: {process.pid}).")
            
            await process.wait()
        
        await websocket.send_text("⏳ Process finished. Determining final status...")
        print(f"Process (PID: {process.pid}) finished with exit code {process.returncode}.")

        if process.returncode == 0:
            crud.update_group_status(db, group_id, 'completed')
        elif process.returncode < 0:
            try:
                signal_name = signal.Signals(-process.returncode).name
                print(f"Process terminated by signal: {signal_name}")
            except ValueError:
                print(f"Process terminated by unknown signal: {-process.returncode}")
            crud.update_group_status(db, group_id, 'terminated')
        else:
            crud.update_group_status(db, group_id, 'failed')
            
    except Exception as e:
        print(f"An error occurred during experiment execution: {e}")
        crud.update_group_status(db, group_id, 'failed')
    finally:
        db.refresh(group)
        write_metadata(group.launch_config_path, {
            "pid": group.pid,
            "status": group.status,
            "created_at": group.created_at,
            "finished_at": group.finished_at
        })
        final_status_message = f"Analysis complete. Final status: {group.status}."
        await websocket.send_text(final_status_message)
