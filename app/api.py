import os
import glob
from typing import List
import psutil
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, FastAPI, Request
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from . import crud, schemas, services, models
from .database import get_db, PROJECT_ROOT, engine, Base, STATIC_DIR, TEMPLATES_DIR
from cvlabkit.core.config import Config
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

# --- FastAPI App Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- 🚀 CVLab-Kit Web Helper Starting up... ---")
    services.setup_directories()
    Base.metadata.create_all(bind=engine)
    
    db = next(get_db())
    try:
        services.sync_database_with_filesystem(db)
    finally:
        db.close()
        
    yield
    print("--- 👋 CVLab-Kit Web Helper Shutting down... ---")

app = FastAPI(lifespan=lifespan)
router = APIRouter()

# --- API Endpoints ---
@router.get("/api/v1/experiment_groups", response_model=List[schemas.ExperimentGroup])
def read_experiment_groups(db: Session = Depends(get_db)):
    return crud.get_experiment_groups(db)

@router.post("/api/v1/experiment_groups/{group_id}/stop", status_code=200)
def stop_experiment_group(group_id: int, db: Session = Depends(get_db)):
    group = crud.get_experiment_group(db, group_id)
    if not (group and group.pid and psutil.pid_exists(group.pid)):
        if group:
            crud.update_group_status(db, group.id, 'failed')
            db.refresh(group)
            services.write_metadata(group.launch_config_path, {
                "pid": group.pid,
                "status": group.status,
                "created_at": group.created_at,
                "finished_at": group.finished_at
            })
        raise HTTPException(status_code=404, detail="Process not found or already terminated.")
    
    try:
        parent = psutil.Process(group.pid)
        for child in parent.children(recursive=True): child.kill()
        parent.kill()
        crud.update_group_status(db, group_id, 'terminated')
        db.refresh(group)
        services.write_metadata(group.launch_config_path, {
            "pid": group.pid,
            "status": group.status,
            "created_at": group.created_at,
            "finished_at": group.finished_at
        })
        return {"message": f"Successfully stopped group {group_id} (PID: {group.pid})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/v1/visualizer/experiments", response_model=List[schemas.VisualizerExperiment])
def get_visualizer_experiments():
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    experiments = {}
    if os.path.isdir(logs_dir):
        for csv_path in glob.glob(f"{logs_dir}/**/*.csv", recursive=True):
            exp_name = os.path.basename(os.path.dirname(csv_path))
            if exp_name not in experiments:
                experiments[exp_name] = {"name": exp_name, "runs": []}
            experiments[exp_name]["runs"].append({"path": os.path.relpath(csv_path, PROJECT_ROOT)})
    return list(experiments.values())

@router.websocket("/run")
async def run_experiment_ws(websocket: WebSocket):
    await websocket.accept()
    db: Session = next(get_db())
    group_id = -1
    try:
        data = await websocket.receive_json()
        config_content = data['content']
        
        korea_time = datetime.utcnow() + timedelta(hours=9)
        timestamp = korea_time.strftime("%Y%m%d_%H%M%S")
        
        temp_path = f"temp_launch_{timestamp}.yaml"
        with open(temp_path, "w") as f: f.write(config_content)

        try:
            temp_config = Config(temp_path)
            run_name = temp_config.get('run_name', f"group_{timestamp}")
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

        launch_config_filename = f"{run_name}_{timestamp}.yaml"
        launch_config_path = os.path.join(services.LAUNCH_CONFIG_DIR, launch_config_filename)
        with open(launch_config_path, 'w') as f: f.write(config_content)

        terminal_log_filename = f"group_{run_name}_{timestamp}.log"
        terminal_log_path = os.path.join(services.TERMINAL_LOG_DIR, terminal_log_filename)
        
        group_schema = schemas.ExperimentGroupCreate(
            name=run_name,
            launch_config_path=os.path.relpath(launch_config_path, PROJECT_ROOT),
            terminal_log_path=os.path.relpath(terminal_log_path, PROJECT_ROOT)
        )
        group = crud.create_experiment_group(db, group_schema)
        group_id = group.id

        await websocket.send_text(f"🚀 Group '{run_name}' created. Launching...")
        await services.run_experiment_process(db, group_id, websocket)

    except Exception as e:
        if group_id != -1: crud.update_group_status(db, group_id, 'failed')
        await websocket.send_text(f"❌ Error: {e}")
    finally:
        db.close()
        if websocket.client_state.name == 'CONNECTED': await websocket.close()

# Include router in the app
app.include_router(router)

# --- Static Files and Templates ---
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/files/{file_path:path}")
async def get_any_file_content(file_path: str):
    full_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        return {"content": f.read()}

@app.get("/configs")
async def get_configs():
    config_dir = os.path.join(PROJECT_ROOT, "config")
    configs = glob.glob(f"{config_dir}/**/*.yaml", recursive=True)
    return {"configs": [os.path.relpath(p, PROJECT_ROOT) for p in configs]}
