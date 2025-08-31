from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# ExperimentGroup Schemas
class ExperimentGroupBase(BaseModel):
    name: str
    status: str
    launch_config_path: str
    terminal_log_path: Optional[str] = None
    pid: Optional[int] = None

class ExperimentGroupCreate(BaseModel):
    name: str
    launch_config_path: str
    terminal_log_path: Optional[str] = None
    created_at: Optional[datetime] = None

class ExperimentGroup(ExperimentGroupBase):
    id: int
    created_at: datetime
    finished_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Visualizer(Runs) Schemas
class VisualizerRun(BaseModel):
    path: str

class VisualizerExperiment(BaseModel):
    name: str
    runs: List[VisualizerRun]
