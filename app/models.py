from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from .database import Base

class ExperimentGroup(Base):
    __tablename__ = 'experiment_groups'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String, default='running') # e.g., 'running', 'completed', 'failed', 'terminated'
    launch_config_path = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True) # 종료 시간 기록
    pid = Column(Integer, nullable=True)
    terminal_log_path = Column(String, nullable=True)
