from sqlalchemy.orm import Session
from . import models, schemas
import psutil
from datetime import datetime

def get_experiment_group(db: Session, group_id: int):
    return db.query(models.ExperimentGroup).filter(models.ExperimentGroup.id == group_id).first()

def get_experiment_groups(db: Session, skip: int = 0, limit: int = 100):
    running_groups = db.query(models.ExperimentGroup).filter(models.ExperimentGroup.status == 'running').all()
    for group in running_groups:
        if group.pid and not psutil.pid_exists(group.pid):
            update_group_status(db, group.id, 'failed')
    
    return (
        db.query(models.ExperimentGroup)
        .order_by(models.ExperimentGroup.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

def create_experiment_group(db: Session, group: schemas.ExperimentGroupCreate):
    db_group = models.ExperimentGroup(**group.dict())
    db.add(db_group)
    db.commit()
    db.refresh(db_group)
    return db_group

def update_group_pid(db: Session, group_id: int, pid: int):
    db_group = get_experiment_group(db, group_id)
    if db_group:
        db_group.pid = pid
        db.commit()
        db.refresh(db_group)
    return db_group

def update_group_status(db: Session, group_id: int, status: str):
    db_group = get_experiment_group(db, group_id)
    if db_group and db_group.status != status:
        db_group.status = status
        if status in ['completed', 'failed', 'terminated']:
            db_group.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(db_group)
    return db_group

def delete_experiment_group(db: Session, group_id: int):
    db_group = get_experiment_group(db, group_id)
    if db_group:
        db.delete(db_group)
        db.commit()
    return db_group
