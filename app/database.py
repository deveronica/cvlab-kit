import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_ROOT = os.path.join(PROJECT_ROOT, "web")

DATABASE_URL = f"sqlite:///{os.path.join(WEB_ROOT, 'web_helper.db')}"
LAUNCH_CONFIG_DIR = os.path.join(WEB_ROOT, "launch_configs")
TERMINAL_LOG_DIR = os.path.join(WEB_ROOT, "terminal_logs")
STATIC_DIR = os.path.join(WEB_ROOT, "static")
TEMPLATES_DIR = os.path.join(WEB_ROOT, "templates")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
