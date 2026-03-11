import ast
import random
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from ..models.data_type import DataTypeColor

logger = logging.getLogger(__name__)

# 미리 정의된 기본 색상 팔레트
PRESET_COLORS = [
    "#22d3ee", # Cyan
    "#f59e0b", # Amber
    "#4ade80", # Green
    "#f472b6", # Pink
    "#a855f7", # Purple
    "#fb7185", # Rose
    "#818cf8", # Indigo
    "#c084fc", # Fuchsia
    "#2dd4bf", # Teal
]

class TypeDiscoveryService:
    """Scans agent files to discover Python types and assign persistent colors."""
    
    def __init__(self, project_root: Path, db: Session):
        self.project_root = project_root
        self.db = db

    def scan_and_register_types(self):
        """Scans all agents and registers discovered types in DB."""
        agent_dir = self.project_root / "cvlabkit" / "agent"
        if not agent_dir.exists():
            return

        discovered_types = set(["any", "tensor", "parameters", "module", "scalar", "dataset"])
        
        for py_file in agent_dir.glob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    # 함수 인자 어노테이션 추출
                    if isinstance(node, ast.arg) and node.annotation:
                        discovered_types.add(self._get_type_name(node.annotation))
                    # 함수 반환값 어노테이션 추출
                    if isinstance(node, ast.FunctionDef) and node.returns:
                        discovered_types.add(self._get_type_name(node.returns))
            except Exception as e:
                logger.error(f"Failed to scan {py_file}: {e}")

        self._persist_types(discovered_types)

    def _get_type_name(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node).strip().lower()
        except:
            return "any"

    def _persist_types(self, type_names: set[str]):
        """Saves new types to DB with a random preset color if not exists."""
        existing = {t.name: t for t in self.db.query(DataTypeColor).all()}
        
        for name in type_names:
            if name not in existing:
                color = random.choice(PRESET_COLORS)
                new_type = DataTypeColor(name=name, color=color)
                self.db.add(new_type)
                logger.info(f"Registered new type: {name} with color {color}")
        
        self.db.commit()

def get_type_colors(db: Session) -> dict[str, str]:
    """Returns a map of type name to color from DB."""
    return {t.name: t.color for t in db.query(DataTypeColor).all()}
