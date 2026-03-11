import logging
import libcst as cst
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ParsedComponent:
    """Represents a component extracted from the source code via CST."""
    id: str
    name: str  # e.g., "self.model"
    type: str  # e.g., "model" (from create.model)
    args: List[str]
    kwargs: Dict[str, str]
    source_range: Dict[str, int]  # start_line, end_line

class AgentCSTVisitor(cst.CSTVisitor):
    """
    Visits the CST to extract agent components and logic flows.
    Focuses on 'setup' and 'train_step' methods.
    """
    
    def __init__(self):
        self.components: List[ParsedComponent] = []
        self.current_method: Optional[str] = None
        self.in_setup = False
        
    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.current_method = node.name.value
        if self.current_method == "setup":
            self.in_setup = True
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self.current_method == "setup":
            self.in_setup = False
        self.current_method = None

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        """
        Detects `self.create.X(...)` calls inside `setup()`.
        """
        if not self.in_setup:
            return True
            
        # Check if it's a call to self.create...
        # Structure: Attribute(value=Attribute(value=Name(value='self'), attr=Name(value='create')), attr=Name(value='COMPONENT_TYPE'))
        if self._is_create_call(node):
            self._extract_component(node)
            
        return True

    def _is_create_call(self, node: cst.Call) -> bool:
        # Simple heuristic check for self.create.foo()
        # This needs robust attribute chain traversing
        try:
            # We expect structure like: self.create.model()
            # node.func -> Attribute
            # node.func.value -> Attribute (self.create)
            # node.func.value.value -> Name (self)
            
            func = node.func
            if isinstance(func, cst.Attribute):
                parent = func.value
                if isinstance(parent, cst.Attribute):
                    if isinstance(parent.value, cst.Name) and parent.value.value == "self":
                        if parent.attr.value == "create":
                            return True
        except Exception:
            pass
        return False

    def _extract_component(self, node: cst.Call):
        try:
            component_type = node.func.attr.value
            
            # Extract args/kwargs simply as strings for now
            args = []
            kwargs = {}
            
            for arg in node.args:
                if arg.keyword:
                    kwargs[arg.keyword.value] = cst.Module([]).code_for_node(arg.value).strip()
                else:
                    args.append(cst.Module([]).code_for_node(arg.value).strip())
            
            # Generate a temporary ID or use the variable name if it's an assignment
            # (Note: assignment handling needs `visit_Assign`)
            
            self.components.append(ParsedComponent(
                id=f"comp_{len(self.components)}", # Placeholder
                name=f"anonymous_{component_type}", # Placeholder
                type=component_type,
                args=args,
                kwargs=kwargs,
                source_range={} # MetadataProvider needed for line numbers
            ))
            
        except Exception as e:
            logger.error(f"Failed to extract component: {e}")

class CSTParserService:
    def parse_agent_code(self, source_code: str) -> Dict[str, Any]:
        """
        Parses the agent source code and returns the graph structure.
        """
        try:
            module = cst.parse_module(source_code)
            visitor = AgentCSTVisitor()
            module.visit(visitor)
            
            return {
                "components": [vars(c) for c in visitor.components],
                # "flow": ... (TODO: Implement train_step parsing)
            }
        except Exception as e:
            logger.error(f"CST Parsing failed: {e}")
            return {"error": str(e)}

