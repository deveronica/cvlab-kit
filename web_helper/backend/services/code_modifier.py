"""
Code Modifier Service

AST-based code modification for Agent source files.
Handles bidirectional sync between node graph and Python code.

Key Operations:
- Remove component definitions (self.x = self.create.category())
- Remove config usages (cfg.get(), cfg.x)
- Add new components
- Update YAML config files

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     Code Modification Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Node Edit ──► AST Analysis ──► Line Removal ──► Code Update    │
│                                                                  │
│  Example:                                                        │
│  Delete "model" node                                             │
│      ↓                                                           │
│  Find: self.model = self.create.model()                         │
│      ↓                                                           │
│  Remove lines 45-45                                              │
│      ↓                                                           │
│  Remove from YAML: model: resnet18                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

import ast
import re
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging
import yaml

logger = logging.getLogger(__name__)


class CodeModifier:
    """
    Modifies Python source code and YAML config files
    based on node graph changes.

    Usage:
        modifier = CodeModifier()

        # Remove a component
        result = modifier.remove_component(
            "cvlabkit/agent/my_agent.py",
            "model",  # variable name
        )

        # Update YAML
        modifier.update_yaml(
            "config/experiment.yaml",
            {"model": None},  # Set to None to remove
        )
    """

    def __init__(self, backup_dir: str = "./.code_backups"):
        """
        Initialize the code modifier.

        Args:
            backup_dir: Directory to store backups before modification
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Backup Management
    # =========================================================================

    def _backup_file(self, file_path: str) -> Path:
        """
        Create a backup of a file before modification.

        Args:
            file_path: Path to file to backup

        Returns:
            Path to backup file
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.stem}_{timestamp}{source.suffix}"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(source, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path

    # =========================================================================
    # Component Removal
    # =========================================================================

    def remove_component(
        self,
        agent_path: str,
        component_name: str,
        create_backup: bool = True,
    ) -> dict:
        """
        Remove a component definition from an agent file.

        Finds and removes lines like:
            self.{component_name} = self.create.{category}(...)

        Args:
            agent_path: Path to the agent Python file
            component_name: Name of the component variable (e.g., "model")
            create_backup: Whether to create a backup before modifying

        Returns:
            Dict with modification results
        """
        path = Path(agent_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {agent_path}"}

        if create_backup:
            self._backup_file(agent_path)

        # Read source
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error in file: {e}"}

        # Find lines to remove
        lines_to_remove = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if target is self.{component_name}
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == component_name
                    ):
                        # Check if value is self.create.{category}(...)
                        if self._is_create_call(node.value):
                            lines_to_remove.append((node.lineno, node.end_lineno or node.lineno))
                            logger.info(f"Found component at line {node.lineno}: self.{component_name}")

        if not lines_to_remove:
            return {
                "success": False,
                "error": f"Component '{component_name}' not found in {agent_path}",
            }

        # Remove lines (reverse order to maintain line numbers)
        removed_code = []
        for start, end in sorted(lines_to_remove, reverse=True):
            removed_lines = lines[start - 1 : end]
            removed_code.extend(removed_lines)
            del lines[start - 1 : end]

        # Write modified source
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "file": str(path),
            "component": component_name,
            "lines_removed": lines_to_remove,
            "removed_code": "".join(removed_code).strip(),
        }

    def _is_create_call(self, node: ast.expr) -> bool:
        """Check if an expression is a self.create.{category}(...) call."""
        if not isinstance(node, ast.Call):
            return False

        # Could be self.create.model() or self.create.model.variant()
        func = node.func
        while isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Attribute):
                if (
                    isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "self"
                    and func.value.attr == "create"
                ):
                    return True
                func = func.value
            elif isinstance(func.value, ast.Name) and func.value.id == "self":
                return func.attr == "create"
            else:
                break
        return False

    # =========================================================================
    # Config Removal
    # =========================================================================

    def remove_config_usage(
        self,
        agent_path: str,
        config_key: str,
        create_backup: bool = True,
    ) -> dict:
        """
        Remove usages of a config key from an agent file.

        Finds and removes lines like:
            x = cfg.get("{config_key}", ...)
            x = self.cfg.{config_key}

        Args:
            agent_path: Path to the agent Python file
            config_key: Config key to remove usages of
            create_backup: Whether to create backup

        Returns:
            Dict with modification results
        """
        path = Path(agent_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {agent_path}"}

        if create_backup:
            self._backup_file(agent_path)

        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        lines_to_remove = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check for cfg.get("key", ...) or self.cfg.get("key", ...)
                if self._is_config_get(node.value, config_key):
                    lines_to_remove.append((node.lineno, node.end_lineno or node.lineno))

        if not lines_to_remove:
            return {
                "success": False,
                "error": f"Config key '{config_key}' usage not found",
            }

        # Remove lines
        removed_code = []
        for start, end in sorted(lines_to_remove, reverse=True):
            removed_lines = lines[start - 1 : end]
            removed_code.extend(removed_lines)
            del lines[start - 1 : end]

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "file": str(path),
            "config_key": config_key,
            "lines_removed": lines_to_remove,
            "removed_code": "".join(removed_code).strip(),
        }

    def _is_config_get(self, node: ast.expr, key: str) -> bool:
        """Check if expression is cfg.get("key", ...) for the given key."""
        if not isinstance(node, ast.Call):
            return False

        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get":
            # Check first argument matches key
            if node.args and isinstance(node.args[0], ast.Constant):
                return node.args[0].value == key
        return False

    # =========================================================================
    # YAML Modification
    # =========================================================================

    def update_yaml(
        self,
        yaml_path: str,
        changes: dict,
        create_backup: bool = True,
        preserve_format: bool = True,
    ) -> dict:
        """
        Update a YAML config file.

        Supports nested paths with dot notation:
            {"loss.supervised": "cross_entropy"}
            → config["loss"]["supervised"] = "cross_entropy"

        Args:
            yaml_path: Path to YAML file
            changes: Dict of changes. Set value to None to remove a key.
                     Keys can use dot notation for nested paths.
            create_backup: Whether to create backup
            preserve_format: Try to preserve original formatting (comments, order)

        Returns:
            Dict with modification results
        """
        path = Path(yaml_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {yaml_path}"}

        # Read original content
        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Validate YAML syntax before modification
        try:
            yaml.safe_load(original_content)
        except yaml.YAMLError as e:
            return {"success": False, "error": f"Invalid YAML syntax: {e}"}

        if create_backup:
            self._backup_file(yaml_path)

        keys_removed = []
        keys_updated = []
        modified_content = original_content

        for key, value in changes.items():
            if value is None:
                # Remove key
                result = self._remove_yaml_key_preserving(modified_content, key)
                if result["success"]:
                    modified_content = result["content"]
                    keys_removed.append(key)
            else:
                # Update existing key
                result = self._update_yaml_key_preserving(modified_content, key, value)
                if result["success"]:
                    modified_content = result["content"]
                    keys_updated.append(key)
                else:
                    # Key doesn't exist - try to add it
                    result = self._add_yaml_key_preserving(modified_content, key, value)
                    if result["success"]:
                        modified_content = result["content"]
                        keys_updated.append(key)
                    elif not preserve_format:
                        # Fallback to parse-modify-stringify
                        modified_content = self._update_yaml_fallback(modified_content, key, value)
                        keys_updated.append(key)

        # Validate result
        try:
            yaml.safe_load(modified_content)
        except yaml.YAMLError as e:
            return {"success": False, "error": f"YAML syntax error after modification: {e}"}

        # Write modified content
        with open(path, "w", encoding="utf-8") as f:
            f.write(modified_content)

        return {
            "success": True,
            "file": str(path),
            "keys_removed": keys_removed,
            "keys_updated": keys_updated,
        }

    def _format_yaml_value(self, value: any) -> str:
        """Format a Python value for YAML output."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            # Check if string needs quoting
            needs_quote = (
                ":" in value or "#" in value or
                "'" in value or '"' in value or
                value.startswith(" ") or value.endswith(" ") or
                value == ""
            )
            if needs_quote:
                escaped = value.replace('"', '\\"')
                return f'"{escaped}"'
            return value
        if isinstance(value, list):
            # Format as inline array: [val1, val2, val3]
            items = ", ".join(self._format_yaml_value(v) for v in value)
            return f"[{items}]"
        if isinstance(value, dict):
            # Use yaml.dump for complex objects
            return yaml.dump(value, default_flow_style=True).strip()
        return str(value)

    def _update_yaml_key_preserving(self, content: str, key: str, value: any) -> dict:
        """
        Update a YAML key while preserving original formatting.

        Supports nested paths: "loss.supervised" -> config["loss"]["supervised"]

        Returns:
            {"success": bool, "content": str}
        """
        formatted_value = self._format_yaml_value(value)
        parts = key.split(".")

        if len(parts) == 1:
            # Top-level key: find "key: value" at root level
            pattern = rf"(^{re.escape(key)}\s*:\s*)([^#\n]+)(#.*)?$"
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                # Preserve inline comment if present
                comment = match.group(3) or ""
                replacement = f"{match.group(1)}{formatted_value}{' ' + comment if comment else ''}"
                new_content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
                return {"success": True, "content": new_content}
        else:
            # Nested key: loss.supervised -> find "supervised:" under "loss:"
            # Look for pattern with proper indentation
            parent_key = parts[0]
            child_key = parts[-1]

            # Pattern: find child_key at indented level within parent block
            # e.g., "  supervised: cross_entropy" within "loss:" block
            pattern = rf"(^[ \t]+{re.escape(child_key)}\s*:\s*)([^#\n]+)(#.*)?$"
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                comment = match.group(3) or ""
                replacement = f"{match.group(1)}{formatted_value}{' ' + comment if comment else ''}"
                new_content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
                return {"success": True, "content": new_content}

        # Pattern not found - key may need to be added
        return {"success": False, "content": content}

    def _remove_yaml_key_preserving(self, content: str, key: str) -> dict:
        """
        Remove a YAML key while preserving formatting.

        Returns:
            {"success": bool, "content": str}
        """
        parts = key.split(".")

        if len(parts) == 1:
            # Top-level key: remove entire line
            pattern = rf"^{re.escape(key)}\s*:.*$\n?"
            new_content, count = re.subn(pattern, "", content, flags=re.MULTILINE)
            return {"success": count > 0, "content": new_content}
        else:
            # Nested key: remove indented line
            child_key = parts[-1]
            pattern = rf"^[ \t]+{re.escape(child_key)}\s*:.*$\n?"
            new_content, count = re.subn(pattern, "", content, flags=re.MULTILINE)
            return {"success": count > 0, "content": new_content}

    def _add_yaml_key_preserving(self, content: str, key: str, value: any) -> dict:
        """
        Add a new YAML key while preserving formatting.

        For top-level keys, appends at end of file.
        For nested keys, appends under the parent block.

        Returns:
            {"success": bool, "content": str}
        """
        formatted_value = self._format_yaml_value(value)
        parts = key.split(".")

        if len(parts) == 1:
            # Top-level key: append at end
            if not content.endswith("\n"):
                content += "\n"
            content += f"{key}: {formatted_value}\n"
            return {"success": True, "content": content}
        else:
            # Nested key: find parent and add under it
            parent_key = parts[0]
            child_key = parts[-1]

            # Find the parent block
            parent_pattern = rf"^{re.escape(parent_key)}\s*:"
            parent_match = re.search(parent_pattern, content, re.MULTILINE)

            if parent_match:
                # Determine indentation by looking at existing children
                indent_pattern = rf"^([ \t]+)\w+\s*:"
                indent_match = re.search(indent_pattern, content[parent_match.end():], re.MULTILINE)
                indent = indent_match.group(1) if indent_match else "  "

                # Find where to insert (after parent line)
                lines = content.split("\n")
                parent_line_idx = content[:parent_match.start()].count("\n")

                # Find the end of parent block
                insert_idx = parent_line_idx + 1
                for i in range(parent_line_idx + 1, len(lines)):
                    line = lines[i]
                    if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                        # Hit next top-level key
                        break
                    if line.strip():
                        insert_idx = i + 1

                # Insert new child
                new_line = f"{indent}{child_key}: {formatted_value}"
                lines.insert(insert_idx, new_line)
                return {"success": True, "content": "\n".join(lines)}
            else:
                # Parent doesn't exist, create it with child
                if not content.endswith("\n"):
                    content += "\n"
                content += f"{parent_key}:\n  {child_key}: {formatted_value}\n"
                return {"success": True, "content": content}

    def _update_yaml_fallback(self, content: str, key: str, value: any) -> str:
        """
        Fallback: parse-modify-stringify (loses comments).
        Supports nested paths.
        """
        try:
            config = yaml.safe_load(content) or {}
        except yaml.YAMLError:
            return content

        parts = key.split(".")
        current = config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value
        current[parts[-1]] = value

        return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def remove_yaml_key(
        self,
        yaml_path: str,
        key: str,
        create_backup: bool = True,
    ) -> dict:
        """
        Remove a key from a YAML file.

        Args:
            yaml_path: Path to YAML file
            key: Key to remove (supports nested keys with dot notation)
            create_backup: Whether to create backup

        Returns:
            Dict with modification results
        """
        return self.update_yaml(yaml_path, {key: None}, create_backup)

    # =========================================================================
    # Component Addition
    # =========================================================================

    def _get_existing_roles(self, tree: ast.AST) -> set[str]:
        """Get all existing role names (self.xxx = self.create.yyy()) from AST."""
        roles = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and self._is_create_call(node.value)
                    ):
                        roles.add(target.attr)
        return roles

    def _generate_unique_role(self, base_role: str, existing_roles: set[str]) -> str:
        """Generate unique role name by appending _2, _3, etc. if needed."""
        if base_role not in existing_roles:
            return base_role

        counter = 2
        while f"{base_role}_{counter}" in existing_roles:
            counter += 1
        return f"{base_role}_{counter}"

    def add_component(
        self,
        agent_path: str,
        category: str,
        impl: str,
        role: Optional[str] = None,
        config: Optional[dict] = None,
        create_backup: bool = True,
    ) -> dict:
        """
        Add a new component to an agent file.

        Adds a line like:
            self.{role} = self.create.{category}()
        or with impl:
            self.{role} = self.create.{category}(impl="{impl}")

        Inside the setup() method.

        Args:
            agent_path: Path to agent file
            category: Component category (model, optimizer, etc.)
            impl: Implementation name (e.g., "resnet18")
            role: Variable name (auto-generated from category if None)
            config: Optional config params (not used in code, for YAML)
            create_backup: Whether to create backup

        Returns:
            Dict with modification results including actual role used
        """
        path = Path(agent_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {agent_path}"}

        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        # Get existing roles to handle duplicates
        existing_roles = self._get_existing_roles(tree)

        # Generate role if not provided
        base_role = role or category
        actual_role = self._generate_unique_role(base_role, existing_roles)

        if create_backup:
            self._backup_file(agent_path)

        # Find setup method
        setup_end_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                # Find the last statement in setup
                if node.body:
                    last_stmt = node.body[-1]
                    setup_end_line = last_stmt.end_lineno or last_stmt.lineno

        if setup_end_line is None:
            return {"success": False, "error": "setup() method not found"}

        # Generate new line
        # Determine indentation from existing code
        indent = "        "  # Default 8 spaces
        for line in lines[setup_end_line - 5 : setup_end_line]:
            if line.strip().startswith("self."):
                indent = line[: len(line) - len(line.lstrip())]
                break

        # Build the create call
        if impl:
            new_line = f'{indent}self.{actual_role} = self.create.{category}(impl="{impl}")\n'
        else:
            new_line = f"{indent}self.{actual_role} = self.create.{category}()\n"

        # Insert after last setup statement
        lines.insert(setup_end_line, new_line)

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "file": str(path),
            "category": category,
            "impl": impl,
            "role": actual_role,
            "role_auto_generated": role is None or actual_role != base_role,
            "inserted_at": setup_end_line + 1,
            "code": new_line.strip(),
        }

    # =========================================================================
    # Component Update
    # =========================================================================

    def update_component_impl(
        self,
        agent_path: str,
        role: str,
        new_impl: str,
        create_backup: bool = True,
    ) -> dict:
        """
        Update implementation in existing self.create.xxx() call.

        Before: self.model = self.create.model()
        After:  self.model = self.create.model(impl="resnet50")

        Or update existing impl:
        Before: self.model = self.create.model(impl="resnet18")
        After:  self.model = self.create.model(impl="resnet50")

        Args:
            agent_path: Path to agent file
            role: Variable name (e.g., "model")
            new_impl: New implementation name
            create_backup: Whether to create backup

        Returns:
            Dict with modification results
        """
        path = Path(agent_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {agent_path}"}

        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        # Find the assignment for this role
        target_line = None
        old_code = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == role
                        and self._is_create_call(node.value)
                    ):
                        target_line = node.lineno
                        end_line = node.end_lineno or node.lineno
                        old_code = "".join(lines[target_line - 1 : end_line])
                        break

        if target_line is None:
            return {
                "success": False,
                "error": f"Component '{role}' not found in {agent_path}",
            }

        if create_backup:
            self._backup_file(agent_path)

        # Get the original line and modify it
        original_line = lines[target_line - 1]
        indent = original_line[: len(original_line) - len(original_line.lstrip())]

        # Use regex to update/add impl argument
        # Pattern: self.create.category(...) or self.create.category.subcategory(...)
        # We need to find the category from the original line
        match = re.search(r"self\.create\.(\w+(?:\.\w+)?)\s*\(", original_line)
        if not match:
            return {"success": False, "error": "Could not parse create call"}

        category_path = match.group(1)

        # Build new line
        new_line = f'{indent}self.{role} = self.create.{category_path}(impl="{new_impl}")\n'

        # Replace the line
        lines[target_line - 1] = new_line

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "file": str(path),
            "role": role,
            "old_impl": old_code.strip() if old_code else None,
            "new_impl": new_impl,
            "line": target_line,
            "code": new_line.strip(),
        }

    # =========================================================================
    # Edge Addition (Dependency)
    # =========================================================================

    def add_dependency_edge(
        self,
        agent_path: str,
        source_role: str,
        target_role: str,
        source_port: str = "parameters",
        create_backup: bool = True,
    ) -> dict:
        """
        Add data dependency in setup() by modifying create call.

        Example: Add model -> optimizer dependency with "parameters" port
        Before: self.optimizer = self.create.optimizer()
        After:  self.optimizer = self.create.optimizer(self.model.parameters())

        Args:
            agent_path: Path to agent file
            source_role: Source component role (e.g., "model")
            target_role: Target component role (e.g., "optimizer")
            source_port: Port/method name (e.g., "parameters")
            create_backup: Whether to create backup

        Returns:
            Dict with modification results
        """
        path = Path(agent_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {agent_path}"}

        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        # Find the target component's create call
        target_line = None
        target_category = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == target_role
                        and self._is_create_call(node.value)
                    ):
                        target_line = node.lineno
                        # Extract category from create call
                        func = node.value.func
                        while isinstance(func, ast.Attribute):
                            if func.attr != "create":
                                target_category = func.attr
                                break
                            func = func.value
                        break

        if target_line is None:
            return {
                "success": False,
                "error": f"Target component '{target_role}' not found",
            }

        if create_backup:
            self._backup_file(agent_path)

        # Get the original line and modify it
        original_line = lines[target_line - 1]
        indent = original_line[: len(original_line) - len(original_line.lstrip())]

        # Build the dependency argument
        dependency_arg = f"self.{source_role}.{source_port}()"

        # Build new line with dependency
        if target_category:
            new_line = f"{indent}self.{target_role} = self.create.{target_category}({dependency_arg})\n"
        else:
            # Fallback - try to preserve original structure
            match = re.search(r"self\.create\.(\w+(?:\.\w+)?)\s*\(", original_line)
            if match:
                cat = match.group(1)
                new_line = f"{indent}self.{target_role} = self.create.{cat}({dependency_arg})\n"
            else:
                return {"success": False, "error": "Could not parse create call"}

        # Replace the line
        old_code = lines[target_line - 1].strip()
        lines[target_line - 1] = new_line

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "file": str(path),
            "source_role": source_role,
            "target_role": target_role,
            "source_port": source_port,
            "line": target_line,
            "old_code": old_code,
            "new_code": new_line.strip(),
        }


    # =========================================================================
    # Edge Removal (Dependency)
    # =========================================================================

    def remove_dependency_edge(
        self,
        agent_path: str,
        source_role: str,
        target_role: str,
        source_port: str = "parameters",
        create_backup: bool = True,
    ) -> dict:
        """
        Remove data dependency from a create call.

        Example: Remove model -> optimizer dependency
        Before: self.optimizer = self.create.optimizer(self.model.parameters())
        After:  self.optimizer = self.create.optimizer()

        Args:
            agent_path: Path to agent file
            source_role: Source component role (e.g., "model")
            target_role: Target component role (e.g., "optimizer")
            source_port: Port/method name (e.g., "parameters")
            create_backup: Whether to create backup

        Returns:
            Dict with modification results
        """
        path = Path(agent_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {agent_path}"}

        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
            lines = source.splitlines(keepends=True)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        # Find the target component's create call with the dependency
        target_line = None
        target_category = None
        if source_port == "self":
            dependency_pattern = f"self.{source_role}"
        else:
            dependency_pattern = f"self.{source_role}.{source_port}()"

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == target_role
                        and self._is_create_call(node.value)
                    ):
                        target_line = node.lineno

                        # Check if this create call has the dependency argument
                        original_line = lines[target_line - 1]
                        if dependency_pattern in original_line:
                            has_dependency = True

                        # Extract category from create call
                        func = node.value.func
                        while isinstance(func, ast.Attribute):
                            if func.attr != "create":
                                target_category = func.attr
                                break
                            func = func.value
                        break

        if target_line is None:
            return {
                "success": False,
                "error": f"Target component '{target_role}' not found",
            }

        if not has_dependency:
            return {
                "success": False,
                "error": f"Dependency from '{source_role}' to '{target_role}' not found",
            }

        if create_backup:
            self._backup_file(agent_path)

        # Get the original line and modify it
        original_line = lines[target_line - 1]
        indent = original_line[: len(original_line) - len(original_line.lstrip())]
        old_code = original_line.strip()

        # Remove the dependency argument
        # Pattern 1: self.create.category(self.source.port())
        # Pattern 2: self.create.category(impl="xxx", self.source.port())
        # Pattern 3: self.create.category(self.source.port(), other_arg)

        # Use regex to remove the dependency argument
        if source_port == "self":
            dep_regex = rf",?\s*(?:\w+\s*=\s*)?self\.{re.escape(source_role)}\s*,?"
        else:
            dep_regex = rf",?\s*(?:\w+\s*=\s*)?self\.{re.escape(source_role)}\.{re.escape(source_port)}\(\)\s*,?"

        # Find where the arguments start and end
        match = re.search(r"self\.create\.(\w+(?:\.\w+)?)\s*\((.*)\)", original_line)
        if not match:
            return {"success": False, "error": "Could not parse create call"}

        category_path = match.group(1)
        args_content = match.group(2)

        # Remove the dependency from arguments
        new_args = re.sub(dep_regex, "", args_content).strip()
        # Clean up any leading/trailing commas
        new_args = re.sub(r"^,\s*", "", new_args)
        new_args = re.sub(r",\s*$", "", new_args)
        new_args = re.sub(r",\s*,", ",", new_args)

        # Build new line
        if new_args:
            new_line = f"{indent}self.{target_role} = self.create.{category_path}({new_args})\n"
        else:
            new_line = f"{indent}self.{target_role} = self.create.{category_path}()\n"

        # Replace the line
        lines[target_line - 1] = new_line

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "file": str(path),
            "source_role": source_role,
            "target_role": target_role,
            "source_port": source_port,
            "line": target_line,
            "old_code": old_code,
            "new_code": new_line.strip(),
        }


# Singleton instance
_code_modifier: Optional[CodeModifier] = None


def get_code_modifier(backup_dir: str = "./.code_backups") -> CodeModifier:
    """Get the singleton CodeModifier instance."""
    global _code_modifier
    if _code_modifier is None:
        _code_modifier = CodeModifier(backup_dir)
    return _code_modifier
