"""Hierarchical Graph Builder - Simulink-style Node Graph Construction.

This module implements the Simulink-style hierarchical design:
- Level 0: Components directly from setup() (skip method level)
- Double-click drill-down into component internals (model layers, loss sub-components)
- Edges extracted from BOTH setup() AND train_step() (code-driven, not config-guessed)
- Breadcrumb navigation via hierarchy.path

Key principle: If it's not in the code, it doesn't become a node!
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from web_helper.backend.models.hierarchy import (
    ComponentCategory,
    DrillTarget,
    HierarchicalNodeGraph,
    Hierarchy,
    HierarchyLevel,
    HierarchyNode,
    HierarchyPath,
    SourceLocation,
)
from web_helper.backend.services.code_analyzer import AgentCodeAnalyzer
from web_helper.backend.services.model_layer_analyzer import analyze_model_file
from web_helper.backend.services.loss_analyzer import analyze_loss_file
from web_helper.backend.services.transform_analyzer import analyze_transform_file
from web_helper.backend.services.agent_graph_builder import AgentGraphBuilder
from web_helper.backend.services.config_node_generator import (
    inject_config_node,
    ConfigNodeGenerator,
)

if TYPE_CHECKING:
    pass


class HierarchicalGraphBuilder:
    """Hierarchical graph builder implementing Simulink-style design.

    Key principles:
    - Level 0 shows ONLY components from setup() (not methods)
    - Edges are extracted from ACTUAL CODE (setup + train_step)
    - NO config-based edge guessing
    - Double-click drill-down reveals component internals
    """

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self._agent_graph_builder = AgentGraphBuilder(self.project_root)

    def _create_analyzer(self, source_code: str, source_file: str) -> AgentCodeAnalyzer:
        """Create an analyzer for the given source code."""
        return AgentCodeAnalyzer(source_code, source_file)

    def _resolve_impl_from_config(self, agent_name: str, config_path: str | None = None) -> dict[str, str]:
        """Resolve implementation names from YAML config for drillable components.

        Args:
            agent_name: Agent name to find matching config.
            config_path: Optional specific config path.

        Returns:
            Map of category -> implementation name (e.g., {"model": "resnet18"}).
        """
        try:
            config_gen = ConfigNodeGenerator(self.project_root)
            config_data = config_gen._load_config(config_path, agent_name)
            return config_gen._extract_implementations(config_data)
        except Exception:
            return {}

    def _update_node_with_impl(
        self,
        node: HierarchyNode,
        impl_map: dict[str, str],
    ) -> HierarchyNode:
        """Update node metadata with resolved implementation name.

        Args:
            node: The node to update.
            impl_map: Map of category -> implementation name.

        Returns:
            Updated node with impl in metadata.
        """
        if not node.category or not node.can_drill:
            return node

        category = node.category.value
        impl = impl_map.get(category)

        if not impl:
            return node

        # Update metadata with impl
        updated_metadata = dict(node.metadata) if node.metadata else {}
        updated_metadata["impl"] = impl

        return HierarchyNode(
            id=node.id,
            label=node.label,
            level=node.level,
            origin=node.origin,
            can_drill=node.can_drill,
            drill_target=node.drill_target,
            category=node.category,
            inputs=node.inputs,
            outputs=node.outputs,
            source=node.source,
            properties=node.properties,
            property_summary=node.property_summary,
            metadata=updated_metadata,
        )

    def build_for_path(
        self,
        agent_name: str,
        path: str = "",
        phase: str | None = None,
        method: str | None = None,
        impl: str | None = None,
        config_path: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Build hierarchical graph for the given path."""
        if not path:
            # Level 0: Agent components
            return self._build_component_level(agent_name, phase=phase, method=method, config_path=config_path)
        else:
            # Level 1+: Drill-down
            path_parts = path.split("/")
            first_part = path_parts[0]

            # Check if this is a method drill-down (path starts with "method/")
            if first_part == "method" and len(path_parts) > 1:
                method_name = path_parts[1]
                return self._drill_method(agent_name, method_name)

            # Otherwise, drill into component
            component_id = first_part
            return self._build_drill_level(
                agent_name, component_id, path_parts[1:], impl=impl
            )

    def build_from_source(
        self,
        agent_name: str,
        source_code: str,
        source_file: str,
        phase: str | None = None,
        method: str | None = None,
        config_path: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Build Level 0 graph from provided source code."""
        return self._build_component_level(
            agent_name,
            phase=phase,
            method=method,
            source_code=source_code,
            source_file=source_file,
            config_path=config_path,
        )

    def _build_component_level(
        self,
        agent_name: str,
        phase: str | None = None,
        method: str | None = None,
        source_code: str | None = None,
        source_file: str | None = None,
        config_path: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Build Level 0: Agent components from setup().

        This shows:
        - All components created via self.xxx = self.create.yyy()
        - Edges from setup() (e.g., model.parameters() -> optimizer)
        - Edges from train_step() (e.g., model output -> loss_fn)

        Args:
            agent_name: Agent class name (e.g., "classification")
            phase: Optional phase filter ("initialize", "flow", or "method")
                   - "initialize": Show only setup() edges (component creation/wiring)
                   - "flow": Show only train_step() edges (data flow during training)
                   - "method": Show Agent methods as nodes (setup, train_step, val_step)
                   - None: Show all edges
            method: Method name for "flow" phase (e.g., "train_step", "val_step")
            source_code: Override source code for analysis
            source_file: Override source file path for analysis
        """
        if source_code is None or source_file is None:
            agent_path = self.project_root / "cvlabkit" / "agent" / f"{agent_name}.py"
            if not agent_path.exists():
                raise FileNotFoundError(f"Agent not found: {agent_path}")

            source_code = agent_path.read_text(encoding="utf-8")
            source_file = str(agent_path.relative_to(self.project_root))

        # Always use AgentGraphBuilder for Level 0 components
        import logging
        builder_logger = logging.getLogger(__name__)
        builder_logger.info(f"Building Level 0 for {agent_name} with config_path: {config_path}")

        graph = self._agent_graph_builder.build(
            agent_source=source_code,
            agent_name=agent_name,
            phase=phase if phase else "initialize",
            method=method, 
            source_file=source_file,
        )
        
        builder_logger.info(f"Graph build complete. Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

        # Resolve impl from YAML config for drill-down navigation
        impl_map = self._resolve_impl_from_config(agent_name, config_path=config_path)

        # Update drill targets and impl for drillable nodes
        updated_nodes = []
        for node in graph.nodes:
            if node.can_drill and node.category:
                category = node.category.value
                drill_target = DrillTarget(
                    type="component",
                    path=self._find_component_path(category, node.id),
                )

                # Add impl to metadata for drill-down
                impl = impl_map.get(category)
                updated_metadata = dict(node.metadata) if node.metadata else {}
                if impl:
                    updated_metadata["impl"] = impl

                updated_node = HierarchyNode(
                    id=node.id,
                    label=node.label,
                    level=node.level,
                    origin=node.origin,
                    can_drill=node.can_drill,
                    drill_target=drill_target,
                    category=node.category,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    source=node.source,
                    properties=node.properties,
                    property_summary=node.property_summary,
                    used_config_keys=node.used_config_keys,
                    metadata=updated_metadata,
                )
                updated_nodes.append(updated_node)
            else:
                updated_nodes.append(node)

        # Inject YAML config node for Level 0 initialize phase
        if (not phase or phase == "initialize"):
            final_nodes, final_edges = inject_config_node(
                graph_nodes=updated_nodes,
                graph_edges=graph.edges,
                agent_name=agent_name,
                config_path=config_path,
                project_root=self.project_root,
            )
        else:
            final_nodes = updated_nodes
            final_edges = graph.edges

        return HierarchicalNodeGraph(
            id=graph.id,
            label=graph.label,
            level=graph.level,
            hierarchy=graph.hierarchy,
            nodes=final_nodes,
            edges=final_edges,
            agent_name=graph.agent_name,
            source_file=graph.source_file,
            method_range=graph.method_range,
            covered_lines=graph.covered_lines,
            uncovered_lines=graph.uncovered_lines,
        )

    def _build_drill_level(
        self,
        agent_name: str,
        component_id: str,
        remaining_path: list[str],
        impl: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Build Level 1+: Drill into component internals.

        Supports:
        - model: Extract PyTorch layers from __init__ and forward()
        - loss: Extract sub-loss components
        - transform: Extract pipeline steps
        - Others: Return placeholder
        """
        # Get component category from Level 0
        agent_path = self.project_root / "cvlabkit" / "agent" / f"{agent_name}.py"
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent not found: {agent_path}")

        source_code = agent_path.read_text(encoding="utf-8")
        source_file = str(agent_path.relative_to(self.project_root))

        # Find component category using analyzer
        analyzer = self._create_analyzer(source_code, source_file)
        analyzer.parse()
        result = analyzer.analyze()

        category = None
        code_impl = None  # impl from code analysis (HARDCODE or DEFAULT)
        impl_source = None
        for node in result.nodes:
            if node.id == component_id:
                category = node.category
                # Extract impl from metadata if available (Builder context)
                if node.metadata:
                    code_impl = node.metadata.get("impl")
                    impl_source = node.metadata.get("impl_source")
                break

        if not category:
            return self._placeholder_graph(
                agent_name, component_id, "Unknown component"
            )

        # Determine effective impl:
        # 1. API impl parameter takes precedence (Execute context with YAML)
        # 2. Fall back to code-derived impl (Builder context: HARDCODE or DEFAULT)
        effective_impl = impl if impl else code_impl

        # Build parent path for breadcrumb with implementation info
        parent_path = [
            HierarchyPath(
                level=HierarchyLevel.COMPONENT,
                label=self._format_agent_label(agent_name),
                node_id=agent_name,
                graph_id=agent_name,
                category=category,
                implementation=effective_impl,
            )
        ]

        # Dispatch to category-specific drill method
        if category == ComponentCategory.MODEL:
            return self._drill_model(
                agent_name,
                component_id,
                parent_path,
                remaining_path,
                impl=effective_impl,
            )
        elif category == ComponentCategory.LOSS:
            return self._drill_loss(
                agent_name, component_id, parent_path, impl=effective_impl
            )
        elif category == ComponentCategory.TRANSFORM:
            return self._drill_transform(
                agent_name, component_id, parent_path, impl=effective_impl
            )
        else:
            return self._placeholder_graph(
                agent_name,
                component_id,
                f"{category.value} internals",
                parent_path,
            )

    def _drill_model(
        self,
        agent_name: str,
        component_id: str,
        parent_path: list[HierarchyPath],
        remaining_path: list[str],
        impl: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Drill into model component - extract PyTorch layers.

        Uses ModelLayerAnalyzer to parse:
        1. Layer definitions from __init__ (self.xxx = nn.Yyy(...))
        2. Data flow from forward() method
        3. Input/Output nodes for function boundaries
        """
        # Find model source file (prefer impl over component_id)
        search_name = impl if impl else component_id
        model_path = self._find_component_file("model", search_name)

        if not model_path or not model_path.exists():
            # Fallback to placeholder if model file not found
            return self._placeholder_graph(
                agent_name, component_id, f"Model not found: {search_name}", parent_path
            )

        try:
            # Use model layer analyzer to extract layers and data flow
            return analyze_model_file(
                model_path=model_path,
                agent_name=agent_name,
                component_id=component_id,
                parent_path=parent_path,
                project_root=self.project_root,
            )
        except Exception as e:
            # On analysis error, return placeholder with error info
            source_file = str(model_path.relative_to(self.project_root))
            return self._placeholder_graph(
                agent_name,
                component_id,
                f"Analysis error: {str(e)[:50]}",
                parent_path,
                source_file,
            )

    def _drill_loss(
        self,
        agent_name: str,
        component_id: str,
        parent_path: list[HierarchyPath],
        impl: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Drill into loss component - extract sub-losses.

        Uses LossAnalyzer to parse:
        1. Sub-loss definitions from __init__ (self.xxx = SomeLoss(...))
        2. Data flow from forward() method
        3. Loss combination patterns
        """
        # Find loss source file (prefer impl over component_id)
        search_name = impl if impl else component_id
        loss_path = self._find_component_file("loss", search_name)

        if not loss_path or not loss_path.exists():
            return self._placeholder_graph(
                agent_name, component_id, f"Loss not found: {search_name}", parent_path
            )

        try:
            return analyze_loss_file(
                loss_path=loss_path,
                agent_name=agent_name,
                component_id=component_id,
                parent_path=parent_path,
                project_root=self.project_root,
            )
        except Exception as e:
            source_file = str(loss_path.relative_to(self.project_root))
            return self._placeholder_graph(
                agent_name,
                component_id,
                f"Analysis error: {str(e)[:50]}",
                parent_path,
                source_file,
            )

    def _drill_transform(
        self,
        agent_name: str,
        component_id: str,
        parent_path: list[HierarchyPath],
        impl: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Drill into transform - extract pipeline steps.

        Uses TransformAnalyzer to parse:
        1. Pipeline composition (transforms.Compose)
        2. Sub-transform definitions
        3. Sequential transform chain
        """
        # Find transform source file (prefer impl over component_id)
        search_name = impl if impl else component_id
        transform_path = self._find_component_file("transform", search_name)

        if not transform_path or not transform_path.exists():
            return self._placeholder_graph(
                agent_name,
                component_id,
                f"Transform not found: {search_name}",
                parent_path,
            )

        try:
            return analyze_transform_file(
                transform_path=transform_path,
                agent_name=agent_name,
                component_id=component_id,
                parent_path=parent_path,
                project_root=self.project_root,
            )
        except Exception as e:
            source_file = str(transform_path.relative_to(self.project_root))
            return self._placeholder_graph(
                agent_name,
                component_id,
                f"Analysis error: {str(e)[:50]}",
                parent_path,
                source_file,
            )

    def _drill_method(
        self,
        agent_name: str,
        method_name: str,
    ) -> HierarchicalNodeGraph:
        """Drill into a method - show internal component usage flow.

        Args:
            agent_name: Agent class name (e.g., "classification")
            method_name: Method name to drill into (e.g., "train_step")

        Returns:
            HierarchicalNodeGraph showing component flow within the method
        """
        # Load agent source
        agent_path = self.project_root / "cvlabkit" / "agent" / f"{agent_name}.py"
        if not agent_path.exists():
            return self._placeholder_graph(agent_name, method_name, "Agent not found")

        source_code = agent_path.read_text(encoding="utf-8")
        source_file = str(agent_path.relative_to(self.project_root))

        # Create analyzer and analyze method internal flow
        analyzer = self._create_analyzer(source_code, source_file)
        analyzer.parse()
        result = analyzer.analyze_method_internal(method_name)

        if result.errors:
            return self._placeholder_graph(agent_name, method_name, result.errors[0])

        # Build parent path for breadcrumb
        parent_path = [
            HierarchyPath(
                level=HierarchyLevel.METHOD,
                label=self._format_agent_label(agent_name),
                node_id=agent_name,
                graph_id=agent_name,
            )
        ]

        return HierarchicalNodeGraph(
            id=f"{agent_name}.method.{method_name}",
            label=f"{method_name}() flow",
            level=HierarchyLevel.OPERATION,
            hierarchy=Hierarchy(
                parent_graph_id=agent_name,
                parent_node_id=method_name,
                depth=1,
                path=parent_path,
            ),
            nodes=result.nodes,
            edges=result.edges,
            agent_name=agent_name,
            source_file=source_file,
            metadata={"method_name": method_name},
        )

    def _placeholder_graph(
        self,
        agent_name: str,
        component_id: str,
        detail: str,
        parent_path: list[HierarchyPath] | None = None,
        source_file: str | None = None,
    ) -> HierarchicalNodeGraph:
        """Create placeholder graph for components without drill support."""
        return HierarchicalNodeGraph(
            id=f"{agent_name}.{component_id}",
            label=f"{component_id} ({detail})",
            level=HierarchyLevel.LAYER,
            hierarchy=Hierarchy(
                parent_graph_id=agent_name,
                parent_node_id=component_id,
                depth=1,
                path=parent_path or [],
            ),
            nodes=[
                HierarchyNode(
                    id=f"{component_id}_internal",
                    label=f"{component_id} {detail}",
                    level=HierarchyLevel.LAYER,
                    can_drill=False,
                    metadata={"placeholder": True, "detail": detail},
                )
            ],
            edges=[],
            agent_name=agent_name,
            source_file=source_file,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _can_drill(self, category: ComponentCategory | None) -> bool:
        """Determine if a component category supports drill-down."""
        drillable = {
            ComponentCategory.MODEL,
            ComponentCategory.LOSS,
            ComponentCategory.TRANSFORM,
            ComponentCategory.OPTIMIZER,
            ComponentCategory.SCHEDULER,
            ComponentCategory.DATASET,
        }
        return category in drillable

    def _format_agent_label(self, agent_name: str) -> str:
        """Format agent name for display."""
        # Convert snake_case to Title Case
        return agent_name.replace("_", " ").title()

    def _find_component_path(self, category: str, component_id: str) -> str:
        """Find component file path."""
        file_path = self._find_component_file(category, component_id)
        if file_path:
            return str(file_path.relative_to(self.project_root))
        return ""

    def _find_component_file(
        self,
        category: str,
        component_id: str,
    ) -> Path | None:
        """Find source file for a component.

        Searches in cvlabkit/component/{category}/ directory.
        """
        component_dir = self.project_root / "cvlabkit" / "component" / category
        if not component_dir.exists():
            return None

        # Try exact match first
        for py_file in component_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            if py_file.stem.lower() == component_id.lower():
                return py_file
            # Try without underscores
            if py_file.stem.lower().replace("_", "") == component_id.lower().replace(
                "_", ""
            ):
                return py_file

        # Try partial match (component_id might be shortened)
        for py_file in component_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            if component_id.lower() in py_file.stem.lower():
                return py_file

        return None
