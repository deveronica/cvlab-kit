"""Agent 소스 → 노드 그래프 변환 (CVLab-Kit 패턴 기반)

CVLab-Kit Agent를 일관된 규칙으로 파싱하여 노드 그래프로 변환:
1. setup() → Level 0 컴포넌트 노드 (self.create.* 호출)
2. train_step() → 실행 흐름 노드 (Forward, Loss, Backward 등)
"""

import ast
import logging
import re
from pathlib import Path
from typing import Literal, Optional, cast

from ..models.hierarchy import (
    CodeFlowEdge,
    ComponentCategory,
    FlowType,
    HierarchicalNodeGraph,
    Hierarchy,
    HierarchyLevel,
    HierarchyNode,
    NodeOrigin,
    OriginType,
    Port,
    PropertyInfo,
    PropertySummary,
    SourceLocation,
    ValueSource,
)
from .create_call_parser import CreateCallInfo, CreateCallParser
from .setup_dependency_parser import DependencyEdge, SetupDependencyParser
from .train_step_parser import FlowNode, FlowNodeType, TrainStepParser

logger = logging.getLogger(__name__)

# drill-down 가능한 카테고리
DRILLABLE_CATEGORIES = {"model", "loss", "transform"}

CATEGORY_DEFAULT_PORTS: dict[str, dict[str, list[Port]]] = {
    "model": {
        "inputs": [],
        "outputs": [
            Port(name="parameters", type="parameters"),
            Port(name="out", type="tensor"),
        ],
    },
    "optimizer": {
        "inputs": [Port(name="parameters", type="parameters")],
        "outputs": [],
    },
    "loss": {
        "inputs": [
            Port(name="pred", type="tensor"),
            Port(name="target", type="tensor"),
        ],
        "outputs": [Port(name="loss", type="scalar")],
    },
    "dataset": {
        "inputs": [],
        "outputs": [Port(name="data", type="dataset")],
    },
    "dataloader": {
        "inputs": [Port(name="dataset", type="dataset")],
        "outputs": [Port(name="batch", type="tensor")],
    },
    "transform": {
        "inputs": [Port(name="data", type="tensor")],
        "outputs": [Port(name="data", type="tensor")],
    },
    "metric": {
        "inputs": [
            Port(name="pred", type="tensor"),
            Port(name="target", type="tensor"),
        ],
        "outputs": [Port(name="value", type="scalar")],
    },
    "scheduler": {
        "inputs": [Port(name="optimizer", type="optimizer")],
        "outputs": [],
    },
    "sampler": {
        "inputs": [Port(name="dataset", type="dataset")],
        "outputs": [Port(name="indices", type="list")],
    },
}


class AgentGraphBuilder:
    """CVLab-Kit Agent를 노드 그래프로 변환"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    def _build_properties(
        self,
        impl: str | None,
        impl_source: str | None,
    ) -> tuple[list[PropertyInfo], PropertySummary]:
        properties: list[PropertyInfo] = []
        summary = PropertySummary()

        if impl_source == "positional":
            value_source = ValueSource.HARDCODE
        elif impl_source == "default":
            value_source = ValueSource.DEFAULT if impl else ValueSource.REQUIRED
        else:
            value_source = ValueSource.CONFIG if impl else ValueSource.REQUIRED

        properties.append(PropertyInfo(name="impl", value=impl, source=value_source))

        if value_source == ValueSource.REQUIRED:
            summary.required_count += 1
        elif value_source == ValueSource.CONFIG:
            summary.config_count += 1
        elif value_source == ValueSource.DEFAULT:
            summary.default_count += 1
        elif value_source == ValueSource.HARDCODE:
            summary.hardcode_count += 1

        return properties, summary

    def build(
        self,
        agent_source: str,
        agent_name: str = "",
        phase: str = "initialize",
        method: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> HierarchicalNodeGraph:
        logger.info(
            f"Building graph for agent: {agent_name} (phase: {phase}, method: {method})"
        )

        create_parser = CreateCallParser()
        create_calls = create_parser.parse_setup(agent_source)
        component_attrs = {call.attr_name for call in create_calls}

        # --- Requirement 6: Extract Config Keys from setup() method ---
        setup_config_keys = []
        try:
            tree = ast.parse(agent_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "setup":
                    # setup 메서드 바디 소스 전체 스캔
                    method_source = ast.unparse(node)
                    # self.cfg.get("key")
                    setup_config_keys.extend(
                        re.findall(
                            r"self\.cfg\.get\((\"|')([^\"']+)(\"|')", method_source
                        )
                    )
                    # self.cfg.key
                    setup_config_keys.extend(
                        [
                            (None, m, None)
                            for m in re.findall(
                                r"self\.cfg\.([a-zA-Z_][a-zA-Z0-9_]*)", method_source
                            )
                            if m != "get"
                        ]
                    )
                    # self.cfg["key"]
                    setup_config_keys.extend(
                        re.findall(r"self\.cfg\[(\"|')([^\"']+)(\"|')\]", method_source)
                    )
        except Exception as e:
            logger.error(f"Failed to parse cfg keys: {e}")

        # Flatten and unique
        final_cfg_keys = []
        for match in setup_config_keys:
            if isinstance(match, tuple):
                final_cfg_keys.append(match[1])
            else:
                final_cfg_keys.append(match)
        final_cfg_keys = list(dict.fromkeys(final_cfg_keys))

        if phase == "initialize":
            dep_parser = SetupDependencyParser()
            dep_edges = dep_parser.parse(agent_source, component_attrs)
            return self._build_component_graph(
                create_calls,
                agent_name,
                source_file,
                agent_source,
                dep_edges,
                final_cfg_keys,
            )

        method_name = method or "train_step"
        flow_parser = TrainStepParser(component_attrs)
        flow_nodes = flow_parser.parse(agent_source, method_name=method_name)
        resolved_method = method_name

        if (
            phase == "flow"
            and not flow_nodes
            and method_name in {"val_step", "validate_step", "validation_step"}
        ):
            for candidate in ("validate_step", "validation_step", "val_step"):
                if candidate == method_name:
                    continue
                candidate_nodes = flow_parser.parse(agent_source, method_name=candidate)
                if candidate_nodes:
                    flow_nodes = candidate_nodes
                    resolved_method = candidate
                    break

        if phase == "flow":
            return self._build_flow_graph(
                create_calls,
                flow_nodes,
                agent_name,
                source_file,
                agent_source,
                method_name=method_name,
                extracted_from=resolved_method,
            )

        dep_parser = SetupDependencyParser()
        dep_edges = dep_parser.parse(agent_source, component_attrs)
        return self._build_component_graph(
            create_calls,
            agent_name,
            source_file,
            agent_source,
            dep_edges,
            final_cfg_keys,
        )

    def _build_component_graph(
        self,
        calls: list[CreateCallInfo],
        agent_name: str,
        source_file: Optional[str],
        agent_source: Optional[str] = None,
        dep_edges: Optional[list[DependencyEdge]] = None,
        used_cfg_keys: Optional[list[str]] = None,
    ) -> HierarchicalNodeGraph:
        nodes = []
        edges = []
        used_cfg_keys = used_cfg_keys or []

        sorted_calls = sorted(calls, key=lambda call: call.line)

        role_counters: dict[str, int] = {}
        call_node_ids: list[str] = []
        for call in sorted_calls:
            count = role_counters.get(call.attr_name, 0) + 1
            role_counters[call.attr_name] = count
            call_node_ids.append(
                call.attr_name if count == 1 else f"{call.attr_name}__{count}"
            )

        # First pass: collect which ports are actually used in dependencies
        used_source_ports: dict[str, set[str]] = {}  # node_name -> set of output ports
        used_target_ports: dict[str, set[str]] = {}  # node_name -> set of input ports

        if dep_edges:
            for dep in dep_edges:
                # Determine source port from dependency
                if dep.dependency_type == "method_call":
                    # Extract method name from label like ".parameters()" → "parameters"
                    # Or ".children().parameters()" -> "parameters"
                    label_text = dep.label or ""
                    parts = label_text.strip(".").split("(")
                    source_port = parts[0].split(".")[-1] if parts else "self"
                else:
                    source_port = "self"

                if dep.source not in used_source_ports:
                    used_source_ports[dep.source] = set()
                used_source_ports[dep.source].add(source_port)

                # Determine target port (init parameter name)
                target_category = next(
                    (
                        call.category
                        for call in sorted_calls
                        if call.attr_name == dep.target
                    ),
                    None,
                )
                target_port = self._infer_init_port(
                    target_category or "unknown", dep.source
                )

                if dep.target not in used_target_ports:
                    used_target_ports[dep.target] = set()
                used_target_ports[dep.target].add(target_port)

        for i, call in enumerate(sorted_calls):
            unique_id = call_node_ids[i]
            category = self._map_category(call.category)
            can_drill = call.category in DRILLABLE_CATEGORIES

            is_local_call = getattr(call, "is_local", False)

            if is_local_call:
                # Local component bypasses registry logic
                category = None
                can_drill = False
                inputs = [
                    Port(name=f"arg{idx}", type="any", kind="data")
                    for idx in range(len(call.args))
                ]
                outputs = [Port(name="out", type="any", kind="data")]
                properties, property_summary = [], PropertySummary()
                node_cfg_keys = []
                origin_type = "local_call"
            else:
                category = self._map_category(call.category)
                can_drill = call.category in DRILLABLE_CATEGORIES
                inputs = []
                outputs = []

                # 1. Input Ports (Dependency Injection)
                if call.attr_name in used_target_ports:
                    for port_name in sorted(used_target_ports[call.attr_name]):
                        inputs.append(Port(name=port_name, type="config", kind="data"))

                # Add only used output ports
                if call.attr_name in used_source_ports:
                    for port_name in used_source_ports[call.attr_name]:
                        # Determine type based on common patterns
                        p_type = (
                            "parameters" if "parameter" in port_name.lower() else "any"
                        )
                        outputs.append(Port(name=port_name, type=p_type, kind="data"))

                # Always add "self" output port for instance reference
                if not any(p.name == "self" for p in outputs):
                    outputs.append(Port(name="self", type="module", kind="data"))

                properties, property_summary = self._build_properties(
                    call.impl, call.impl_source
                )

                # Smart config key mapping for component
                node_cfg_keys = []
                for key in used_cfg_keys:
                    if (
                        key in call.attr_name.lower()
                        or call.category in key.lower()
                        or key in ["device", "logger", "epochs", "lr", "batch_size"]
                    ):
                        node_cfg_keys.append(key)

                origin_type = OriginType.CREATE_CALL

            # 1. Input Ports (Dependency Injection)
            if call.attr_name in used_target_ports:
                for port_name in sorted(used_target_ports[call.attr_name]):
                    inputs.append(Port(name=port_name, type="config", kind="data"))

            # Add only used output ports
            if call.attr_name in used_source_ports:
                for port_name in used_source_ports[call.attr_name]:
                    # Determine type based on common patterns
                    p_type = "parameters" if "parameter" in port_name.lower() else "any"
                    outputs.append(Port(name=port_name, type=p_type, kind="data"))

            # Always add "self" output port for instance reference
            if not any(p.name == "self" for p in outputs):
                outputs.append(Port(name="self", type="module", kind="data"))

            properties, property_summary = self._build_properties(
                call.impl, call.impl_source
            )

            # Smart config key mapping for component
            node_cfg_keys = []
            for key in used_cfg_keys:
                if (
                    key in call.attr_name.lower()
                    or call.category in key.lower()
                    or key in ["device", "logger", "epochs", "lr", "batch_size"]
                ):
                    node_cfg_keys.append(key)

            node = HierarchyNode(
                id=unique_id,
                label=call.attr_name,
                object_name=call.attr_name,
                level=HierarchyLevel.COMPONENT,
                category=category,
                can_drill=can_drill,
                inputs=inputs,
                outputs=outputs,
                source=SourceLocation(
                    file=source_file or "", line=call.line, end_line=call.end_line
                ),
                origin=NodeOrigin(
                    type=origin_type,
                    create_path=[call.category]
                    + ([call.variant] if call.variant else []),
                    code_snippet=call.code_snippet,
                ),
                properties=properties,
                property_summary=property_summary,
                used_config_keys=list(dict.fromkeys(node_cfg_keys))
                if node_cfg_keys
                else [],
                metadata={
                    "attr_name": call.attr_name,
                    "order": i + 1,
                    "is_local": is_local_call,
                },
            )
            nodes.append(node)

        def find_node_id(var_name: str, current_line: int) -> Optional[str]:
            matches = [n for n in nodes if n.metadata.get("attr_name") == var_name]
            if not matches:
                return None
            valid = [n for n in matches if n.source.line <= current_line]
            return (
                sorted(valid, key=lambda n: n.source.line, reverse=True)[0].id
                if valid
                else matches[0].id
            )

        edge_counter = 1
        if dep_edges:
            for dep in dep_edges:
                src_id = find_node_id(dep.source, dep.line)
                tgt_id = find_node_id(dep.target, dep.line)
                if src_id and tgt_id:
                    # Req 1+2: Extract method name from dependency_type
                    source_port = "self"
                    flow_type = FlowType.REFERENCE

                    if dep.dependency_type == "method_call":
                        # Extract method name from label like ".parameters()" → "parameters"
                        label_text = dep.label or ""
                        source_port = (
                            label_text.strip(".").split("(")[0]
                            if label_text
                            else "self"
                        )
                        flow_type = (
                            FlowType.PARAMETERS
                            if "parameter" in source_port.lower()
                            else FlowType.REFERENCE
                        )

                    target_category = next(
                        (c.category for c in sorted_calls if c.attr_name == dep.target),
                        None,
                    )
                    target_port = self._infer_init_port(
                        target_category or "unknown", dep.source
                    )

                    edge_id = f"{src_id}_{source_port}_to_{tgt_id}_{target_port}"

                    # Req 3: Assign sequence_index from edge_counter
                    edges.append(
                        CodeFlowEdge(
                            id=edge_id,
                            source_node=src_id,
                            source_port=source_port,
                            target_node=tgt_id,
                            target_port=target_port,
                            flow_type=flow_type,
                            edge_type=("data"),
                            sequence_index=edge_counter,
                            extracted_from="setup",
                        )
                    )
                    edge_counter += 1

        for i in range(len(sorted_calls) - 1):
            src_id, tgt_id = call_node_ids[i], call_node_ids[i + 1]
            edge_id = f"{src_id}_out_to_{tgt_id}_in"
            edges.append(
                CodeFlowEdge(
                    id=edge_id,
                    source_node=src_id,
                    source_port="out",
                    target_node=tgt_id,
                    target_port="in",
                    flow_type=FlowType.CONTROL,
                    edge_type=("execution"),
                    sequence_index=edge_counter,
                    extracted_from="setup",
                    source=SourceLocation(
                        file=source_file or "", line=sorted_calls[i + 1].line
                    ),
                )
            )
            edge_counter += 1

        return HierarchicalNodeGraph(
            id=f"{agent_name}_setup",
            label=agent_name,
            level=HierarchyLevel.COMPONENT,
            hierarchy=Hierarchy(depth=0, path=[]),
            nodes=nodes,
            edges=edges,
            agent_name=agent_name,
            source_file=source_file,
        )

    def _build_flow_graph(
        self,
        create_calls: list[CreateCallInfo],
        flow_nodes: list[FlowNode],
        agent_name: str,
        source_file: Optional[str],
        agent_source: str,
        method_name: str = "train_step",
        extracted_from: Optional[str] = None,
    ) -> HierarchicalNodeGraph:
        extracted_from = extracted_from or method_name
        component_graph = self._build_component_graph(
            create_calls,
            agent_name,
            source_file,
            agent_source,
            dep_edges=[],
            used_cfg_keys=[],
        )
        control_nodes, control_edges = self._build_flow_control_graph(
            flow_nodes, source_file, extracted_from, create_calls
        )

        edges = control_edges

        # P0: Define referenced_components clearly
        referenced_components = set()
        for edge in edges:
            referenced_components.add(edge.source_node)
            referenced_components.add(edge.target_node)

        flow_component_nodes = []
        for node in component_graph.nodes:
            if node.id in referenced_components:
                flow_node = node.model_copy()
                flow_node.inputs = [Port(name="in", type="execution", kind="exec")]
                flow_node.outputs = [Port(name="out", type="execution", kind="exec")]
                flow_component_nodes.append(flow_node)

        return HierarchicalNodeGraph(
            id=f"{agent_name}_{method_name}",
            label=f"{agent_name} - {method_name}()",
            level=HierarchyLevel.OPERATION,
            hierarchy=Hierarchy(depth=0, path=[]),
            nodes=flow_component_nodes + control_nodes,
            edges=edges,
            agent_name=agent_name,
            source_file=source_file,
        )

    def _build_flow_control_graph(
        self,
        flow_nodes: list[FlowNode],
        source_file: Optional[str],
        extracted_from: str,
        create_calls: list[CreateCallInfo],
    ) -> tuple[list[HierarchyNode], list[CodeFlowEdge]]:
        execution_types = {
            FlowNodeType.FORWARD,
            FlowNodeType.LOSS_COMPUTE,
            FlowNodeType.METRIC_UPDATE,
            FlowNodeType.METRIC_COMPUTE,
            FlowNodeType.METRIC_RESET,
            FlowNodeType.OPTIMIZER_STEP,
            FlowNodeType.OPTIMIZER_ZERO,
            FlowNodeType.MODE_CHANGE,
            FlowNodeType.METHOD_CALL,
        }
        control_nodes, control_edges = [], []
        last_producer: dict[str, tuple[str, str]] = {
            "batch": ("method_inputs", "batch")
        }

        control_nodes.append(
            HierarchyNode(
                id="method_inputs",
                label="METHOD INPUTS",
                level=HierarchyLevel.OPERATION,
                category=ComponentCategory.UNKNOWN,
                can_drill=False,
                inputs=[],
                outputs=[Port(name="batch", type="tensor", kind="data")],
                source=SourceLocation(file=source_file or "", line=None),
                metadata={"node_type": "inputs"},  # line=0 -> line=None
            )
        )

        def add_control_node(
            node_type,
            label,
            line,
            end_line,
            inputs=None,
            outputs=None,
            code_snippet="",
            category=None,
            extra_metadata=None,
            data_inputs=None,
            data_outputs=None,
        ):
            node_id = f"{node_type}_{line}"
            input_ports = [
                Port(
                    name=name,
                    type="execution" if name in ("in", "true", "false") else "any",
                    kind="exec" if name in ("in", "true", "false") else "data",
                )
                for name in (inputs or ["in"])
            ]
            output_ports = [
                Port(
                    name=name,
                    type="execution" if name in ("out", "true", "false") else "any",
                    kind="exec" if name in ("out", "true", "false") else "data",
                )
                for name in (outputs or ["out"])
            ]
            if data_inputs:
                input_ports.extend(
                    [Port(name=name, type="any", kind="data") for name in data_inputs]
                )
            if data_outputs:
                output_ports.extend(
                    [Port(name=name, type="any", kind="data") for name in data_outputs]
                )
            control_nodes.append(
                HierarchyNode(
                    id=node_id,
                    label=label,
                    level=HierarchyLevel.OPERATION,
                    category=category,
                    can_drill=False,
                    inputs=input_ports,
                    outputs=output_ports,
                    source=SourceLocation(
                        file=source_file or "", line=line, end_line=end_line
                    ),
                    origin=NodeOrigin(
                        type=OriginType.METHOD_CALL, code_snippet=code_snippet
                    ),
                    metadata={
                        "node_type": node_type,
                        "data_inputs": data_inputs or [],
                        "data_outputs": data_outputs or [],
                        **(extra_metadata or {}),
                    },
                )
            )
            return node_id

        def add_edge(
            src,
            tgt,
            line,
            src_p,
            tgt_p,
            kind: Literal["execution", "data"] = "execution",
            flow=FlowType.CONTROL,
            var=None,
        ):
            control_edges.append(
                CodeFlowEdge(
                    id=f"{kind}_{src}_{tgt}_{line}_{src_p}_{tgt_p}",
                    source_node=src,
                    source_port=src_p,
                    target_node=tgt,
                    target_port=tgt_p,
                    flow_type=flow,
                    edge_type=cast(Literal["execution", "data"], kind),
                    variable_name=var,
                    extracted_from=extracted_from,
                    source=SourceLocation(file=source_file or "", line=line),
                )
            )

        def build_sequence(
            nodes_list, entry_id, entry_port="out", component_map=None, last_line=0
        ):
            prev_id, curr_p, prev_l = entry_id, entry_port, last_line
            for flow in nodes_list:
                if flow.type == FlowNodeType.CONDITIONAL:
                    node_id = add_control_node(
                        "control_if",
                        flow.label,
                        flow.line,
                        flow.end_line or flow.line,
                        outputs=["true", "false"],
                        code_snippet=flow.code_snippet,
                        data_inputs=flow.input_vars,
                    )
                    then_end, then_p, _ = build_sequence(
                        flow.children, node_id, "true", component_map, flow.line
                    )
                    else_end, else_p, _ = build_sequence(
                        flow.else_children, node_id, "false", component_map, flow.line
                    )
                    merge_id = add_control_node(
                        "control_merge",
                        "merge",
                        flow.end_line or flow.line,
                        flow.end_line or flow.line,
                    )
                    add_edge(
                        then_end or node_id,
                        merge_id,
                        flow.line,
                        then_p if then_end else "true",
                        "in",
                    )
                    add_edge(
                        else_end or node_id,
                        merge_id,
                        flow.line,
                        else_p if else_end else "false",
                        "in",
                    )
                    prev_id, curr_p, prev_l = (
                        merge_id,
                        "out",
                        flow.end_line or flow.line,
                    )
                    continue
                elif flow.type in execution_types:
                    category = self._flow_type_to_category(
                        flow.type, flow.component_ref, component_map or {}
                    )
                    node_id = add_control_node(
                        f"process_{flow.type.value}",
                        flow.component_ref or "module",
                        flow.line,
                        flow.end_line or flow.line,
                        code_snippet=flow.code_snippet,
                        category=category,
                        data_inputs=flow.input_vars,
                        data_outputs=flow.output_vars,
                    )
                else:
                    node_id = add_control_node(
                        "control_step",
                        flow.label,
                        flow.line,
                        flow.end_line or flow.line,
                        code_snippet=flow.code_snippet,
                        data_inputs=flow.input_vars,
                        data_outputs=flow.output_vars,
                    )

                if prev_id:
                    add_edge(prev_id, node_id, flow.line, curr_p, "in")
                if node_id:
                    for v in flow.input_vars:
                        if v in last_producer:
                            p_id, p_p = last_producer[v]
                            if p_id != node_id:
                                add_edge(
                                    p_id,
                                    node_id,
                                    flow.line,
                                    p_p,
                                    v,
                                    "data",
                                    FlowType.TENSOR,
                                    v,
                                )
                    for v in flow.output_vars:
                        last_producer[v] = (node_id, v)
                    prev_id, curr_p, prev_l = node_id, "out", flow.end_line or flow.line
            return prev_id, curr_p, prev_l

        comp_map = {c.attr_name: c for c in create_calls}
        build_sequence(flow_nodes, None, "out", comp_map)
        return control_nodes, control_edges

    def _map_category(self, raw):
        m = {
            "model": ComponentCategory.MODEL,
            "optimizer": ComponentCategory.OPTIMIZER,
            "loss": ComponentCategory.LOSS,
            "dataset": ComponentCategory.DATASET,
            "dataloader": ComponentCategory.DATALOADER,
            "transform": ComponentCategory.TRANSFORM,
            "metric": ComponentCategory.METRIC,
            "sampler": ComponentCategory.SAMPLER,
            "scheduler": ComponentCategory.SCHEDULER,
        }
        return m.get(raw, ComponentCategory.UNKNOWN)

    def _infer_init_port(self, cat, src):
        m = {
            "optimizer": "params",
            "dataloader": "dataset",
            "scheduler": "optimizer",
            "sampler": "dataset",
        }
        if cat in m:
            return m[cat]
        for t in ["dataset", "model", "optimizer", "transform"]:
            if t in src.lower():
                return t
        return "in"

    def _flow_type_to_category(self, flow_t, ref, comp_map):
        if ref in comp_map:
            return self._map_category(comp_map[ref].category)
        m = {
            FlowNodeType.FORWARD: ComponentCategory.MODEL,
            FlowNodeType.LOSS_COMPUTE: ComponentCategory.LOSS,
            FlowNodeType.BACKWARD: ComponentCategory.LOSS,
            FlowNodeType.OPTIMIZER_STEP: ComponentCategory.OPTIMIZER,
            FlowNodeType.OPTIMIZER_ZERO: ComponentCategory.OPTIMIZER,
            FlowNodeType.METRIC_UPDATE: ComponentCategory.METRIC,
            FlowNodeType.METRIC_COMPUTE: ComponentCategory.METRIC,
            FlowNodeType.METRIC_RESET: ComponentCategory.METRIC,
            FlowNodeType.MODE_CHANGE: ComponentCategory.MODEL,
        }
        return m.get(flow_t)
