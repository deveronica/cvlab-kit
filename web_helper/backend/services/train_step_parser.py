"""
train_step() 메서드의 실행 흐름을 파싱

CVLab-Kit Agent의 train_step()에서 발생하는 실행 흐름을 노드로 변환:
- self.model(x) → ForwardNode
- self.loss_fn(pred, y) → LossComputeNode
- loss.backward() → BackwardNode
- self.optimizer.step() → OptimizerStepNode
- with torch.no_grad(): → NoGradContext
- if condition: → ConditionalNode
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FlowNodeType(Enum):
    """실행 흐름 노드 타입"""

    FORWARD = "forward"  # self.model(x)
    LOSS_COMPUTE = "loss_compute"  # self.loss_fn(pred, y)
    BACKWARD = "backward"  # loss.backward()
    OPTIMIZER_STEP = "optimizer_step"  # self.optimizer.step()
    OPTIMIZER_ZERO = "optimizer_zero"  # self.optimizer.zero_grad()
    NO_GRAD = "no_grad"  # with torch.no_grad():
    CONDITIONAL = "conditional"  # if condition:
    LOOP = "loop"  # for x in y:
    ASSIGNMENT = "assignment"  # 일반 할당
    METHOD_CALL = "method_call"  # 기타 메서드 호출
    # P0 추가 타입
    RETURN = "return"  # return {...}
    METRIC_UPDATE = "metric_update"  # self.metric.update()
    METRIC_COMPUTE = "metric_compute"  # self.metric.compute()
    METRIC_RESET = "metric_reset"  # self.metric.reset()
    MODE_CHANGE = "mode_change"  # self.model.train(), self.model.eval()
    DATA_UNPACK = "data_unpack"  # inputs, labels = batch
    # P1 추가 타입
    DEVICE_TRANSFER = "device_transfer"  # x.to(device), x.cuda(), x.cpu()
    CONFIG_ACCESS = "config_access"  # self.cfg.get("key"), self.cfg.key


@dataclass
class FlowNode:
    """실행 흐름 노드"""

    type: FlowNodeType
    label: str
    line: int
    end_line: int = 0
    component_ref: Optional[str] = None  # self.{component} 참조
    children: list["FlowNode"] = field(default_factory=list)  # 중첩 구조
    else_children: list["FlowNode"] = field(default_factory=list)  # else 브랜치
    condition: Optional[str] = None  # if/while 조건
    code_snippet: str = ""
    input_vars: list[str] = field(default_factory=list)  # 입력 변수명 목록
    output_vars: list[str] = field(default_factory=list)  # 출력 변수명 목록


class TrainStepParser(ast.NodeVisitor):
    """train_step() 실행 흐름 추출"""

    BACKWARD_PATTERNS = {"backward"}
    STEP_PATTERNS = {"step"}
    ZERO_GRAD_PATTERNS = {"zero_grad"}
    NO_GRAD_PATTERNS = {"no_grad", "inference_mode"}
    # P0 추가 패턴
    METRIC_UPDATE_PATTERNS = {"update"}
    METRIC_COMPUTE_PATTERNS = {"compute"}
    METRIC_RESET_PATTERNS = {"reset"}
    MODE_PATTERNS = {"train", "eval"}
    # P1 추가 패턴
    DEVICE_PATTERNS = {"to", "cuda", "cpu"}

    def __init__(self, component_attrs: Optional[set[str]] = None):
        """
        Args:
            component_attrs: setup()에서 정의된 컴포넌트 속성명 집합
                             e.g., {"model", "optimizer", "sup_loss_fn", ...}
                             None이면 모든 self.* 호출 감지
        """
        self.component_attrs = component_attrs or set()
        self.flow_nodes: list[FlowNode] = []
        self._source_lines: list[str] = []

    def parse(self, source: str, method_name: str = "train_step") -> list[FlowNode]:
        """메서드의 실행 흐름 추출"""
        self.flow_nodes = []
        self._source_lines = source.splitlines()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                for stmt in node.body:
                    flow_node = self._visit_stmt(stmt)
                    if flow_node:
                        self.flow_nodes.append(flow_node)
                break

        return self.flow_nodes

    def _visit_stmt(self, node: ast.stmt) -> Optional[FlowNode]:
        """문장을 FlowNode로 변환"""
        # with torch.no_grad(): → NoGrad 컨테이너
        if isinstance(node, ast.With):
            return self._handle_with(node)

        # if condition: → Conditional
        if isinstance(node, ast.If):
            return self._handle_if(node)

        # for x in y: → Loop
        if isinstance(node, ast.For):
            return self._handle_loop(node)

        # 표현식 (메서드 호출)
        if isinstance(node, ast.Expr):
            return self._handle_expr(node)

        # 할당문
        if isinstance(node, ast.Assign):
            return self._handle_assign(node)

        # P0: return 문
        if isinstance(node, ast.Return):
            return self._handle_return(node)

        return None

    def _handle_with(self, node: ast.With) -> Optional[FlowNode]:
        """with 문 처리 (torch.no_grad() 등)"""
        # with torch.no_grad(): 감지
        for item in node.items:
            context = item.context_expr
            if isinstance(context, ast.Call):
                func = context.func
                # torch.no_grad() 또는 torch.inference_mode()
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in self.NO_GRAD_PATTERNS
                ):
                    children = []
                    for stmt in node.body:
                        child = self._visit_stmt(stmt)
                        if child:
                            children.append(child)

                    return FlowNode(
                        type=FlowNodeType.NO_GRAD,
                        label=f"with torch.{func.attr}():",
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        children=children,
                        code_snippet=self._get_code_snippet(node.lineno, node.lineno),
                    )

        return None

    def _handle_if(self, node: ast.If) -> Optional[FlowNode]:
        """if 문 처리"""
        try:
            condition = ast.unparse(node.test)
            # 조건식에서 변수 추출
            input_vars = self._extract_vars(node.test)
        except Exception:
            condition = "<condition>"
            input_vars = []

        # then 브랜치
        then_children = []
        for stmt in node.body:
            child = self._visit_stmt(stmt)
            if child:
                then_children.append(child)

        # else 브랜치
        else_children = []
        for stmt in node.orelse:
            child = self._visit_stmt(stmt)
            if child:
                else_children.append(child)

        return FlowNode(
            type=FlowNodeType.CONDITIONAL,
            label=f"if {condition[:30]}{'...' if len(condition) > 30 else ''}",
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            condition=condition,
            children=then_children,  # else는 별도 처리 필요시 확장
            else_children=else_children,
            code_snippet=self._get_code_snippet(node.lineno, node.lineno),
            input_vars=input_vars,
        )

    def _handle_loop(self, node: ast.For) -> Optional[FlowNode]:
        """for 문 처리"""
        try:
            target = ast.unparse(node.target)
            iter_expr = ast.unparse(node.iter)
            input_vars = self._extract_vars(node.iter)
            output_vars = self._extract_vars(node.target)
        except Exception:
            target = "<var>"
            iter_expr = "<iter>"
            input_vars = []
            output_vars = []

        children = []
        for stmt in node.body:
            child = self._visit_stmt(stmt)
            if child:
                children.append(child)

        return FlowNode(
            type=FlowNodeType.LOOP,
            label=f"for {target} in {iter_expr[:20]}{'...' if len(iter_expr) > 20 else ''}",
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            children=children,
            code_snippet=self._get_code_snippet(node.lineno, node.lineno),
            input_vars=input_vars,
            output_vars=output_vars,
        )

    def _handle_expr(self, node: ast.Expr) -> Optional[FlowNode]:
        """표현식 문장 처리 (메서드 호출)"""
        if not isinstance(node.value, ast.Call):
            return None

        call = node.value
        return self._analyze_call(call, node.lineno, node.end_lineno or node.lineno)

    def _handle_assign(self, node: ast.Assign) -> Optional[FlowNode]:
        """할당문 처리"""
        # P0: 튜플 언패킹 감지 (inputs, labels = batch)
        target = node.targets[0]
        if isinstance(target, ast.Tuple):
            try:
                target_names = [ast.unparse(elt) for elt in target.elts]
                value_str = ast.unparse(node.value)
                input_vars = self._extract_vars(node.value)
                label = f"{', '.join(target_names)} = {value_str}"
                return FlowNode(
                    type=FlowNodeType.DATA_UNPACK,
                    label=label,
                    line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    code_snippet=self._get_code_snippet(node.lineno, node.lineno),
                    input_vars=input_vars,
                    output_vars=target_names,
                )
            except Exception:
                pass

        # 우변이 호출인 경우
        if isinstance(node.value, ast.Call):
            flow_node = self._analyze_call(
                node.value, node.lineno, node.end_lineno or node.lineno
            )
            if flow_node:
                # 좌변 정보 추가
                try:
                    target_str = ast.unparse(node.targets[0])
                    flow_node.label = f"{target_str} = {flow_node.label}"
                    flow_node.output_vars = [target_str]
                except Exception:
                    pass
            return flow_node

        # 일반 변수 할당/연산 (e.g. x = y + 1)
        # Call 분석에서 처리되지 않은 경우
        try:
            target_str = ast.unparse(node.targets[0])
            value_str = ast.unparse(node.value)
            input_vars = self._extract_vars(node.value)
            return FlowNode(
                type=FlowNodeType.ASSIGNMENT,
                label=f"{target_str} = {value_str}",
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                code_snippet=self._get_code_snippet(node.lineno, node.lineno),
                input_vars=input_vars,
                output_vars=[target_str],
            )
        except Exception:
            return None

    def _handle_return(self, node: ast.Return) -> Optional[FlowNode]:
        """P0: return 문 처리"""
        input_vars = []
        try:
            if node.value:
                value_str = ast.unparse(node.value)
                input_vars = self._extract_vars(node.value)
                # 긴 값은 줄임
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                label = f"return {value_str}"
            else:
                label = "return"
        except Exception:
            label = "return <...>"

        return FlowNode(
            type=FlowNodeType.RETURN,
            label=label,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            code_snippet=self._get_code_snippet(node.lineno, node.lineno),
            input_vars=input_vars,
        )

    def _analyze_call(
        self, call: ast.Call, line: int, end_line: int
    ) -> Optional[FlowNode]:
        """호출 분석"""
        func = call.func
        # 인자에서 입력 변수 추출
        input_vars = self._extract_vars_from_args(call.args, call.keywords)

        # loss.backward()
        if isinstance(func, ast.Attribute) and func.attr in self.BACKWARD_PATTERNS:
            # backward의 주체(loss)도 입력으로 처리
            if isinstance(func.value, ast.Name):
                input_vars.insert(0, func.value.id)
            return FlowNode(
                type=FlowNodeType.BACKWARD,
                label="backward()",
                line=line,
                end_line=end_line,
                code_snippet=self._get_code_snippet(line, line),
                input_vars=input_vars,
            )

        # optimizer.step()
        if isinstance(func, ast.Attribute) and func.attr in self.STEP_PATTERNS:
            component = self._get_component_name(func.value)
            return FlowNode(
                type=FlowNodeType.OPTIMIZER_STEP,
                label=f"{component}.step()" if component else "step()",
                line=line,
                end_line=end_line,
                component_ref=component,
                code_snippet=self._get_code_snippet(line, line),
                input_vars=input_vars,
            )

        # optimizer.zero_grad()
        if isinstance(func, ast.Attribute) and func.attr in self.ZERO_GRAD_PATTERNS:
            component = self._get_component_name(func.value)
            return FlowNode(
                type=FlowNodeType.OPTIMIZER_ZERO,
                label=f"{component}.zero_grad()" if component else "zero_grad()",
                line=line,
                end_line=end_line,
                component_ref=component,
                code_snippet=self._get_code_snippet(line, line),
                input_vars=input_vars,
            )

        # P0: self.metric.update(), self.metric.compute(), self.metric.reset()
        if isinstance(func, ast.Attribute):
            method_name = func.attr
            component = self._get_component_name(func.value)
            if component and "metric" in component.lower():
                if method_name in self.METRIC_UPDATE_PATTERNS:
                    return FlowNode(
                        type=FlowNodeType.METRIC_UPDATE,
                        label=f"{component}.update()",
                        line=line,
                        end_line=end_line,
                        component_ref=component,
                        code_snippet=self._get_code_snippet(line, line),
                        input_vars=input_vars,
                    )
                if method_name in self.METRIC_COMPUTE_PATTERNS:
                    return FlowNode(
                        type=FlowNodeType.METRIC_COMPUTE,
                        label=f"{component}.compute()",
                        line=line,
                        end_line=end_line,
                        component_ref=component,
                        code_snippet=self._get_code_snippet(line, line),
                        input_vars=input_vars,
                    )
                if method_name in self.METRIC_RESET_PATTERNS:
                    return FlowNode(
                        type=FlowNodeType.METRIC_RESET,
                        label=f"{component}.reset()",
                        line=line,
                        end_line=end_line,
                        component_ref=component,
                        code_snippet=self._get_code_snippet(line, line),
                        input_vars=input_vars,
                    )

        # P0: self.model.train(), self.model.eval()
        if isinstance(func, ast.Attribute) and func.attr in self.MODE_PATTERNS:
            component = self._get_component_name(func.value)
            if component:
                return FlowNode(
                    type=FlowNodeType.MODE_CHANGE,
                    label=f"{component}.{func.attr}()",
                    line=line,
                    end_line=end_line,
                    component_ref=component,
                    code_snippet=self._get_code_snippet(line, line),
                    input_vars=input_vars,
                )

        # P1: x.to(device), x.cuda(), x.cpu()
        if isinstance(func, ast.Attribute) and func.attr in self.DEVICE_PATTERNS:
            try:
                # 변수명 추출 (x.to(device)에서 x)
                var_name = ast.unparse(func.value)
                # x도 입력으로 추가
                if var_name not in input_vars:
                    input_vars.insert(0, var_name)
                    
                # 인자 추출 (to(self.device)에서 self.device)
                if call.args:
                    device_arg = ast.unparse(call.args[0])
                    label = f"{var_name}.{func.attr}({device_arg})"
                else:
                    label = f"{var_name}.{func.attr}()"
                return FlowNode(
                    type=FlowNodeType.DEVICE_TRANSFER,
                    label=label,
                    line=line,
                    end_line=end_line,
                    code_snippet=self._get_code_snippet(line, line),
                    input_vars=input_vars,
                    # Returns the transformed tensor (same name usually, but technically new object if not in-place)
                    output_vars=[var_name], 
                )
            except Exception:
                pass

        # P1: self.cfg.get("key") or self.cfg("key")
        if isinstance(func, ast.Attribute) and func.attr == "get":
            # Check if it's self.cfg.get()
            if (
                isinstance(func.value, ast.Attribute)
                and func.value.attr == "cfg"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
            ):
                try:
                    # 첫 번째 인자 (key) 추출
                    if call.args:
                        key = ast.unparse(call.args[0])
                        label = f"cfg.get({key})"
                    else:
                        label = "cfg.get()"
                    return FlowNode(
                        type=FlowNodeType.CONFIG_ACCESS,
                        label=label,
                        line=line,
                        end_line=end_line,
                        code_snippet=self._get_code_snippet(line, line),
                        input_vars=input_vars,
                    )
                except Exception:
                    pass

        # self.{component}(...) → Forward 또는 LossCompute
        # Pattern: self.model(x) where func is Name('model') and func.value is Attribute(self.model)
        # But actually for self.model(x), func IS the Attribute node
        if isinstance(func, ast.Attribute):
            if (
                func.attr == "forward"
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
            ):
                component = func.value.attr
                if component and (
                    not self.component_attrs or component in self.component_attrs
                ):
                    is_loss = "loss" in component.lower()
                    node_type = (
                        FlowNodeType.LOSS_COMPUTE if is_loss else FlowNodeType.FORWARD
                    )
                    return FlowNode(
                        type=node_type,
                        label=f"{component}.forward()",
                        line=line,
                        end_line=end_line,
                        component_ref=component,
                        code_snippet=self._get_code_snippet(line, line),
                        input_vars=input_vars,
                    )

            # Direct self.component() call: self.model(x)
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                component = func.attr
                if component and (
                    not self.component_attrs or component in self.component_attrs
                ):
                    # loss 관련인지 확인
                    is_loss = "loss" in component.lower()
                    node_type = (
                        FlowNodeType.LOSS_COMPUTE if is_loss else FlowNodeType.FORWARD
                    )

                    return FlowNode(
                        type=node_type,
                        label=f"{component}()",
                        line=line,
                        end_line=end_line,
                        component_ref=component,
                        code_snippet=self._get_code_snippet(line, line),
                        input_vars=input_vars,
                    )
        
        # 일반 메서드 호출 (Fallback)
        try:
            label = ast.unparse(call)
            # 너무 길면 자름
            if len(label) > 40:
                label = label[:37] + "..."
        except Exception:
            label = "method_call()"
            
        return FlowNode(
            type=FlowNodeType.METHOD_CALL,
            label=label,
            line=line,
            end_line=end_line,
            code_snippet=self._get_code_snippet(line, line),
            input_vars=input_vars,
        )

    def _extract_vars(self, node: ast.AST) -> list[str]:
        """노드에서 변수명 추출"""
        vars = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                vars.append(child.id)
        return list(dict.fromkeys(vars))  # 중복 제거

    def _extract_vars_from_args(self, args: list[ast.expr], keywords: list[ast.keyword]) -> list[str]:
        """인자 목록에서 변수 추출"""
        vars = []
        for arg in args:
            vars.extend(self._extract_vars(arg))
        for kw in keywords:
            vars.extend(self._extract_vars(kw.value))
        return list(dict.fromkeys(vars))

    def _get_component_name(self, node: ast.expr) -> Optional[str]:
        """self.{component}에서 component 이름 추출"""
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None

    def _get_code_snippet(self, start_line: int, end_line: int) -> str:
        """소스 코드 스니펫 추출"""
        if 0 < start_line <= len(self._source_lines):
            lines = self._source_lines[start_line - 1 : end_line]
            return "\n".join(lines).strip()
        return ""


def parse_train_step(
    source: str, component_attrs: Optional[set[str]] = None
) -> list[FlowNode]:
    """train_step() 실행 흐름 추출 (편의 함수)"""
    parser = TrainStepParser(component_attrs)
    return parser.parse(source)
