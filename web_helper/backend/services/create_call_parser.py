"""CVLab-Kit Agent의 self.create.* 호출을 파싱하여 노드로 변환

CVLab-Kit의 모든 컴포넌트는 일관된 패턴으로 생성됨:
    self.{attr} = self.create.{category}[.{variant}](**kwargs)
    {local_var} = self.create.{category}[.{variant}](**kwargs)  # 로컬 변수도 지원

예시:
    self.model = self.create.model()
    self.weak_transform = self.create.transform.weak()
    self.sup_loss_fn = self.create.loss.supervised()
    self.labeled_loader = self.create.dataloader.labeled(dataset=train_dataset)
    train_dataset = self.create.dataset.train()  # 로컬 변수 할당도 캡처
"""

import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CreateCallInfo:
    """self.create.* 호출 정보"""

    attr_name: str  # 할당된 속성명 (e.g., "model", "weak_transform")
    category: str  # 카테고리 (e.g., "model", "transform")
    variant: Optional[str]  # 변형 (e.g., "weak", "supervised")
    impl: Optional[str] = None  # 구현체명 (e.g., "resnet18" from create.model("resnet18"))
    impl_source: Optional[str] = None  # impl 출처: "positional" | "default" | "yaml" | None
    kwargs: dict = field(default_factory=dict)  # 전달된 키워드 인자
    args: list = field(default_factory=list)  # 전달된 위치 인자
    line: int = 0  # 소스 라인 번호
    end_line: int = 0
    code_snippet: str = ""  # 원본 코드
    is_local: bool = False  # 로컬 변수 여부 (self.x vs x)


class CreateCallParser(ast.NodeVisitor):
    """setup() 메서드에서 self.create.* 호출 추출"""

    def __init__(self):
        self.calls: list[CreateCallInfo] = []
        self._source_lines: list[str] = []

    def parse_setup(self, source: str) -> list[CreateCallInfo]:
        """Agent 소스에서 setup() 내 create 호출 추출"""
        self.calls = []
        self._source_lines = source.splitlines()

        tree = ast.parse(source)

        # setup 메서드 찾기
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "setup":
                # setup 메서드 내부만 방문
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        self.visit_Assign(child)
                break

        return self.calls

    def parse_method(self, source: str, method_name: str) -> list[CreateCallInfo]:
        """특정 메서드에서 create 호출 추출"""
        self.calls = []
        self._source_lines = source.splitlines()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        self.visit_Assign(child)
                break

        return self.calls

    def visit_Assign(self, node: ast.Assign):
        """self.{attr} = self.create.* 또는 {local_var} = self.create.* 패턴 감지"""
        if len(node.targets) != 1:
            return

        target = node.targets[0]
        attr_name: str
        is_local: bool = False

        # Case 1: self.{attr} = self.create.*
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            attr_name = target.attr
            is_local = False
        # Case 2: {local_var} = self.create.*
        elif isinstance(target, ast.Name):
            attr_name = target.id
            is_local = True
        else:
            return

        # Case 3: self.<method>(...) on RHS (local helper) -> LocalComponent
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "self"
            and node.value.func.attr != "create"
        ):
            method_name = node.value.func.attr

            # kwargs extraction
            kwargs = {}
            for kw in node.value.keywords:
                if kw.arg:
                    try:
                        kwargs[kw.arg] = ast.unparse(kw.value)
                    except Exception:
                        kwargs[kw.arg] = "<complex>"

            # args extraction
            args = []
            for arg in node.value.args:
                try:
                    args.append(ast.unparse(arg))
                except Exception:
                    args.append("<complex>")

            # code snippet
            code_snippet = ""
            if 0 < node.lineno <= len(self._source_lines):
                code_lines = self._source_lines[node.lineno - 1 : node.end_lineno or node.lineno]
                code_snippet = "\n".join(code_lines).strip()

            call_info = CreateCallInfo(
                attr_name=attr_name,
                category=method_name,  # Promote method name as category
                variant=None,
                impl=None,
                impl_source=None,
                kwargs=kwargs,
                args=args,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                code_snippet=code_snippet,
                is_local=is_local,
            )
            self.calls.append(call_info)
            return

        # 우변: self.create.{category}[.{variant}](...) 체인 파싱
        call_info = self._parse_create_chain(
            node.value, attr_name, node.lineno, node.end_lineno or node.lineno, is_local
        )
        if call_info:
            self.calls.append(call_info)

    def _parse_create_chain(
        self, node: ast.expr, attr_name: str, line: int, end_line: int, is_local: bool = False
    ) -> Optional[CreateCallInfo]:
        """self.create.category.variant() 체인 파싱

        Handles chained calls like:
        - self.create.model()
        - self.create.model().to(device)
        - self.create.transform.weak()
        - train_dataset = self.create.dataset.train() (local variable)
        """
        if not isinstance(node, ast.Call):
            return None

        # Unwrap chained method calls: self.create.model().to(device) → self.create.model()
        actual_call = self._unwrap_chained_calls(node)
        if actual_call is None:
            return None

        # 체인 수집: self.create.transform.weak() → ["self", "create", "transform", "weak"]
        chain = []
        current = actual_call.func
        while isinstance(current, ast.Attribute):
            chain.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            chain.insert(0, current.id)

        # 검증: self.create.{category}[.{variant}]
        if len(chain) < 3 or chain[0] != "self" or chain[1] != "create":
            return None

        category = chain[2]
        variant = chain[3] if len(chain) > 3 else None

        # Use actual_call for kwargs/args extraction
        node = actual_call

        # kwargs 추출
        kwargs = {}
        for kw in node.keywords:
            if kw.arg:
                try:
                    kwargs[kw.arg] = ast.unparse(kw.value)
                except Exception:
                    kwargs[kw.arg] = "<complex>"

        # args 추출
        args = []
        for arg in node.args:
            try:
                args.append(ast.unparse(arg))
            except Exception:
                args.append("<complex>")

        # impl 추출: 구현체명 결정
        # 우선순위: 1) 첫 번째 위치 인자 (문자열) 2) default= 키워드 인자
        impl: Optional[str] = None
        impl_source: Optional[str] = None

        # Case 1: self.create.model("resnet18") - 첫 번째 위치 인자
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            impl = node.args[0].value
            impl_source = "positional"
        # Case 2: self.create.model(default="resnet18") - default 키워드 인자
        elif "default" in kwargs:
            default_val = kwargs["default"]
            # 따옴표 제거 (ast.unparse가 "'resnet18'" 형태로 반환)
            if default_val.startswith(("'", '"')) and default_val.endswith(("'", '"')):
                impl = default_val[1:-1]
                impl_source = "default"

        # 코드 스니펫 추출
        code_snippet = ""
        if 0 < line <= len(self._source_lines):
            code_lines = self._source_lines[line - 1 : end_line]
            code_snippet = "\n".join(code_lines).strip()

        return CreateCallInfo(
            attr_name=attr_name,
            category=category,
            variant=variant,
            impl=impl,
            impl_source=impl_source,
            kwargs=kwargs,
            args=args,
            line=line,
            end_line=end_line,
            code_snippet=code_snippet,
            is_local=is_local,
        )


    def _unwrap_chained_calls(self, node: ast.Call) -> Optional[ast.Call]:
        """Unwrap chained method calls to find the innermost create call.

        Example: self.create.model().to(device) → self.create.model()

        Returns the innermost call if it's a self.create.* pattern, None otherwise.
        """
        # Check if this node itself is a create call
        if self._is_create_call(node):
            return node

        # If not, check if the func is an Attribute whose value is a Call
        # Pattern: xxx().to(device) → func is Attribute(attr='to', value=Call(...))
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Call):
                # Recurse into the inner call
                return self._unwrap_chained_calls(node.func.value)

        return None

    def _is_create_call(self, call_node: ast.Call) -> bool:
        """Check if a call is self.create.xxx() or self.create.xxx.yyy()."""
        func = call_node.func

        if not isinstance(func, ast.Attribute):
            return False

        # Walk back to find self.create pattern
        current = func.value
        while isinstance(current, ast.Attribute):
            if current.attr == "create":
                if isinstance(current.value, ast.Name) and current.value.id == "self":
                    return True
            current = current.value

        # Also check if func itself ends with create chain
        # e.g., self.create.model where func.attr = 'model', func.value.attr = 'create'
        if isinstance(func.value, ast.Attribute):
            if func.value.attr == "create":
                if isinstance(func.value.value, ast.Name) and func.value.value.id == "self":
                    return True

        return False


def parse_agent_setup(source: str) -> list[CreateCallInfo]:
    """Agent 소스에서 setup() 내 모든 create 호출 추출 (편의 함수)"""
    parser = CreateCallParser()
    return parser.parse_setup(source)
