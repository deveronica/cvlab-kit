# Creator 동작 방식 상세 설명

`Creator`는 `cvlab-kit` 프레임워크의 핵심적인 역할을 담당하는 동적 컴포넌트 팩토리입니다. 이 문서에서는 `Creator`가 YAML 설정 파일을 기반으로 어떻게 에이전트와 컴포넌트를 생성하는지 그 내부 동작 순서와 방식을 `creator.py` 코드와 함께 상세히 설명합니다.

## **1. Creator의 역할과 초기화**

`Creator`의 주된 역할은 다음과 같습니다.

1.  실험을 총괄하는 메인 **에이전트(Agent)를 생성**합니다.
2.  에이전트가 필요로 하는 하위 **컴포넌트(Model, Optimizer, DataLoader 등)를 생성**할 수 있는 인터페이스를 제공합니다.

모든 과정은 `main.py`에서 `Creator` 객체를 생성하는 것으로 시작됩니다.

```python
# main.py
cfg = Config(args.config)    # 1. YAML 설정 로드
create = Creator(cfg)        # 2. Creator 초기화
agent = create.agent()       # 3. 메인 에이전트 생성
agent.run()                  # 4. 에이전트 실행
```

`Creator`가 초기화될 때, 내부에 `ComponentCreator` 인스턴스를 함께 생성합니다. `Creator` 자신은 에이전트 생성만 담당하고, 나머지 모든 컴포넌트 생성 작업은 `ComponentCreator`에게 위임하는 구조입니다.

```python
# cvlabkit/core/creator.py
class Creator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # ComponentCreator가 실제 컴포넌트 생성을 담당
        self.component_creator = ComponentCreator(cfg)
```

## **2. 컴포넌트 생성 요청 (Agent -> Creator)**

에이전트는 자신의 `setup()` 메소드 내에서 필요한 컴포넌트들을 `Creator`에게 요청합니다. 이 때 `create`는 에이전트에 주입된 `ComponentCreator`의 프록시(proxy) 역할을 합니다.

```python
# agent.py (예시)
class MyAgent(Agent):
    def setup(self):
        # self.create는 ComponentCreator를 가리킴
        self.model = self.create.model() 
        self.optimizer = self.create.optimizer(self.model.parameters())
        self.train_loader = self.create.dataloader.train()
```

`self.create.model()`과 같은 호출이 발생하면 `ComponentCreator`의 `__getattr__` 메소드가 실행되어, `model`이라는 카테고리를 처리할 `_ComponentCategoryLoader`를 반환합니다.

## **3. Creator의 내부 동작 순서 (상세)**

`create.dataloader.train()`을 예시로 `Creator`의 내부 동작을 코드와 함께 순서대로 따라가 보겠습니다.

### **1단계: 컴포넌트 카테고리 로더 가져오기 (`create.dataloader`)**

1.  **호출**: 에이전트 코드에서 `create.dataloader`가 처음 접근될 때, `ComponentCreator`의 `__getattr__` 메소드가 `category='dataloader'` 인자와 함께 호출됩니다.

    ```python
    # ComponentCreator.__getattr__
    def __getattr__(self, category: str) -> "_ComponentCategoryLoader":
        # ...
        base_class = self._base_classes.get(category) # 'dataloader' -> DataLoader 베이스 클래스
        # ...
        return _ComponentCategoryLoader(self.cfg, category, base_class)
    ```

2.  **베이스 클래스 탐색**: `_base_classes` 딕셔너리에서 `'dataloader'` 키에 해당하는 `cvlabkit.component.base.DataLoader` 베이스 클래스를 찾습니다. 이 딕셔너리는 `ComponentCreator` 초기화 시 `_get_all_base_classes` 메소드를 통해 미리 채워져 있습니다.

3.  **카테고리 로더 생성**: `dataloader` 카테고리만 전담하여 처리할 `_ComponentCategoryLoader` 인스턴스를 생성하여 반환합니다. 이 로더는 `cfg`와 `DataLoader` 베이스 클래스 정보를 가지고 있습니다.

### **2단계: 특정 옵션 로더 가져오기 (`.train`)**

1.  **호출**: `create.dataloader` 뒤에 붙은 `.train`이 접근될 때, 이전 단계에서 반환된 `_ComponentCategoryLoader` 객체의 `__getattr__` 메소드가 `option='train'` 인자와 함께 호출됩니다.

    ```python
    # _ComponentCategoryLoader.__getattr__
    def __getattr__(self, option: str) -> Callable[..., Any]:
        key = f"{self.category}.{option}" # "dataloader.train"
        config_value = self.cfg.get(key) # "cifar10(split=train, shuffle=true)"
        # ...
        def creator_lambda(*args, **kwargs):
            # ... (3단계에서 설명)
        return creator_lambda
    ```

2.  **설정 값 조회**: `cfg.get("dataloader.train")`을 호출하여 YAML 파일에서 해당 키의 값을 찾습니다. 이 예시에서는 `"cifar10(split=train, shuffle=true)"` 문자열이 `config_value`가 됩니다.

3.  **람다 함수 반환**: 실제 컴포넌트 생성을 지연시키고, 런타임 인자(예: `optimizer` 생성 시 `model.parameters()`)를 받을 수 있도록 `creator_lambda`라는 내부 함수(클로저)를 정의하여 반환합니다. 아직 컴포넌트가 생성된 시점은 아닙니다.

### **3단계: 컴포넌트 인스턴스 생성 (`()`)**

1.  **호출**: 에이전트 코드에서 `create.dataloader.train()`의 마지막 `()`가 호출될 때, 비로소 이전 단계에서 반환된 `creator_lambda`가 실행됩니다.

    ```python
    # creator_lambda 내부 로직
    def creator_lambda(*args, **kwargs):
        # config_value가 '|'를 포함하는지 확인 -> 여기서는 아님
        # ...
        # 'cifar10(split=train, shuffle=true)' 문자열 파싱
        impl_name, component_cfg = self._get_component_info(config_value)
        # impl_name = 'cifar10', component_cfg = {'split': 'train', 'shuffle': True}
        
        # 'cifar10' 구현체 로드
        constructor = self._load_implementation(impl_name)
        
        # 최종 인스턴스 생성
        return self._create_instance(constructor, component_cfg, *args, **kwargs)
    ```

2.  **설정 파싱**: `_get_component_info` 메소드가 `"cifar10(split=train, shuffle=true)"` 문자열을 파싱합니다. AST(Abstract Syntax Tree)를 사용하여 함수 호출 구문을 분석하고, 구현 이름(`cifar10`)과 파라미터(`{'split': 'train', 'shuffle': True}`)를 분리하여 `Config` 객체로 만듭니다.

3.  **구현 로딩**: `_load_implementation('cifar10')` 메소드가 호출됩니다.
    -   `package_path`를 `cvlabkit.component.dataloader`로 설정합니다.
    -   `importlib.import_module("cvlabkit.component.dataloader.cifar10")`를 통해 해당 모듈을 동적으로 임포트합니다.
    -   모듈 내부를 검사하여 `cvlabkit.component.base.DataLoader`를 상속하는 클래스(`CIFAR10Loader`)를 찾아 생성자(`constructor`)를 반환합니다.

4.  **인스턴스 생성**: `_create_instance(constructor, ...)` 메소드가 호출됩니다.
    -   컴포넌트별 설정(`component_cfg`)과 전역 설정(`self.cfg`)을 병합하여 최종 설정 객체 `final_cfg`를 만듭니다.
    -   `CIFAR10Loader`의 생성자 시그니처를 분석하여, `final_cfg` 객체와 런타임 인자(`*args`, `**kwargs`)를 주입하여 최종 인스턴스를 생성합니다.

### **4단계: 최종 반환**

-   생성된 `CIFAR10Loader` 인스턴스가 에이전트의 `setup()` 메소드로 최종 반환됩니다.
-   `self.train_loader = create.dataloader.train()` 라인이 완료됩니다.

## **요약: Creator의 마법**

`Creator`는 `__getattr__`와 클로저(closure)를 적극적으로 활용하여 `create.model.generator()`와 같은 연쇄적인 호출을 가능하게 합니다. 각 단계마다 특정 카테고리나 옵션을 전담하는 로더 객체나 람다 함수를 반환하며, 최종적으로 `()` 호출이 이루어질 때 실제 컴포넌트 생성 작업이 트리거됩니다.

이러한 동적 생성 메커니즘 덕분에 사용자는 파이썬 코드를 직접 수정하지 않고도 YAML 설정 변경만으로 프레임워크의 거의 모든 동작을 제어할 수 있습니다.
