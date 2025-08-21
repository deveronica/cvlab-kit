# 아키텍처 개요

이 섹션에서는 CVLab-Kit 프레임워크의 내부 작동 방식과 아키텍처에 대해 자세히 설명합니다.


## 작동 방식

1. **진입점 (`main.py`)**: 사용자가 `python main.py --config config/main.yaml`과 같이 실행하면, `main.py`는 지정된 설정 파일을 읽어옵니다.

2. **설정 파싱 (`cvlabkit/core/config.py`)**: `Config` 클래스는 YAML 파일을 읽고 파싱하여 파이썬 객체처럼 점(.)으로 접근할 수 있는 형태(예: `config.model.name`)로 변환합니다.

3. **생성자 인스턴스화 (`cvlabkit/core/creator.py`)**: `Creator` 클래스는 `Config` 객체를 받아 필요한 모든 "부품"들을 동적으로 생성하는 팩토리 역할을 합니다. 일반적으로 `create` 변수에 할당되어, `create.model()`, `create.optimizer()` 등의 메서드를 사용하여 컴포넌트를 생성할 수 있습니다.
    - 예를 들어, `config.model` 섹션을 기반으로 `cvlabkit/component/model` 폴더에서 `FasterRCNN`과 같은 모델 클래스를 찾아 인스턴스화합니다.
    - 옵티마이저, 손실 함수, 데이터셋 등 다른 모든 컴포넌트도 동일한 방식으로 생성됩니다.

4. **에이전트 실행 (`cvlabkit/agent/`)**: `Creator`는 설정 파일에 지정된 에이전트(예: `ClassificationAgent`)를 인스턴스화합니다. 이 에이전트는 생성된 모델, 데이터 로더, 손실 함수 및 기타 컴포넌트를 결합하여 학습, 평가, 테스트의 전체 프로세스를 조정하고 실행하는 주요 엔티티입니다.
    - `ClassificationAgent`: 일반적인 학습/평가 파이프라인을 처리합니다.

5. **모듈형 컴포넌트 (`cvlabkit/component/`)**: 이 프로젝트의 핵심 기능은 각 기능이 독립적인 "부품"(컴포넌트)으로 분리되어 있다는 점입니다.
    - `model`: Faster R-CNN과 같은 딥러닝 모델.
    - `loss`: Cross-Entropy, Focal Loss와 같은 손실 함수.
    - `dataset`, `dataloader`: VOC, Cityscapes와 같은 데이터셋 및 데이터를 공급하는 로더.
    - `optimizer`, `scheduler`: Adam, SGD와 같은 최적화 도구 및 학습률 스케줄러.

이러한 구조는 확장을 매우 용이하게 합니다. 새로운 컴포넌트 구현을 추가하려면 해당 폴더에 새 Python 파일만 추가하고 YAML 파일에 이름을 지정하면 됩니다.


## 컴포넌트 설계 철학: InterfaceMeta

CVLab-Kit의 모든 컴포넌트는 `InterfaceMeta`라는 커스텀 메타클래스를 기반으로 설계되어 유연성과 재사용성을 극대화합니다. 이 시스템은 개발자가 최소한의 코드로 컴포넌트를 구현할 수 있도록 두 가지 주요 패턴을 지원합니다.

1. **직접 구현 (Direct Implementation)**
    - **설명:** 완전히 새로운 기능을 구현할 때 사용합니다. `cvlabkit.component.base`의 추상 클래스(예: `Model`, `Loss`)를 상속받고, 필요한 모든 메서드를 직접 구현합니다.
    - **예시:**
        ```python
        from cvlabkit.component.base import Model
        import torch.nn as nn

        class MyCustomModel(Model):
            def __init__(self, cfg):
                super().__init__()
                self.linear = nn.Linear(10, cfg.num_classes)

            def forward(self, x):
                return self.linear(x)
        ```

2. **위임 (Delegation / Composition)**
    * **설명:** `torch.optim.Adam`과 같은 기존 라이브러리 구현체를 재사용하면서, 일부 동작만 변경하거나 확장하고 싶을 때 사용합니다. 컴포넌트 클래스의 `__init__` 메서드 안에서 기존 라이브러리 객체를 `self`의 속성으로 할당하면, `InterfaceMeta`가 자동으로 해당 객체로 메서드 호출을 위임합니다. 사용자가 오버라이드한 메서드는 위임보다 우선합니다.
    * **예시:**
        ```python
        from cvlabkit.component.base import Optimizer
        import torch.optim as optim

        class CustomAdam(Optimizer):
            def __init__(self, cfg, parameters):
                # 기존 Adam 옵티마이저를 위임 대상으로 지정
                # 무한 재귀를 피하기 위해 self.opt 와 같이 다른 이름으로 할당합니다.
                self.opt = optim.Adam(parameters, lr=cfg.get("lr", 1e-3))

            def step(self):
                # step 메서드는 직접 오버라이드하여 추가 로직 구현
                print("Custom step logic before Adam step")
                self.opt.step()

            # zero_grad()와 같은 다른 메서드는 자동으로 self.opt로 위임됩니다.
        ```

이 `InterfaceMeta` 시스템 덕분에, 개발자는 보일러플레이트 코드 없이 상황에 가장 적합한 방식으로 컴포넌트를 구현하는 데만 집중할 수 있으며, PyTorch의 유연성을 최대한 활용할 수 있습니다.


## 컴포넌트 의존성 해결

CVLab-Kit은 `Agent`를 중심으로 컴포넌트 간의 의존성을 해결합니다. `Agent`는 지휘자(Orchestrator)로서, 필요한 컴포넌트들을 `Creator`를 통해 생성하고 이들을 적절히 조합하여 전체 워크플로우를 구성합니다.


### 의존성 주입 방식

의존성 해결은 주로 **생성자 주입(Constructor Injection)** 방식으로 이루어집니다. 즉, 특정 컴포넌트를 생성할 때 필요한 다른 컴포넌트나 설정 값을 생성자의 인자로 전달합니다.

1. **필수 의존성**: `Optimizer`는 항상 `Model`의 파라미터를 필요로 합니다. `Agent`는 먼저 `Model`을 생성한 뒤, `model.parameters()`를 `Optimizer` 생성자에 직접 전달합니다.

    ```python
    # Agent 내부 로직 예시
    model = self.create.model()
    optimizer = self.create.optimizer(model.parameters())
    ```

2. **선택적 의존성**: `Dataset`은 `Transform`을 필요로 할 수도, 아닐 수도 있습니다. `Agent`는 설정 파일(`config.yaml`)을 확인하여 `transform`이 명시된 경우에만 `Transform` 컴포넌트를 생성하고 `Dataset`에 주입합니다. 설정에 없으면 `None`을 전달하여 `Dataset`이 기본 동작을 수행하도록 합니다.

    ```python
    # Agent 내부 로직 예시
    # 설정에 'transform'이 있을 때만 transform 객체를 생성, 없으면 None
    transform = self.create.transform() if 'transform' in self.cfg else None
    
    # Dataset 생성 시 transform 객체(또는 None)를 인자로 전달
    dataset = self.create.dataset.train(transform=transform)
    ```

3. **단순 인터페이스 (Simple Interface)**
    - **설명:** 여러 컴포넌트가 따라야 할 메서드 시그니처만 정의하고 싶을 때 사용합니다. 추상 클래스를 정의하기만 하고, 인스턴스화하지 않으면 됩니다. 이 인터페이스를 상속받는 모든 자식 클래스는 위 1번 또는 2번 방식으로 구현해야 합니다.

이러한 유기적인 의존성 관리 방식 덕분에, 사용자는 YAML 설정 변경만으로 각 컴포넌트의 구현을 수정하지 않고도 다양한 조합의 실험을 유연하게 구성할 수 있습니다.


## 컴포넌트 역할과 책임 (Component Roles & Responsibilities)

CVLab-Kit의 유연성은 각 컴포넌트가 자신의 **명확한 역할에만 집중**하는 것에서 비롯됩니다. 이는 "책임의 분리(Separation of Concerns)" 원칙을 따르며, 사용자가 새로운 아이디어를 실험할 때 어떤 컴포넌트를 수정하거나 새로 만들어야 할지 쉽게 판단할 수 있도록 돕습니다.

각 컴포넌트의 역할과 책임 범위는 다음과 같습니다.

### **`Agent` (The Orchestrator)**
- **책임 (Why)**: 실험의 **전체 시나리오를 정의하고 실행**합니다. 학습, 평가, 데이터 처리 등 모든 워크플로우를 지휘하는 **"실험의 뇌"** 와 같습니다.
- **범위 (Scope)**:
    - 어떤 컴포넌트(Model, Dataset, Optimizer 등)를 사용할지 `Creator`를 통해 생성하고 조립합니다.
    - 데이터를 어떻게 분할하고(e.g., Labeled/Unlabeled 분리), 어떤 `Sampler` 전략을 사용할지 결정하여 `DataLoader`를 구성합니다.
    - 학습 루프를 관리하며, 모델의 출력을 `Loss` 함수에 전달하고 `Optimizer`를 실행합니다.
    - 특정 학습 기법(e.g., Semi-Supervised Learning, Knowledge Distillation)에 필요한 고유한 로직은 **전적으로 `Agent`의 책임**입니다.

### **`Dataset` (The Source)**
- **책임 (Why)**: 특정 데이터셋(e.g., MSTAR, CIFAR-10)의 **데이터가 어디에 있고, `i`번째 데이터가 무엇인지(e.g., 이미지와 레이블)를 정의**합니다.
- **범위 (Scope)**:
    - 데이터의 원본 위치(path)를 관리하고, 인덱스를 통해 데이터 샘플과 정답을 반환하는 역할에만 집중합니다.
    - **금지된 역할**: 데이터 분할, 샘플링 방식, 특정 학습 기법에 대해 알아서는 안 됩니다. `mstar.py`는 MSTAR 데이터를 표현할 뿐, 이 데이터가 어떻게 사용될지는 전혀 관여하지 않습니다.

### **`Transform` (The Processor)**
- **책임 (Why)**: `Dataset`에서 나온 원본 데이터를 모델에 입력하기 적합한 형태로 **가공(처리)**합니다.
- **범위 (Scope)**:
    - 이미지 증강(Augmentation), 정규화(Normalization), 텐서 변환(To-Tensor) 등 단일 데이터 샘플에 대한 모든 변환 작업을 수행합니다.
    - `adaptive_rand_augment.py`와 같이, `Agent`로부터 추가 정보(e.g., 난이도)를 받아 동적으로 변환 강도를 조절하는 복잡한 로직을 포함할 수 있습니다.

### **`Sampler` & `DataLoader` (The Strategy & Provider)**
- **`Sampler` 책임 (Why)**: `DataLoader`가 `Dataset`에서 데이터를 어떤 **순서와 조합으로 가져올지 그 전략을 정의**합니다.
- **`DataLoader` 책임 (Why)**: `Dataset`과 `Sampler` 전략을 바탕으로 **배치(batch) 단위로 데이터를 공급**합니다.
- **범위 (Scope)**:
    - `Sampler`는 인덱스 순서(e.g., `SequentialSampler`), 무작위 추출(`RandomSampler`), 또는 여러 인덱스 그룹을 특정 비율로 섞는(`MixedBatchSampler`) 등 **인덱스 레벨의 전략**에만 집중합니다.
    - `DataLoader`는 이 전략을 받아 실제 데이터를 가져오고, 멀티프로세싱, 배치화(batching) 등 **효율적인 데이터 공급**을 책임집니다. 대부분의 경우 `cvlabkit`의 기본 `basic` 로더로 충분합니다.

### **`Model` & `Loss` (The Problem-Solver & The Judge)**
- **`Model` 책임 (Why)**: 입력 데이터를 받아 **문제를 해결하고 예측(prediction)을 출력**합니다.
- **`Loss` 책임 (Why)**: 모델의 예측이 **얼마나 정답과 다른지를 측정(평가)**합니다.
- **범위 (Scope)**:
    - `Model`은 순전파(forward pass) 로직에만 집중합니다.
    - `Loss`는 예측과 정답을 입력받아 스칼라(scalar) 손실 값을 계산하는 역할만 수행합니다. 어떤 `Optimizer`가 이 손실을 사용할지는 `Loss`의 관심사가 아닙니다.

### **`Optimizer` & `Scheduler` (The Tuner & The Regulator)**
- **`Optimizer` 책임 (Why)**: `Loss`가 계산한 오차를 바탕으로 모델의 파라미터를 **개선(업데이트)**합니다.
- **`Scheduler` 책임 (Why)**: 학습 과정에 따라 `Optimizer`의 학습률(learning rate)을 **동적으로 조절**합니다.
- **범위 (Scope)**:
    - `Optimizer`는 `model.parameters()`를 받아 `step()`, `zero_grad()` 등 파라미터 업데이트에만 관여합니다.
    - `Scheduler`는 `Optimizer` 인스턴스를 받아 학습률을 조절하는 역할만 수행합니다.

### **`Metric`, `Logger`, `Checkpoint` (The Recorders)**
- **`Metric` 책임 (Why)**: 학습 결과를 **정량적으로 측정**합니다 (e.g., Accuracy, mAP).
- **`Logger` 책임 (Why)**: 실험 과정과 `Metric` 결과를 **외부(e.g., Wandb)에 기록**합니다.
- **`Checkpoint` 책임 (Why)**: 모델의 가중치 등 실험 상태를 **저장하고 복원**합니다.
- **범위 (Scope)**: 이들은 학습 과정 자체에 영향을 주지 않고, 실험의 진행 상황과 결과를 관찰하고 기록하는 독립적인 역할을 수행합니다.