# CVLab-Kit

PyTorch 기반 단순하고 확장 가능한 프로토타이핑 프레임워크


## Overview

이 프로젝트는 PyTorch 기반의 경량 모듈형 실험 프레임워크로, **에이전트(Agent) 중심**으로 컴포넌트(Component)들을 조합하여, 손쉽게 확장하고 실험을 반복할 수 있도록 설계된 프레임워크입니다. 빠른 아이디어 검증 및 모델 구조 실험이 가능한 **프로토타이핑 환경**을 제공하며, YAML 설정 파일과 Creator 클래스를 통해 실험 설정 및 학습 과정을 간결하게 통합하여 실험 환경을 구성하고 관리할 수 있습니다.


## Key Feature

| 기능 | 설명 |
|------|------|
| **Agent‑Centric Workflow** | 학습, 검증, 평가 루프를 에이전트 중심으로 관리 |
| **Dynamic Component Factory** | 모델·옵티마이저·데이터셋 등의 구성 요소를 컴포넌트 단위로 <br>`create.<component>.<key>()` 동적으로 로딩 |
| **Dry‑run Validation** | 학습 전에 구성이 올바른지 검증하여, 학습 도중 중단 문제를 사전에 방지 |
| **Grid Search** | YAML에 구성된 다중 값들이 자동으로 실험 조합으로 확장되어 반복 실험 |
| **Zero‑Boilerplate** | 신규 컴포넌트 구현체는 컴포넌트 추상 클래스의 상속을 통해 일관되게 관리 |


## Installation

### 1. uv 설치

```bash
pip install uv
```

### 2. 프로젝트 클론

```bash
git clone https://github.com/deveronica/cvlab-kit.git && cd cvlab-kit
```

### 3. 의존성 설치

```bash
uv sync
```

> **uv**는 Poetry·pip‑tools와 비슷한 UX를 제공하면서도 의존성 해석과 빌드를 Rust로 가속화한 도구입니다.

## Quick Start

### 1. Dry-run or Generate Template

- ~~구성이 완전하지 않으면 Dry-run 과정에서 자동으로 `templates/generated.yaml` 생성~~(미구현)
- `python3 config/generate_template.py`를 통해, `config/templates` 폴더에 `generated_basic.yaml` 파일 생성

### 2. Write YAML Configuration

YAML 설정 파일을 작성합니다.

### 3. Run Experiment

설정을 검증하고 실험을 진행합니다. (현재 generate_template.py를 통한 수동 생성 후 --fast 옵션으로 진행)

```python
uv run main.py --config config/example.yaml --fast
```

## Working Process

1. 진입점 (`main.py`): 사용자가 python main.py --config config/main.yaml과 같이 실행하면, main.py는 지정된 설정 파일을 읽어옵니다.

2. 설정 파싱 (`cvlabkit/core/config.py`): Config 클래스가 YAML 파일을 읽고 파싱하여 파이썬 객체처럼 점(.)으로 접근할 수 있는 형태로 변환합니다. (예: config.model.name)

3. 생성자 생성 (`cvlabkit/core/creator.py`): Creator 클래스는 설정(Config) 객체를 받아 필요한 모든 "부품"들을 동적으로 생성하는 팩토리(Factory) 역할을 합니다. 일반적으로 create 변수에 할당하여, create.model(), create.optimizer() 등의 메서드를 사용하여 필요한 컴포넌트 들을 생성할 수 있습니다.
    - 대표적인 예시로, config.model 섹션을 보고 `cvlabkit/component/model` 폴더에서 FasterRCNN 같은 모델 클래스를 찾아 인스턴스화합니다.
    - optimizer, loss, dataset 등 다른 모든 구성 요소도 같은 방식으로 생성합니다.

4. 에이전트 실행 (`cvlabkit/agent/`): Creator가 설정 파일에 명시된 에이전트(예: BasicAgent)를 생성합니다. 이 에이전트는 생성된 모델, 데이터 로더, 손실 함수 등을 조합하여 실제 학습, 평가, 테스트의 전체 과정을 조율하고 실행하는 주체입니다.
    - BasicAgent: 일반적인 학습/평가 파이프라인을 담당합니다.

5. 모듈형 컴포넌트 (`cvlabkit/component/`): 이 프로젝트의 가장 큰 특징은 각 기능이 독립적인 부품(컴포넌트)으로 분리되어 있다는 점입니다.
    - model: Faster R-CNN과 같은 딥러닝 모델
    - loss: Cross-Entropy, Focal Loss 등 손실 함수
    - dataset, dataloader: VOC, Cityscapes 등 데이터셋과 데이터를 공급하는 로더
    - optimizer, scheduler: Adam, SGD 등 최적화 도구 및 학습률 스케줄러
    - 이러한 구조 덕분에 새로운 모델이나 손실 함수를 추가하고 싶을 때, 해당 폴더에 새로운 파이썬 파일만 추가하고 YAML 파일에서 이름을 지정해주면 되므로 확장이 매우 용이합니다.

## Development Philosophy

PyTorch의 자유로운 모듈 작성 방식은 실험 구성 간 인터페이스 불일치와 모듈 간 의존성을 높일 수 있습니다. CVLab-Kit은 PyTorch를 기반으로도 실험 환경의 반복 실험과 빠른 프로토타이핑을 위해 다음과 같은 개발 철학을 바탕으로 설계되었습니다:

### 1. Automatic Grid Search

- YAML 안의 리스트 타입으로 설정된 값들이 자동으로 조합되어 반복 실험을 수행합니다.

### 2. Component Interface

- `cvlabkit/component/base` 디렉토리에 컴포넌트의 추상 클래스를 정의함으로써, 모든 하위 클래스가 동일한 인터페이스를 가지도록 합니다. 이는 코드 일관성을 유지하고 확장성을 높입니다.

### 3. Dependency Resolution

- 상속 의존성 해결: 추상 컴포넌트에 상속을 통해 모델, 옵티마이저, 데이터셋 등이 가지고 있는 공통된 상속을 강제하여 PyTorch의 상속 의존성을 일괄적으로 해결합니다. 이를 통해 사용자 정의 모듈이 자동 탐색·호출될 때도 일관성 있게 작동하도록 보장합니다.

- 컴포넌트 간 의존성 해결: Creator를 통한 단계적 생성과 명시적 변수 입력 방식으로 수행됩니다. 예를 들어 모델 파라미터(`model.parameters()`)는 모델 객체가 생성된 이후에야 얻을 수 있으므로, Creator가 먼저 모델을 생성한 뒤 해당 파라미터를 옵티마이저 생성 시 인자(`create.optimizer(model.parameters())`)으로 전달합니다.

- 객체 속성 의존성 해결: Creator가 cfg객체를 기반으로 생성되고, 이를 컴포넌트 생성 시 스탬프 결합을 통해, 모델, 옵티마이저, 데이터 로더 간 필요한 공통된 의존성을 자동으로 주입해 관리하며, 개별 파라미터와 공통 파라미터(e.g. cfg.num_class, cfg.channel 등)를 일관되게 전달합니다.

모든 컴포넌트는 cfg 객체를 생성자를 통해 주입받아 설정값 기반으로 일관되게 동작하며, 객체 간 의존성(모델 파라미터 등)은 직접 의존성 주입을 통해 관리합니다. 이를 통해 설정값과 의존성 인자를 명확히 분리해 프로토타이핑 범용성을 높입니다.

### 4. Dynamic Loading

- 모든 컴포넌트는 component/base의 추상 컴포넌트 클래스를 상속하여 핵심 메서드와 속성을 일관되게 유지하며, 모든 구현체는 cvlabkit/component/`componentName` 디렉토리에 위치해야 합니다.

- Creator는 YAML에서 **컴포넌트의 설정값**이 cvlabkit/component/`componentName` 디렉토리에 있는 **파일명**과 일치하는 파일을 찾아 해당 추상 컴포넌트를 상속 받은 클래스의 인스턴스를 동적으로 생성합니다. 만약, 컴포넌트가 여러 가지 사용 되었을 경우를 대비하여, create.`componentName`.`key`의 형태로 호출되었을 경우, cfg.`componentName`.`key` 값을 설정값으로 전달합니다.

따라서 사용자는 YAML 설정만으로도 불필요한 import 없이 가볍게 컴포넌트를 로드해 실험 코드를 간결하게 유지합니다.

이러한 철학들을 통해 CVLab-Kit은 빠르게 실험을 반복하며, 확장성과 유지보수성이 높은 PyTorch 기반 프로토타이핑 환경을 제공합니다.

## Adding New Component Implementation

### 1. **인터페이스 상속**
- `cvlabkit/component/base` 모듈의 추상 클래스를 상속받아 필요한 메서드를 구현합니다.
- 기존에 구현된 컴포넌트를 활용할 경우, 합성(Composition) 패턴을 활용하여 객체를 생성한 뒤, 필요한 메서드를 재정의하여 사용할 수 있습니다.

```python
# cvlabkit/component/optimizer/adamw.py
import torch.optim as optim
from cvlabkit.component.base import Optimizer

class Optimizer(Optimizer):
    def __init__(self, cfg, parameters):
        super().__init__()
        self.opt = optim.AdamW(parameters, lr=cfg.get("lr", 1e-3))

    def step(self):
        self.opt.step()
```

### 2. **설정 값 반영**
- YAML 파일에 구현한 컴포넌트명을 값으로 등록합니다.

```yaml
optimizer: adamw
```

### 3. **자동 로딩**
- 학습 스크립트에서는 별도 import 없이 YAML 수정만으로, 기존에 구성된 구조를 유지하며 새로운 컴포넌트를 자동으로 로드할 수 있습니다.
- 예를 들어, `create.optimizer` 함수를 유지해도, yaml의 optimizer에 해당하는 값이 수정되었으므로, `AdamW`에 해당하는 `Optimizer` 구현체를 생성할 수 있습니다.

```python
opt = create.optimizer(model.parameters())
```

### 4. **자동 로딩 테스트**
- `main.py`을 수행하여 새로운 컴포넌트가 올바르게 로드되는지 확인합니다.

## Additional Library

프로젝트는 다음과 같은 주요 라이브러리를 사용하고자 합니다:

| Component  | 대표 라이브러리                                                          |
| ---------- | -------------------------------------------------------------------- |
| Transform  | `torchvision.transforms`, `albumentations`, `kornia`                 |
| Dataset    | `torchvision.datasets`, `HF datasets`, `webdataset`                  |
| Model      | `torchvision.models`, `timm`, `transformers`                         |
| Loss       | `torch.nn`, `pytorch‑metric‑learning` |
| Optimizer  | `torch.optim`, `timm.optim`                                          |
| Scheduler  | `torch.optim.lr_scheduler`, `timm.scheduler`                         |
| Metrics    | `torchmetrics`, `sklearn.metrics`                                    |
| Checkpoint | `torch.save/load`, `safetensors`                                     |
| Logger     | `wandb`, `tensorboard`, `mlflow`                                    |