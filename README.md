# CVLab-Kit

이 프로젝트는 PyTorch 기반 Computer Vision Laboratory를 위한 에이전트(Agent) 중심 프레임워크입니다. 학습 파이프라인을 구성하기 위해 Agent와 Component들을 동적으로 로딩합니다.

## 설치 가이드

### 1. uv 설치
```bash
pip install uv
```

### 2. 프로젝트 클론
```bash
git clone https://github.com/deveronica/cvlab-kit.git && cd cvlab-kit
```

## 빠른 시작 가이드
### 1. Dry-run or Generate template
구성이 완전하지 않으면 `templates/generated.yaml` 자동 생성 또는  `python3 config/generate_template.py`를 통해, `config/templates` 폴더에 `generated_basic.yaml` 파일 생성

### 2. 모든 키를 채운 후, 실제 학습 실행

프로젝트를 실행하려면 다음 명령어를 사용합니다:
(uv 환경 아직 미구현)
```bash
uv run main.py --config config/voc.yaml --fast
```

## 주요 특징

* **Agent-centric Workflow**: `create = Creator(cfg)`를 통해 원하는 에이전트를 즉시 인스턴스화.
  ```text
  create.agent.<key>()
  ```
* **Component Factory**: 모델·옵티마이저·데이터셋 등 모든 요소를 Component 단위로 동적 로딩.
  ```text
  create.<component>.<key>()
  ```

## 기술 스펙

### Grid Search
- YAML 안의 리스트로 설정된 값들이 자동으로 조합되어 반복 실험을 수행합니다.

### Zero-Boilerplate
- 신규 컴포넌트 구현체는 component/base 내부의 클래스를 상속하고, 적합한 폴더에 넣기만 하면 자동으로 탐색됩니다. 공통된 라이브러리 의존성은 이곳에서 해결됩니다.

## 새로운 컴포넌트 추가 방법

### 1. **추상 인터페이스 상속**
- `cvlabkit/component/base` 모듈에서 템플릿 클래스를 상속하고, 필요한 메서드를 구현합니다.

```python
# cvlabkit/component/optimizer/adamw.py
import torch.optim as optim

from cvlabkit.component.base import Optimizer


class Optimizer(Optimizer):
    def __init__(self, cfg, params):
        super().__init__()
        self.opt = optim.AdamW(params, lr=cfg.get("lr", 1e-3))
```

### 2. **YAML 설정 추가**
- YAML 파일에 구현한 컴포넌트명을 값으로 등록합니다.

```yaml
optimizer: adamw
```

### 3. **자동 로딩**
학습 스크립트에서는 별도 import 없이,

```python
opt = create.optimizer(model.parameters())
```

### 4. **자동 로딩 테스트**
- `main.py`을 수행하여 새로운 컴포넌트가 올바르게 로드되는지 확인합니다.

## 라이브러리 및 의존성

프로젝트는 다음과 같은 주요 라이브러리를 사용하고 있습니다:

| Component  | 대표 라이브러리                                 |
| ---------- | --------------------------------------------|
| Transform  | torch, albumentations, kornia               |
| Dataset    | torchvision.datasets, webdataset            |
| Model      | timm, transformers                          |
| Loss       | torch.nn, segmentation_models_pytorch     |
| Optimizer  | torch.optim                                 |
| Scheduler  | torch.optim.lr_scheduler                    |
| Metrics    | torchmetrics                                |
| Logger     | wandb, tensorboard                          |
