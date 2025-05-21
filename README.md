# CVLab‑Kit (🚀 Simplified FlexTorch)

PyTorch 프로젝트를 **에이전트(Agent) 중심**으로 손쉽게 확장·실험할 수 있도록 설계된 경량 프레임워크입니다.  
Component를 “레고 블록”처럼 끼워 넣고, `create = Creator(cfg)`로 생성자를 마련하고, `create.<category>.<key>()` 의 형식으로 `cfg.category.key`를 참조하여, 동적 로딩하여 학습 파이프라인을 구성합니다.

---

## 🔑 Key Feature

| 기능 | 설명 |
|------|------|
| **Agent‑centric Workflow** | `Creator`를 통해 원하는 에이전트를 즉시 인스턴스화.<br>`creator.agent.<key>()` |
| **Component Factory** | 모델·옵티마이저·데이터셋 등 모든 요소를 Component 단위로 <br>`creator.<component>.<key>()` 호출 한 줄로 동적 로딩 |
| **Dry‑run Validation** | 1 iteration 그래프 검증 → 누락 Config 자동 템플릿 생성 |
| **Grid Search** | YAML 안의 리스트로 설정된 값들이 자동으로 조합으로 확장되어 실험 반복 |
| **Zero‑Boilerplate** | 신규 컴포넌트 구현체는 추상 클래스 상속 후 폴더에 넣기만 하면 자동으로 탐색 진행 |

---

## ⚙️ Installation


### 1. uv 설치
```bash
pip install uv
```

### 2. 프로젝트 클론
```bash
git clone https://github.com/deveronica/cvlabkit.git
cd cvlabkit
```

### 3. 의존성 설치 및 자동 실행
```bash
uv run main.py --config config/example.yaml
```

> **uv**는 Poetry·pip‑tools와 비슷한 UX를 제공하면서도 의존성 해석과 빌드를 Rust로 가속화한 도구입니다.

## 🚀 빠른 시작

### 1. Dry-run
구성이 완전하지 않으면 templates/generated.yaml 생성

```bash
python main.py --config config/cls_resnet.yaml
```

### 2. 모든 키를 채운 후, 실제 학습 실행
```bash
python main.py --config config/cls_resnet.yaml --fast
```

## 📂 프로젝트 구조

```text
config/               # YAML 실험 설정
    cls_resnet.yaml
    ...
cvlabkit/
    core/             # Config, Proxy, Creator
        agent.py      # 추상 Agent 클래스
    agent/            # 사용자 정의 Agent
        myagent.py
    component/        # 각 카테고리별 컴포넌트
        base/         # 추상 Component 클래스
        model/
            mymodel.py
            ...
        optimizer/
        ...
data/                 # 데이터셋 경로
logs/                 # 학습 로그 & 체크포인트
main.py               # Start Point
README.md
```

## 🛠️ 새 컴포넌트 작성하기

### 1. **추상 인터페이스 상속**

```python
# cvlabkit/component/optimizer/adamw.py
import torch.optim as optim

from cvlabkit.component.base import Optimizer


class Optimizer(Optimizer):
    def __init__(self, cfg, params):
        super().__init__()
        self.opt = optim.AdamW(params, lr=cfg.get("lr", 1e-3))
```

### 2. **YAML 지정**

```yaml
optimizer: adamw
```

### 3. **자동 로딩**
학습 스크립트에서는 별도 import 없이,

```python
opt = create.optimizer(model.parameters())
```

## 📚 참고 라이브러리 (요약)

| Component  | 대표 라이브러리                                                             |
| ---------- | -------------------------------------------------------------------- |
| Transform  | `torchvision.transforms`, `albumentations`, `kornia`                 |
| Dataset    | `torchvision.datasets`, `HF datasets`, `webdataset`                  |
| Model      | `torchvision.models`, `timm`, `transformers`                         |
| Loss       | `torch.nn`, `segmentation_models_pytorch`, `pytorch‑metric‑learning` |
| Optimizer  | `torch.optim`, `timm.optim`                                          |
| Scheduler  | `torch.optim.lr_scheduler`, `timm.scheduler`                         |
| Metrics    | `torchmetrics`, `sklearn.metrics`                                    |
| Checkpoint | `torch.save/load`, `safetensors`                                     |
| Logger     | `tensorboardX`, `wandb`, `mlflow`                                    |

---

## ✨ 기여 방법

1. 새로운 Agent/Component는 **base 추상 클래스**를 상속합니다.
2. 모듈 파일명을 YAML에서 참조할 키와 동일하게 설정합니다.
3. Pull Request 전에 `main.py --config tests/dry_run.yaml` 로 Dry‑run을 통과해야 합니다.

---
