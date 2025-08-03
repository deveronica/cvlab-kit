# Adding New Component Implementation

## 1. 컴포넌트 추가 위치
- cvlabkit/component 디렉토리 하위에 컴포넌트 타입에 맞는 디렉토리를 생성하고, 그 안에 구현체를 추가합니다.
- 예를 들어, 새로운 loss 함수를 추가하려면 `cvlabkit/component/loss` 디렉토리에 my_loss.py 파일을 생성하고, 그 안에 MyLoss 클래스를 구현합니다.

```
cvlabkit/
└── component/
    ├── loss/
    │   ├── cross_entropy.py
    │   └── my_loss.py  # <--- 여기에 추가
    └── model/
        └── ...
```

## 2. **추상 클래스 상속**
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

## 3. **설정 값 반영**
- YAML 파일에 구현한 컴포넌트명을 값으로 등록합니다.

```yaml
optimizer: adamw
```

## 4. **자동 로딩**
- 학습 스크립트에서는 별도 import 없이 YAML 수정만으로, 기존에 구성된 구조를 유지하며 새로운 컴포넌트를 자동으로 로드할 수 있습니다.
- 예를 들어, Agent에서 `create.optimizer` 함수를 그대로 유지해도, yaml의 optimizer에 해당하는 값이 "adamw"로 변경되었으므로, `AdamW`에 해당하는 `Optimizer` 구현체를 생성할 수 있습니다.

```python
opt = create.optimizer(model.parameters())
```

## 5. **자동 로딩 테스트**
- `main.py`을 수행하여 새로운 컴포넌트가 올바르게 로드되는지 확인합니다.