# 개발자 가이드

> 커스텀 컴포넌트 개발 및 Agent 확장 방법

## 핵심 철학 이해

### Why in Key, How in Value
- **Key(목적)**: Agent가 "왜 이 컴포넌트가 필요한가" (용도/목적)
- **Value(방법)**: "어떤 구현체로 해결할 것인가" (구체적 방법)

```yaml
## 단일 스키마: component_type: implementation
model: resnet18       # "모델이 필요하다(Why)" + "ResNet18로(How)"
optimizer: adam       # "최적화가 필요하다(Why)" + "Adam으로(How)"

## 다중 스키마: component_type: {key: implementation}
dataloader:
  train: basic        # "학습용 데이터가 필요하다(Why)" + "basic으로(How)"
  val: basic          # "검증용 데이터가 필요하다(Why)" + "basic으로(How)"
  
transform:
  weak: "resize | normalize"     # "약한 증강이 필요하다(Why)" + "resize|normalize로(How)"
  strong: "resize | randaugment" # "강한 증강이 필요하다(Why)" + "resize|randaugment로(How)"
```

### 다중 스키마 특징
- **평평한 1계층 구조**: 컴포넌트 이름과 파라미터가 모두 동일 레벨
- **Key는 목적**: train/val/test, weak/strong 등은 Agent 관점의 용도
- **기본값**: 명시하지 않으면 basic 사용

## 1. 새로운 컴포넌트 추가

### 모델 추가
```python
## cvlabkit/component/model/my_model.py
from cvlabkit.component.base import Model
import torch.nn as nn

class MyModel(Model):
    def __init__(self, cfg):
        super().__init__()
        # 필수: namespace 접근 (없으면 AttributeError)
        num_classes = cfg.num_classes
        
        # 선택적: get() 사용 (기본값 제공)
        hidden_size = cfg.get("hidden_size", 512)
        dropout_p = cfg.get("dropout_p", 0.1)
        
        self.fc = nn.Linear(784, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return self.classifier(x)
```

**YAML 설정**:
```yaml
model: my_model
num_classes: 10        # 필수 (없으면 AttributeError)
hidden_size: 256       # 선택적 (없으면 512 사용)
dropout_p: 0.2         # 선택적 (없으면 0.1 사용)
```

### 기존 라이브러리 래핑
```python
## cvlabkit/component/optimizer/my_adam.py
from cvlabkit.component.base import Optimizer
import torch.optim as optim

class MyAdam(Optimizer):
    def __init__(self, cfg, parameters):
        # 평평한 구조에서 파라미터 가져오기
        lr = cfg.get("lr", 0.001)
        weight_decay = cfg.get("weight_decay", 0)
        
        # 기존 라이브러리를 내부 객체로 생성
        self.opt = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    
    def step(self):
        # 커스텀 로직 추가 가능
        print(f"Learning rate: {self.opt.param_groups[0]['lr']}")
        self.opt.step()
    
    # zero_grad() 등은 InterfaceMeta가 자동으로 self.opt로 위임
```

## 2. 새로운 Agent 추가

```python
## cvlabkit/agent/my_agent.py
from cvlabkit.core.agent import Agent

class MyAgent(Agent):
    def __init__(self, cfg, component_creator):
        super().__init__(cfg, component_creator)
        # self.create = component_creator 자동 설정됨
        
    def setup(self):
        # 지휘자 역할: 의존성 순서대로 컴포넌트 생성
        
        # 1. 독립적인 컴포넌트들 먼저
        self.model = self.create.model()
        self.loss_fn = self.create.loss()
        
        # 2. 의존성이 있는 컴포넌트는 명시적 주입
        self.optimizer = self.create.optimizer(self.model.parameters())
        
        # 3. 다중 스키마: Key로 목적 구분
        self.train_loader = self.create.dataloader.train()
        self.val_loader = self.create.dataloader.val()
        
        # 4. 선택적 컴포넌트 (YAML에 있을 때만)
        if 'scheduler' in self.cfg:
            self.scheduler = self.create.scheduler(self.optimizer)
        
        # 5. 변환: 용도별로 구분
        if 'transform' in self.cfg:
            self.weak_transform = self.create.transform.weak()
            self.strong_transform = self.create.transform.strong()
    
    def train_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
```

## 3. Config 접근 가이드

### 기본 원칙
- **필수 파라미터**: `cfg.key` (AttributeError로 누락 감지)
- **선택적 파라미터**: `cfg.get("key", default)` (안전한 기본값)

```python
class MyComponent:
    def __init__(self, cfg):
        # 필수: 없으면 즉시 에러
        self.num_classes = cfg.num_classes
        self.model_name = cfg.model_name
        
        # 선택적: 기본값으로 안전하게
        self.learning_rate = cfg.get("lr", 0.001)
        self.batch_size = cfg.get("batch_size", 32)
        self.use_dropout = cfg.get("use_dropout", True)
        
        # 조건부 로직
        if cfg.get("use_pretrained", False):
            self.load_pretrained(cfg.pretrained_path)
```

## 4. YAML 스키마 가이드

### 평평한 구조 (모든 파라미터 1계층)
```yaml
## 컴포넌트와 파라미터가 동일 레벨
model: resnet18
optimizer: adam
loss: cross_entropy

## 모든 파라미터도 1계층에
num_classes: 10
lr: 0.001
batch_size: 32
epochs: 100

## 다중 스키마: Key는 목적
dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)
  test: basic                           # 기본값 사용

transform:
  weak: "resize | normalize"
  strong: "resize | randaugment | normalize"
```

### Creator 사용법
```python
## Agent에서의 접근
self.model = self.create.model()                    # 단일 스키마
self.train_loader = self.create.dataloader.train()  # 다중 스키마 - Key로 목적 지정
self.val_loader = self.create.dataloader.val()      # 다중 스키마 - Key로 목적 지정
self.weak_aug = self.create.transform.weak()        # 다중 스키마 - 약한 증강
self.strong_aug = self.create.transform.strong()    # 다중 스키마 - 강한 증강
```

## 5. 개발 규칙

### DO ✅
- **Config 접근**: 필수는 `cfg.key`, 선택은 `cfg.get()`
- **의존성 주입**: Agent가 명시적으로 관리 (`optimizer(model.parameters())`)
- **다중 스키마**: Key로 목적 구분 (train/val, weak/strong)
- **평평한 구조**: 모든 파라미터를 1계층에 배치
- **기본값**: 명시하지 않으면 basic 사용

### DON'T ❌
- **cvlabkit/core/ 수정**: 안정성 정책 위반
- **직접 Creator 생성**: Agent에서 `Creator()` 직접 호출 금지
- **중첩 구조**: 파라미터를 여러 계층으로 나누지 말 것
- **컴포넌트 간 직접 의존**: 모든 의존성은 Agent가 관리

## 6. 테스트 방법

```bash
## 설정 검증 (Dry-run)
uv run main.py --config config/my_experiment.yaml

## 실제 실행
uv run main.py --config config/my_experiment.yaml --fast

## 템플릿 생성으로 필요한 모든 파라미터 확인
python config/generate_template.py
```

## 7. 자주 사용하는 패턴

### 조건부 컴포넌트
```python
class MyAgent(Agent):
    def setup(self):
        # 기본 컴포넌트
        self.model = self.create.model()
        self.optimizer = self.create.optimizer(self.model.parameters())
        
        # 선택적 컴포넌트 (YAML에 있을 때만)
        if 'scheduler' in self.cfg:
            self.scheduler = self.create.scheduler(self.optimizer)
        
        if 'transform' in self.cfg:
            self.transform = self.create.transform.train()
```

### Grid Search 지원
```yaml
## 자동으로 모든 조합 실험 생성
run_name: "experiment_lr{{lr}}_bs{{batch_size}}"
lr: [0.001, 0.01, 0.1]        # 3개 값
batch_size: [32, 64, 128]     # 3개 값
## → 총 3×3 = 9개 실험 자동 생성
```

### 복잡한 파이프라인
```yaml
transform:
  train: "resize(size=224) | random_flip | normalize(mean=[0.485, 0.456, 0.406])"
  val: "resize(size=224) | normalize(mean=[0.485, 0.456, 0.406])"
```

```python
## Agent에서 사용
self.train_transform = self.create.transform.train()  # 파이프라인 자동 구성
self.val_transform = self.create.transform.val()      # 파이프라인 자동 구성
```

---

## 관련 문서

- [컴포넌트 확장](extending_components.md)
- [아키텍처](architecture.md)
- [개발 철학](development_philosophy.md)
