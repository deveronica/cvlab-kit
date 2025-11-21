# Grid Search로 하이퍼파라미터 탐색하기

이 튜토리얼에서는 CVLab-Kit의 자동 Grid Search 기능을 사용하여 최적의 하이퍼파라미터를 찾는 방법을 배웁니다.

## 기본 개념

YAML 설정에서 **최상위 레벨에 리스트를 사용하면 자동으로 Grid Search가 실행**됩니다. 모든 조합이 별도 실험으로 생성됩니다.

**중요**: 괄호 안의 파라미터는 그리드 서치가 되지 않습니다.
- ❌ `optimizer: adam(lr=[0.001, 0.01])` - 작동 안함 (괄호 안)
- ✅ `lr: [0.001, 0.01]` + `optimizer: adam` - 작동함 (최상위 레벨)

## 간단한 예제

### 학습률 탐색

```yaml
# config/lr_search.yaml
project: lr_search

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

model: resnet18(num_classes=10)

# 학습률 3개 값 테스트 (최상위 레벨에 리스트)
lr: [0.0001, 0.001, 0.01]
optimizer: adam

loss: cross_entropy
epochs: 50
batch_size: 128
```

**결과**: 3개의 실험이 자동 생성되어 순차 실행됩니다.

```
실험 1: lr=0.0001
실험 2: lr=0.001
실험 3: lr=0.01
```

### 실행

```bash
uv run main.py --config config/lr_search.yaml --fast
```

## 다중 파라미터 Grid Search

### 2차원 Grid

```yaml
# config/2d_grid.yaml
project: optimizer_search

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

model: resnet18(num_classes=10)

# 학습률 3개 × 배치 크기 2개 = 6개 실험
lr: [0.0001, 0.001, 0.01]
batch_size: [64, 128]
optimizer: adam

loss: cross_entropy
epochs: 50
```

**결과**: 6개의 실험 조합

```
실험 1: lr=0.0001, batch_size=64
실험 2: lr=0.0001, batch_size=128
실험 3: lr=0.001, batch_size=64
실험 4: lr=0.001, batch_size=128
실험 5: lr=0.01, batch_size=64
실험 6: lr=0.01, batch_size=128
```

### 3차원 Grid

```yaml
# config/3d_grid.yaml
project: full_search

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

# 모델 2개 × 학습률 3개 × 배치 크기 2개 = 12개 실험
model: [resnet18(num_classes=10), resnet50(num_classes=10)]
lr: [0.0001, 0.001, 0.01]
batch_size: [64, 128]
optimizer: adam

loss: cross_entropy
epochs: 50
```

**결과**: 12개의 실험 조합

## 고급: 컴포넌트 비교

### Optimizer 비교

```yaml
# config/optimizer_comparison.yaml
project: optimizer_comparison

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

model: resnet18(num_classes=10)

# 3가지 optimizer 비교
optimizer: [
  adam(lr=0.001),
  sgd(lr=0.01, momentum=0.9),
  adamw(lr=0.001, weight_decay=0.01)
]

loss: cross_entropy
epochs: 100
batch_size: 128
```

### Loss 함수 비교

```yaml
# config/loss_comparison.yaml
project: loss_comparison

model: resnet18(num_classes=10)

# 2가지 loss 비교
loss: [
  cross_entropy,
  focal_loss(alpha=0.25, gamma=2.0)
]

optimizer: adam(lr=0.001)
epochs: 100
```

## 고급: 조건부 Grid Search

### Placeholder를 활용한 연동

```yaml
# config/conditional_grid.yaml
project: conditional_search

# 전역 변수 정의
num_classes: 10
base_lr: [0.0001, 0.001, 0.01]

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

model: resnet18(num_classes={{num_classes}})

# base_lr에 따라 weight_decay 자동 조정
optimizer: adam(
  lr={{base_lr}},
  weight_decay=0.0001
)

loss: cross_entropy
epochs: 50
batch_size: 128
```

## Web UI에서 결과 비교

### Projects 탭에서 비교

1. Web UI 접속: `http://localhost:5173`
2. **Projects** 탭으로 이동
3. 프로젝트 선택 (예: `lr_search`)
4. 모든 실험이 테이블로 표시됨

**테이블 컬럼**:
- `run_name`: 실험 이름
- `lr`: 학습률 (하이퍼파라미터)
- `batch_size`: 배치 크기
- `val_acc`: 최종 검증 정확도
- `val_loss`: 최종 검증 손실
- Status: 실행 상태

### 정렬 및 필터링

**최고 성능 찾기**:
- `val_acc` 컬럼 클릭하여 내림차순 정렬

**특정 조건 필터링**:
- 테이블 상단 검색 바 사용
- 예: `lr > 0.001 AND batch_size = 128`

### 시각화

**메트릭 비교**:
1. 여러 실험 선택 (체크박스)
2. **Compare** 버튼 클릭
3. 학습 곡선 오버레이 표시

## 실전 예제: ResNet 아키텍처 탐색

```yaml
# config/architecture_search.yaml
project: architecture_search

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

# ResNet depth 비교: 18, 34, 50
model: [
  resnet18(num_classes=10),
  resnet34(num_classes=10),
  resnet50(num_classes=10)
]

# 각 모델에 맞는 학습률 탐색
lr: [0.0001, 0.001]
optimizer: adam

loss: cross_entropy
epochs: 100
batch_size: 128

# 총 3 × 2 = 6개 실험
```

결과 분석:

| Model | LR | Val Acc | 학습 시간 |
|-------|--------|---------|-----------|
| resnet18 | 0.0001 | 0.89 | 45분 |
| resnet18 | 0.001 | 0.92 | 45분 |
| resnet34 | 0.0001 | 0.90 | 65분 |
| resnet34 | 0.001 | 0.93 | 65분 |
| resnet50 | 0.0001 | 0.91 | 90분 |
| resnet50 | 0.001 | **0.94** | 90분 |

**결론**: ResNet50 + lr=0.001이 최고 성능

## 고급: Random Search (수동)

Grid Search가 너무 많은 조합을 만들 때, 수동으로 Random Search 구현:

```python
# scripts/random_search.py
import random
import yaml
import subprocess

# Search space
lr_space = [0.0001, 0.0003, 0.001, 0.003, 0.01]
batch_size_space = [32, 64, 128, 256]
weight_decay_space = [0, 1e-5, 1e-4, 1e-3]

# Random sampling (20회)
for i in range(20):
    config = {
        "project": "random_search",
        "dataset": {
            "train": "cifar10(split=train, download=true)",
            "val": "cifar10(split=test, download=true)"
        },
        "model": "resnet18(num_classes=10)",
        "optimizer": f"adam(lr={random.choice(lr_space)}, weight_decay={random.choice(weight_decay_space)})",
        "loss": "cross_entropy",
        "batch_size": random.choice(batch_size_space),
        "epochs": 50
    }

    # Save config
    config_path = f"config/random_search_{i}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Submit to queue
    subprocess.run(["uv", "run", "main.py", "--config", config_path, "--fast"])
```

## 큐를 활용한 대규모 Grid Search

### 자동 큐 제출

```bash
# 큐에 Grid Search 실험 모두 제출
uv run app.py --dev &
sleep 3

# Grid Search 설정 실행 (큐에 자동 추가)
uv run main.py --config config/3d_grid.yaml --fast
```

**Web UI에서 확인**:
- **Queue** 탭에서 12개 실험이 순차 대기
- **Devices** 탭에서 GPU 할당 상태 확인
- 유휴 GPU에 자동으로 실험 할당됨

### 분산 환경에서 Grid Search

여러 GPU 서버가 있을 때:

```yaml
# config/large_grid.yaml (100개 조합)
project: large_grid_search

model: [resnet18, resnet34, resnet50, wideresnet28_10]
lr: [0.0001, 0.0003, 0.001, 0.003, 0.01]
batch_size: [32, 64, 128, 256, 512]
optimizer: adam

# 4 × 5 × 5 = 100개 실험
```

실행:

**중앙 서버에서**:
```bash
uv run app.py --server-only
```

**GPU 서버 1, 2, 3에서 각각**:
```bash
uv run app.py --client-only --url http://central-server:8000
```

Web UI에서 큐에 제출하면 100개 실험이 3대 GPU 서버에 자동 분산 실행

## 결과 분석

### CSV 내보내기

```python
# scripts/export_results.py
import requests
import pandas as pd

# Get all experiments
url = "http://localhost:8000/api/projects/lr_search/experiments"
response = requests.get(url)
data = response.json()["data"]

# Convert to DataFrame
experiments = data["experiments"]
df = pd.DataFrame([
    {
        "run_name": exp["run_name"],
        **exp["hyperparameters"],
        **exp["final_metrics"]
    }
    for exp in experiments
])

# Save to CSV
df.to_csv("grid_search_results.csv", index=False)
print(df.sort_values("val_acc", ascending=False).head(10))
```

### 최적 하이퍼파라미터 추출

```python
# Find best hyperparameters
best = df.loc[df["val_acc"].idxmax()]

print(f"Best configuration:")
print(f"  Learning rate: {best['lr']}")
print(f"  Batch size: {best['batch_size']}")
print(f"  Val accuracy: {best['val_acc']:.4f}")
```

## 팁

### 1. 실험 개수 제어

Grid Search 전에 조합 개수 확인:

```python
# scripts/count_combinations.py
import itertools

lr = [0.0001, 0.001, 0.01]
batch_size = [64, 128]
model = ["resnet18", "resnet50"]

total = len(lr) * len(batch_size) * len(model)
print(f"Total experiments: {total}")  # 12

# 예상 시간 계산 (각 실험 1시간 가정)
print(f"Estimated time: {total} hours")
```

### 2. 단계적 탐색

**Step 1**: 넓은 범위 탐색

```yaml
lr: [0.0001, 0.001, 0.01, 0.1]
optimizer: adam
```

**Step 2**: 좁은 범위 정밀 탐색

```yaml
# Step 1에서 lr=0.001이 최고였다면
lr: [0.0005, 0.001, 0.0015, 0.002]
optimizer: adam
```

### 3. 조기 종료

짧은 epoch으로 빠르게 탐색:

```yaml
# 초기 탐색 (10 epoch)
epochs: 10
batch_size: [64, 128, 256]
lr: [0.0001, 0.001, 0.01]
optimizer: adam

# 상위 3개 설정만 선택하여 전체 학습 (100 epoch)
```

## 체크리스트

- [ ] YAML에서 리스트로 하이퍼파라미터 정의
- [ ] 조합 개수 계산 (너무 많으면 조정)
- [ ] `--fast` 플래그로 Grid Search 실행
- [ ] Web UI Projects 탭에서 결과 확인
- [ ] 메트릭 기준으로 정렬하여 최고 성능 찾기
- [ ] 최적 설정으로 재학습

## 다음 단계

- [성능 튜닝 가이드](../performance_tuning.md)
- [분산 실행으로 Grid Search 가속화](distributed_setup.md)
- [실험자 가이드](../user_guide_experimenter.md)
