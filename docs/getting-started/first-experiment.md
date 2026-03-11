# 첫 실험 실행하기

설치가 완료되었다면, 첫 번째 실험을 실행해봅시다.

## 1. 예제 설정 파일 확인

```bash
cat config/cifar10_baseline.yaml
```

## 2. 실험 실행

```bash
uv run main.py --config config/cifar10_baseline.yaml
```

### 예상 출력

```
Epoch [1/10] ━━━━━━━━━━━━━━━━━━━━ 100%
  train_loss: 1.234
  train_acc: 0.567
  val_loss: 1.123
  val_acc: 0.612
```

## 3. Web UI로 결과 확인

### Web Helper 시작

```bash
uv run app.py --dev
```

브라우저에서 `http://localhost:5173` 접속

### 결과 탐색

1. **Projects 탭**: 모든 프로젝트와 실험 목록
2. **실험 선택**: 방금 실행한 실험 클릭
3. **메트릭 확인**: 학습 곡선, 최종 정확도, 하이퍼파라미터

## 4. 나만의 설정 만들기

`config/my_experiment.yaml` 생성:

```yaml
project: my_experiments

# 데이터셋
dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

# 모델
model: resnet18(num_classes=10)

# 학습 설정
optimizer: adam(lr=0.001)
loss: cross_entropy
epochs: 5
batch_size: 128

# 디바이스
device: cuda
```

### 실행

```bash
uv run main.py --config config/my_experiment.yaml
```

## 주요 개념

| 개념 | 설명 | 예시 |
|------|------|------|
| **Agent** | 실험 오케스트레이션 | `classification`, `fixmatch` |
| **Component** | 재사용 가능한 ML 블록 | `model`, `loss`, `optimizer` |
| **Creator** | 동적 컴포넌트 생성 | `self.create.model()` |
| **Config** | YAML 기반 설정 | `config/*.yaml` |

## 다음 단계

- [설정 문법 가이드](../가이드/설정-문법.md) - YAML 작성법
- [설정 예제](../가이드/설정-예제.md) - 다양한 실험 설정
- [웹 UI 사용법](../가이드/웹-UI-사용.md) - 결과 분석
