# 빠른 시작 가이드

30분 안에 CVLab-Kit로 첫 번째 실험을 실행하고 결과를 확인해보세요.

**키워드**: 설치, 시작하기, 튜토리얼, CIFAR-10, Web UI, 첫 실험, 빠른 시작

## 사전 준비

- Python 3.8 이상
- CUDA 지원 GPU (선택사항)
- 10GB 이상의 디스크 공간

## 1단계: 설치 (5분)

### 저장소 클론

```bash
git clone https://github.com/deveronica/cvlab-kit.git
cd cvlab-kit
```

### 의존성 설치

**uv 사용** (권장):
```bash
uv sync
```

**pip 사용**:
```bash
pip install -e .
```

**검증**: 설치가 완료되면 다음 명령어가 정상 작동해야 합니다.
```bash
uv run python -c "import cvlabkit; print('설치 완료!')"
```

## 2단계: 첫 실험 실행 (10분)

프로젝트에 제공된 예제 설정 파일을 사용합니다.

**설정 파일 확인**:
```bash
cat config/cifar10_baseline.yaml
```

**실험 실행**:
```bash
uv run main.py --config config/cifar10_baseline.yaml
```

**진행 상황 확인**:
- 터미널에 학습 진행률이 표시됩니다
- `logs/cifar10/` 디렉토리에 결과가 저장됩니다

### 예상 출력

```
Epoch [1/10] ━━━━━━━━━━━━━━━━━━━━ 100%
  train_loss: 1.234
  train_acc: 0.567
  val_loss: 1.123
  val_acc: 0.612
```

## 3단계: Web UI로 결과 확인 (5분)

### Web Helper 시작

백엔드 + 프론트엔드 개발 서버 시작:
```bash
uv run app.py --dev
```

**접속**: 브라우저에서 `http://localhost:5173` 열기

### 실험 결과 탐색

1. **Projects 탭**: 모든 프로젝트와 실험 목록
2. **실험 선택**: `cifar10` 프로젝트에서 방금 실행한 실험 클릭
3. **메트릭 확인**:
   - 학습 곡선 그래프
   - 최종 정확도
   - 하이퍼파라미터 설정

## 4단계: 설정 커스터마이즈 (10분)

### YAML 설정 수정

`config/my_first_experiment.yaml` 파일 생성:

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
device: cuda  # 또는 cpu
```

### 수정된 설정으로 실험 실행

```bash
uv run main.py --config config/my_first_experiment.yaml
```

## 다음 단계

축하합니다! 첫 실험을 성공적으로 완료했습니다. 이제 다음을 탐색해보세요:

### 실험자라면
- [실험자 가이드](user_guide_experimenter.md) - Web UI 활용법
- [설정 예제](config_examples.md) - 다양한 실험 설정
- [분산 실행 빠른 시작](distributed_execution_quickstart.md) - 다중 GPU 활용

### 개발자라면
- [개발자 가이드](user_guide_developer.md) - 컴포넌트 개발
- [아키텍처](architecture.md) - 시스템 구조 이해
- [컴포넌트 확장](extending_components.md) - 커스텀 컴포넌트 작성

### 문제가 발생했나요?

- [문제 해결 가이드](troubleshooting.md)
- [FAQ](faq.md)
- [GitHub Issues](https://github.com/deveronica/cvlab-kit/issues)

## 주요 개념 한눈에 보기

| 개념 | 설명 | 예시 |
|------|------|------|
| **Agent** | 실험 오케스트레이션 | `classification`, `fixmatch` |
| **Component** | 재사용 가능한 ML 빌딩 블록 | `model`, `loss`, `optimizer` |
| **Creator** | 동적 컴포넌트 생성 팩토리 | `self.create.model()` |
| **Config** | YAML 기반 실험 설정 | `config/*.yaml` |

더 자세한 내용은 [핵심 개념](architecture.md)을 참고하세요.
