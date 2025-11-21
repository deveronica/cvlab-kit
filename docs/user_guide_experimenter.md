# 실험자 가이드

> 웹 UI를 통한 실험 실행 및 결과 분석

## 1. 웹UI로 실험하기

```bash
## 웹 인터페이스 시작
uv run app.py
## 브라우저: http://localhost:8000
```

**워크플로우**: Execute → Queue → Devices → Metrics

## 2. 기본 실험 설정

### 간단한 분류 실험
```yaml
agent: classification

model: resnet18
num_classes: 10

transform: to_tensor
dataset:
  train: cifar10(split=train)
  val: cifar10(split=val)
dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

optimizer: adam
lr: 0.001
loss: cross_entropy
metric: accuracy

epochs: 10
batch_size: 128
```

### Grid Search (자동 실험)
```yaml
run_name: "resnet_lr{{lr}}_bs{{batch_size}}"

## 여러 값 → 자동으로 모든 조합 실험
lr: [0.001, 0.01, 0.1]
batch_size: [32, 64, 128]
## → 총 3×3 = 9개 실험 자동 생성
```

## 3. 고급 실험 패턴

### 데이터 증강
```yaml
transform:
  train: "random_crop | random_flip | to_tensor | normalize"
  val: "to_tensor | normalize"
```

## 4. 필수 설정 요소

| 구성요소 | 예시 | 설명 |
|---------|------|------|
| `agent` | `classification`, `fixmatch` | 실험 시나리오 |
| `model` | `resnet18` | 사용할 모델 |
| `dataset` | `cifar10(split=train)` | 데이터셋 |
| `optimizer` | `adam` | 최적화 알고리즘 |
| `loss` | `cross_entropy` | 손실 함수 |
| `metric` | `accuracy` | 평가 지표 |

## 5. 자주 사용하는 설정

```yaml
## 실험 메타데이터
run_name: "my_experiment_{{date}}"
description: "Baseline experiment"
author: "Your Name"

## 재현성
seed: 42

## 데이터 경로
data_root: "./data/cifar10"
download: true

## 하드웨어
device: 0
num_workers: 4
```

## 6. 문제 해결

**실험이 안 돌아갈 때**:

1. Execute 탭에서 설정 검증
2. Devices 탭에서 GPU 메모리 확인
3. CLI로 직접 실행: `uv run main.py --config my_config.yaml`

**일반적인 오류**:

- 컴포넌트 없음 → 오타 확인
- 메모리 부족 → `batch_size` 줄이기
- 파라미터 오류 → 필수 값 누락 확인

## 7. 결과 확인

- **Queue 탭**: 실행 상태
- **Metrics 탭**: 결과 비교 테이블
- **파일**: `./logs/<project>/<run_name>.csv`
---

## 자주 묻는 질문 (FAQ)

### Q: 실험이 큐에 추가되지 않습니다

**A**: 다음을 확인하세요:
1. Config 파일이 유효한지 검증 (`--config` 옵션으로 dry-run)
2. 프로젝트 이름이 올바른지 확인
3. Web Helper 서버가 실행 중인지 확인 (`http://localhost:8000`)

### Q: 결과 차트가 표시되지 않습니다

**A**: 

- CSV 파일에 metrics 열이 있는지 확인
- 로그 파일 경로가 올바른지 확인 (`logs/<project>/<run_name>.csv`)
- Reindex 버튼을 눌러 강제 재색인

### Q: Grid Search로 너무 많은 실험이 생성됩니다

**A**: 리스트 값을 줄이거나 단일 값으로 변경하세요:
```yaml
## 18개 실험 생성
lr: [0.001, 0.01, 0.1]
batch_size: [32, 64, 128]

## 3개 실험만 생성
lr: [0.001, 0.01, 0.1]
batch_size: 64
```

### Q: 실험 도중 멈춘 것 같습니다

**A**: 

1. Queue 탭에서 실험 상태 확인
2. 로그 파일 확인: `tail -f logs/<project>/<run_name>/train.log`
3. GPU 사용률 확인: `nvidia-smi`

---

## 관련 문서

- [배포 가이드](deployment_guide.md) - 서버 설정 및 배포 방법
- [분산 실행](distributed_execution_guide.md) - 여러 GPU 서버 활용
- [설정 가이드](config_guide.md) - YAML 설정 방법
- [문제 해결](troubleshooting.md) - 상세한 에러 해결 방법
