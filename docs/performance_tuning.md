# 성능 튜닝 가이드

> GPU 메모리 최적화, 학습 속도 향상, 디버깅 팁

이 문서는 CVLab-Kit 실험의 성능을 최적화하기 위한 실전 가이드입니다.

## GPU 메모리 최적화

### 1. 배치 크기 조정

**증상**: `RuntimeError: CUDA out of memory`

```yaml
## ❌ 메모리 부족
batch_size: 256

## ✅ 배치 크기 줄이기
batch_size: 128

## ✅ 또는 gradient accumulation 사용
batch_size: 64
gradient_accumulation_steps: 2  # 실질적으로 128과 동일
```


### 2. 혼합 정밀도 학습 (Mixed Precision)

**PyTorch AMP 사용** (자동 메모리 절약):

```python
## cvlabkit/agent/classification.py
import torch.cuda.amp as amp

class ClassificationAgent(Agent):
    def train_step(self, batch):
        with amp.autocast():
            outputs = self.model(batch['x'])
            loss = self.loss(outputs, batch['y'])

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

**효과**: 메모리 사용량 **30-50% 감소**, 학습 속도 **20-30% 향상**

### 3. DataLoader 워커 수 조정

```yaml
## CPU 코어 수 확인: `lscpu` 또는 `nproc`

## ❌ 너무 많으면 CPU 병목
num_workers: 16

## ✅ 최적값: CPU 코어 수의 1/2 ~ 1/4
num_workers: 4
```

### 4. 모델 크기 줄이기

```yaml
## ❌ 큰 모델
model: resnet152

## ✅ 작은 모델로 실험 후 스케일업
model: resnet18

## ✅ 또는 파라미터 줄이기
model: wideresnet(depth=28, width=2)  # 기본값: depth=28, width=10
```

### 5. 이미지 해상도 줄이기

```yaml
transform:
  train: "resize(size=224) | ..."  # ❌ 높은 해상도
  train: "resize(size=128) | ..."  # ✅ 낮은 해상도

## 또는 동적 해상도
transform:
  train: "random_resized_crop(size=224, scale=(0.8, 1.0)) | ..."
```

## 학습 속도 향상

### 1. 데이터 로딩 병목 제거

**CPU-GPU 병렬화 확인**:

```bash
## GPU 사용률 모니터링
nvidia-smi dmon -s u

## GPU 사용률이 100% 미만이면 CPU 병목
## → num_workers 늘리기
```

```yaml
## 데이터 로딩 최적화
num_workers: 4
pin_memory: true  # CUDA 메모리 고정 (속도 향상)
persistent_workers: true  # 워커 재사용 (초기화 오버헤드 감소)
```

### 2. 모델 컴파일 (PyTorch 2.0+)

```python
## cvlabkit/agent/classification.py
import torch

class ClassificationAgent(Agent):
    def setup(self):
        self.model = torch.compile(self.model)  # JIT 컴파일
```

**효과**: 학습 속도 **10-30% 향상** (큰 모델일수록 효과 큼)

### 3. Gradient Checkpointing (메모리 <-> 속도 트레이드오프)

```python
## 메모리를 절약하되 속도는 약간 느려짐
model = resnet50(num_classes=10)
model.gradient_checkpointing_enable()
```

**사용 시나리오**: 메모리가 부족하지만 배치 크기를 유지하고 싶을 때

### 4. 프리페칭 (Prefetching)

```yaml
## DataLoader에서 자동으로 다음 배치 준비
dataloader:
  train: basic(split=train, shuffle=true, prefetch_factor=2)
```

### 5. 캐싱 (작은 데이터셋)

```python
## cvlabkit/component/dataset/cifar10.py
class CIFAR10Dataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        # 전체 데이터를 메모리에 로드
        self.data = self._load_all()

    def __getitem__(self, idx):
        # 디스크 I/O 없이 메모리에서 직접 반환
        return self.data[idx]
```

**효과**: CIFAR-10 같은 작은 데이터셋에서 학습 속도 **2-3배 향상**

## 분산 실험 최적화

### 1. 다중 GPU 활용

```yaml
## config/distributed_experiment.yaml
device: [0, 1, 2, 3]  # 4개 GPU에 자동 분배
batch_size: 256       # 각 GPU당 64

## PyTorch DistributedDataParallel 자동 활성화
```

### 2. 원격 클라이언트 부하 분산

```yaml
## 서버: 로컬 GPU 없이 큐 관리만
## uv run app.py --server-only

## 클라이언트 1: GPU 0-1 사용
## uv run app.py --client-only --url http://server:8000

## 클라이언트 2: GPU 0-3 사용
## uv run app.py --client-only --url http://server:8000
```

**효과**: 여러 GPU 서버를 하나의 큐로 통합 관리

### 3. Grid Search 병렬 실행

```yaml
## config/parallel_grid.yaml
lr: [0.0001, 0.001, 0.01]  # 3개 실험
batch_size: [64, 128]      # 2개 실험
## → 총 6개 실험

device: [0, 1, 2, 3]  # 4개 GPU에 분배
## → 동시에 4개 실험 실행, 나머지 2개는 큐 대기
```

## 메모리 프로파일링

### PyTorch 메모리 추적

```python
import torch

## 메모리 스냅샷 저장
torch.cuda.memory._record_memory_history()

## 학습 코드 실행
agent.train()

## 메모리 사용 분석
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

```bash
## 분석 도구
python -m torch.utils.memory_profiler memory_snapshot.pickle
```

### nvidia-smi 모니터링

```bash
## 실시간 GPU 사용률 모니터링
watch -n 1 nvidia-smi

## 또는 자세한 정보
nvidia-smi dmon -s ucm -d 1

## 특정 프로세스 추적
nvidia-smi pmon -i 0
```

## 디버깅 및 프로파일링

### 1. 학습 속도 병목 찾기

```python
import time

class ClassificationAgent(Agent):
    def train_epoch(self):
        for i, batch in enumerate(self.train_loader):
            t0 = time.time()

            # 데이터 로딩 시간
            data_time = time.time() - t0

            # Forward/Backward 시간
            t1 = time.time()
            loss = self.train_step(batch)
            compute_time = time.time() - t1

            if i % 100 == 0:
                print(f"Data: {data_time:.3f}s | Compute: {compute_time:.3f}s")
```

**분석**:

- `data_time > compute_time` → **CPU 병목** (num_workers 늘리기)
- `compute_time > data_time` → **GPU 병목** (정상)

### 2. PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    agent.train_epoch()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 3. 느린 Transform 찾기

```python
import time

class DebugTransform:
    def __init__(self, transform, name):
        self.transform = transform
        self.name = name

    def __call__(self, x):
        t0 = time.time()
        result = self.transform(x)
        elapsed = time.time() - t0
        if elapsed > 0.01:  # 10ms 이상이면 출력
            print(f"{self.name}: {elapsed:.3f}s")
        return result
```

## 자주 사용하는 최적화 조합

### 시나리오 1: 메모리 부족

```yaml
## 메모리 절약 최우선
batch_size: 64              # 배치 크기 줄이기
gradient_accumulation_steps: 2
num_workers: 2              # 워커 수 줄이기
transform:
  train: "resize(size=128) | ..."  # 해상도 줄이기
```


## 성능 체크리스트

실험 전 다음 항목들을 확인하세요:

### GPU 설정
- [ ] `nvidia-smi`로 GPU 가용 메모리 확인
- [ ] 다른 프로세스가 GPU를 사용 중인지 확인
- [ ] 배치 크기가 메모리에 맞는지 확인

### 데이터 로딩
- [ ] `num_workers` 적절히 설정 (CPU 코어의 1/4)
- [ ] `pin_memory: true` 설정
- [ ] 데이터셋이 로컬 디스크에 있는지 확인 (NFS는 느림)

### 모델 최적화
- [ ] 작은 모델로 먼저 실험
- [ ] 혼합 정밀도 학습 고려 (`use_amp: true`)
- [ ] PyTorch 2.0+에서 `torch.compile` 사용

### 실험 관리
- [ ] Grid search는 분산 실행으로
- [ ] 긴 실험은 체크포인트 저장 설정
- [ ] 로그 파일 크기 주기적으로 확인

---

## 관련 문서

- [분산 실행 가이드](distributed_execution_guide.md) - 다중 GPU 및 원격 클라이언트 설정
- [설정 예제](config_examples.md) - 최적화된 YAML 설정 템플릿
- [문제 해결](troubleshooting.md) - 메모리 및 성능 관련 에러 해결
- [배포 가이드](deployment_guide.md) - 서버 하드웨어 선택 및 설정
