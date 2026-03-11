# 설정 예제

> 실전 YAML 설정 템플릿 모음

이 문서는 자주 사용되는 실험 시나리오별 YAML 설정 예제를 제공합니다. 각 예제는 복사하여 바로 사용할 수 있도록 작성되었습니다.

## 기본 분류 실험

### CIFAR-10 베이스라인

```yaml
## config/cifar10_baseline.yaml
run_name: "cifar10_resnet18_baseline"
description: "CIFAR-10 classification baseline with ResNet18"
author: "Your Name"

## 데이터셋
dataset:
  train: cifar10(split=train)
  val: cifar10(split=test)

## 데이터로더
dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

## 데이터 증강
transform:
  train: "random_crop(size=32, padding=4) | random_flip | to_tensor | normalize"
  val: "to_tensor | normalize"

## 모델 및 학습
model: resnet18
num_classes: 10

optimizer: adam
lr: 0.001
weight_decay: 0.0001

loss: cross_entropy
metric: accuracy

## 하이퍼파라미터
epochs: 100
batch_size: 128

## 하드웨어
device: 0
num_workers: 4

## 재현성
seed: 42
```

### ImageNet 전이학습

```yaml
## config/custom_dataset_transfer.yaml
run_name: "transfer_resnet50_{{dataset_name}}"
description: "Transfer learning from ImageNet pretrained ResNet50"

## 커스텀 데이터셋
dataset:
  train: image_folder(root=./data/my_dataset/train)
  val: image_folder(root=./data/my_dataset/val)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "resize(size=256) | random_crop(size=224) | random_flip | to_tensor | normalize"
  val: "resize(size=256) | center_crop(size=224) | to_tensor | normalize"

model: resnet50(pretrained=true)
num_classes: 10

## 전이학습용 낮은 학습률
optimizer: adam
lr: 0.0001
weight_decay: 0.0001

loss: cross_entropy
metric: "accuracy | f1(average=macro)"

epochs: 50
batch_size: 64

device: 0
num_workers: 4
seed: 42
```

## Grid Search 실험

### 학습률 탐색

```yaml
## config/grid_search_lr.yaml
run_name: "resnet18_lr{{lr}}_wd{{weight_decay}}"
description: "Grid search for learning rate and weight decay"

dataset:
  train: cifar10(split=train)
  val: cifar10(split=test)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "random_crop(size=32, padding=4) | random_flip | to_tensor | normalize"
  val: "to_tensor | normalize"

model: resnet18
num_classes: 10

optimizer: adam
## Grid search: 3 × 3 = 9개 실험 자동 생성
lr: [0.0001, 0.001, 0.01]
weight_decay: [0.0, 0.0001, 0.001]

loss: cross_entropy
metric: accuracy

epochs: 50
batch_size: 128

device: 0
num_workers: 4
seed: 42
```

### 모델 및 배치 크기 비교

```yaml
## config/grid_search_model_batch.yaml
run_name: "{{model}}_bs{{batch_size}}"
description: "Compare different models and batch sizes"

dataset:
  train: cifar10(split=train)
  val: cifar10(split=test)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "random_crop(size=32, padding=4) | random_flip | to_tensor | normalize"
  val: "to_tensor | normalize"

## 2 × 3 = 6개 실험
model: [resnet18, wideresnet]
num_classes: 10

optimizer: adam
lr: 0.001
weight_decay: 0.0001

loss: cross_entropy
metric: accuracy

epochs: 50
## 다양한 배치 크기 테스트
batch_size: [64, 128, 256]

device: 0
num_workers: 4
seed: 42
```

## 다중 컴포넌트 설정

### 여러 Loss 사용

```yaml
## config/multi_loss.yaml
run_name: "multi_loss_experiment"
description: "Experiment using multiple loss functions"

dataset:
  train: custom_dataset(split=train)
  val: custom_dataset(split=val)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "resize | augment | to_tensor | normalize"
  val: "resize | to_tensor | normalize"

model: resnet18
num_classes: 10

optimizer: adam
lr: 0.001

## 여러 loss를 사용하는 경우
loss:
  main: cross_entropy(reduction="mean")
  auxiliary: focal_loss(alpha=0.25, gamma=2.0)

metric: "accuracy | f1(average=macro)"

epochs: 100
batch_size: 64

device: 0
num_workers: 4
seed: 42
```

### 여러 Metric 파이프라인

```yaml
## config/multi_metric.yaml
run_name: "multi_metric_experiment"
description: "Track multiple evaluation metrics"

dataset:
  train: cifar10(split=train)
  val: cifar10(split=test)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "random_crop | random_flip | to_tensor | normalize"
  val: "to_tensor | normalize"

model: resnet18
num_classes: 10

optimizer: adam
lr: 0.001

loss: cross_entropy

## 여러 메트릭을 파이프라인으로 조합
metric:
  train: "accuracy"
  val: "accuracy | f1(average=macro) | f1(average=micro)"

epochs: 50
batch_size: 128

device: 0
num_workers: 4
seed: 42
```

## 분산 실행

### 다중 GPU 서버

```yaml
## config/distributed_resnet.yaml
run_name: "distributed_resnet18_{{device}}"
description: "Distributed training across multiple GPUs"

dataset:
  train: imagenet(split=train)
  val: imagenet(split=val)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "resize(size=256) | random_crop(size=224) | random_flip | to_tensor | normalize"
  val: "resize(size=256) | center_crop(size=224) | to_tensor | normalize"

model: resnet18
num_classes: 1000

optimizer: adam
lr: 0.001
weight_decay: 0.0001

loss: cross_entropy
metric: "accuracy | f1(average=macro)"

epochs: 90
batch_size: 256

## 분산 실행: 여러 GPU에 자동 분배
device: [0, 1, 2, 3]
num_workers: 8

seed: 42
```

### 원격 GPU 클라이언트

```yaml
## config/remote_client_experiment.yaml
run_name: "remote_{{client_name}}_resnet"
description: "Experiment for remote GPU client"

dataset:
  train: cifar10(split=train)
  val: cifar10(split=test)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

transform:
  train: "random_crop(size=32, padding=4) | random_flip | to_tensor | normalize"
  val: "to_tensor | normalize"

model: resnet18
num_classes: 10

optimizer: adam
lr: 0.001
weight_decay: 0.0001

loss: cross_entropy
metric: accuracy

epochs: 100
batch_size: 128

## 원격 클라이언트용 설정
device: 0  # 클라이언트의 GPU 0번 사용
num_workers: 4

## 클라이언트 메타데이터
client_name: "gpu_server_01"
remote_log_sync: true

seed: 42
```

## 고급 설정 패턴

### 조건부 설정 (플레이스홀더)

```yaml
## config/conditional_config.yaml
run_name: "{{model}}_{{dataset}}_lr{{lr}}"
description: "Experiment with {{model}} on {{dataset}}"

## 플레이스홀더를 사용한 동적 설정
model: "{{model}}"
dataset:
  train: "{{dataset}}(split=train)"
  val: "{{dataset}}(split=test)"

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

## CLI에서 오버라이드 가능
## uv run main.py --config conditional_config.yaml --model resnet50 --dataset cifar100
num_classes: 100

optimizer: adam
lr: 0.001

loss: cross_entropy
metric: accuracy

epochs: 50
batch_size: 128

device: 0
num_workers: 4
seed: 42
```

### 다양한 Augmentation 파이프라인

```yaml
## config/augmentation_pipeline.yaml
run_name: "augmentation_experiment"
description: "Testing different augmentation strategies"

dataset:
  train: cifar10(split=train)
  val: cifar10(split=test)

dataloader:
  train: basic(split=train, shuffle=true)
  val: basic(split=val, shuffle=false)

## 복잡한 augmentation 파이프라인
transform:
  train: "resize(size=32) | random_crop(size=32, padding=4) | random_flip(prob=0.5) | color_jitter(brightness=0.2, contrast=0.2) | to_tensor | normalize"
  val: "resize(size=32) | to_tensor | normalize"

model: wideresnet
num_classes: 10

optimizer: adam
lr: 0.002
weight_decay: 0.0005

loss: cross_entropy
metric: accuracy

epochs: 200
batch_size: 64

device: 0
num_workers: 4
seed: 42
```

---

## 관련 문서

- [설정 가이드](config_guide.md) - YAML 설정 규칙 및 문법
- [실험자 가이드](user_guide_experimenter.md) - 웹 UI를 통한 실험 실행
- [분산 실행 가이드](distributed_execution_guide.md) - 다중 GPU 및 원격 클라이언트 설정
- [문제 해결](troubleshooting.md) - Config 관련 에러 해결
