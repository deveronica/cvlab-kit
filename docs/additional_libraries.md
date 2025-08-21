# Recommended Libraries

이 프로젝트는 각 컴포넌트 타입을 구현할 때 다음과 같은 주요 라이브러리 사용을 권장하거나 목표로 합니다.

| Component  | 대표 라이브러리                                                          |
| ---------- | -------------------------------------------------------------------- |
| Transform  | `torchvision.transforms`, `albumentations`, `kornia`                 |
| Dataset    | `torchvision.datasets`, `HF datasets`, `webdataset`                  |
| Model      | `torchvision.models`, `timm`, `transformers`                         |
| Loss       | `torch.nn`, `pytorch‑metric‑learning`                                |
| Optimizer  | `torch.optim`, `timm.optim`                                          |
| Scheduler  | `torch.optim.lr_scheduler`, `timm.scheduler`                         |
| Metrics    | `torchmetrics`, `sklearn.metrics`                                    |
| Checkpoint | `torch.save/load`, `safetensors`                                     |
| Logger     | `wandb`, `tensorboard`, `mlflow`                                     |
