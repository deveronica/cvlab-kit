# 권장 라이브러리

각 컴포넌트 타입 구현 시 다음 라이브러리를 권장합니다.

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
