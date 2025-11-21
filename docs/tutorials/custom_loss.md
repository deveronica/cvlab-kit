# 커스텀 Loss 작성하기

이 튜토리얼에서는 CVLab-Kit에 새로운 Loss 컴포넌트를 추가하는 방법을 배웁니다.

## 시나리오

Focal Loss를 CVLab-Kit에서 사용할 수 있도록 커스텀 컴포넌트로 작성해봅시다. Focal Loss는 클래스 불균형 문제를 해결하기 위한 손실 함수입니다.

## 1단계: 기본 Loss 작성

```python
# cvlabkit/component/loss/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvlabkit.component.base import Loss


class FocalLoss(Loss):
    """Focal Loss for addressing class imbalance.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Args:
        cfg: Configuration with:
            - alpha (float): Weighting factor (default: 0.25)
            - gamma (float): Focusing parameter (default: 2.0)
            - reduction (str): 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.get("alpha", 0.25)
        self.gamma = cfg.get("gamma", 2.0)
        self.reduction = cfg.get("reduction", "mean")

    def forward(self, preds, targets):
        """Compute focal loss.

        Args:
            preds (torch.Tensor): Model predictions [B, C]
            targets (torch.Tensor): Ground truth labels [B]

        Returns:
            torch.Tensor: Computed loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(preds, targets, reduction="none")

        # Get predicted probabilities
        p = torch.exp(-ce_loss)

        # Focal loss formula: -(1-p)^gamma * log(p)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
```

## 2단계: YAML 설정에서 사용

```yaml
# config/focal_loss_example.yaml
project: imbalanced_classification

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

model: resnet18(num_classes=10)

# Focal Loss 사용
loss: focal_loss(alpha=0.25, gamma=2.0, reduction=mean)

optimizer: adam(lr=0.001)
epochs: 100
batch_size: 128
```

## 고급: 보조 Loss (Auxiliary Loss)

다중 출력 모델을 위한 보조 Loss:

```python
# cvlabkit/component/loss/auxiliary_loss.py
from cvlabkit.component.base import Loss


class AuxiliaryLoss(Loss):
    """Loss for models with multiple outputs.

    Args:
        cfg: Configuration with:
            - main_weight (float): Weight for main task (default: 1.0)
            - aux_weight (float): Weight for auxiliary task (default: 0.3)
    """

    def __init__(self, cfg):
        super().__init__()
        self.main_weight = cfg.get("main_weight", 1.0)
        self.aux_weight = cfg.get("aux_weight", 0.3)

        # Use cross entropy for both tasks
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """Compute auxiliary loss.

        Args:
            outputs (dict): {"main": main_preds, "auxiliary": aux_preds}
            targets (tuple): (main_targets, aux_targets)

        Returns:
            torch.Tensor: Combined loss value
        """
        main_targets, aux_targets = targets

        main_loss = self.ce_loss(outputs["main"], main_targets)
        aux_loss = self.ce_loss(outputs["auxiliary"], aux_targets)

        total_loss = (
            self.main_weight * main_loss +
            self.aux_weight * aux_loss
        )

        return total_loss
```

Agent에서 사용:

```python
class MultiTaskAgent(Agent):
    def train_step(self, batch):
        x, targets = batch

        # Model returns dict: {"main": ..., "auxiliary": ...}
        outputs = self.model(x)

        # Loss expects dict outputs and tuple targets
        loss = self.loss(outputs, targets)

        return loss
```

## 실전 예제: Contrastive Loss

Self-supervised learning을 위한 Contrastive Loss:

```python
# cvlabkit/component/loss/contrastive_loss.py
import torch
import torch.nn.functional as F
from cvlabkit.component.base import Loss


class ContrastiveLoss(Loss):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

    Used in SimCLR and other contrastive learning methods.

    Args:
        cfg: Configuration with:
            - temperature (float): Temperature parameter (default: 0.5)
    """

    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.get("temperature", 0.5)

    def forward(self, z_i, z_j):
        """Compute contrastive loss between two views.

        Args:
            z_i (torch.Tensor): Embeddings of view 1 [B, D]
            z_j (torch.Tensor): Embeddings of view 2 [B, D]

        Returns:
            torch.Tensor: Contrastive loss value
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # [2B, 2B]

        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)

        # Remove diagonal (self-similarity)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Positive pairs: (i, i+B) and (i+B, i)
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0).reshape(2 * batch_size, 1)

        # Negatives: all other pairs
        negatives = similarity_matrix

        # InfoNCE loss
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        loss = F.cross_entropy(logits, labels)

        return loss
```

사용 예시:

```yaml
# config/simclr.yaml
project: self_supervised

loss: contrastive_loss(temperature=0.5)

model: resnet18_encoder(output_dim=128)
optimizer: adam(lr=0.001)
```

## 디버깅 팁

**1. Loss 값 확인**

```python
class FocalLoss(Loss):
    def forward(self, preds, targets):
        focal_loss = ...

        # 디버깅: Loss 값 출력
        if self.training:
            print(f"Focal Loss: {focal_loss.item():.4f}")

        return focal_loss
```

**2. Gradient 확인**

```python
class CustomLoss(Loss):
    def forward(self, preds, targets):
        loss = ...

        # Gradient 체크
        if preds.requires_grad:
            print(f"Predictions require grad: True")

        return loss
```

**3. Loss 컴포넌트 발견 확인**

```bash
uv run python -c "from cvlabkit.core.creator import Creator; print(Creator._registry['loss'].keys())"
```

## 테스트

Loss를 작성한 후 간단히 테스트:

```python
# test_focal_loss.py
import torch
from cvlabkit.core.config import Config
from cvlabkit.component.loss.focal_loss import FocalLoss

# Create config
cfg = Config({"alpha": 0.25, "gamma": 2.0, "reduction": "mean"})

# Initialize loss
loss_fn = FocalLoss(cfg)

# Test with dummy data
preds = torch.randn(4, 10)  # [batch_size=4, num_classes=10]
targets = torch.tensor([0, 1, 2, 3])  # [batch_size=4]

loss = loss_fn(preds, targets)
print(f"Loss value: {loss.item()}")
```

## 체크리스트

- [ ] `cvlabkit/component/loss/` 디렉토리에 파일 생성
- [ ] `Loss` 베이스 클래스 상속
- [ ] `__init__(self, cfg)` 메서드 구현
- [ ] `forward(self, preds, targets)` 메서드 구현
- [ ] YAML 설정에서 loss 이름 지정
- [ ] 간단한 테스트 스크립트로 검증
- [ ] 실제 학습에서 동작 확인

## 다음 단계

- [커스텀 모델 작성하기](custom_model.md)
- [커스텀 Dataset 작성하기](custom_dataset.md)
- [컴포넌트 확장 가이드](../extending_components.md)
