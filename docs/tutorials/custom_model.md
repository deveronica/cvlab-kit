# 커스텀 모델 작성하기

이 튜토리얼에서는 CVLab-Kit에 새로운 모델 컴포넌트를 추가하는 방법을 단계별로 배웁니다.

## 시나리오

Vision Transformer (ViT)를 CVLab-Kit에서 사용할 수 있도록 커스텀 컴포넌트로 작성해봅시다.

## 1단계: 파일 생성

`cvlabkit/component/model/` 디렉토리에 새 파일을 생성합니다:

```bash
touch cvlabkit/component/model/vision_transformer.py
```

## 2단계: 기본 구조 작성

모든 모델은 `Model` 베이스 클래스를 상속받아야 합니다:

```python
# cvlabkit/component/model/vision_transformer.py
import torch
import torch.nn as nn
from cvlabkit.component.base import Model


class VisionTransformer(Model):
    """Vision Transformer (ViT) for image classification.

    Args:
        cfg: Configuration object containing:
            - num_classes (int): Number of output classes
            - image_size (int): Input image size (default: 224)
            - patch_size (int): Size of image patches (default: 16)
            - embed_dim (int): Embedding dimension (default: 768)
            - num_heads (int): Number of attention heads (default: 12)
            - num_layers (int): Number of transformer layers (default: 12)
    """

    def __init__(self, cfg):
        super().__init__()

        # Extract configuration parameters
        self.num_classes = cfg.get("num_classes", 1000)
        self.image_size = cfg.get("image_size", 224)
        self.patch_size = cfg.get("patch_size", 16)
        self.embed_dim = cfg.get("embed_dim", 768)
        self.num_heads = cfg.get("num_heads", 12)
        self.num_layers = cfg.get("num_layers", 12)

        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Build model components
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Classification head
        self.head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input images [B, 3, H, W]

        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        B = x.shape[0]

        # Patch embedding: [B, 3, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.patch_embed(x)

        # Flatten patches: [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoding
        x = self.transformer(x)

        # Classification (use CLS token)
        x = self.head(x[:, 0])

        return x
```

## 3단계: YAML 설정에서 사용

이제 설정 파일에서 새 모델을 사용할 수 있습니다:

```yaml
# config/vit_cifar10.yaml
project: vit_experiments

dataset:
  train: cifar10(split=train, download=true)
  val: cifar10(split=test, download=true)

# 커스텀 ViT 모델 사용
model: vision_transformer(
  num_classes=10,
  image_size=224,
  patch_size=16,
  embed_dim=768,
  num_heads=12,
  num_layers=12
)

optimizer: adam(lr=0.001)
loss: cross_entropy
epochs: 100
batch_size: 128
device: cuda
```

## 4단계: 실행 및 테스트

```bash
# 설정 검증
uv run main.py --config config/vit_cifar10.yaml

# 실제 학습 시작
uv run main.py --config config/vit_cifar10.yaml --fast
```

## 고급: Pretrained 모델 사용

timm 라이브러리를 사용하여 사전학습된 모델을 로드할 수도 있습니다:

```python
# cvlabkit/component/model/vit_pretrained.py
import timm
from cvlabkit.component.base import Model


class ViTPretrained(Model):
    """Vision Transformer with pretrained weights from timm.

    Args:
        cfg: Configuration with:
            - model_name (str): timm model name (default: 'vit_base_patch16_224')
            - num_classes (int): Number of classes
            - pretrained (bool): Use pretrained weights (default: True)
    """

    def __init__(self, cfg):
        super().__init__()
        model_name = cfg.get("model_name", "vit_base_patch16_224")
        num_classes = cfg.get("num_classes", 1000)
        pretrained = cfg.get("pretrained", True)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
```

사용 예시:

```yaml
model: vit_pretrained(
  model_name=vit_base_patch16_224,
  num_classes=10,
  pretrained=true
)
```

## 고급: 다중 출력 모델

여러 개의 출력을 반환하는 모델도 작성할 수 있습니다:

```python
class MultiHeadModel(Model):
    """Model with multiple prediction heads."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)

        # Remove original FC layer
        self.backbone.fc = nn.Identity()

        # Multiple heads
        self.classifier = nn.Linear(512, cfg.get("num_classes", 10))
        self.aux_head = nn.Linear(512, cfg.get("aux_classes", 5))

    def forward(self, x):
        features = self.backbone(x)

        # Return dictionary for multiple outputs
        return {
            "main": self.classifier(features),
            "auxiliary": self.aux_head(features)
        }
```

Agent에서 사용:

```python
class MultiTaskAgent(Agent):
    def train_step(self, batch):
        x, (y_main, y_aux) = batch

        outputs = self.model(x)

        loss_main = self.loss_main(outputs["main"], y_main)
        loss_aux = self.loss_aux(outputs["auxiliary"], y_aux)

        total_loss = loss_main + 0.3 * loss_aux
        return total_loss
```

## 디버깅 팁

**1. 컴포넌트가 인식되지 않을 때**

```bash
# Creator가 인식하는 모든 모델 확인
uv run python -c "from cvlabkit.core.creator import Creator; print(Creator._registry['model'].keys())"
```

**2. 설정 파라미터 확인**

```python
class VisionTransformer(Model):
    def __init__(self, cfg):
        super().__init__()

        # 디버깅: 받은 설정 출력
        print(f"Received config: {cfg.to_dict()}")

        self.num_classes = cfg.get("num_classes", 1000)
        print(f"Using num_classes: {self.num_classes}")
```

**3. Forward 출력 확인**

```python
def forward(self, x):
    print(f"Input shape: {x.shape}")
    x = self.model(x)
    print(f"Output shape: {x.shape}")
    return x
```

## 체크리스트

- [ ] `cvlabkit/component/model/` 디렉토리에 파일 생성
- [ ] `Model` 베이스 클래스 상속
- [ ] `__init__(self, cfg)` 메서드 구현
- [ ] `forward(self, x)` 메서드 구현
- [ ] YAML 설정에서 모델 이름 지정
- [ ] `--config` 플래그로 실행 테스트
- [ ] 실제 학습 진행 확인

## 다음 단계

- [커스텀 Loss 작성하기](custom_loss.md)
- [커스텀 Dataset 작성하기](custom_dataset.md)
- [컴포넌트 확장 가이드](../extending_components.md)
