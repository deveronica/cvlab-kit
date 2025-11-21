# 커스텀 Dataset 작성하기

이 튜토리얼에서는 CVLab-Kit에 새로운 Dataset 컴포넌트를 추가하는 방법을 배웁니다.

## 시나리오

커스텀 이미지 분류 데이터셋을 CVLab-Kit에서 사용할 수 있도록 작성해봅시다.

## 1단계: 기본 Dataset 작성

```python
# cvlabkit/component/dataset/custom_image_dataset.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset
from cvlabkit.component.base import Dataset


class CustomImageDataset(Dataset):
    """Custom image classification dataset.

    Directory structure:
        root/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
        │   └── class2/
        │       ├── img1.jpg
        │       └── img2.jpg
        └── test/
            └── ...

    Args:
        cfg: Configuration with:
            - root (str): Root directory path
            - split (str): 'train' or 'test'
            - extensions (list): Image file extensions (default: ['.jpg', '.png'])
    """

    def __init__(self, cfg):
        super().__init__()
        self.root = Path(cfg.get("root", "data"))
        self.split = cfg.get("split", "train")
        self.extensions = cfg.get("extensions", [".jpg", ".png", ".jpeg"])

        # Build file list and class mapping
        self.samples = []
        self.class_to_idx = {}
        self._build_dataset()

    def _build_dataset(self):
        """Scan directory and build file list."""
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        # Get all class directories
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        # Build class mapping
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx

            # Find all images in this class
            for ext in self.extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((img_path, idx))

        print(f"Found {len(self.samples)} images in {len(self.class_to_idx)} classes")

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)

        return image, label
```

## 2단계: YAML 설정에서 사용

```yaml
# config/custom_dataset.yaml
project: custom_classification

dataset:
  train: custom_image_dataset(
    root=data/my_dataset,
    split=train,
    extensions=[.jpg, .png]
  )
  val: custom_image_dataset(
    root=data/my_dataset,
    split=test
  )

transform:
  train: "resize(size=224) | random_horizontal_flip | to_tensor | normalize"
  val: "resize(size=224) | to_tensor | normalize"

model: resnet18(num_classes=10)
optimizer: adam(lr=0.001)
loss: cross_entropy
```

## 고급: CSV 기반 Dataset

CSV 파일에서 메타데이터를 읽는 Dataset:

```python
# cvlabkit/component/dataset/csv_dataset.py
import pandas as pd
from pathlib import Path
from PIL import Image
from cvlabkit.component.base import Dataset


class CSVDataset(Dataset):
    """Dataset that loads image paths and labels from CSV.

    CSV format:
        image_path,label
        train/img1.jpg,0
        train/img2.jpg,1

    Args:
        cfg: Configuration with:
            - csv_file (str): Path to CSV file
            - root (str): Root directory for images
            - image_col (str): Column name for image paths (default: 'image_path')
            - label_col (str): Column name for labels (default: 'label')
    """

    def __init__(self, cfg):
        super().__init__()
        self.csv_file = cfg.get("csv_file")
        self.root = Path(cfg.get("root", "."))
        self.image_col = cfg.get("image_col", "image_path")
        self.label_col = cfg.get("label_col", "label")

        # Load CSV
        self.data = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.data)} samples from {self.csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get image path and label
        img_path = self.root / row[self.image_col]
        label = int(row[self.label_col])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label
```

사용 예시:

```yaml
dataset:
  train: csv_dataset(
    csv_file=data/train.csv,
    root=data/images,
    image_col=image_path,
    label_col=label
  )
```

## 고급: 다중 입력 Dataset

여러 입력을 반환하는 Dataset (예: 이미지 + 메타데이터):

```python
# cvlabkit/component/dataset/multimodal_dataset.py
import torch
from cvlabkit.component.base import Dataset


class MultimodalDataset(Dataset):
    """Dataset with image and tabular features.

    Args:
        cfg: Configuration with:
            - image_dataset: Image dataset config
            - features_file (str): Path to tabular features (CSV/NPY)
    """

    def __init__(self, cfg):
        super().__init__()

        # Load image dataset
        from cvlabkit.core.creator import Creator
        creator = Creator(cfg)
        self.image_dataset = creator.dataset.image_dataset()

        # Load tabular features
        features_file = cfg.get("features_file")
        self.features = torch.load(features_file)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Get image and label
        image, label = self.image_dataset[idx]

        # Get tabular features
        features = self.features[idx]

        # Return tuple of inputs and label
        return (image, features), label
```

Agent에서 사용:

```python
class MultimodalAgent(Agent):
    def train_step(self, batch):
        (images, features), labels = batch

        # Model takes both inputs
        outputs = self.model(images, features)

        loss = self.loss(outputs, labels)
        return loss
```

## 고급: 데이터 증강이 포함된 Dataset

Dataset 내부에서 증강을 수행:

```python
# cvlabkit/component/dataset/augmented_dataset.py
import torch
import torchvision.transforms as T
from cvlabkit.component.base import Dataset


class AugmentedDataset(Dataset):
    """Dataset with built-in strong augmentations.

    Useful for semi-supervised learning (준지도 학습) methods like FixMatch.

    Args:
        cfg: Configuration with:
            - base_dataset: Base dataset config
            - num_augments (int): Number of augmented versions per image
    """

    def __init__(self, cfg):
        super().__init__()

        # Load base dataset
        from cvlabkit.core.creator import Creator
        creator = Creator(cfg)
        self.base_dataset = creator.dataset.base_dataset()

        self.num_augments = cfg.get("num_augments", 2)

        # Strong augmentation
        self.strong_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # Create multiple augmented versions
        augmented = [self.strong_transform(image) for _ in range(self.num_augments)]

        # Return list of augmented images and label
        return augmented, label
```

## 실전 예제: 캐싱 Dataset

느린 I/O를 개선하기 위한 메모리 캐싱:

```python
# cvlabkit/component/dataset/cached_dataset.py
import torch
from tqdm import tqdm
from cvlabkit.component.base import Dataset


class CachedDataset(Dataset):
    """Dataset wrapper that caches samples in memory.

    Useful for small datasets with expensive preprocessing.

    Args:
        cfg: Configuration with:
            - base_dataset: Base dataset to cache
            - cache_all (bool): Cache all samples at init (default: False)
    """

    def __init__(self, cfg):
        super().__init__()

        # Load base dataset
        from cvlabkit.core.creator import Creator
        creator = Creator(cfg)
        self.base_dataset = creator.dataset.base_dataset()

        # Initialize cache
        self.cache = {}
        self.cache_all = cfg.get("cache_all", False)

        if self.cache_all:
            print("Caching all samples...")
            for idx in tqdm(range(len(self.base_dataset))):
                self.cache[idx] = self.base_dataset[idx]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Return from cache if available
        if idx in self.cache:
            return self.cache[idx]

        # Load and cache
        sample = self.base_dataset[idx]
        self.cache[idx] = sample

        return sample
```

사용 예시:

```yaml
dataset:
  train: cached_dataset(
    base_dataset: custom_image_dataset(root=data, split=train),
    cache_all=true
  )
```

## 디버깅 팁

**1. Dataset 크기 확인**

```python
class CustomDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        # ... build dataset ...
        print(f"Dataset size: {len(self.samples)}")
        print(f"Number of classes: {len(self.class_to_idx)}")
```

**2. 샘플 시각화**

```python
# test_dataset.py
from cvlabkit.core.config import Config
from cvlabkit.component.dataset.custom_image_dataset import CustomImageDataset
import matplotlib.pyplot as plt

cfg = Config({"root": "data/my_dataset", "split": "train"})
dataset = CustomImageDataset(cfg)

# Show first 5 samples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    image, label = dataset[i]
    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.savefig("dataset_samples.png")
```

**3. DataLoader 테스트**

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
images, labels = batch

print(f"Batch images shape: {images.shape}")
print(f"Batch labels shape: {labels.shape}")
```

## 체크리스트

- [ ] `cvlabkit/component/dataset/` 디렉토리에 파일 생성
- [ ] `Dataset` 베이스 클래스 상속
- [ ] `__init__(self, cfg)` 메서드 구현
- [ ] `__len__(self)` 메서드 구현
- [ ] `__getitem__(self, idx)` 메서드 구현
- [ ] YAML 설정에서 dataset 이름 지정
- [ ] 샘플 로딩 테스트
- [ ] DataLoader와 함께 동작 확인

## 다음 단계

- [커스텀 모델 작성하기](custom_model.md)
- [커스텀 Loss 작성하기](custom_loss.md)
- [컴포넌트 확장 가이드](../extending_components.md)
