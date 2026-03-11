# 설정 가이드

> YAML 기반 동적 컴포넌트 로딩 및 Grid Search 설정

`Creator`는 YAML 설정 파일을 기반으로 필요한 컴포넌트를 동적으로 로딩합니다. 사용자는 `import` 구문 없이 YAML 파일만 수정하여 실험 구성을 변경할 수 있습니다. 이를 위해 다음 설정 규칙을 따라야 합니다.

## **규칙 1: 기본 컴포넌트 로딩**

가장 간단한 형태는 `{컴포넌트_타입}: {컴포넌트_이름}` 형식입니다. `컴포넌트_이름`은 `cvlabkit/component/{컴포넌트_타입}/` 디렉토리 안에 있는 구현체의 파이썬 파일명과 일치해야 합니다.

- **`config.yaml` 예시:**
    ```yaml
    optimizer: adam
    loss: cross_entropy
    ```
- **Agent 코드:**
    ```python
    # cvlabkit/component/optimizer/adam.py 를 로드
    opt = create.optimizer()

    # cvlabkit/component/loss/cross_entropy.py 를 로드
    loss_fn = create.loss()
    ```

## **규칙 2: 컴포넌트에 파라미터 전달**

컴포넌트 구현에 특정 파라미터가 필요한 경우, 해당 파라미터를 YAML 파일의 **최상위 레벨**에 정의합니다. `Creator`는 컴포넌트를 생성할 때 전체 설정 객체(`cfg`)를 주입하므로, 컴포넌트 내부에서 `cfg.파라미터_이름` 형태로 접근할 수 있습니다.

- **`config.yaml` 예시:**
    ```yaml
    model: resnet18
    num_classes: 10  # resnet18 모델을 위한 파라미터

    lr: 1e-3         # adam 옵티마이저를 위한 파라미터
    optimizer: adam
    ```
- **`resnet18.py` 구현 예시:**
    ```python
    # cvlabkit/component/model/resnet18.py
    from cvlabkit.component.base import Model
    from torchvision.models import resnet18

    class ResNet18(Model):
        def __init__(self, cfg):
            super().__init__()
            # YAML 최상위에 정의된 num_classes 값을 가져옴
            num_classes = cfg.get("num_classes", 1000)
            self.model = resnet18(num_classes=num_classes)

        def forward(self, x):
            return self.model(x)
    ```

## **규칙 3: 한 타입의 여러 컴포넌트 사용 (고급)**

하나의 에이전트가 같은 타입의 여러 컴포넌트를 사용해야 할 경우 (예: GAN의 생성자/판별자 모델), YAML에서 논리적인 이름을 키로 사용하여 중첩 구조로 정의할 수 있습니다.

- **`config.yaml` 예시 (중첩 구조):**
    ```yaml
    model:
      generator: unet
      discriminator: patch_gan
    ```
- **Agent 코드:**
    ```python
    # cvlabkit/component/model/unet.py 를 로드
    gen_model = create.model.generator()

    # cvlabkit/component/model/patch_gan.py 를 로드
    disc_model = create.model.discriminator()
    ```

## **규칙 4: 동일 컴포넌트에 다른 옵션 전달 (신규)**

동일한 컴포넌트를 사용하지만, 인스턴스마다 다른 설정을 적용하고 싶을 때 (예: 학습용과 검증용 데이터로더에 각각 다른 설정을 적용) `이름(파라미터=값)` 형식을 사용할 수 있습니다. 이 형식은 `_type`과 `_params` 키를 가진 딕셔너리로 파싱되어 `Creator`에 의해 처리됩니다.

- **`config.yaml` 예시:**
    ```yaml
    dataloader:
      train: cifar10(split=train, shuffle=true)
      val: cifar10(split=val, shuffle=false)
    ```
- **Agent 코드:**
    ```python
    # cvlabkit/component/dataloader/cifar10.py 를 로드 (split='train', shuffle=True 설정으로)
    train_loader = create.dataloader.train()

    # cvlabkit/component/dataloader/cifar10.py 를 로드 (split='val', shuffle=False 설정으로)
    val_loader = create.dataloader.val()
    ```
- **컴포넌트 구현 예시 (`cifar10.py`):**
    ```python
    # cvlabkit/component/dataloader/cifar10.py
    from cvlabkit.component.base import DataLoader

    class CIFAR10Loader(DataLoader):
        def __init__(self, cfg):
            # 'cifar10(split=train)' 에서 파싱된 'split' 파라미터를 가져옴. 기본값은 'train'
            split = cfg.get("split", "train") 
            shuffle = cfg.get("shuffle", False)
            # ... 데이터로더 설정 ...
    ```

## **규칙 5: `|`를 이용한 컴포넌트 파이프라인 구성**

여러 컴포넌트를 순차적으로 실행하는 파이프라인을 구성할 때 `|` (파이프) 문자를 사용할 수 있습니다. 이 DSL(Domain-Specific Language) 구문은 주로 `transform`이나 `metric`과 같이 여러 단계를 조합해야 하는 경우에 유용합니다.

`Creator`는 `|`로 연결된 문자열을 파싱하여 각 컴포넌트의 인스턴스를 생성한 뒤, 이들을 `cvlabkit/component/{컴포넌트_타입}/compose.py`에 정의된 `Compose` 컴포넌트로 감싸서 최종 객체를 반환합니다.

- **`config.yaml` 예시:**
    ```yaml
    transform:
      # 여러 트랜스폼을 파이프로 연결
      strong: "resize(size=128) | adaptive_rand_augment(magnitude_min=0, magnitude_max=15) | to_tensor | normalize"

    metric:
      # 여러 메트릭을 파이프로 연결
      val: "accuracy | f1(average=macro)"
    ```

- **Agent 코드:**
    ```python
    # resize, adaptive_rand_augment, to_tensor, normalize 인스턴스가 담긴
    # Compose 객체가 생성됨
    strong_transform = create.transform.strong()

    # accuracy, f1 인스턴스가 담긴 Compose 객체가 생성됨
    val_metric = create.metric.val()
    ```

### Compose 작동 원리

`Compose` 컴포넌트는 여러 컴포넌트를 순차적으로 실행하는 컨테이너입니다. Creator가 `|`로 연결된 설정을 만나면 자동으로 각 컴포넌트를 인스턴스화한 뒤 `Compose`로 래핑합니다.

**Transform Compose 예시:**
```python
# cvlabkit/component/transform/compose.py
class Compose(Transform):
    def __init__(self, cfg: Config, components: List[Transform]):
        # PRE-INSTANTIATED transform 리스트를 받음
        self.pipeline = transforms.Compose(transforms=components)

    def __call__(self, sample, **kwargs):
        # 각 transform을 순차 실행하며 kwargs 전달
        for t in self.pipeline.transforms:
            if isinstance(t, Transform):
                sample = t(sample, **kwargs)  # 커스텀 kwargs 전달
            else:
                sample = t(sample)  # torchvision 표준 transform
        return sample
```

**Metric Compose 예시:**
```python
# cvlabkit/component/metric/compose.py
class Compose(Metric):
    def __init__(self, cfg: Config, components: List[Metric]):
        self.metrics = components  # PRE-INSTANTIATED metric 리스트

    def update(self, **kwargs):
        for metric in self.metrics:
            metric.update(**kwargs)  # 모든 metric 업데이트

    def compute(self) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            results.update(metric.compute())  # 결과 병합
        return results
```

**핵심 특징:**
- `Compose`는 이미 인스턴스화된 컴포넌트 리스트를 받음 (`components` 파라미터)
- Transform: kwargs를 통한 추가 정보 전달 지원 (예: `difficulty_score`)
- Metric: 여러 메트릭의 결과를 하나의 딕셔너리로 병합
- 각 컴포넌트 타입은 자체 `compose.py` 구현 필요 (`transform/`, `metric/`)

**실행 흐름:**
1. YAML: `transform: "resize(size=128) | to_tensor | normalize"`
2. Creator: 각 컴포넌트 인스턴스화 → `[resize_obj, to_tensor_obj, normalize_obj]`
3. Compose 생성: `Compose(cfg, components=[resize_obj, to_tensor_obj, normalize_obj])`
4. 사용: `compose_obj(image)` → resize → to_tensor → normalize 순차 실행

---

## 관련 문서

- [Creator 워크플로우](creator_workflow.md) - Creator의 동작 원리와 컴포넌트 로딩 메커니즘
- [컴포넌트 확장](extending_components.md) - 새로운 컴포넌트 추가 방법
- [Build 개념](build_concept_explained.md) - 설정 템플릿 자동 생성
- [문제 해결](troubleshooting.md) - Config 관련 일반적인 에러
