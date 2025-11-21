# 자주 묻는 질문 (FAQ)

CVLab-Kit 사용 중 자주 묻는 질문들을 모았습니다.

## 설치 및 환경 설정

**Q:** Python 버전은 무엇을 사용해야 하나요?**

**A:** Python 3.8 이상이 필요합니다. Python 3.10 이상을 권장합니다.

```bash
python --version  # Python 3.10.x 이상 권장
```

**Q:** GPU가 없어도 사용할 수 있나요?

**A:** 네, CPU만으로도 실험을 실행할 수 있습니다. 설정 파일에서 `device: cpu`로 지정하세요.

```yaml
device: cpu
```

단, 학습 속도는 GPU 대비 느립니다. 작은 모델이나 데이터셋으로 테스트하는 것을 권장합니다.

**Q:** CUDA out of memory 에러가 발생합니다

**A:** GPU 메모리 부족 문제입니다. 다음 방법을 시도하세요:

1. **배치 크기 줄이기**:
```yaml
batch_size: 64  # 기존 128에서 줄임
```

2. **모델 크기 줄이기**:
```yaml
model: resnet18  # resnet50 대신
```

3. **Gradient accumulation 사용**:
```yaml
batch_size: 32
gradient_accumulation_steps: 4  # 실질적으로 128과 동일
```

자세한 내용은 [성능 튜닝 가이드](performance_tuning.md#gpu-메모리-최적화)를 참고하세요.

## 실험 실행

**Q:** 실험이 시작되지 않습니다

**A:** 다음 사항을 확인하세요:

1. **설정 파일 경로 확인**:
```bash
ls config/your_config.yaml  # 파일이 존재하는지 확인
```

2. **YAML 문법 오류 확인**:
```bash
uv run main.py --config config/your_config.yaml  # 에러 메시지 확인
```

3. **필수 필드 누락 확인**:
   - `project`, `dataset`, `model`, `optimizer`, `loss`는 필수입니다

**Q:** 여러 실험을 동시에 실행하려면?

**A:** Web Helper의 Queue 기능을 사용하세요:

1. Web UI에서 **Execute** 탭으로 이동
2. **Add to Queue** 버튼으로 실험 추가
3. 여러 실험을 큐에 추가하면 자동으로 순차 실행

또는 분산 실행 모드를 사용하면 여러 GPU에 병렬 실행됩니다:
```bash
uv run app.py --dev
# Execute 탭에서 실험 추가
```

자세한 내용은 [분산 실행 가이드](distributed_execution_guide.md)를 참고하세요.

**Q:** Grid search는 어떻게 하나요?

**A:** YAML 설정에서 리스트를 사용하면 자동으로 grid search가 실행됩니다:

```yaml
lr: [0.001, 0.01, 0.1]       # 3개 값
batch_size: [64, 128]        # 2개 값
# → 총 6개 실험 자동 생성
```

각 조합이 별도 실험으로 실행되며, Web UI의 Projects 탭에서 결과를 비교할 수 있습니다.

## Web Helper 사용

**Q:** Web UI에 접속할 수 없습니다

**A:** 다음을 확인하세요:

1. **서버가 실행 중인지 확인**:
```bash
uv run app.py --dev  # 개발 모드
# 또는
uv run app.py        # 프로덕션 모드
```

2. **포트 확인**:
   - 개발 모드: `http://localhost:5173` (프론트엔드)
   - 프로덕션 모드: `http://localhost:8000`

3. **방화벽 확인**: 로컬 포트가 차단되어 있지 않은지 확인

**Q:** 실험 결과가 Web UI에 나타나지 않습니다

**A:** 다음을 시도하세요:

1. **Reindex 실행**:
   - Projects 탭에서 **Reindex** 버튼 클릭
   - 또는 API 호출: `POST /api/projects/reindex`

2. **로그 파일 확인**:
```bash
ls logs/your_project/  # CSV, YAML 파일이 있는지 확인
```

3. **파일 권한 확인**: Web Helper가 `logs/` 디렉토리를 읽을 수 있는지 확인

**Q:** 원격 서버에서 실행 중인데 접속이 안됩니다

**A:** 서버를 `0.0.0.0`으로 바인딩하세요:

```bash
uv run app.py --host 0.0.0.0 --port 8000
```

그리고 클라이언트에서 서버 IP로 접속:
```
http://your-server-ip:8000
```

방화벽에서 8000 포트를 열어야 할 수도 있습니다.

## 설정 파일

**Q:** 설정 파일 작성이 어렵습니다

**A:** 예제 설정 파일을 참고하세요:

```bash
ls config/          # 제공된 예제 확인
cp config/cifar10_baseline.yaml config/my_experiment.yaml
# my_experiment.yaml을 수정하여 사용
```

또는 [설정 예제](config_examples.md) 문서를 참고하세요.

**Q:** 컴포넌트에 파라미터를 전달하려면?

**A:** 괄호 안에 파라미터를 지정합니다:

```yaml
# 파라미터 없음
model: resnet18

# 파라미터 있음
model: resnet18(num_classes=10, pretrained=true)

# 여러 파라미터
optimizer: adam(lr=0.001, weight_decay=0.0001)
```

**Q:** Placeholder는 어떻게 사용하나요?

**A:** `{{변수명}}` 형식으로 사용합니다:

```yaml
# 전역 변수 정의
num_classes: 10

# Placeholder로 참조
model: resnet18(num_classes={{num_classes}})
loss: cross_entropy(num_classes={{num_classes}})
```

## 커스텀 컴포넌트

**Q:** 커스텀 모델을 추가하려면?

**A:** `cvlabkit/component/model/` 디렉토리에 파일을 추가하세요:

```python
# cvlabkit/component/model/my_model.py
from cvlabkit.component.base.model import Model

class MyModel(Model):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.get("num_classes", 10)
        # 모델 구현

    def forward(self, x):
        # Forward pass 구현
        pass
```

그리고 YAML에서 사용:
```yaml
model: my_model(num_classes=10)
```

자세한 내용은 [컴포넌트 확장 가이드](extending_components.md)를 참고하세요.

**Q:** 기존 PyTorch 모델을 사용할 수 있나요?

**A:** 네, Model 베이스 클래스를 상속받아 래핑하면 됩니다:

```python
from cvlabkit.component.base.model import Model
import torchvision.models as models

class PretrainedResNet(Model):
    def __init__(self, cfg):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, cfg.num_classes)

    def forward(self, x):
        return self.model(x)
```

## 분산 실행

**Q:** 다중 GPU를 사용하려면?

**A:** 설정 파일에서 GPU 리스트를 지정하세요:

```yaml
device: [0, 1, 2, 3]  # 4개 GPU 사용
```

PyTorch DistributedDataParallel이 자동으로 활성화됩니다.

**Q:** 원격 GPU 서버를 클라이언트로 연결하려면?

**A:** 서버와 클라이언트를 분리 실행합니다:

```bash
# 서버 (중앙 관리)
uv run app.py --server-only

# 클라이언트 (GPU 서버)
uv run app.py --client-only --url http://server-ip:8000
```

자세한 내용은 [분산 실행 가이드](distributed_execution_guide.md#4-클라이언트-연결)를 참고하세요.

## 문제 해결

**Q:** ImportError가 발생합니다

**A:** 의존성을 다시 설치하세요:

```bash
uv sync
# 또는
pip install -e .
```

특정 패키지가 누락된 경우:
```bash
uv add package-name
```

**Q:** 학습이 매우 느립니다

**A:** 다음을 확인하세요:

1. **GPU 사용 확인**:
```python
import torch
print(torch.cuda.is_available())  # True여야 함
```

2. **DataLoader 워커 수 조정**:
```yaml
num_workers: 4  # CPU 코어의 1/4 정도
```

3. **Mixed Precision 사용**:
```yaml
use_amp: true  # Automatic Mixed Precision
```

자세한 내용은 [성능 튜닝 가이드](performance_tuning.md)를 참고하세요.

**Q:** 실험이 중간에 멈춥니다

**A:** Web UI의 Queue에서 실험 상태를 확인하고 재시작하세요.

## 더 많은 도움이 필요하신가요?

- [문제 해결 가이드](troubleshooting.md) - 상세한 에러 해결 방법
- [GitHub Issues](https://github.com/deveronica/cvlab-kit/issues) - 버그 리포트 및 기능 요청
- [문서 카탈로그](README.md) - 전체 문서 목록
