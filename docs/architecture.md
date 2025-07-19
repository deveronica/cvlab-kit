# 아키텍처 개요

이 섹션에서는 CVLab-Kit 프레임워크의 내부 작동 방식과 아키텍처에 대해 자세히 설명합니다.

## 작동 방식

1. **진입점 (`main.py`)**: 사용자가 `python main.py --config config/main.yaml`과 같이 실행하면, `main.py`는 지정된 설정 파일을 읽어옵니다.

2. **설정 파싱 (`cvlabkit/core/config.py`)**: `Config` 클래스는 YAML 파일을 읽고 파싱하여 파이썬 객체처럼 점(.)으로 접근할 수 있는 형태(예: `config.model.name`)로 변환합니다.

3. **생성자 인스턴스화 (`cvlabkit/core/creator.py`)**: `Creator` 클래스는 `Config` 객체를 받아 필요한 모든 "부품"들을 동적으로 생성하는 팩토리 역할을 합니다. 일반적으로 `create` 변수에 할당되어, `create.model()`, `create.optimizer()` 등의 메서드를 사용하여 컴포넌트를 생성할 수 있습니다.
    - 예를 들어, `config.model` 섹션을 기반으로 `cvlabkit/component/model` 폴더에서 `FasterRCNN`과 같은 모델 클래스를 찾아 인스턴스화합니다.
    - 옵티마이저, 손실 함수, 데이터셋 등 다른 모든 컴포넌트도 동일한 방식으로 생성됩니다.

4. **에이전트 실행 (`cvlabkit/agent/`)**: `Creator`는 설정 파일에 지정된 에이전트(예: `BasicAgent`)를 인스턴스화합니다. 이 에이전트는 생성된 모델, 데이터 로더, 손실 함수 및 기타 컴포넌트를 결합하여 학습, 평가, 테스트의 전체 프로세스를 조정하고 실행하는 주요 엔티티입니다.
    - `BasicAgent`: 일반적인 학습/평가 파이프라인을 처리합니다.

5. **모듈형 컴포넌트 (`cvlabkit/component/`)**: 이 프로젝트의 핵심 기능은 각 기능이 독립적인 "부품"(컴포넌트)으로 분리되어 있다는 점입니다.
    - `model`: Faster R-CNN과 같은 딥러닝 모델.
    - `loss`: Cross-Entropy, Focal Loss와 같은 손실 함수.
    - `dataset`, `dataloader`: VOC, Cityscapes와 같은 데이터셋 및 데이터를 공급하는 로더.
    - `optimizer`, `scheduler`: Adam, SGD와 같은 최적화 도구 및 학습률 스케줄러.

이러한 구조는 확장을 매우 용이하게 합니다. 새로운 컴포넌트 구현을 추가하려면 해당 폴더에 새 Python 파일만 추가하고 YAML 파일에 이름을 지정하면 됩니다.