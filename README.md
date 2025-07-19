# CVLab-Kit

PyTorch 기반 단순하고 확장 가능한 프로토타이핑 프레임워크


## Overview

이 프로젝트는 PyTorch 기반의 경량 모듈형 실험 프레임워크로, **에이전트(Agent) 중심**으로 컴포넌트(Component)들을 조합하여, 손쉽게 확장하고 실험을 반복할 수 있도록 설계된 프레임워크입니다. 빠른 아이디어 검증 및 모델 구조 실험이 가능한 **프로토타이핑 환경**을 제공하며, YAML 설정 파일과 Creator 클래스를 통해 실험 설정 및 학습 과정을 간결하게 통합하여 실험 환경을 구성하고 관리할 수 있습니다.


## Key Feature

| 기능 | 설명 |
|------|------|
| **Agent‑Centric Workflow** | 학습, 검증, 평가 루프를 에이전트 중심으로 관리 |
| **Dynamic Component Factory** | 모델·옵티마이저·데이터셋 등의 구성 요소를 컴포넌트 단위로 <br>`create.<component>.<key>()` 동적으로 로딩 |
| **Dry‑run Validation** | 학습 전에 구성이 올바른지 검증하여, 학습 도중 중단 문제를 사전에 방지 |
| **Grid Search** | YAML에 구성된 다중 값들이 자동으로 실험 조합으로 확장되어 반복 실험 |
| **Zero‑Boilerplate** | 신규 컴포넌트 구현체는 컴포넌트 추상 클래스의 상속을 통해 일관되게 관리 |


## Installation

### 1. uv 설치

```bash
pip install uv
```

### 2. 프로젝트 클론

```bash
git clone https://github.com/deveronica/cvlab-kit.git && cd cvlab-kit
```

### 3. 의존성 설치

```bash
uv sync
```

> **uv**는 Poetry·pip‑tools와 비슷한 UX를 제공하면서도 의존성 해석과 빌드를 Rust로 가속화한 도구입니다.

## Quick Start

### 1. Dry-run or Generate Template

- ~~구성이 완전하지 않으면 Dry-run 과정에서 자동으로 `templates/generated.yaml` 생성~~(미구현)
- `python3 config/generate_template.py`를 통해, `config/templates` 폴더에 `generated_basic.yaml` 파일 생성

### 2. Write YAML Configuration

YAML 설정 파일을 작성합니다.

### 3. Run Experiment

설정을 검증하고 실험을 진행합니다. (현재 generate_template.py를 통한 수동 생성 후 --fast 옵션으로 진행)

```python
uv run main.py --config config/example.yaml --fast
```

## Documentation

더 자세한 내용은 다음 문서를 참조하세요:

*   [아키텍처 개요](docs/architecture.md)
*   [개발 철학](docs/development_philosophy.md)
*   [설정 가이드](docs/config_guide.md)
*   [컴포넌트 확장](docs/extending_components.md)
*   [추가 라이브러리](docs/additional_libraries.md)
