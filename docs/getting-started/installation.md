# 설치 가이드

CVLab-Kit 설치 및 환경 설정 방법을 안내합니다.

## 사전 준비

- Python 3.8 이상
- CUDA 지원 GPU (선택사항)
- 10GB 이상의 디스크 공간

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/deveronica/cvlab-kit.git
cd cvlab-kit
```

### 2. 의존성 설치

**uv 사용** (권장):
```bash
uv sync
```

**pip 사용**:
```bash
pip install -e .
```

### 3. 설치 확인

```bash
uv run python -c "import cvlabkit; print('설치 완료!')"
```

## GPU 설정

### CUDA 버전 확인

```bash
nvidia-smi
```

### PyTorch CUDA 지원 확인

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 디렉토리 구조

설치 후 디렉토리 구조:

```
cvlab-kit/
├── cvlabkit/           # 핵심 프레임워크
│   ├── core/           # Creator, Config, Agent
│   ├── component/      # Model, Loss, Optimizer 등
│   └── agent/          # 실험 Agent들
├── web_helper/         # Web UI
├── config/             # 설정 파일 예제
├── logs/               # 실험 결과 (자동 생성)
└── main.py             # 실험 진입점
```

## 다음 단계

- [첫 실험 실행하기](./첫-실험.md)
- [분산 실행 설정](./분산-실행.md)
