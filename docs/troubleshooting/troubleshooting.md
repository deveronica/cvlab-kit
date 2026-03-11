# 문제 해결 가이드

CVLab-Kit 사용 중 발생할 수 있는 일반적인 문제와 해결 방법을 정리했습니다.

## 설정 관련 문제

### Component not found 에러

**증상**:
```
Error: Component 'resnet50' not found in cvlabkit/component/model/
```

**원인**:

- YAML 설정 파일에 존재하지 않는 컴포넌트 이름 입력
- 오타 또는 대소문자 불일치
- 컴포넌트 파일이 실제로 없음

**해결 방법**:

**1. 컴포넌트 이름 확인**:
```bash
ls cvlabkit/component/model/
```

**2. 대소문자 정확히 일치하는지 확인**

**3. 사용 가능한 컴포넌트 목록은 `/api/components/list` 엔드포인트에서 확인**

### Config 검증 실패

**증상**:
```
ValidationError: Required key 'num_classes' not found
```

**원인**:

- 필수 파라미터 누락
- YAML 문법 오류

**해결 방법**:

**1. 에러 메시지에서 누락된 키 확인**

**2. Dry-run 모드로 검증**:
```bash
uv run main.py --config config/my_config.yaml
```

**3. 자동 생성된 템플릿(`generated_basic.yaml`) 참고**

---

## GPU 관련 문제

### GPU 메모리 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**원인**:

- 배치 크기가 너무 큼
- 모델이 너무 큼
- 이전 프로세스가 GPU 메모리 점유

**해결 방법**:

**1. 배치 크기 줄이기**:
```yaml
batch_size: 32
```

**2. GPU 메모리 초기화**:

GPU 사용 중인 프로세스 확인:
```bash
nvidia-smi
```

해당 프로세스 종료:
```bash
kill <PID>
```

**3. 혼합 정밀도 학습 사용**:
```yaml
mixed_precision: true
```

### CUDA device mismatch

**증상**:
```
RuntimeError: Expected all tensors to be on the same device
```

**원인**:

- 일부 텐서가 CPU에 있고 일부는 GPU에 있음
- 멀티 GPU 환경에서 디바이스 불일치

**해결 방법**:

**1. 모든 입력 데이터를 같은 디바이스로**:
```python
inputs = inputs.to(device)
targets = targets.to(device)
```

**2. 특정 GPU 지정**:
```bash
CUDA_VISIBLE_DEVICES=0 uv run main.py --config ...
```

---

## 분산 실행 문제

### 서버-클라이언트 연결 실패

**증상**:
```
ConnectionError: Failed to connect to server at http://server:8000
```

**원인**:

- 서버가 실행 중이 아님
- 방화벽 또는 네트워크 문제
- 잘못된 서버 주소

**해결 방법**:

**1. 서버 실행 확인** (서버에서):
```bash
uv run app.py --server-only
```

**2. 네트워크 연결 테스트**:
```bash
curl http://server-ip:8000/api/devices
```

**3. 방화벽 포트 열기**:
```bash
sudo ufw allow 8000
```

### Heartbeat 타임아웃

**증상**:

- 클라이언트가 오프라인으로 표시됨
- "Device not responding" 경고

**원인**:

- 클라이언트가 종료됨
- 네트워크 불안정
- Heartbeat 간격이 너무 길음

**해결 방법**:

**1. 클라이언트 재시작**:
```bash
uv run app.py --client-only --url http://server:8000
```

**2. Heartbeat 간격 조정** (app.py):
```python
HEARTBEAT_INTERVAL = 3
```

---

## 데이터 로딩 문제

### 데이터셋을 찾을 수 없음

**증상**:
```
FileNotFoundError: Dataset not found at ./data/mstar
```

**원인**:

- 데이터셋 경로가 잘못됨
- 데이터셋이 다운로드되지 않음

**해결 방법**:

**1. 데이터셋 경로 확인**:
```yaml
dataset:
  train: mstar(root="./data/mstar", split="train")
```

**2. 절대 경로 사용**:
```yaml
dataset:
  train: mstar(root="/absolute/path/to/data")
```

### DataLoader 느림

**증상**:

- 학습이 매우 느림
- GPU 사용률이 낮음

**원인**:

- `num_workers=0`으로 설정됨
- 디스크 I/O 병목

**해결 방법**:

**1. `num_workers` 증가**:
```yaml
dataloader:
  train: basic(batch_size=64, num_workers=4)
```

**2. 데이터를 SSD로 이동**

**3. 데이터 증강 최소화**

---

## Import 에러

### ModuleNotFoundError

**증상**:
```
ModuleNotFoundError: No module named 'cvlabkit'
```

**원인**:

- 패키지가 설치되지 않음
- Python 환경 문제

**해결 방법**:

**1. 의존성 재설치**:
```bash
uv sync
```

**2. 가상환경 활성화 확인** (uv 환경인지 확인):
```bash
which python
```

---

## 로그/결과 관련 문제

### 로그 파일이 생성되지 않음

**증상**:

- `logs/` 디렉토리가 비어 있음
- Web UI에서 실험이 표시되지 않음

**원인**:

- 로그 디렉토리 권한 문제
- `log_dir` 설정 오류

**해결 방법**:

1. 로그 디렉토리 권한 확인:
   ```bash
   ls -la logs/
   chmod 755 logs/
   ```
2. 설정에서 `log_dir` 확인:
   ```yaml
   log_dir: ./logs
   run_name: my_experiment
   ```

### Metrics가 기록되지 않음

**증상**:

- CSV 파일에 metrics 열이 비어 있음
- Web UI에서 차트가 표시되지 않음

**원인**:

- Metric 컴포넌트가 설정되지 않음
- `log_metrics()` 호출 누락

**해결 방법**:

1. Metric 설정 추가:
   ```yaml
   metric:
     accuracy: accuracy()
   ```
2. Agent에서 metric 로깅 확인

---

## 자주 묻는 질문 (FAQ)

### Q: Grid search가 너무 많은 실험을 생성합니다

**A**: 리스트 항목 수를 줄이거나, 특정 조합만 실험하도록 설정하세요:
```yaml
## Before: 3 x 3 x 2 = 18 experiments
lr: [0.001, 0.01, 0.1]
batch_size: [32, 64, 128]
optimizer: [adam, sgd]

## After: 3 experiments only
lr: [0.001, 0.01, 0.1]
batch_size: 64
optimizer: adam
```

### Q: 실험이 큐에서 실행되지 않습니다

**A**: 다음을 확인하세요:
1. 사용 가능한 GPU가 있는지 (Devices 탭)
2. 다른 실험이 실행 중인지 (Queue 탭)
3. Config 파일이 유효한지

### Q: Web UI가 업데이트되지 않습니다

**A**: 브라우저를 새로고침하거나 SSE 연결을 확인하세요:
```bash
curl http://localhost:8000/events/stream
```

### Q: 체크포인트를 어디에서 찾을 수 있나요?

**A**: 기본 경로는 `logs/<project>/<run_name>/checkpoints/`입니다.

---

## 추가 도움말

### 디버그 모드 활성화

상세한 로그를 보려면:
```bash
uv run main.py --config config.yaml --verbose
```

### 로그 파일 확인

에러 발생 시 로그 파일 확인:
```bash
tail -f logs/<project>/<run_name>/train.log
```

### 이슈 보고

문제가 해결되지 않으면 GitHub Issues에 보고하세요:
- Repository: https://github.com/deveronica/cvlab-kit/issues
- 에러 메시지와 재현 단계를 포함해주세요

---

## 관련 문서

- [설정 가이드](config_guide.md) - YAML 설정 방법
- [배포 가이드](deployment_guide.md) - 서버 설정
- [사용자 가이드 (실험자)](user_guide_experimenter.md) - 기본 사용법
- [사용자 가이드 (개발자)](user_guide_developer.md) - 컴포넌트 개발
