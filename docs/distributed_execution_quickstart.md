# 빠른 시작

> 5분 안에 분산 실행 환경 구축하기

## 🚀 빠른 시작

### 서버 설정 (1분)

**웹 서버 시작**:
```bash
cd /path/to/cvlab-kit
uv run app.py --dev
```

**브라우저에서 확인**:
```bash
open http://localhost:8000
```

### Middleend 설정 (2분)

**Middleend 실행** (하트비트 + 작업 실행 + 로그 동기화):
```bash
uv run app.py --client-only --url http://lab-server:8000
```

**Daemon 모드** (SSH 세션 독립):
```bash
uv run app.py --client-only --url http://lab-server:8000 --daemon
```

### 실험 실행 (2분)

1. 웹 UI **Devices** 탭에서 GPU 서버 확인 (status: healthy)
2. **Execute** 탭에서 실험 추가:
   - Config Path: `config/my_experiment.yaml`
   - Project: `my_project`
   - **Add to Queue** 클릭
3. **Projects** 탭에서 실시간 로그 확인

완료! 🎉

---

## 🔧 트러블슈팅

### Q: 디바이스가 "disconnected" 상태로 표시됨

**Middleend에서 로그 확인**:

Daemon 모드:
```bash
tail -f logs/middleend.log
```

일반 모드 - 터미널 출력 확인

**네트워크 확인**:
```bash
ping lab-server
curl http://lab-server:8000/api/devices
```

### Q: 작업이 디스패치되지 않음

- Devices 탭에서 status가 "healthy"인지 확인 (3초 내 하트비트 필요)
- 이미 다른 작업이 실행 중인지 확인 (디바이스당 1개 작업만)

### Q: 로그가 동기화되지 않음

**Middleend 상태 확인**:
```bash
uv run app.py --status  # Daemon 모드
ps aux | grep "app.py"  # 일반 모드
```

**Middleend 재시작**:
```bash
uv run app.py --stop  # Daemon 중지
uv run app.py --client-only --url http://server:8000 --daemon  # 재시작
```

---

## 📚 더 알아보기

- [전체 가이드](distributed_execution_guide.md) - 상세 설정, API, 고급 기능
- [Architecture](architecture.md) - 프로젝트 아키텍처
- [User Guide](user_guide_experimenter.md) - 실험자 가이드

---

## 관련 문서

- [분산 실행 가이드 (전체)](distributed_execution_guide.md)
- [배포 가이드](deployment_guide.md)
