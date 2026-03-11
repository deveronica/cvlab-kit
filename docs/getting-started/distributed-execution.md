# 분산 실행 가이드

여러 GPU 서버에서 실험을 자동 분배하고 실시간으로 로그를 동기화하는 방법을 안내합니다.

## 아키텍처 개요

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│      Backend (Server)           │      │      Middleend (GPU 서버)        │
│  - FastAPI + SQLite             │◄────►│  - Device Agent (하트비트)       │
│  - Config/로그 저장              │      │  - Log Synchronizer             │
│  - 웹 UI                        │      │  - 작업 실행                     │
└─────────────────────────────────┘      └─────────────────────────────────┘
```

### 주요 기능

- **실시간 하트비트**: 3초마다 GPU 상태 전송
- **자동 작업 디스패치**: 유휴 GPU에 자동 배정
- **증분 로그 동기화**: CSV는 delta만, YAML/PT는 전체 전송
- **재연결 복구**: 네트워크 단절 후 자동 재개

---

## 빠른 시작 (5분)

### 1. 서버 시작 (1분)

```bash
cd /path/to/cvlab-kit
uv run app.py --dev
```

브라우저에서 `http://localhost:8000` 확인

### 2. Middleend 실행 (2분)

GPU 서버에서:

```bash
uv run app.py --client-only --url http://lab-server:8000
```

**Daemon 모드** (SSH 세션 독립):
```bash
uv run app.py --client-only --url http://lab-server:8000 --daemon
```

### 3. 실험 실행 (2분)

1. 웹 UI **Devices** 탭에서 GPU 서버 확인 (status: healthy)
2. **Execute** 탭에서 실험 추가
3. **Projects** 탭에서 실시간 로그 확인

---

## 서버 설정

### 웹 서버 실행

```bash
uv run app.py --dev     # 개발 모드 (HMR)
uv run app.py           # 프로덕션 모드
```

### 방화벽 설정 (선택사항)

**Linux**:
```bash
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

**macOS**: `/etc/pf.conf`에 추가
```
pass in proto tcp from any to any port 8000
```

---

## Middleend 설정

### 기본 실행

```bash
uv run app.py --client-only --url http://lab-server:8000
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--url` | 서버 URL | 필수 |
| `--client-host-id` | 커스텀 호스트 ID | hostname |
| `--client-interval` | 하트비트 주기(초) | 3 |
| `--poll-interval` | 작업 폴링 주기(초) | 5 |
| `--daemon` | 백그라운드 실행 | false |

### Daemon 모드

```bash
# 시작
uv run app.py --client-only --url http://server:8000 --daemon

# 상태 확인
uv run app.py --status

# 중지
uv run app.py --stop

# 로그 확인
tail -f logs/middleend.log
```

### systemd 서비스 (Linux)

```bash
sudo tee /etc/systemd/system/cvlab-middleend.service > /dev/null <<EOF
[Unit]
Description=CVLab-Kit Middleend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(which uv) run app.py --client-only --url http://server:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable cvlab-middleend
sudo systemctl start cvlab-middleend
```

---

## 사용 방법

### 디바이스 확인

웹 UI **Devices** 탭:

| 상태 | 의미 |
|------|------|
| healthy | 정상 (3초 내 하트비트) |
| stale | 연결 불안정 |
| disconnected | 오프라인 |

### 실험 실행

1. **Execute** 탭에서 Config Path, Project 입력
2. **Add to Queue** 클릭
3. Queue Manager가 유휴 GPU에 자동 배정

### 로그 확인

**서버 측**:
```
logs/{project}/{run_name}.csv|yaml|pt
```

**클라이언트 측**:
```
logs_{server}/{project}/{run_name}.csv|yaml|pt
```

---

## 트러블슈팅

### 디바이스가 "disconnected" 상태

```bash
# 네트워크 확인
ping lab-server
curl http://lab-server:8000/api/devices

# Middleend 로그 확인
tail -f logs/middleend.log

# 재시작
uv run app.py --stop
uv run app.py --client-only --url http://server:8000 --daemon
```

### 작업이 디스패치되지 않음

- Devices 탭에서 status가 "healthy"인지 확인
- 이미 다른 작업이 실행 중인지 확인 (디바이스당 1개)

### 로그가 동기화되지 않음

```bash
# 동기화 상태 확인
cat logs_{server}/.sync_state.json

# 수동 업로드
curl -X POST http://server:8000/api/sync/full/{uid}/{file} \
     -F "file=@logs_{server}/project/run.csv"
```

---

## API 참조

### 동기화 엔드포인트

| 엔드포인트 | 용도 |
|-----------|------|
| `POST /api/sync/delta/{uid}/{file}` | CSV 증분 업로드 |
| `POST /api/sync/full/{uid}/{file}` | 파일 전체 업로드 |
| `GET /api/sync/status/{uid}` | 동기화 상태 질의 |

---

## FAQ

**Q: 여러 GPU를 가진 서버에서 병렬 실행 가능?**
A: 현재는 디바이스당 1개 작업만 실행됩니다.

**Q: 서버/클라이언트 버전이 달라도 되나요?**
A: 권장하지 않습니다. 호환성 문제가 발생할 수 있습니다.

**Q: 한 서버에 여러 Agent 실행 가능?**
A: 네, 각각 다른 `--client-host-id`를 지정하세요.

---

## 다음 단계

- [웹 UI 사용법](../가이드/웹-UI-사용.md)
- [배포 가이드](../배포/배포-가이드.md)
