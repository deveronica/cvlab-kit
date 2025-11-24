# Component Sync System

CVLab-Kit의 Agent와 Component를 버전 관리하고 Worker 간 동기화하는 시스템입니다.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Image (Base)                     │
│  cvlabkit/core/          - 항상 포함 (DO NOT MODIFY)         │
│  cvlabkit/component/base/ - 항상 포함                        │
└─────────────────────────────────────────────────────────────┘
                              ↓ 동적 동기화
┌─────────────────────────────────────────────────────────────┐
│                   Component Store (Server DB)                │
│  content-addressable storage (xxhash3 기반)                  │
│  파일 삭제 없음, 추가만 (데이터 유실 방지)                    │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### Data Model

```python
# Component 버전 (content-addressable storage)
ComponentVersion:
    hash: str           # PK, xxhash3 of content
    path: str           # "agent/classification.py"
    category: str       # "agent", "model", "transform", etc.
    name: str           # "classification", "resnet", etc.
    content: str        # 코드 내용
    is_active: bool     # 현재 활성 버전 여부
    created_at: datetime

# 실험별 사용 버전 (재현성)
ExperimentComponentManifest:
    experiment_uid: str
    component_path: str
    component_hash: str
    created_at: datetime
```

### API Endpoints

```
# Component 관리
GET  /api/components/versions                    # 전체 목록
GET  /api/components/versions/{category}         # 카테고리별 목록
GET  /api/components/versions/{category}/{name}  # 버전 히스토리
GET  /api/components/versions/hash/{hash}        # 코드 조회
POST /api/components/versions/upload             # 새 버전 업로드
POST /api/components/versions/activate           # 활성화 (롤백)
POST /api/components/versions/scan               # 로컬 스캔 후 등록

# 재현성
GET  /api/components/manifest/{experiment_uid}   # 실험에 사용된 버전
POST /api/components/manifest/{experiment_uid}   # 버전 저장
```

## Usage

### 1. 로컬 Component 등록

서버 시작 후 최초 1회 실행:

```bash
# API 호출
curl -X POST http://localhost:8000/api/components/versions/scan

# 또는 Web UI에서
Components 탭 → "Scan Local" 버튼 클릭
```

### 2. Worker에서 Component 동기화

실험 실행 시 자동으로 동기화됩니다:

```python
# DeviceAgent._execute_job() 내부에서 자동 실행
sync_result = self.component_manager.sync_from_version_store(config_path)
```

동기화 과정:
1. Config YAML에서 필요한 agent/component 파싱
2. 로컬 해시와 서버 활성 버전 해시 비교
3. 변경된 파일만 다운로드
4. 실험 manifest 저장 (재현성)

### 3. 버전 롤백

Web UI에서:
1. Components 탭 이동
2. 원하는 component 선택
3. Version History에서 원하는 버전의 "Activate" 클릭

API로:
```bash
curl -X POST http://localhost:8000/api/components/versions/activate \
  -H "Content-Type: application/json" \
  -d '{"hash": "a1b2c3d4e5f6..."}'
```

### 4. 실험 재현

과거 실험에서 사용한 component 버전으로 복원:

```python
from web_helper.middleend.component_manager import ComponentManager

manager = ComponentManager(server_url="http://localhost:8000")
result = manager.restore_from_manifest("experiment_uid_here")
print(f"Restored: {result['synced']}")
```

## File Structure

```
web_helper/
├─ backend/
│  ├─ models/component.py        # DB 모델
│  ├─ api/components.py          # API 엔드포인트
│  └─ services/component_store.py # Store 서비스
├─ middleend/
│  ├─ component_manager.py       # Worker 동기화 클라이언트
│  └─ device_agent.py            # Component sync 통합
└─ frontend/src/components/views/
   └─ ComponentsView.tsx         # Web UI
```

## Version Control Strategy

1. **Content-addressable**: 동일 해시 = 동일 파일, 덮어쓰기 불가
2. **Append-only**: component store에서 삭제 없음
3. **Manifest 보존**: 모든 실험의 component 버전 기록
4. **복원 가능**: 언제든 과거 버전으로 롤백 가능

## Sync Strategy

Server와 Worker 간 코드 버전이 다를 때의 동기화 전략입니다.

### Conflict Types

| 타입 | 설명 | 기본 동작 |
|------|------|----------|
| `server_only` | 서버에만 존재 (로컬 없음) | 서버에서 다운로드 |
| `server_newer` | 서버가 더 최신 (로컬은 구버전) | 서버에서 다운로드 |
| `local_newer` | 로컬이 서버에 없는 새 버전 | **사용자 선택 필요** |
| `local_only` | 로컬에만 존재 (서버 없음) | 서버에 업로드 |

### Strategy Options

```python
from web_helper.middleend.component_manager import ComponentManager, SyncStrategy

manager = ComponentManager(
    server_url="http://localhost:8000",
    sync_strategy=SyncStrategy.INTERACTIVE,  # 기본값
)
```

| 전략 | 설명 |
|------|------|
| `SERVER_AUTHORITATIVE` | 항상 서버 버전 사용 (로컬 변경 덮어씀) |
| `LOCAL_PRIORITY` | 로컬 변경사항 서버에 업로드 |
| `INTERACTIVE` | 충돌 시 사용자에게 알림 (기본값) |
| `DRY_RUN` | 변경사항만 보고, 실제 동기화 안 함 |

### Sync Decision Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Sync Decision Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  로컬 파일 존재?                                              │
│      ├─ No → 서버에서 다운로드                                │
│      └─ Yes → 해시 비교                                      │
│                 ├─ 동일 → Skip (이미 최신)                    │
│                 └─ 다름 → 서버에 해시 존재?                   │
│                            ├─ Yes (inactive) → 서버 다운로드  │
│                            └─ No → 로컬이 새 버전!            │
│                                    ├─ [auto] 업로드+활성화    │
│                                    ├─ [ask] 사용자 선택       │
│                                    └─ [skip] 로컬 유지        │
└─────────────────────────────────────────────────────────────┘
```

### Interactive Mode with Notifications

`INTERACTIVE` 전략 사용 시, 충돌이 발생하면:

1. Web UI 우측 상단 알림 아이콘에 표시
2. 알림 패널에서 각 충돌에 대해 선택 가능:
   - **Upload to Server**: 로컬 버전을 서버에 업로드
   - **Download from Server**: 서버 버전으로 덮어쓰기
   - **Skip**: 동기화하지 않고 로컬 유지

미응답 알림은 자동으로 저장되어 다음 세션에서도 유지됩니다.

### Usage Examples

#### 1. 로컬 개발 보호 (Interactive)

```python
# 기본 설정: 로컬 변경사항 감지 시 사용자에게 알림
manager = ComponentManager(
    server_url="http://localhost:8000",
    sync_strategy=SyncStrategy.INTERACTIVE,
)
result = manager.sync_with_strategy(config_path)

# 충돌이 있으면 result.conflicts에 저장됨
if result.conflicts:
    print(f"Pending conflicts: {len(result.conflicts)}")
```

#### 2. 자동 업로드 (Local Priority)

```python
# 로컬 변경사항 자동으로 서버에 업로드
manager = ComponentManager(
    server_url="http://localhost:8000",
    sync_strategy=SyncStrategy.LOCAL_PRIORITY,
)
result = manager.sync_with_strategy(config_path)
print(f"Uploaded: {result.uploaded}")
```

#### 3. 강제 서버 동기화 (Server Authoritative)

```python
# 항상 서버 버전 사용 (CI/CD 환경에 적합)
manager = ComponentManager(
    server_url="http://localhost:8000",
    sync_strategy=SyncStrategy.SERVER_AUTHORITATIVE,
)
result = manager.sync_with_strategy(config_path)
```

#### 4. Dry Run (변경사항 확인)

```python
# 실제 동기화 없이 변경사항만 확인
manager = ComponentManager(
    server_url="http://localhost:8000",
    sync_strategy=SyncStrategy.DRY_RUN,
)
result = manager.sync_with_strategy(config_path)
print(f"Would sync: {result.skipped}")  # 모든 항목이 skipped로 표시됨
```

## Valid Categories

- `agent` - 실험 orchestration
- `model` - 신경망 모델
- `dataset` - 데이터셋 로더
- `dataloader` - 배치 로더
- `transform` - 데이터 변환
- `optimizer` - 최적화 알고리즘
- `loss` - 손실 함수
- `metric` - 평가 지표
- `scheduler` - 학습률 스케줄러
- `sampler` - 데이터 샘플러
- `solver` - 솔버
- `checkpoint` - 체크포인트 관리
- `logger` - 로깅

## Security

- Content hash 검증으로 전송 중 변조 방지
- 파일 삭제 없음으로 데이터 유실 방지
- API Key 인증 지원 (CVLABKIT_API_KEY 환경변수)
