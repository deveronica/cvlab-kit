# REST API Reference

Web Helper의 전체 REST API 명세입니다. FastAPI 기반 RESTful API로 실험 관리, 큐 제어, 메트릭 조회 등을 제공합니다.

## API 서버 설정

Swagger UI 상단의 "Servers" 드롭다운에서 서버를 선택할 수 있습니다:

- **Local development**: `http://localhost:8000` (기본값)
- **Local (127.0.0.1)**: `http://127.0.0.1:8000`
- **Custom server**: 원격 서버 테스트용 - protocol, host, port를 직접 입력

원격 서버를 테스트하려면 "Custom server"를 선택하고 각 변수를 입력하세요.

## 응답 포맷

모든 응답은 다음 표준 포맷을 따릅니다:

```json
{
  "success": true,
  "data": { /* 요청 결과 데이터 */ },
  "meta": { "message": "작업 완료 메시지" }
}
```

에러 응답:

```json
{
  "title": "Not Found",
  "status": 404,
  "detail": "상세 에러 메시지"
}
```

## 주요 엔드포인트

### Projects

**프로젝트 목록 조회**

```http
GET /api/projects
```

모든 프로젝트와 해당 프로젝트의 실험 목록을 반환합니다. 프로젝트는 생성 시간 기준 최신순으로 정렬됩니다.

응답 예시:

```json
{
  "success": true,
  "data": [
    {
      "name": "cifar10_classification",
      "runs": [
        {
          "run_name": "resnet18_baseline",
          "status": "completed",
          "started_at": "2025-11-19T10:30:00",
          "finished_at": "2025-11-19T11:45:00",
          "config_path": "config/cifar10.yaml"
        }
      ]
    }
  ]
}
```

**프로젝트 실험 상세 조회**

```http
GET /api/projects/{project_name}/experiments
```

특정 프로젝트의 모든 실험과 하이퍼파라미터, 메트릭을 조회합니다.

응답 예시:

```json
{
  "success": true,
  "data": {
    "project": "cifar10_classification",
    "experiment_count": 5,
    "experiments": [
      {
        "run_name": "resnet18_lr001",
        "status": "completed",
        "hyperparameters": {
          "lr": 0.001,
          "batch_size": 128,
          "optimizer": "adam"
        },
        "final_metrics": {
          "val_acc": 0.92,
          "val_loss": 0.245
        },
        "max_metrics": { "val_acc": 0.93 },
        "notes": "베이스라인 실험",
        "tags": ["baseline", "resnet"]
      }
    ]
  }
}
```

**프로젝트 재색인**

```http
POST /api/projects/reindex
POST /api/projects/reindex/{project_name}
```

파일시스템에서 실험 로그를 다시 스캔하여 데이터베이스에 반영합니다. `logs/` 디렉토리의 변경사항이 UI에 반영되지 않을 때 사용하세요.

### Runs

**실험 상세 정보**

```http
GET /api/runs/{project}/{run_name}
```

특정 실험의 상세 정보와 관련 파일 경로를 반환합니다.

**실험 메트릭 조회**

```http
GET /api/runs/{project}/{run_name}/metrics?downsample=100
```

CSV 파일에서 step-wise 메트릭을 읽어옵니다. `downsample` 파라미터로 데이터 포인트 수를 줄일 수 있습니다.

응답 예시:

```json
{
  "success": true,
  "data": {
    "data": [
      { "step": 1, "train_loss": 2.3, "train_acc": 0.15, "val_loss": 2.1, "val_acc": 0.18 },
      { "step": 2, "train_loss": 1.8, "train_acc": 0.35, "val_loss": 1.7, "val_acc": 0.40 }
    ],
    "total_steps": 2,
    "columns": ["step", "train_loss", "train_acc", "val_loss", "val_acc"]
  }
}
```

**설정 파일 조회**

```http
GET /api/runs/{project}/{run_name}/config
```

실험의 YAML 설정 파일 내용을 반환합니다.

**로그 파일 조회**

```http
GET /api/runs/{project}/{run_name}/logs
```

실험의 모든 로그 파일(CSV, stdout, stderr 등)을 결합하여 반환합니다.

**노트 및 태그 수정**

```http
PATCH /api/runs/{project}/{run_name}/notes
PATCH /api/runs/{project}/{run_name}/tags
```

실험에 사용자 노트나 태그를 추가/수정합니다.

요청 예시:

```json
// Notes
{
  "notes": "하이퍼파라미터 튜닝 후 성능 개선됨"
}

// Tags
{
  "tags": ["best", "baseline", "published"]
}
```

**아티팩트 관리**

```http
GET /api/runs/{project}/{run_name}/artifacts
GET /api/runs/{project}/{run_name}/artifacts/download?file_path=/path/to/file
```

실험의 체크포인트, 로그 파일 등 모든 아티팩트를 조회하거나 다운로드합니다.

### Queue

**큐 목록 조회**

```http
GET /api/queue/list?status=running&limit=50&offset=0
```

실행 대기 중이거나 실행 중인 작업 목록을 조회합니다.

파라미터:
- `status`: `pending`, `running`, `completed`, `failed`, `cancelled`
- `project`: 프로젝트 이름으로 필터링
- `limit`: 최대 반환 개수 (기본값: 50)
- `offset`: 페이지네이션 오프셋

**큐 통계**

```http
GET /api/queue/stats
```

큐 전체 통계를 반환합니다.

응답 예시:

```json
{
  "success": true,
  "data": {
    "total_jobs": 10,
    "pending": 3,
    "running": 2,
    "completed": 5,
    "failed": 0
  }
}
```

**작업 제출**

```http
POST /api/queue/add
```

새 실험을 큐에 추가합니다. YAML 설정을 인라인으로 전송합니다.

요청 예시:

```json
{
  "config": "project: my_experiment\nmodel: resnet18\noptimizer: adam(lr=0.001)\n...",
  "device": "cuda:0",
  "priority": "normal",
  "name": "resnet18_experiment",
  "project": "my_project"
}
```

**작업 제어**

```http
POST /api/queue/experiment/{experiment_uid}/cancel
POST /api/queue/job/{job_id}/pause
POST /api/queue/job/{job_id}/resume
POST /api/queue/job/{job_id}/priority
```

실행 중인 작업을 취소, 일시정지, 재개하거나 우선순위를 변경합니다.

**실시간 로그 스트리밍**

```http
WS /api/queue/job/{job_id}/ws/logs
```

WebSocket을 통해 작업의 실시간 로그를 스트리밍합니다.

사용 예시 (JavaScript):

```javascript
const ws = new WebSocket(`ws://localhost:8000/api/queue/job/${jobId}/ws/logs`);
ws.onmessage = (event) => {
  console.log(event.data);  // 로그 라인
};
```

**실험 히스토리**

```http
GET /api/queue/experiments?status=completed&project=cifar10&limit=100
```

`web_helper/queue_logs/` 디렉토리의 모든 실험 실행 기록을 조회합니다. Run 데이터와 독립적으로 관리됩니다.

### Devices

**디바이스 등록**

```http
POST /api/devices/heartbeat
```

클라이언트가 3초마다 heartbeat를 전송하여 디바이스 상태를 업데이트합니다.

요청 예시:

```json
{
  "device_id": "gpu-server-01",
  "hostname": "ml-workstation",
  "gpu_info": [
    { "id": 0, "name": "NVIDIA RTX 4090", "memory_total": 24576, "memory_used": 2048 }
  ],
  "cpu_percent": 25.3,
  "memory_percent": 45.2
}
```

### Events (SSE)

**이벤트 스트림**

```http
GET /api/events/stream
```

Server-Sent Events를 통해 실시간 업데이트를 수신합니다.

지원 이벤트:
- `device_update`: 디바이스 상태 변경
- `run_update`: 실험 상태 변경
- `queue_update`: 큐 상태 변경
- `metric_update`: 메트릭 업데이트

사용 예시 (JavaScript):

```javascript
const eventSource = new EventSource('/api/events/stream');
eventSource.addEventListener('device_update', (event) => {
  const data = JSON.parse(event.data);
  console.log('Device updated:', data);
});
```

## 에러 코드

| 코드 | 의미 | 일반적인 원인 |
|------|------|--------------|
| 400 | Bad Request | 잘못된 요청 파라미터, 유효하지 않은 YAML |
| 403 | Forbidden | 디렉토리 접근 권한 없음 |
| 404 | Not Found | 리소스(프로젝트, 실험, 파일)를 찾을 수 없음 |
| 500 | Internal Server Error | 서버 내부 오류, 파일 읽기 실패 |

## 일반적인 사용 패턴

**1. 실험 제출 및 모니터링**

```bash
# 1. 큐에 실험 추가
curl -X POST http://localhost:8000/api/queue/add \
  -H "Content-Type: application/json" \
  -d '{"config": "...", "device": "any", "priority": "normal"}'

# 2. 큐 상태 확인
curl http://localhost:8000/api/queue/stats

# 3. WebSocket으로 실시간 로그 확인
# (JavaScript/Python WebSocket 클라이언트 사용)
```

**2. 실험 결과 분석**

```bash
# 1. 프로젝트의 모든 실험 조회
curl http://localhost:8000/api/projects/cifar10/experiments

# 2. 특정 실험의 메트릭 다운샘플링 조회
curl "http://localhost:8000/api/runs/cifar10/run_001/metrics?downsample=50"

# 3. 설정 파일 확인
curl http://localhost:8000/api/runs/cifar10/run_001/config
```

**3. 파일 재색인**

```bash
# logs/ 디렉토리 변경 후 UI에 반영되지 않을 때
curl -X POST http://localhost:8000/api/projects/reindex
```

## Swagger UI

아래 Swagger UI에서 모든 엔드포인트를 직접 테스트할 수 있습니다.

<swagger-ui src="openapi.json" url-deepLinking="true" tryItOutEnabled="true"/>
