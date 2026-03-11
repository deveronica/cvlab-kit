# CVLab-Kit 사용자 문서

CVLab-Kit 사용을 위한 가이드 문서입니다.

> **개발 문서**는 [PRD/](../PRD/) 폴더를 참고하세요.

---

## 문서 구조

```
docs/
├── 시작하기/           # 설치 및 첫 실험
│   ├── 설치.md
│   ├── 첫-실험.md
│   └── 분산-실행.md
├── 가이드/             # 상세 사용법
│   ├── 설정-문법.md
│   ├── 설정-예제.md
│   ├── 컴포넌트-사용.md
│   ├── 웹-UI-사용.md
│   └── 성능-튜닝.md
├── 배포/               # 서버 배포
│   └── 배포-가이드.md
├── 참조/               # 레퍼런스
│   └── FAQ.md
├── 문제해결/           # 트러블슈팅
│   └── 트러블슈팅.md
├── api/                # API 문서
├── tutorials/          # 튜토리얼
└── legacy/             # 구 문서
```

---

## 읽기 순서

### 처음 시작하는 경우

1. [시작하기/설치.md](./시작하기/설치.md) - 환경 설정
2. [시작하기/첫-실험.md](./시작하기/첫-실험.md) - 첫 실험 실행
3. [가이드/설정-문법.md](./가이드/설정-문법.md) - YAML 작성법

### 분산 환경을 구축하는 경우

1. [시작하기/분산-실행.md](./시작하기/분산-실행.md) - 분산 실행 설정
2. [배포/배포-가이드.md](./배포/배포-가이드.md) - 서버 배포

### 문제가 발생한 경우

1. [문제해결/트러블슈팅.md](./문제해결/트러블슈팅.md)
2. [참조/FAQ.md](./참조/FAQ.md)

---

## 관련 문서

### 개발 문서 (PRD/)

CVLab-Kit 내부 아키텍처와 설계 결정을 이해하려면:

- [PRD/L0-vision/](../PRD/L0-vision/) - 프로젝트 존재 이유
- [PRD/L1-goals/](../PRD/L1-goals/) - 버전별 목표
- [PRD/L2-architecture/](../PRD/L2-architecture/) - 시스템 구조
- [PRD/L3-spec/](../PRD/L3-spec/) - 상세 명세
- [PRD/L4-decisions/](../PRD/L4-decisions/) - 아키텍처 결정 기록

### Node System 명세 (Builder 관련)

- [PRD/L3-spec/node-system/포트-시스템.md](../PRD/L3-spec/node-system/포트-시스템.md) - 포트 스키마, Handle ID
- [PRD/L3-spec/node-system/엣지-플로우-타입.md](../PRD/L3-spec/node-system/엣지-플로우-타입.md) - FlowType, CodeFlowEdge
- [PRD/L3-spec/node-system/데이터-스키마.md](../PRD/L3-spec/node-system/데이터-스키마.md) - 전체 데이터 모델
- [PRD/L3-spec/node-system/AST-노드-변환.md](../PRD/L3-spec/node-system/AST-노드-변환.md) - 파서 파이프라인

### AI 개발 가이드

- [CLAUDE.md](../CLAUDE.md) - Claude Code 개발 규칙
