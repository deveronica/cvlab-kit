# CVLab-Kit Documentation

문서 카탈로그 - 필요한 문서를 빠르게 찾으세요.

## 🚀 시작하기 (필수)

처음 시작하거나 배포할 때 읽어야 할 문서:

| 문서 | 설명 | 대상 |
|------|------|------|
| [배포 가이드](deployment_guide.md) | **GitHub 배포부터 서버-클라이언트 연결까지 전체 가이드** | 초보자, 배포 담당자 |
| [빠른 시작](distributed_execution_quickstart.md) | **5분 빠른 시작 가이드** | 빨리 시작하고 싶은 사람 |
| [분산 실행](distributed_execution_guide.md) | **분산 실행 상세 가이드** (API, 트러블슈팅) | 고급 사용자 |
| [실험자 가이드](user_guide_experimenter.md) | 실험자용 가이드 (웹 UI 사용법) | 실험자 |
| [설정 가이드](config_guide.md) | YAML 설정 파일 작성법 | 실험자 |

## 🏗️ 아키텍처 & 설계 (개발자용)

CVLab-Kit 구조를 이해하고 싶다면:

| 문서 | 설명 | 대상 |
|------|------|------|
| [아키텍처](architecture.md) | 전체 시스템 아키텍처 | 개발자 |
| [개발 철학](development_philosophy.md) | 설계 철학 (What vs How) | 개발자 |
| [Creator 동작 방식 상세 설명](creator_workflow.md) | Creator 동작 방식 | 개발자 |
| [컴포넌트 확장](extending_components.md) | Component 확장 방법 | Component 개발자 |
| [개발자 가이드](user_guide_developer.md) | 웹 UI 개발자 가이드 | 프론트엔드/백엔드 개발자 |
| [권장 라이브러리](additional_libraries.md) | 추가 라이브러리 정보 | 개발자 |

## 🔧 실무 가이드

성능 최적화 및 문제 해결:

| 문서 | 설명 | 사용 시점 |
|------|------|----------|
| [설정 예제](config_examples.md) | 실전 YAML 설정 예제 모음 | 설정 작성 참고 |
| [성능 튜닝 가이드](performance_tuning.md) | GPU 최적화 및 성능 향상 | 학습 속도 개선 시 |
| [문제 해결 가이드](troubleshooting.md) | 일반적인 에러 해결 방법 | 문제 발생 시 |
| [Web Helper 가이드](web_helper_guide.md) | 웹 UI 상세 사용 가이드 | 웹 기능 활용 시 |
| [Build 개념](build_concept_explained.md) | Build 개념 설명 | 빌드 이해 필요 시 |
| [문서화 가이드](documentation_guide.md) | 문서 작성 가이드 | 문서 기여 시 |
| [GitHub 가이드](github_guide.md) | GitHub 워크플로우 가이드 | Git/GitHub 작업 시 |

---

## 📖 읽는 순서 추천

### 초보자

1. [배포 가이드](deployment_guide.md) - GitHub 배포부터 시작
2. [빠른 시작](distributed_execution_quickstart.md) - 빠른 실행
3. [실험자 가이드](user_guide_experimenter.md) - 웹 UI 사용법
4. [설정 가이드](config_guide.md) - 실험 설정 작성

### 개발자

1. [아키텍처](architecture.md) - 전체 구조 이해
2. [개발 철학](development_philosophy.md) - 설계 철학
3. [Creator 동작 방식 상세 설명](creator_workflow.md) - Creator 동작
4. [컴포넌트 확장](extending_components.md) - Component 확장
5. [개발자 가이드](user_guide_developer.md) - 웹 UI 개발

### 실험자

1. [실험자 가이드](user_guide_experimenter.md) - 웹 UI 사용법
2. [설정 가이드](config_guide.md) - YAML 설정
3. 필요시 특정 기능 가이드 참조

---

## 📊 전체 문서 목록

총 17개 문서 (프로덕션 수준):

- **시작하기**: 5개
- **아키텍처**: 5개
- **실무 가이드**: 7개
- **API 참조**: mkdocs.yml 참조

모든 문서는 [MkDocs 사이트](https://deveronica.github.io/cvlab-kit)에서 확인하세요.
