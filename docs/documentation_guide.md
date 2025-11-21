# 문서화 가이드

> MkDocs 기반 프로젝트 문서 작성 및 관리

## 📝 문서 작성 원칙

### 1. 구조

```markdown
# 제목 (H1은 하나만)

간단한 요약 (1-2문장)

## 개요

무엇을 하는 문서인지 설명

## 빠른 시작

복붙 가능한 코드 예시

## 상세 설명

### 섹션 1
### 섹션 2

## 예제

실전 예시

## 트러블슈팅

## FAQ

## 참고 자료
```

### 2. 스타일 가이드

**DO ✅**
```markdown
## 설치 방법

다음 명령어로 설치합니다:

\`\`\`bash
uv sync
\`\`\`

설치가 완료되면 다음과 같이 실행하세요:

\`\`\`bash
uv run app.py
\`\`\`
```

**DON'T ❌**
```markdown
## 설치 방법
일단 uv sync를 실행하고, 그 다음에 app.py를 실행하면 됩니다.
근데 만약 에러가 나면...
```

### 3. 코드 블록

**언어 지정 필수**:
```markdown
\`\`\`bash
uv run app.py
\`\`\`

\`\`\`yaml
model: resnet18
\`\`\`

\`\`\`python
def example():
    pass
\`\`\`
```

### 4. 링크

**상대 링크 사용**:
```markdown
[설정 가이드](config_guide.md)  # ✅
[설정 가이드](/docs/config_guide.md)  # ❌
```

### 5. 이미지

```markdown
![Architecture](assets/architecture.png)
```

이미지는 `docs/assets/` 또는 `assets/`에 저장

---

## 🛠️ MkDocs 사용법

### 로컬 미리보기

```bash
mkdocs serve
# → http://localhost:8000 접속
```

### 빌드

```bash
mkdocs build
# → site/ 폴더에 HTML 생성
```

### GitHub Pages 배포

```bash
mkdocs gh-deploy
```

---

## 🎨 문서 유지보수

### Legacy 관리

```bash
# 6개월 이상 안 본 문서는 legacy로
mkdir -p docs/legacy
mv docs/old_file.md docs/legacy/
```

### 정기 점검

```bash
# 깨진 링크 확인
grep -r "\[.*\](.*\.md)" docs/

# 오래된 문서 찾기
find docs/ -mtime +180 -name "*.md"
```

---

## 🔗 참고 자료

- [MkDocs 공식 문서](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Technical Writing Guide](https://developers.google.com/tech-writing)
