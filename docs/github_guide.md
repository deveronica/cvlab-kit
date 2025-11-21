# GitHub 가이드

> Git 워크플로우와 협업 규칙

## 🎯 핵심 원칙

### ✅ GitHub에 올려야 하는 것 (소스 코드)

**"다른 사람이 프로젝트를 실행하기 위해 필요한 것"**

- 📝 **코드 파일**: `.py`, `.js`, `.tsx`, `.ts`
- 📋 **설정 파일**: `pyproject.toml`, `package.json`, `*.yaml`
- 📚 **문서**: `README.md`, `docs/*.md`
- 🔧 **의존성 목록**: `uv.lock`, `package-lock.json`
- 📜 **라이선스**: `LICENSE`

**크기**: 보통 수십 MB 이하

---

### ❌ GitHub에 올리면 안 되는 것 (생성 파일)

**"실행하면 자동으로 만들어지는 것"**

| 폴더/파일 | 크기 | 이유 | 어떻게 생성? |
|----------|------|------|-------------|
| `checkpoints/` | 32GB | 실험 체크포인트 | 실험 실행 시 자동 생성 |
| `outputs/` | 1.8GB | 실험 결과 | 실험 실행 시 자동 생성 |
| `web_helper*.zip` | 1GB+ | 백업 파일 | 수동으로 만든 압축 파일 |
| `node_modules/` | 101MB | npm 패키지 | `npm install` 시 자동 |
| `web_helper/frontend/dist/` | ~10MB | 빌드 결과물 | `npm run build` 시 자동 |
| `data/` | 722MB | 데이터셋 | 별도 다운로드 |
| `logs/` | 변동 | 실험 로그 | 실험 실행 시 자동 |
| `.trash/` | 변동 | 휴지통 | 삭제한 파일들 |
| `reports/` | 1MB | 임시 리포트 | 분석 스크립트 실행 시 |
| `tests/` | 380KB | 테스트 출력 | 테스트 실행 시 |

**크기**: 총 34GB+ (거의 모두 불필요)

---

## 📊 현재 프로젝트 분석

### 전체 크기: ~34GB

| 항목 | 크기 | GitHub에? |
|------|------|----------|
| **소스 코드** (cvlabkit, web_helper 코드) | ~50MB | ✅ 올려야 함 |
| **문서** (docs, README) | ~1MB | ✅ 올려야 함 |
| **설정 파일** (pyproject.toml, configs) | ~1MB | ✅ 올려야 함 |
| **의존성** (uv.lock, package-lock.json) | ~1MB | ✅ 올려야 함 |
| |||
| **checkpoints** (실험 체크포인트) | 32GB | ❌ 제외 |
| **outputs** (실험 결과) | 1.8GB | ❌ 제외 |
| **web_helper 압축 파일들** | 1GB+ | ❌ 제외 |
| **node_modules** (npm 패키지) | 101MB | ❌ 제외 |
| **data** (데이터셋) | 722MB | ❌ 제외 |
| **기타** (.trash, logs, tests 등) | ~100MB | ❌ 제외 |

**GitHub에 올릴 크기**: ~50-100MB ✅
**제외할 크기**: ~34GB ❌

---

## 🛠️ .gitignore 수정

현재 `.gitignore`에 **누락된 것**들을 추가해야 합니다:

```bash
## outputs 폴더 추가
outputs/

## 압축 파일 제외
*.zip

## 프론트엔드 빌드 결과
web_helper/frontend/dist/
```

### 수정된 .gitignore

```gitignore
## ... 기존 내용 ...

## Project Directory
checkpoints/
data/
logs/
tests/
wandb/
legacy/
results/
.trash/
tools/
.playwright-mcp/
temp_configs/
outputs/           # ← 추가

## Documentation
reports/
test-results/

## Backup files
*.zip              # ← 추가
*.tar.gz

## Web Helper
web_helper/web_helper.db
web_helper/frontend/dist/    # ← 추가
web_helper/frontend/node_modules/  # ← 추가 (명확히)
```

---

## 📝 각 항목 설명

### dist/ (빌드 결과물)

**무엇?**: 프론트엔드 빌드 결과 (HTML, CSS, JS 번들)

```bash
web_helper/frontend/dist/
├── index.html
├── assets/
│   ├── index-abc123.js
│   └── index-def456.css
```

**왜 제외?**: `npm run build` 실행하면 자동 생성됨

**언제 필요?**: 서버 배포 시 (하지만 GitHub Actions에서 자동 빌드 가능)

---

### node_modules/ (npm 패키지)

**무엇?**: npm 패키지들 (React, TypeScript 등)

```bash
node_modules/
├── react/
├── react-dom/
├── typescript/
└── ... (수천 개 폴더)
```

**왜 제외?**: `npm install` 실행하면 자동 설치됨

**크기**: 100MB ~ 수백 MB (프로젝트마다 다름)

**대신 올리는 것**: `package.json` + `package-lock.json`

---

### .trash/ (휴지통)

**무엇?**: 삭제한 파일들의 임시 저장소

**왜 있나?**: 실수로 삭제한 파일 복구용

**왜 제외?**: 개인 개발 환경의 임시 파일

**해결**: 진짜 삭제해도 되는 파일이면 완전히 삭제
```bash
rm -rf .trash
```

---

### reports/ (리포트)

**무엇?**: 실험 분석 리포트, 스크립트 출력

**왜 제외?**:

- 자동 생성됨
- 개인 분석 결과 (팀과 공유 불필요)
- 코드 실행하면 다시 만들 수 있음

**예외**: 최종 논문용 리포트는 `docs/` 폴더에 따로 저장

---

### tests/ (테스트 출력)

**무엇?**: pytest 결과, 스크린샷, 테스트 아티팩트

```bash
tests/
├── __pycache__/
├── .pytest_cache/
└── screenshots/
```

**왜 제외?**: `pytest` 실행하면 자동 생성

**혼동 주의**:

- `tests/` 폴더 자체 ❌ (제외)
- `tests/*.py` 테스트 코드 ✅ (올려야 함)

→ `.gitignore`에 `tests/`가 있지만, 테스트 **코드**는 올려야 합니다!

**해결**:
```gitignore
## 잘못된 방법
tests/  # 테스트 코드까지 제외됨

## 올바른 방법
tests/__pycache__/
tests/.pytest_cache/
tests/screenshots/
tests/test-results/
```

---

### outputs/ (실험 결과)

**무엇?**: 실험 분석 결과, 그래프, 통계

**크기**: 1.8GB

**왜 제외?**:

- 실험 실행하면 재생성 가능
- 크기가 큼
- 개인 실험 결과

**예외**: 중요한 결과는 별도로 정리해서 문서화

---

### checkpoints/ (체크포인트)

**무엇?**: 학습된 모델 가중치 (.pt, .pth 파일)

**크기**: 32GB (!)

**왜 제외?**:

- GitHub 파일 크기 제한 (100MB/파일, 1GB/저장소 권장)
- 학습하면 재생성 가능

**대신 사용**:

- Hugging Face Hub
- Google Drive
- AWS S3
- wandb Artifacts

---

### *.zip (압축 파일)

**무엇?**: 백업용 압축 파일 (`web_helper.zip`, `web_helper 2.zip` 등)

**왜 있나?**: 수동으로 백업한 것

**왜 제외?**:

- Git이 이미 버전 관리 중
- 중복 백업 불필요
- 크기가 큼

**해결**:
```bash
## 로컬 백업 폴더로 이동
mkdir -p ~/Backups/cvlab-kit
mv web_helper*.zip ~/Backups/cvlab-kit/

## 또는 삭제
rm web_helper*.zip
```

---

## ✅ GitHub에 올리기 전 체크리스트

### 1. .gitignore 업데이트

```bash
## .gitignore에 추가
cat >> .gitignore <<EOF

## Additional exclusions
outputs/
*.zip
*.tar.gz
web_helper/frontend/dist/
EOF
```

### 2. Git 캐시 정리

이미 Git에 추적된 파일 제거:

```bash
## 추적 중인 불필요한 파일 제거 (파일은 유지)
git rm -r --cached outputs/
git rm -r --cached web_helper/frontend/dist/
git rm --cached *.zip

## 커밋
git add .gitignore
git commit -m "Update .gitignore: exclude outputs, dist, zip files"
```

### 3. 크기 확인

```bash
## GitHub에 올릴 파일 크기 확인
git ls-files | xargs du -ch | tail -1

## 100MB 이하여야 함 (권장)
## 1GB 이하여야 함 (최대)
```

### 4. 큰 파일 찾기

```bash
## 100MB 이상 파일 찾기
find . -type f -size +100M -not -path "./.git/*"

## Git에 추적 중인 큰 파일
git ls-files | xargs du -h | sort -hr | head -20
```

---

## 🚀 GitHub 푸시 순서

### Step 1: .gitignore 업데이트

```bash
## outputs/ 추가
echo "outputs/" >> .gitignore

## *.zip 추가
echo "*.zip" >> .gitignore

## dist/ 추가
echo "web_helper/frontend/dist/" >> .gitignore
```

### Step 2: 불필요한 파일 제거

```bash
## Git 추적 제거 (파일은 유지)
git rm -r --cached outputs/ 2>/dev/null
git rm --cached *.zip 2>/dev/null
git rm -r --cached web_helper/frontend/dist/ 2>/dev/null
```

### Step 3: 커밋 & 푸시

```bash
## 변경사항 확인
git status

## 모두 추가
git add .

## 커밋
git commit -m "chore: Update .gitignore and remove large files"

## 푸시
git push origin main
```

---

## 📦 큰 파일 관리 (선택)

### Git LFS (Large File Storage)

체크포인트 같은 큰 파일을 올려야 한다면:

```bash
## Git LFS 설치
brew install git-lfs  # macOS
## 또는 https://git-lfs.com/

## 초기화
git lfs install

## .pt 파일을 LFS로 관리
git lfs track "*.pt"
git lfs track "*.pth"

## .gitattributes 커밋
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

**주의**: GitHub LFS는 유료 (1GB 무료, 이후 $5/50GB)

---

## 🎓 요약

### GitHub에 올리는 것

```
소스 코드 (50MB)
├── cvlabkit/          # Python 코드
├── web_helper/        # 백엔드/프론트엔드 코드
├── docs/              # 문서
├── config/            # 설정 파일
├── pyproject.toml     # Python 의존성
├── package.json       # Node.js 의존성
├── uv.lock           # 의존성 잠금 파일
├── README.md         # 프로젝트 설명
└── .gitignore        # Git 제외 목록
```

### GitHub에 올리지 않는 것

```
생성 파일 (34GB)
├── checkpoints/       # 모델 가중치 (32GB)
├── outputs/          # 실험 결과 (1.8GB)
├── *.zip             # 백업 파일 (1GB)
├── node_modules/     # npm 패키지 (101MB)
├── data/             # 데이터셋 (722MB)
├── dist/             # 빌드 결과 (10MB)
├── logs/             # 로그
├── .trash/           # 휴지통
└── reports/          # 임시 리포트
```

---

## 🤔 자주 묻는 질문

### Q: tests/ 폴더는 왜 .gitignore에 있나요?

**A**: 혼동입니다. 테스트 **코드**는 올려야 하지만, 테스트 **출력**은 제외해야 합니다.

**해결**:
```gitignore
## 잘못됨
tests/

## 올바름
tests/__pycache__/
tests/.pytest_cache/
tests/test-results/
```

### Q: dist/ 폴더를 올려야 하지 않나요?

**A**: 서버에서 배포 시 `npm run build`로 자동 빌드할 수 있으므로 불필요합니다.

**예외**: GitHub Pages 배포 시에는 dist를 커밋해야 할 수도 있지만, 보통 GitHub Actions로 자동화합니다.

### Q: 실험 결과를 팀과 공유하려면?

**A**: Git 대신 다음 사용:
- 📊 wandb (실험 추적)
- 📁 Google Drive / Dropbox
- 🤗 Hugging Face Hub (모델)
- 📝 Notion / Confluence (분석 리포트)

### Q: 데이터셋은 어떻게 공유하나요?

**A**:

1. 공개 데이터셋: README에 다운로드 링크 + 스크립트
2. 비공개 데이터셋: 별도 저장소 (S3, Drive)

```bash
## download_data.sh
wget https://example.com/dataset.zip
unzip dataset.zip -d data/
```

---

## 🎯 다음 단계

1. ✅ `.gitignore` 업데이트
2. ✅ 불필요한 파일 제거
3. ✅ GitHub에 푸시
4. 📚 README.md 작성
5. 🚀 GitHub Pages 배포 (선택)

**예상 GitHub 저장소 크기**: 50-100MB ✅
