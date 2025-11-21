#!/usr/bin/env bash
# MkDocs 문서 내 링크 검증 스크립트
# 사용법: ./docs/check_links.sh

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DOCS_DIR")"

echo -e "${BLUE}=== CVLab-Kit Documentation Link Checker ===${NC}\n"

# 카운터 초기화
TOTAL_LINKS=0
BROKEN_LINKS=0
MISSING_FILES=0

# 결과 저장 배열
declare -a BROKEN_RESULTS
declare -a MISSING_RESULTS

# Markdown 파일에서 링크 추출 함수
check_markdown_file() {
    local file=$1
    local relative_path="${file#$DOCS_DIR/}"

    # [text](link.md) 형태의 마크다운 링크 추출
    while IFS= read -r line; do
        # 링크 추출 (정규표현식)
        if [[ $line =~ \]\(([^\)]+)\) ]]; then
            local link="${BASH_REMATCH[1]}"

            # URL은 건너뛰기 (http://, https://, mailto:)
            if [[ $link =~ ^https?:// ]] || [[ $link =~ ^mailto: ]]; then
                continue
            fi

            # 앵커 제거 (#section)
            local link_without_anchor="${link%%#*}"

            # 빈 링크는 건너뛰기 (앵커만 있는 경우)
            if [[ -z "$link_without_anchor" ]]; then
                continue
            fi

            TOTAL_LINKS=$((TOTAL_LINKS + 1))

            # 절대 경로 링크 처리 (/path/to/file.md)
            if [[ $link_without_anchor == /* ]]; then
                local target_file="${PROJECT_ROOT}${link_without_anchor}"
            else
                # 상대 경로 링크 처리 (../other/file.md)
                local dir="$(dirname "$file")"
                local target_file="$(cd "$dir" && realpath -m "$link_without_anchor" 2>/dev/null || echo "")"
            fi

            # 파일 존재 확인
            if [[ -n "$target_file" ]] && [[ ! -f "$target_file" ]]; then
                BROKEN_LINKS=$((BROKEN_LINKS + 1))
                BROKEN_RESULTS+=("${RED}✗${NC} $relative_path → $link")
            fi
        fi
    done < "$file"
}

# mkdocs.yml에서 참조된 파일 검증 함수
check_mkdocs_nav() {
    local mkdocs_file="$PROJECT_ROOT/mkdocs.yml"

    if [[ ! -f "$mkdocs_file" ]]; then
        echo -e "${YELLOW}⚠ mkdocs.yml not found${NC}"
        return
    fi

    echo -e "${BLUE}Checking mkdocs.yml navigation...${NC}"

    # nav 섹션에서 .md 파일 추출 (YAML 파싱 간소화)
    while IFS= read -r line; do
        if [[ $line =~ :\ ([a-zA-Z0-9_/-]+\.md) ]]; then
            local nav_file="${BASH_REMATCH[1]}"
            local full_path="$DOCS_DIR/$nav_file"

            if [[ ! -f "$full_path" ]]; then
                MISSING_FILES=$((MISSING_FILES + 1))
                MISSING_RESULTS+=("${RED}✗${NC} mkdocs.yml → $nav_file (missing)")
            fi
        fi
    done < "$mkdocs_file"
}

# 1. mkdocs.yml 검증
check_mkdocs_nav

echo -e "\n${BLUE}Checking markdown files...${NC}"

# 2. 모든 Markdown 파일 검사
while IFS= read -r md_file; do
    check_markdown_file "$md_file"
done < <(find "$DOCS_DIR" -type f -name "*.md")

# 3. 결과 출력
echo -e "\n${BLUE}=== Results ===${NC}\n"

if [[ $MISSING_FILES -gt 0 ]]; then
    echo -e "${RED}Missing files in mkdocs.yml:${NC}"
    printf '%s\n' "${MISSING_RESULTS[@]}"
    echo ""
fi

if [[ $BROKEN_LINKS -gt 0 ]]; then
    echo -e "${RED}Broken links in documents:${NC}"
    printf '%s\n' "${BROKEN_RESULTS[@]}"
    echo ""
fi

# 4. 요약
echo -e "${BLUE}Summary:${NC}"
echo -e "  Total links checked: ${TOTAL_LINKS}"

if [[ $MISSING_FILES -eq 0 ]] && [[ $BROKEN_LINKS -eq 0 ]]; then
    echo -e "  ${GREEN}✓ All links are valid!${NC}"
    exit 0
else
    echo -e "  ${RED}✗ Broken links: ${BROKEN_LINKS}${NC}"
    echo -e "  ${RED}✗ Missing files: ${MISSING_FILES}${NC}"
    exit 1
fi
