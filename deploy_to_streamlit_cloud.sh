#!/bin/bash

echo "🚀 Streamlit Cloud 배포 준비 스크립트"
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 필요한 파일들 확인
echo -e "${BLUE}📁 필요한 파일들 확인 중...${NC}"

required_files=(
    "streamlit_cloud_app.py"
    "requirements_cloud.txt"
    ".streamlit/config.toml"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file (누락)${NC}"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "${RED}❌ 누락된 파일이 있습니다. 배포를 진행할 수 없습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 모든 필요한 파일이 준비되었습니다!${NC}"

# Git 상태 확인
echo -e "\n${BLUE}🔍 Git 상태 확인 중...${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git이 설치되어 있지 않습니다.${NC}"
    echo "Git 설치: https://git-scm.com/downloads"
    exit 1
fi

# Git 저장소 초기화 여부 확인
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Git 저장소가 초기화되지 않았습니다.${NC}"
    echo -e "${BLUE}Git 저장소를 초기화하시겠습니까? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${BLUE}🔧 Git 저장소 초기화 중...${NC}"
        git init
        echo -e "${GREEN}✅ Git 저장소 초기화 완료${NC}"
    else
        echo -e "${YELLOW}⚠️  Git 저장소 초기화를 건너뜁니다.${NC}"
    fi
fi

# .gitignore 생성
echo -e "\n${BLUE}📝 .gitignore 파일 생성 중...${NC}"
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Excel files (클라우드에서는 샘플 데이터 사용)
*.xlsx
*.xls

# Output directories
rag_anything_output/
test_excels_vlm_output/
test_results/

# Docker
Dockerfile
docker-compose.yml
deploy.sh

# macOS
.DS_Store

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
EOF

echo -e "${GREEN}✅ .gitignore 파일 생성 완료${NC}"

# 배포 가이드 출력
echo -e "\n${GREEN}🎉 배포 준비 완료!${NC}"
echo -e "\n${BLUE}📋 다음 단계를 따라 Streamlit Cloud에 배포하세요:${NC}"
echo -e "\n${YELLOW}1. GitHub에서 새 저장소 생성${NC}"
echo -e "${YELLOW}2. 다음 명령어로 코드 업로드:${NC}"
echo -e "${BLUE}   git add .${NC}"
echo -e "${BLUE}   git commit -m \"Initial commit: Test Excels VLM System\"${NC}"
echo -e "${BLUE}   git branch -M main${NC}"
echo -e "${BLUE}   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git${NC}"
echo -e "${BLUE}   git push -u origin main${NC}"
echo -e "\n${YELLOW}3. [Streamlit Cloud](https://share.streamlit.io/) 접속${NC}"
echo -e "${YELLOW}4. GitHub 계정으로 로그인${NC}"
echo -e "${YELLOW}5. \"New app\" 클릭${NC}"
echo -e "${YELLOW}6. 저장소 선택 후 Main file path: streamlit_cloud_app.py 입력${NC}"
echo -e "${YELLOW}7. \"Deploy!\" 클릭${NC}"
echo -e "\n${GREEN}🎯 자세한 가이드는 STREAMLIT_CLOUD_DEPLOY.md 파일을 참조하세요!${NC}"

echo -e "\n${BLUE}💡 팁:${NC}"
echo -e "${YELLOW}• requirements_cloud.txt 파일이 requirements.txt보다 우선됩니다${NC}"
echo -e "${YELLOW}• .streamlit/config.toml 파일로 테마와 설정을 커스터마이징할 수 있습니다${NC}"
echo -e "${YELLOW}• 배포 후 자동으로 HTTPS가 적용됩니다${NC}"
