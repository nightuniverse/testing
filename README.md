# Test Excels VLM System - Streamlit Interface

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-FF6B6B?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com/)

**엑셀 파일의 이미지와 데이터를 분석하는 VLM(Vision Language Model) 시스템**

Excel 파일들을 분석하고 VLM(Vision Language Model)을 포함한 쿼리 시스템을 웹 인터페이스로 제공합니다.

## 🚀 시작하기

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Qdrant 서버 실행

Qdrant 벡터 데이터베이스가 필요합니다. Docker를 사용하여 실행하세요:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

또는 Qdrant를 직접 설치하여 실행:

```bash
# Qdrant 설치 (macOS)
brew install qdrant

# Qdrant 서버 실행
qdrant
```

### 3. Streamlit 앱 실행

```bash
streamlit run streamlit_vlm_interface.py
```

브라우저에서 `http://localhost:8501`로 접속하여 사용하세요.

## 📊 주요 기능

### 🔍 질문하기
- 자연어로 Excel 파일 내용에 대해 질문
- 수치 데이터, 이미지, 공정 정보 등 다양한 유형의 질문 지원
- 실시간 검색 결과 제공

### 🖼️ 이미지 분석
- Excel 파일에서 추출된 이미지 자동 분석
- 이미지 파일 경로 및 메타데이터 표시
- 미리보기/Finder에서 이미지 열기 기능

### 📈 데이터 현황
- 총 데이터 포인트 수
- 컬렉션 정보
- Excel 파일 목록 및 크기

### 🗂️ 파일 관리
- Excel 파일 목록 및 상세 정보
- 이미지 파일 목록
- 파일 크기 및 경로 정보

## 💡 예시 질문들

### 수치 데이터 관련
- "월별 생산량은 얼마인가요?"
- "품질 검사 합격률은 몇 퍼센트인가요?"
- "조립 공정의 소요시간은 얼마인가요?"
- "메인보드의 단가는 얼마인가요?"
- "현재고 수량은 몇 개인가요?"

### 이미지/시각적 내용
- "조립 공정도 이미지를 보여주세요"
- "품질검사표 이미지를 분석해주세요"
- "작업순서도 이미지를 보여주세요"

### 공정/작업 관련
- "조립 공정의 단계별 과정을 보여주세요"
- "품질 검사 기준은 무엇인가요?"
- "수입검사 과정에서 주의할 점은?"

### 부품/파트 관련
- "조립파트 목록과 가격을 보여주세요"
- "메인보드의 공급업체는 어디인가요?"

## 🏗️ 시스템 구조

```
test_excels_vlm_system.py    # 핵심 VLM 시스템
streamlit_vlm_interface.py   # Streamlit 웹 인터페이스
test_excels/                 # Excel 파일들
├── SM-F741U(B6) FRONT DECO SUB 조립 작업표준서_20240708(조립수정) (1).xlsx
└── 생성형 AI 연동을 위한 조립파트 관련 자료.xlsx
rag_anything_output/         # 파싱된 이미지 파일들
└── [파일명]/
    └── docling/
        └── images/
            └── *.png
```

## 🌐 배포 및 호스팅

### 빠른 배포 (Docker)

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/test-excels-vlm.git
cd test-excels-vlm

# 2. Docker 배포
./deploy.sh docker

# 3. 접속
# Streamlit: http://localhost:8501
# Qdrant: http://localhost:6333
```

### Docker Compose 배포

```bash
# Docker Compose로 배포
./deploy.sh docker-compose

# 또는 직접 실행
docker-compose up -d
```

### Streamlit Cloud 배포 (무료)

```bash
# 1. GitHub에 코드 업로드
git add .
git commit -m "Initial commit"
git push origin main

# 2. Streamlit Cloud에서 배포
# - https://share.streamlit.io 접속
# - GitHub 계정 연결
# - 저장소 선택 후 배포
```

### 클라우드 플랫폼 배포

#### AWS EC2
```bash
# AWS CLI 설정 후
./deploy.sh aws

# 수동 설정:
# 1. EC2 인스턴스 생성 (t3.medium 이상)
# 2. 보안 그룹 설정 (포트 8501, 6333)
# 3. Docker 설치 및 컨테이너 실행
```

#### Google Cloud Platform
```bash
# gcloud CLI 설정 후
./deploy.sh gcp

# 수동 설정:
# 1. Compute Engine 인스턴스 생성
# 2. 방화벽 규칙 설정
# 3. Docker 설치 및 컨테이너 실행
```

#### Azure
```bash
# Azure CLI 설정 후
./deploy.sh azure

# 수동 설정:
# 1. VM 생성
# 2. 네트워크 보안 그룹 설정
# 3. Docker 설치 및 컨테이너 실행
```

### 프로덕션 환경 설정

```bash
# Nginx와 함께 프로덕션 배포
docker-compose --profile production up -d

# SSL 인증서 설정
mkdir ssl
# 인증서 파일들을 ssl/ 디렉토리에 배치
```

## 📊 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   사용자        │    ┌─────────────────┐    │   Qdrant        │
│   (브라우저)    │◄──►│   Streamlit     │◄──►│   Vector DB     │
│                 │    │   VLM Interface │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Excel Files   │
                       │   + Images      │
                       └─────────────────┘
```

## 🔧 설정 옵션

### Qdrant 연결 설정
기본적으로 `http://localhost:6333`에 연결됩니다. 다른 주소를 사용하려면:

```python
system = TestExcelsVLMSystem(qdrant_url="http://your-qdrant-server:6333")
```

### 벡터 차원 설정
현재 384차원 벡터를 사용합니다. 변경하려면 `test_excels_vlm_system.py`의 `_init_collection` 메서드를 수정하세요.

## 🐛 문제 해결

### Qdrant 연결 오류
- Qdrant 서버가 실행 중인지 확인
- 포트 6333이 열려있는지 확인
- Docker 컨테이너가 정상 실행 중인지 확인

### 이미지 파일 없음
- `rag_anything_output` 폴더에 이미지 파일들이 있는지 확인
- Excel 파일 파싱이 완료되었는지 확인

### 메모리 부족
- 대용량 Excel 파일 처리 시 메모리 사용량 모니터링
- 필요시 파일 크기 제한 설정

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트나 기능 요청은 이슈를 통해 제출해주세요.
Pull Request도 환영합니다!
