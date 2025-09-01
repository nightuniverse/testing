# Test Excels VLM System - Streamlit Cloud Edition

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)

**엑셀 파일의 이미지와 데이터를 분석하는 VLM(Vision Language Model) 시스템 - Streamlit Cloud 배포 버전**

Excel 파일들을 분석하고 VLM(Vision Language Model)을 포함한 쿼리 시스템을 웹 인터페이스로 제공합니다. 이 버전은 Streamlit Cloud에서 24/7 접속 가능하도록 최적화되었습니다.

## 🚀 빠른 시작

### Streamlit Cloud에서 바로 실행
1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. GitHub 계정으로 로그인
3. 이 저장소 선택
4. Main file path: `streamlit_cloud_app.py` 입력
5. "Deploy!" 클릭

### 로컬에서 실행
```bash
# 1. 저장소 클론
git clone https://github.com/nightuniverse/testing.git
cd testing

# 2. 의존성 설치
pip install -r requirements_cloud.txt

# 3. Streamlit 앱 실행
streamlit run streamlit_cloud_app.py
```

## 📊 주요 기능

### 🔍 자연어 질문
- "월별 생산량은 얼마인가요?"
- "품질검사 합격률은 몇 퍼센트인가요?"
- "조립 공정의 소요시간은 얼마인가요?"

### 📈 데이터 시각화
- 월별 생산량 차트
- 품질검사 합격률 그래프
- 조립 공정 소요시간 분석

### 🖼️ 이미지 분석
- 품질검사표 이미지
- 조립공정도 이미지
- 부품도면 이미지

### 💰 비즈니스 인사이트
- 메인보드 단가 정보
- 현재고 수량 현황
- 공정별 소요시간 분석

## 🌐 24/7 접속 가능

이 시스템은 Streamlit Cloud에 배포되어 있어서:
- ✅ **언제든지 접속 가능** - 컴퓨터를 끄고 있어도 접속 가능
- ✅ **모든 기기에서 접속** - PC, 태블릿, 스마트폰
- ✅ **자동 HTTPS 보안** - 안전한 연결
- ✅ **무료 호스팅** - 기본 기능은 무료

## 📁 파일 구조

```
streamlit_cloud_app.py          # 메인 애플리케이션
requirements_cloud.txt          # 클라우드용 의존성
.streamlit/config.toml          # Streamlit 설정
STREAMLIT_CLOUD_DEPLOY.md       # 배포 가이드
deploy_to_streamlit_cloud.sh    # 배포 스크립트
```

## 🔧 설정

### 테마 커스터마이징
`.streamlit/config.toml` 파일에서 테마를 변경할 수 있습니다:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### 의존성 관리
`requirements_cloud.txt`에서 필요한 패키지들을 관리합니다:

```
streamlit>=1.28.0
pandas>=1.5.0
Pillow>=9.0.0
openpyxl>=3.0.0
xlrd>=2.0.0
```

## 🚀 배포 방법

### Streamlit Cloud 배포 (추천)
1. GitHub에 코드 업로드
2. [Streamlit Cloud](https://share.streamlit.io/) 접속
3. 저장소 선택 후 배포

### 자동 배포 스크립트 사용
```bash
./deploy_to_streamlit_cloud.sh
```

## 📊 샘플 데이터

이 버전은 실제 Excel 파일 대신 샘플 데이터를 사용합니다:

- **월별 생산량**: 1월~12월 생산량 데이터
- **품질검사 합격률**: 분기별 합격률 통계
- **조립 공정**: 단계별 소요시간
- **단가 정보**: 메인보드 단가
- **재고 현황**: 현재고 수량

## 🎯 사용 예시

### 1. 생산량 확인
질문: "월별 생산량은 얼마인가요?"
- 월별 생산량 차트 표시
- 총 연간 생산량 계산
- 상세 데이터 테이블 제공

### 2. 품질 관리
질문: "품질검사 합격률은 몇 퍼센트인가요?"
- 분기별 합격률 그래프
- 평균 합격률 계산
- 품질검사표 이미지 제공

### 3. 공정 분석
질문: "조립 공정의 소요시간은 얼마인가요?"
- 공정별 소요시간 차트
- 총 조립 시간 계산
- 조립공정도 이미지 제공

## 🔄 업데이트

코드를 수정한 후 자동으로 재배포됩니다:

```bash
git add .
git commit -m "Update: 기능 개선"
git push origin main
```

## 🐛 문제 해결

### 일반적인 오류들
1. **Import Error**: `requirements_cloud.txt`에 패키지 추가
2. **File Not Found**: 상대 경로 사용 확인
3. **Memory Error**: 데이터 처리 최적화

### 로그 확인
Streamlit Cloud 대시보드에서 "View app logs" 클릭하여 오류 로그 확인

## 📞 지원

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Streamlit Community**: 커뮤니티 포럼
- **Documentation**: [STREAMLIT_CLOUD_DEPLOY.md](STREAMLIT_CLOUD_DEPLOY.md)

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트나 기능 요청은 이슈를 통해 제출해주세요.
Pull Request도 환영합니다!

---

**🎉 이제 24/7 접속 가능한 VLM 시스템을 즐겨보세요!**
