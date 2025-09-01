# 🚀 Streamlit Cloud 배포 가이드

## 📋 개요
이 가이드는 Test Excels VLM System을 Streamlit Cloud에 배포하여 24/7 접속 가능하게 만드는 방법을 설명합니다.

## ✨ Streamlit Cloud의 장점
- 🌐 **24/7 접속 가능**: 컴퓨터를 끄고 있어도 접속 가능
- 💰 **무료 호스팅**: 기본 기능은 무료
- 🔄 **자동 배포**: GitHub 연동으로 코드 변경 시 자동 업데이트
- 📱 **모바일 친화적**: 모든 기기에서 접속 가능
- 🔒 **보안**: HTTPS 자동 적용

## 🛠️ 배포 준비

### 1. GitHub 계정 준비
- GitHub 계정이 필요합니다 (없다면 [가입하기](https://github.com/join))

### 2. 코드 업로드
```bash
# 1. GitHub에서 새 저장소 생성
# 2. 로컬에서 Git 초기화
git init
git add .
git commit -m "Initial commit: Test Excels VLM System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 🚀 Streamlit Cloud 배포 단계

### 1단계: Streamlit Cloud 접속
1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. GitHub 계정으로 로그인

### 2단계: 새 앱 생성
1. **"New app"** 버튼 클릭
2. **Repository**: 방금 생성한 GitHub 저장소 선택
3. **Branch**: `main` 선택
4. **Main file path**: `streamlit_cloud_app.py` 입력

### 3단계: 고급 설정 (선택사항)
- **App URL**: 원하는 URL 설정 (예: `test-excels-vlm`)
- **Requirements file**: `requirements_cloud.txt` 선택

### 4단계: 배포
1. **"Deploy!"** 버튼 클릭
2. 배포 완료까지 2-3분 대기

## 📁 배포에 필요한 파일들

### 핵심 파일
- ✅ `streamlit_cloud_app.py` - 메인 애플리케이션
- ✅ `requirements_cloud.txt` - 의존성 패키지
- ✅ `.streamlit/config.toml` - Streamlit 설정

### 제외할 파일들
- ❌ `Dockerfile` - Docker 관련 (불필요)
- ❌ `docker-compose.yml` - Docker 관련 (불필요)
- ❌ `*.xlsx` - Excel 파일 (클라우드에서는 샘플 데이터 사용)
- ❌ `__pycache__/` - Python 캐시 파일

## 🔧 배포 후 설정

### 1. 앱 URL 확인
배포 완료 후 다음과 같은 URL이 생성됩니다:
```
https://your-app-name-your-username.streamlit.app
```

### 2. 공개 설정
- **Public**: 모든 사람이 접속 가능
- **Private**: GitHub 팀원만 접속 가능

### 3. 자동 배포 확인
GitHub에 코드를 푸시하면 자동으로 재배포됩니다.

## 🧪 테스트

### 배포된 앱 테스트
1. 생성된 URL로 접속
2. "시스템 재초기화" 버튼 클릭
3. 예시 질문들 테스트:
   - "월별 생산량은 얼마인가요?"
   - "품질검사 합격률은 몇 퍼센트인가요?"
   - "조립 공정도 이미지를 보여주세요"

## 🚨 문제 해결

### 일반적인 오류들

#### 1. Import Error
```
ModuleNotFoundError: No module named 'xxx'
```
**해결**: `requirements_cloud.txt`에 해당 패키지 추가

#### 2. File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory
```
**해결**: 클라우드 환경에서는 파일 경로가 다를 수 있음, 상대 경로 사용

#### 3. Memory Error
```
MemoryError: Unable to allocate array
```
**해결**: 대용량 파일 처리 코드 최적화

### 로그 확인
Streamlit Cloud 대시보드에서 **"View app logs"** 클릭하여 오류 로그 확인

## 🔄 업데이트 방법

### 코드 수정 후 재배포
```bash
git add .
git commit -m "Update: 기능 개선"
git push origin main
```
→ Streamlit Cloud에서 자동으로 재배포

## 📊 모니터링

### Streamlit Cloud 대시보드에서 확인 가능
- 📈 **사용자 접속 통계**
- ⚡ **성능 메트릭**
- 🚨 **오류 로그**
- 🔄 **배포 상태**

## 💡 팁과 트릭

### 1. 성능 최적화
- 대용량 데이터는 샘플링하여 사용
- 이미지는 압축하여 로딩 속도 향상
- 캐싱 기능 활용

### 2. 보안 고려사항
- 민감한 정보는 환경변수로 관리
- API 키 등은 Streamlit Cloud의 Secrets 기능 사용

### 3. 사용자 경험
- 로딩 스피너 추가
- 에러 메시지 친화적으로 작성
- 반응형 디자인 적용

## 🎯 다음 단계

배포 완료 후:
1. **사용자 테스트**: 팀원들과 함께 테스트
2. **피드백 수집**: 개선점 파악
3. **기능 확장**: 추가 기능 개발
4. **성능 모니터링**: 사용량 및 성능 추적

## 📞 지원

문제가 발생하면:
1. **Streamlit Cloud 로그** 확인
2. **GitHub Issues** 등록
3. **Streamlit Community** 포럼 활용

---

**🎉 축하합니다! 이제 24/7 접속 가능한 VLM 시스템이 완성되었습니다!**
