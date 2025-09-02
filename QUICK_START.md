# 🚀 빠른 시작 가이드

## 1. 시스템 실행하기

### 방법 1: 자동 실행 스크립트 사용 (권장)
```bash
./run_streamlit.sh
```

### 방법 2: 수동 실행
```bash
# 1. Qdrant 서버 실행
docker run -d -p 6333:6333 --name qdrant-server qdrant/qdrant

# 2. Python 패키지 설치
pip install -r requirements.txt

# 3. Streamlit 앱 실행
streamlit run streamlit_vlm_interface.py
```

## 2. 브라우저에서 접속
- URL: http://localhost:8501
- 웹 인터페이스가 자동으로 열립니다

## 3. 사용법

### 🔍 질문하기
1. 텍스트 영역에 질문을 입력
2. "🔍 검색" 버튼 클릭
3. 결과 확인

### 📋 예시 질문들
- "월별 생산량은 얼마인가요?"
- "조립 공정의 소요시간은 얼마인가요?"
- "메인보드의 단가는 얼마인가요?"
- "조립 공정도 이미지를 보여주세요"
- "현재고 현황 데이터를 보여주세요"

### 📊 데이터 현황 확인
- "📊 데이터 현황" 버튼 클릭
- 총 데이터 포인트 수, 컬렉션 정보 확인

### 🗂️ 파일 목록 확인
- "🗂️ 파일 목록" 버튼 클릭
- Excel 파일 및 이미지 파일 목록 확인

## 4. 문제 해결

### Qdrant 연결 오류
```bash
# Qdrant 컨테이너 재시작
docker restart qdrant-server
```

### Streamlit 앱 오류
```bash
# 포트 확인
lsof -i :8501

# 다른 포트로 실행
streamlit run streamlit_vlm_interface.py --server.port 8502
```

### 시스템 테스트
```bash
python test_streamlit_system.py
```

## 5. 시스템 종료
- Streamlit 앱: Ctrl+C
- Qdrant 서버: `docker stop qdrant-server`

---

**🎉 이제 Excel 파일들을 분석하고 질문할 수 있습니다!**
