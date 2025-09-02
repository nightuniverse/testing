#!/usr/bin/env python3
"""
Test Streamlit VLM System
"""

import time
import requests
from test_excels_vlm_system import TestExcelsVLMSystem

def test_qdrant_connection():
    """Qdrant 연결 테스트"""
    print("🔍 Qdrant 연결 테스트 중...")
    try:
        system = TestExcelsVLMSystem()
        collections = system.qdrant_client.get_collections()
        print(f"✅ Qdrant 연결 성공! 컬렉션 수: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"❌ Qdrant 연결 실패: {e}")
        return False

def test_streamlit_app():
    """Streamlit 앱 테스트"""
    print("🌐 Streamlit 앱 테스트 중...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit 앱이 정상적으로 실행 중입니다!")
            print("🌍 브라우저에서 http://localhost:8501 로 접속하세요.")
            return True
        else:
            print(f"❌ Streamlit 앱 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Streamlit 앱에 연결할 수 없습니다.")
        print("💡 다음 명령어로 Streamlit 앱을 실행하세요:")
        print("   streamlit run streamlit_vlm_interface.py")
        return False
    except Exception as e:
        print(f"❌ Streamlit 앱 테스트 오류: {e}")
        return False

def test_vlm_system():
    """VLM 시스템 테스트"""
    print("🤖 VLM 시스템 테스트 중...")
    try:
        system = TestExcelsVLMSystem()
        
        # 테스트 쿼리 실행
        test_queries = [
            "월별 생산량은 얼마인가요?",
            "조립 공정의 소요시간은 얼마인가요?",
            "메인보드의 단가는 얼마인가요?"
        ]
        
        for query in test_queries:
            print(f"\n📝 테스트 질문: {query}")
            result = system.query_with_vlm(query)
            print(f"✅ 답변: {result['answer'][:100]}...")
            if result['images']:
                print(f"🖼️ 이미지: {len(result['images'])}개")
        
        print("\n✅ VLM 시스템 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ VLM 시스템 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Test Excels VLM System - 통합 테스트")
    print("=" * 50)
    
    # 1. Qdrant 연결 테스트
    qdrant_ok = test_qdrant_connection()
    
    # 2. VLM 시스템 테스트
    vlm_ok = test_vlm_system()
    
    # 3. Streamlit 앱 테스트
    streamlit_ok = test_streamlit_app()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약:")
    print(f"🔍 Qdrant 연결: {'✅ 성공' if qdrant_ok else '❌ 실패'}")
    print(f"🤖 VLM 시스템: {'✅ 성공' if vlm_ok else '❌ 실패'}")
    print(f"🌐 Streamlit 앱: {'✅ 성공' if streamlit_ok else '❌ 실패'}")
    
    if all([qdrant_ok, vlm_ok, streamlit_ok]):
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("🌍 브라우저에서 http://localhost:8501 로 접속하여 시스템을 사용하세요.")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다.")
        if not qdrant_ok:
            print("💡 Qdrant 서버를 실행하세요: docker run -p 6333:6333 qdrant/qdrant")
        if not streamlit_ok:
            print("💡 Streamlit 앱을 실행하세요: streamlit run streamlit_vlm_interface.py")

if __name__ == "__main__":
    main()
