#!/usr/bin/env python3
"""
Streamlit VLM Interface for Test Excels System
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import time
from test_excels_vlm_system import TestExcelsVLMSystem

# 페이지 설정
st.set_page_config(
    page_title="Test Excels VLM System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .image-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    .metric-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """시스템 초기화 (캐시됨)"""
    with st.spinner("VLM 시스템을 초기화하는 중..."):
        system = TestExcelsVLMSystem()
        system.process_test_excels()
        return system

def display_image_info(image_path, image_name, file_name):
    """이미지 정보 표시"""
    with st.expander(f"📄 {image_name}", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**이미지 정보:**")
            st.write(f"📁 파일명: {image_name}")
            st.write(f"📄 소스: {file_name}")
        
        with col2:
            if Path(image_path).exists():
                file_size = Path(image_path).stat().st_size
                st.success(f"✅ 파일 존재")
                st.write(f"📏 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                st.write(f"📁 경로: {image_path}")
                
                # 파일 열기 버튼들
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(f"🖼️ 미리보기로 열기", key=f"preview_{image_name}"):
                        import subprocess
                        try:
                            subprocess.run(["open", "-a", "Preview", image_path])
                            st.success("미리보기로 이미지를 열었습니다!")
                        except Exception as e:
                            st.error(f"이미지를 열 수 없습니다: {e}")
                
                with col_btn2:
                    if st.button(f"📁 Finder에서 열기", key=f"finder_{image_name}"):
                        import subprocess
                        try:
                            subprocess.run(["open", image_path])
                            st.success("Finder에서 이미지를 열었습니다!")
                        except Exception as e:
                            st.error(f"Finder를 열 수 없습니다: {e}")
            else:
                st.error(f"❌ 파일 없음: {image_path}")

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">📊 Test Excels VLM System</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 🔧 시스템 설정")
        
        # Qdrant 연결 상태 확인
        qdrant_status = st.empty()
        
        # 시스템 초기화
        if st.button("🔄 시스템 재초기화"):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 예시 질문들")
        
        example_queries = [
            "월별 생산량은 얼마인가요?",
            "품질 검사 합격률은 몇 퍼센트인가요?",
            "조립 공정의 소요시간은 얼마인가요?",
            "메인보드의 단가는 얼마인가요?",
            "현재고 수량은 몇 개인가요?",
            "조립 공정도 이미지를 보여주세요",
            "품질검사표 이미지를 분석해주세요",
            "조립파트 목록과 가격을 보여주세요",
            "현재고 현황 데이터를 보여주세요"
        ]
        
        for i, query in enumerate(example_queries):
            if st.button(f"{i+1}. {query[:30]}...", key=f"example_{i}"):
                st.session_state.query = query
                st.rerun()
    
    # 메인 컨텐츠
    try:
        # 시스템 초기화
        system = initialize_system()
        qdrant_status.success("✅ Qdrant 연결됨")
        
        # 쿼리 입력 섹션
        st.markdown('<h2 class="sub-header">🔍 질문하기</h2>', unsafe_allow_html=True)
        
        # 세션 상태에서 쿼리 가져오기
        if 'query' not in st.session_state:
            st.session_state.query = ""
        
        query = st.text_area(
            "질문을 입력하세요:",
            value=st.session_state.query,
            height=100,
            placeholder="예: 월별 생산량은 얼마인가요? 또는 조립 공정도 이미지를 보여주세요"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            search_button = st.button("🔍 검색", type="primary")
        
        with col2:
            if st.button("📊 데이터 현황"):
                st.session_state.show_stats = True
                st.rerun()
        
        with col3:
            if st.button("🗂️ 파일 목록"):
                st.session_state.show_files = True
                st.rerun()
        
        # 검색 실행
        if search_button and query.strip():
            st.session_state.query = query
            st.session_state.show_stats = False
            st.session_state.show_files = False
            
            with st.spinner("검색 중..."):
                result = system.query_with_vlm(query)
            
            # 결과 표시
            st.markdown('<h2 class="sub-header">📋 검색 결과</h2>', unsafe_allow_html=True)
            
            # 답변 표시
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"**질문:** {result['query']}")
            st.markdown(f"**답변:** {result['answer']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 이미지 정보 표시
            if result['images']:
                st.markdown('<h3 class="sub-header">🖼️ 관련 이미지</h3>', unsafe_allow_html=True)
                
                for i, (image_name, description) in enumerate(zip(result['images'], result['image_descriptions'])):
                    # 이미지 경로 추출
                    if "경로:" in description:
                        image_path = description.split("경로: ")[-1].rstrip(")")
                    else:
                        image_path = ""
                    
                    # 파일명 추출
                    if "파일:" in description:
                        file_name = description.split("파일:" in description)[-1].split("에서")[0]
                    else:
                        file_name = "Unknown"
                    
                    display_image_info(image_path, image_name, file_name)
            
            # 이미지가 직접 포함된 경우 표시
            if 'image_paths' in result and result['image_paths']:
                st.markdown('<h3 class="sub-header">🖼️ 이미지 표시</h3>', unsafe_allow_html=True)
                
                for i, image_path in enumerate(result['image_paths']):
                    if Path(image_path).exists():
                        st.markdown(f"**이미지 {i+1}:** {Path(image_path).name}")
                        st.image(image_path, caption=f"이미지 {i+1}: {Path(image_path).name}", width=400)
                    else:
                        st.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
            # 검색 결과 상세 정보
            if result['search_results']:
                st.markdown('<h3 class="sub-header">📊 참고 정보</h3>', unsafe_allow_html=True)
                
                # 검색 결과를 데이터프레임으로 변환
                search_data = []
                for i, search_result in enumerate(result['search_results']):
                    search_data.append({
                        '순위': i + 1,
                        '파일명': search_result['file_name'],
                        '내용 유형': search_result['content_type'],
                        '관련도': f"{search_result['score']:.3f}",
                        '내용': search_result['text'][:100] + "..." if len(search_result['text']) > 100 else search_result['text']
                    })
                
                df = pd.DataFrame(search_data)
                st.dataframe(df, use_container_width=True)
        
        # 데이터 현황 표시
        if st.session_state.get('show_stats', False):
            st.markdown('<h2 class="sub-header">📊 데이터 현황</h2>', unsafe_allow_html=True)
            
            try:
                # 컬렉션 정보 가져오기
                collection_info = system.qdrant_client.get_collection(system.collection_name)
                total_points = collection_info.points_count
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("총 데이터 포인트", f"{total_points:,}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("컬렉션명", system.collection_name)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("벡터 차원", "384")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Excel 파일 목록
                excel_files = list(Path(".").glob("*.xlsx"))
                st.markdown(f"**📁 Excel 파일 ({len(excel_files)}개):**")
                
                for excel_file in excel_files:
                    file_size = excel_file.stat().st_size
                    st.write(f"• {excel_file.name} ({file_size:,} bytes)")
                
            except Exception as e:
                st.error(f"데이터 현황을 가져올 수 없습니다: {e}")
        
        # 파일 목록 표시
        if st.session_state.get('show_files', False):
            st.markdown('<h2 class="sub-header">🗂️ 파일 목록</h2>', unsafe_allow_html=True)
            
            # Excel 파일들
            excel_files = list(Path(".").glob("*.xlsx"))
            st.markdown(f"**📊 Excel 파일들 ({len(excel_files)}개):**")
            
            for excel_file in excel_files:
                with st.expander(f"📄 {excel_file.name}", expanded=False):
                    file_size = excel_file.stat().st_size
                    st.write(f"**파일 크기:** {file_size:,} bytes ({file_size/1024:.1f} KB)")
                    st.write(f"**절대 경로:** {excel_file.absolute()}")
                    
                    # 파일 내용 미리보기 (간단한 정보만)
                    try:
                        excel_data = pd.read_excel(excel_file, sheet_name=None)
                        st.write(f"**시트 수:** {len(excel_data)}")
                        
                        for sheet_name, df in excel_data.items():
                            st.write(f"• {sheet_name}: {len(df)}행 x {len(df.columns)}열")
                    except Exception as e:
                        st.write(f"파일 읽기 오류: {e}")
            
            # 이미지 파일들
            image_dirs = list(Path(".").glob("rag_anything_output/*/docling/images"))
            if image_dirs:
                st.markdown("**🖼️ 이미지 파일들:**")
                for img_dir in image_dirs:
                    images = list(img_dir.glob("*.png"))
                    if images:
                        folder_name = img_dir.parent.parent.name
                        with st.expander(f"📁 {folder_name} ({len(images)}개 이미지)", expanded=False):
                            for img in images:
                                img_size = img.stat().st_size
                                st.write(f"• {img.name} ({img_size:,} bytes)")
    
    except Exception as e:
        st.error(f"시스템 초기화 중 오류가 발생했습니다: {e}")
        st.info("Qdrant 서버가 실행 중인지 확인해주세요. (기본 포트: 6333)")

if __name__ == "__main__":
    main()
