import streamlit as st
import pandas as pd
from pathlib import Path
import zipfile
import os
from PIL import Image
import io
import base64
import json
from datetime import datetime
import logging

# 페이지 설정
st.set_page_config(
    page_title="Test Excels VLM System - Cloud",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudVLMSystem:
    def __init__(self):
        self.excel_files = []
        self.processed_data = {}
        self.sample_images = {}
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            # 샘플 이미지 생성
            self.create_sample_images()
            
            # Excel 파일 처리 (실제 파일 내용 기반)
            self.process_real_excel_data()
            
            st.success("✅ 시스템 초기화 완료!")
            return True
        except Exception as e:
            st.error(f"❌ 시스템 초기화 중 오류 발생: {str(e)}")
            return False
    
    def create_sample_images(self):
        """샘플 이미지 생성"""
        try:
            # 품질검사표 이미지
            quality_img = self.create_quality_inspection_image()
            self.sample_images["품질검사표"] = quality_img
            
            # 조립공정도 이미지
            assembly_img = self.create_assembly_process_image()
            self.sample_images["조립공정도"] = assembly_img
            
            # 부품도면 이미지
            part_img = self.create_part_drawing_image()
            self.sample_images["부품도면"] = part_img
            
            logger.info("샘플 이미지 생성 완료")
        except Exception as e:
            logger.error(f"샘플 이미지 생성 실패: {e}")
    
    def create_quality_inspection_image(self):
        """품질검사표 이미지 생성"""
        img = Image.new('RGB', (400, 300), color='white')
        return img
    
    def create_assembly_process_image(self):
        """조립공정도 이미지 생성"""
        img = Image.new('RGB', (400, 300), color='lightblue')
        return img
    
    def create_part_drawing_image(self):
        """부품도면 이미지 생성"""
        img = Image.new('RGB', (400, 300), color='lightgreen')
        return img
    
    def process_real_excel_data(self):
        """실제 Excel 파일 내용 기반 데이터 처리"""
        self.processed_data = {
            "조립 공정": {
                "수입검사": "부품 외관 및 치수 검사",
                "이오나이저 작업": "이물제거 및 정전기 제거",
                "DINO 검사": "전장 디노 검사 및 GATE DINO 검사",
                "CU+SPONGE TAPE 조립": "압착 및 경사압착 작업",
                "도전 TAPE 검사": "좌우 편심검사",
                "SPONGE TAPE 검사": "실오라기 육안검사 및 확대경 검사",
                "출하검사": "100% 및 200% 검사",
                "포장": "최종 포장 작업"
            },
            "제품 정보": {
                "모델명": "SM-F741U",
                "제품코드": "GH98-49241A",
                "부품명": "FRONT DECO SUB",
                "문서번호": "SK-WI-001",
                "작성자": "강승지 프로",
                "작성부서": "개발팀"
            },
            "ERP 시스템": {
                "BOM 정보": "제품 생산에 필요한 자재 확인",
                "BOM 정전개 현황": "메뉴 - BOM 정보 - 5. BOM 정전개 현황",
                "자재 관리": "생산에 필요한 자재 현황",
                "공급업체 정보": "부품 공급업체 관리"
            },
            "품질 관리": {
                "검사 기준": "각 공정별 품질 검사 기준",
                "검사 항목": "외관, 치수, 기능 검사",
                "불합격 기준": "품질 기준 미달 시 처리 절차",
                "검사 기록": "검사 결과 기록 및 관리"
            }
        }
    
    def query_system(self, query):
        """쿼리 처리"""
        query_lower = query.lower()
        
        # 조립 공정 관련
        if "조립" in query_lower or "공정" in query_lower:
            return self.get_assembly_process_data()
        
        # 제품 정보 관련
        elif "제품" in query_lower or "모델" in query_lower:
            return self.get_product_info_data()
        
        # ERP 시스템 관련
        elif "erp" in query_lower or "bom" in query_lower or "자재" in query_lower:
            return self.get_erp_data()
        
        # 품질 관리 관련
        elif "품질" in query_lower or "검사" in query_lower:
            return self.get_quality_data()
        
        # 이미지 관련
        elif "이미지" in query_lower or "사진" in query_lower:
            return self.get_image_data(query)
        
        else:
            return self.get_general_response(query)
    
    def get_assembly_process_data(self):
        """조립 공정 데이터 반환"""
        data = self.processed_data["조립 공정"]
        df = pd.DataFrame(list(data.items()), columns=['공정명', '설명'])
        
        return {
            "type": "assembly",
            "title": "⚙️ SM-F741U 조립 공정",
            "data": df,
            "summary": f"총 {len(data)}개 공정",
            "chart_type": "table"
        }
    
    def get_product_info_data(self):
        """제품 정보 데이터 반환"""
        data = self.processed_data["제품 정보"]
        df = pd.DataFrame(list(data.items()), columns=['항목', '내용'])
        
        return {
            "type": "product",
            "title": "📋 제품 정보",
            "data": df,
            "summary": f"모델: {data['모델명']} / 제품코드: {data['제품코드']}",
            "chart_type": "table"
        }
    
    def get_erp_data(self):
        """ERP 시스템 데이터 반환"""
        data = self.processed_data["ERP 시스템"]
        df = pd.DataFrame(list(data.items()), columns=['시스템', '기능'])
        
        return {
            "type": "erp",
            "title": "💻 ERP 시스템 정보",
            "data": df,
            "summary": "제품 생산에 필요한 자재 관리 시스템",
            "chart_type": "table"
        }
    
    def get_quality_data(self):
        """품질 관리 데이터 반환"""
        data = self.processed_data["품질 관리"]
        df = pd.DataFrame(list(data.items()), columns=['항목', '내용'])
        
        return {
            "type": "quality",
            "title": "🔍 품질 관리",
            "data": df,
            "summary": "각 공정별 품질 검사 기준 및 절차",
            "chart_type": "table"
        }
    
    def get_image_data(self, query):
        """이미지 데이터 반환"""
        if "품질" in query:
            return {
                "type": "image",
                "title": "🔍 품질검사표",
                "image": self.sample_images["품질검사표"],
                "description": "품질검사 기준 및 절차"
            }
        elif "조립" in query:
            return {
                "type": "image",
                "title": "⚙️ 조립공정도",
                "image": self.sample_images["조립공정도"],
                "description": "조립 공정 흐름도"
            }
        elif "부품" in query or "도면" in query:
            return {
                "type": "image",
                "title": "📐 부품도면",
                "image": self.sample_images["부품도면"],
                "description": "부품 상세 도면"
            }
        else:
            return {
                "type": "image",
                "title": "🖼️ 관련 이미지",
                "image": self.sample_images["품질검사표"],
                "description": "요청하신 이미지"
            }
    
    def get_general_response(self, query):
        """일반 응답"""
        return {
            "type": "general",
            "title": "💡 일반 정보",
            "content": f"'{query}'에 대한 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주세요.",
            "suggestions": [
                "조립 공정은 어떤 것들이 있나요?",
                "제품 정보를 알려주세요",
                "ERP 시스템 기능은 무엇인가요?",
                "품질 검사 기준은 무엇인가요?",
                "조립 공정도 이미지를 보여주세요"
            ]
        }

def main():
    st.title("📊 Test Excels VLM System - Cloud")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 시스템 설정")
        
        if st.button("🔄 시스템 재초기화", type="primary"):
            st.session_state.system = CloudVLMSystem()
            st.rerun()
        
        st.header("📝 예시 질문들")
        
        example_questions = [
            "조립 공정은 어떤 것들이 있나요?",
            "제품 정보를 알려주세요",
            "ERP 시스템 기능은 무엇인가요?",
            "품질 검사 기준은 무엇인가요?",
            "조립 공정도 이미지를 보여주세요"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question}"):
                st.session_state.query = question
                st.rerun()
    
    # 메인 컨텐츠
    if 'system' not in st.session_state:
        st.session_state.system = CloudVLMSystem()
    
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # 쿼리 입력
    query = st.text_input(
        "🔍 질문을 입력하세요:",
        value=st.session_state.query,
        placeholder="예: 조립 공정은 어떤 것들이 있나요?"
    )
    
    if st.button("🚀 질문하기", type="primary") or st.session_state.query:
        if query:
            st.session_state.query = query
            with st.spinner("질문을 처리하고 있습니다..."):
                result = st.session_state.system.query_system(query)
                display_result(result)
        else:
            st.warning("질문을 입력해주세요.")

def display_result(result):
    """결과 표시"""
    if result["type"] == "assembly":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], use_container_width=True)
        
        with col2:
            st.metric("총 공정 수", result["summary"])
            st.info("SM-F741U 모델의 조립 공정 절차")
    
    elif result["type"] == "product":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], use_container_width=True)
        
        with col2:
            st.metric("모델명", result["summary"])
            st.info("제품 기본 정보")
    
    elif result["type"] == "erp":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], use_container_width=True)
        
        with col2:
            st.metric("시스템", result["summary"])
            st.info("ERP 시스템 기능")
    
    elif result["type"] == "quality":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], use_container_width=True)
        
        with col2:
            st.metric("품질 관리", result["summary"])
            st.info("품질 검사 기준 및 절차")
    
    elif result["type"] == "image":
        st.subheader(result["title"])
        st.image(result["image"], caption=result["description"], width=400)
    
    elif result["type"] == "general":
        st.subheader(result["title"])
        st.write(result["content"])
        
        if "suggestions" in result:
            st.write("💡 추천 질문:")
            for suggestion in result["suggestions"]:
                st.write(f"- {suggestion}")

if __name__ == "__main__":
    main()
