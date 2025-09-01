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
            
            # Excel 파일 처리 (샘플 데이터 사용)
            self.process_sample_excel_data()
            
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
    
    def process_sample_excel_data(self):
        """샘플 Excel 데이터 처리"""
        self.processed_data = {
            "월별 생산량": {
                "1월": 1500, "2월": 1800, "3월": 2200, "4월": 2000,
                "5월": 2500, "6월": 2800, "7월": 3000, "8월": 3200,
                "9월": 3500, "10월": 3800, "11월": 4000, "12월": 4200
            },
            "품질검사 합격률": {
                "1분기": 95.2, "2분기": 96.8, "3분기": 97.1, "4분기": 98.3
            },
            "조립 공정 소요시간": {
                "메인보드 조립": "45분", "부품 결합": "30분", "품질검사": "15분", "포장": "10분"
            },
            "메인보드 단가": "₩125,000",
            "현재고 수량": "2,450개"
        }
    
    def query_system(self, query):
        """쿼리 처리"""
        query_lower = query.lower()
        
        # 월별 생산량 관련
        if "생산량" in query_lower:
            return self.get_production_data()
        
        # 품질검사 관련
        elif "품질" in query_lower or "검사" in query_lower:
            return self.get_quality_data()
        
        # 조립 공정 관련
        elif "조립" in query_lower or "공정" in query_lower:
            return self.get_assembly_data()
        
        # 단가 관련
        elif "단가" in query_lower or "가격" in query_lower:
            return self.get_price_data()
        
        # 재고 관련
        elif "재고" in query_lower or "수량" in query_lower:
            return self.get_inventory_data()
        
        # 이미지 관련
        elif "이미지" in query_lower or "사진" in query_lower:
            return self.get_image_data(query)
        
        else:
            return self.get_general_response(query)
    
    def get_production_data(self):
        """생산량 데이터 반환"""
        data = self.processed_data["월별 생산량"]
        df = pd.DataFrame(list(data.items()), columns=['월', '생산량'])
        
        return {
            "type": "production",
            "title": "📊 월별 생산량 현황",
            "data": df,
            "summary": f"총 연간 생산량: {sum(data.values()):,}개",
            "chart_type": "bar"
        }
    
    def get_quality_data(self):
        """품질검사 데이터 반환"""
        data = self.processed_data["품질검사 합격률"]
        df = pd.DataFrame(list(data.items()), columns=['분기', '합격률(%)'])
        
        return {
            "type": "quality",
            "title": "🔍 품질검사 합격률",
            "data": df,
            "summary": f"평균 합격률: {sum(data.values())/len(data):.1f}%",
            "chart_type": "line"
        }
    
    def get_assembly_data(self):
        """조립 공정 데이터 반환"""
        data = self.processed_data["조립 공정 소요시간"]
        df = pd.DataFrame(list(data.items()), columns=['공정', '소요시간'])
        
        return {
            "type": "assembly",
            "title": "⚙️ 조립 공정 소요시간",
            "data": df,
            "summary": "총 조립 시간: 1시간 40분",
            "chart_type": "bar"
        }
    
    def get_price_data(self):
        """단가 데이터 반환"""
        return {
            "type": "price",
            "title": "💰 메인보드 단가",
            "data": self.processed_data["메인보드 단가"],
            "summary": "현재 시장 평균 대비 15% 저렴",
            "chart_type": "metric"
        }
    
    def get_inventory_data(self):
        """재고 데이터 반환"""
        return {
            "type": "inventory",
            "title": "📦 현재고 수량",
            "data": self.processed_data["현재고 수량"],
            "summary": "안전재고 기준: 1,500개",
            "chart_type": "metric"
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
                "월별 생산량은 얼마인가요?",
                "품질검사 합격률은 몇 퍼센트인가요?",
                "조립 공정의 소요시간은 얼마인가요?",
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
            "월별 생산량은 얼마인가요?",
            "품질검사 합격률은 몇 퍼센트인가요?",
            "조립 공정의 소요시간은 얼마인가요?",
            "메인보드의 단가는 얼마인가요?",
            "현재고 수량은 몇 개인가요?",
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
        placeholder="예: 월별 생산량은 얼마인가요?"
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
    if result["type"] == "production":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(result["data"].set_index('월'))
        
        with col2:
            st.metric("총 연간 생산량", result["summary"])
            st.dataframe(result["data"])
    
    elif result["type"] == "quality":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.line_chart(result["data"].set_index('분기'))
        
        with col2:
            st.metric("평균 합격률", result["summary"])
            st.dataframe(result["data"])
    
    elif result["type"] == "assembly":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(result["data"].set_index('공정'))
        
        with col2:
            st.metric("총 조립 시간", result["summary"])
            st.dataframe(result["data"])
    
    elif result["type"] == "price":
        st.subheader(result["title"])
        st.metric("메인보드 단가", result["data"])
        st.info(result["summary"])
    
    elif result["type"] == "inventory":
        st.subheader(result["title"])
        st.metric("현재고 수량", result["data"])
        st.info(result["summary"])
    
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
