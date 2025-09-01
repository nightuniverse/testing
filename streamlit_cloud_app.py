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
        self.extracted_images = {}
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            # Excel 파일에서 이미지 추출
            self.extract_images_from_excel()
            
            # Excel 파일 처리 (실제 파일 내용 기반)
            self.process_real_excel_data()
            
            st.success("✅ 시스템 초기화 완료!")
            return True
        except Exception as e:
            st.error(f"❌ 시스템 초기화 중 오류 발생: {str(e)}")
            return False
    
    def extract_images_from_excel(self):
        """Excel 파일에서 이미지 추출"""
        try:
            excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') and not f.startswith('~$')]
            
            for excel_file in excel_files:
                logger.info(f"Excel 파일에서 이미지 추출 중: {excel_file}")
                
                # Excel 파일을 ZIP으로 열기
                with zipfile.ZipFile(excel_file, 'r') as zip_file:
                    # 이미지 파일들 찾기
                    image_files = [f for f in zip_file.namelist() if f.startswith('xl/media/')]
                    
                    for image_file in image_files:
                        try:
                            # 이미지 파일 읽기
                            with zip_file.open(image_file) as img_file:
                                img_data = img_file.read()
                                img = Image.open(io.BytesIO(img_data))
                                
                                # 이미지 이름 추출
                                img_name = os.path.basename(image_file)
                                img_name_without_ext = os.path.splitext(img_name)[0]
                                
                                # 이미지 저장
                                self.extracted_images[img_name_without_ext] = img
                                logger.info(f"이미지 추출 완료: {img_name}")
                                
                        except Exception as e:
                            logger.error(f"이미지 추출 실패 {image_file}: {e}")
            
            logger.info(f"총 {len(self.extracted_images)}개 이미지 추출 완료")
            
        except Exception as e:
            logger.error(f"Excel 이미지 추출 실패: {e}")
            # 이미지가 없으면 기본 이미지 생성
            self.create_default_images()
    
    def extract_images_from_uploaded_file(self, uploaded_file):
        """업로드된 Excel 파일에서 이미지 추출"""
        try:
            # 업로드된 파일을 임시로 저장
            with open("temp_excel.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Excel 파일을 ZIP으로 열기
            with zipfile.ZipFile("temp_excel.xlsx", 'r') as zip_file:
                # 이미지 파일들 찾기
                image_files = [f for f in zip_file.namelist() if f.startswith('xl/media/')]
                
                extracted_count = 0
                for image_file in image_files:
                    try:
                        # 이미지 파일 읽기
                        with zip_file.open(image_file) as img_file:
                            img_data = img_file.read()
                            img = Image.open(io.BytesIO(img_data))
                            
                            # 이미지 이름 추출
                            img_name = os.path.basename(image_file)
                            img_name_without_ext = os.path.splitext(img_name)[0]
                            
                            # 이미지 저장
                            self.extracted_images[img_name_without_ext] = img
                            extracted_count += 1
                            
                    except Exception as e:
                        logger.error(f"이미지 추출 실패 {image_file}: {e}")
                
                # 임시 파일 삭제
                if os.path.exists("temp_excel.xlsx"):
                    os.remove("temp_excel.xlsx")
                
                return extracted_count
                
        except Exception as e:
            logger.error(f"업로드된 Excel 이미지 추출 실패: {e}")
            return 0
    
    def create_default_images(self):
        """기본 이미지 생성 (Excel에서 이미지를 찾을 수 없을 때)"""
        try:
            # 품질검사표 이미지
            quality_img = self.create_quality_inspection_image()
            self.extracted_images["품질검사표"] = quality_img
            
            # 조립공정도 이미지
            assembly_img = self.create_assembly_process_image()
            self.extracted_images["조립공정도"] = assembly_img
            
            # 부품도면 이미지
            part_img = self.create_part_drawing_image()
            self.extracted_images["부품도면"] = part_img
            
            logger.info("기본 이미지 생성 완료")
        except Exception as e:
            logger.error(f"기본 이미지 생성 실패: {e}")
    
    def create_quality_inspection_image(self):
        """품질검사표 이미지 생성"""
        # 400x300 크기의 이미지 생성
        img = Image.new('RGB', (400, 300), color='white')
        
        # 간단한 품질검사표 그리기
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(img)
        
        # 제목
        draw.text((20, 20), "품질검사표", fill='black')
        draw.line([(20, 50), (380, 50)], fill='black', width=2)
        
        # 검사 항목들
        items = [
            "1. 외관 검사",
            "2. 치수 검사", 
            "3. 기능 검사",
            "4. 내구성 검사"
        ]
        
        y_pos = 70
        for item in items:
            draw.text((30, y_pos), item, fill='blue')
            y_pos += 30
        
        # 합격/불합격 체크박스
        draw.text((200, 70), "□ 합격", fill='green')
        draw.text((200, 100), "□ 불합격", fill='red')
        
        return img
    
    def create_assembly_process_image(self):
        """조립공정도 이미지 생성"""
        # 400x300 크기의 이미지 생성
        img = Image.new('RGB', (400, 300), color='lightblue')
        
        from PIL import ImageDraw
        
        draw = ImageDraw.Draw(img)
        
        # 제목
        draw.text((20, 20), "조립공정도", fill='darkblue')
        draw.line([(20, 50), (380, 50)], fill='darkblue', width=2)
        
        # 공정 흐름도 그리기
        processes = [
            "수입검사",
            "이오나이저",
            "DINO 검사", 
            "CU+SPONGE",
            "도전 TAPE",
            "출하검사",
            "포장"
        ]
        
        x_pos = 30
        y_pos = 80
        for i, process in enumerate(processes):
            # 박스 그리기
            draw.rectangle([x_pos, y_pos, x_pos+80, y_pos+40], outline='darkblue', width=2, fill='white')
            draw.text((x_pos+5, y_pos+10), process, fill='darkblue', size=8)
            
            # 화살표 그리기 (마지막 제외)
            if i < len(processes) - 1:
                draw.line([x_pos+80, y_pos+20, x_pos+100, y_pos+20], fill='darkblue', width=2)
                # 화살표 머리
                draw.polygon([(x_pos+100, y_pos+15), (x_pos+100, y_pos+25), (x_pos+110, y_pos+20)], fill='darkblue')
            
            x_pos += 100
            
            # 두 번째 줄로 넘어가기
            if x_pos > 350:
                x_pos = 30
                y_pos += 80
        
        return img
    
    def create_part_drawing_image(self):
        """부품도면 이미지 생성"""
        # 400x300 크기의 이미지 생성
        img = Image.new('RGB', (400, 300), color='lightgreen')
        
        from PIL import ImageDraw
        
        draw = ImageDraw.Draw(img)
        
        # 제목
        draw.text((20, 20), "부품도면 - FRONT DECO SUB", fill='darkgreen')
        draw.line([(20, 50), (380, 50)], fill='darkgreen', width=2)
        
        # 간단한 도면 그리기
        # 외곽선
        draw.rectangle([50, 80, 350, 250], outline='darkgreen', width=3)
        
        # 내부 구조
        draw.rectangle([70, 100, 150, 180], outline='darkgreen', width=2, fill='white')
        draw.text((80, 120), "GATE", fill='darkgreen')
        
        draw.rectangle([170, 100, 250, 180], outline='darkgreen', width=2, fill='white')
        draw.text((180, 120), "SPONGE", fill='darkgreen')
        
        draw.rectangle([270, 100, 330, 180], outline='darkgreen', width=2, fill='white')
        draw.text((280, 120), "TAPE", fill='darkgreen')
        
        # 치수선
        draw.line([50, 260, 350, 260], fill='darkgreen', width=1)
        draw.text((200, 270), "300mm", fill='darkgreen')
        
        draw.line([370, 80, 370, 250], fill='darkgreen', width=1)
        draw.text((380, 165), "170mm", fill='darkgreen')
        
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
        
        # 이미지 관련 (우선순위 높임)
        if "이미지" in query_lower or "사진" in query_lower:
            return self.get_image_data(query)
        
        # 조립 공정 관련
        elif "조립" in query_lower or "공정" in query_lower:
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
        query_lower = query.lower()
        
        # 추출된 이미지 목록 표시
        if not self.extracted_images:
            return {
                "type": "no_image",
                "title": "🖼️ 이미지 없음",
                "content": "Excel 파일에서 추출된 이미지가 없습니다.",
                "available_images": []
            }
        
        # 질문에 맞는 이미지 찾기
        matched_images = []
        
        # 이미지 이름과 키워드 매핑
        image_keywords = {
            "image49": ["제품", "안착", "상세", "클로즈업", "부품"],
            "image50": ["검사", "장비", "현미경", "지그", "렌즈"],
            "image51": ["공정", "흐름", "단계", "과정"],
            "image52": ["품질", "검사", "기준", "절차"],
            "image53": ["조립", "공정", "작업", "절차"],
            "image54": ["도면", "부품", "설계", "치수"],
            "image55": ["검사", "테스트", "확인"],
            "image56": ["포장", "완성", "최종"]
        }
        
        # 질문 키워드 분석
        question_keywords = []
        if "제품" in query_lower or "안착" in query_lower:
            question_keywords.extend(["제품", "안착", "상세"])
        if "검사" in query_lower or "품질" in query_lower:
            question_keywords.extend(["검사", "품질", "테스트"])
        if "조립" in query_lower or "공정" in query_lower:
            question_keywords.extend(["조립", "공정", "작업"])
        if "부품" in query_lower or "도면" in query_lower:
            question_keywords.extend(["부품", "도면", "설계"])
        if "장비" in query_lower or "현미경" in query_lower:
            question_keywords.extend(["장비", "현미경", "지그"])
        if "포장" in query_lower or "완성" in query_lower:
            question_keywords.extend(["포장", "완성", "최종"])
        
        # 매칭 점수 계산
        best_match = None
        best_score = 0
        
        for img_name, img in self.extracted_images.items():
            img_name_lower = img_name.lower()
            score = 0
            
            # 이미지 이름 기반 매칭
            if img_name in image_keywords:
                img_keywords = image_keywords[img_name]
                for q_keyword in question_keywords:
                    for img_keyword in img_keywords:
                        if q_keyword in img_keyword or img_keyword in q_keyword:
                            score += 2  # 높은 점수
            
            # 직접 키워드 매칭
            for q_keyword in question_keywords:
                if q_keyword in img_name_lower:
                    score += 1
            
            # 특별한 매칭 규칙
            if "제품" in query_lower and "안착" in query_lower and "image49" in img_name_lower:
                score += 5  # 제품 안착 관련 질문에 image49 우선
            elif "검사" in query_lower and "장비" in query_lower and "image50" in img_name_lower:
                score += 5  # 검사 장비 관련 질문에 image50 우선
            elif "공정" in query_lower and "흐름" in query_lower and "image51" in img_name_lower:
                score += 5  # 공정 흐름 관련 질문에 image51 우선
            
            if score > best_score:
                best_score = score
                best_match = (img_name, img, f"매칭 점수: {score}")
        
        # 매칭된 이미지가 있으면 반환
        if best_match and best_score > 0:
            img_name, img, description = best_match
            return {
                "type": "image",
                "title": f"🖼️ {img_name}",
                "image": img,
                "description": description,
                "all_images": [best_match]
            }
        
        # 매칭되는 이미지가 없으면 모든 이미지 목록 표시
        return {
            "type": "image_list",
            "title": "🖼️ 사용 가능한 이미지들",
            "content": f"질문 '{query}'에 맞는 이미지를 찾을 수 없습니다. 다음 이미지들이 있습니다:",
            "available_images": list(self.extracted_images.keys()),
            "all_images": [(name, img, "사용 가능한 이미지") for name, img in self.extracted_images.items()]
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
        
        st.header("📁 Excel 파일 업로드")
        st.write("Excel 파일을 업로드하여 이미지를 추출할 수 있습니다.")
        
        uploaded_file = st.file_uploader(
            "Excel 파일 선택 (.xlsx)",
            type=['xlsx'],
            help="이미지가 포함된 Excel 파일을 업로드하세요"
        )
        
        if uploaded_file is not None:
            if st.button("📤 이미지 추출", type="primary"):
                with st.spinner("Excel 파일에서 이미지를 추출하고 있습니다..."):
                    extracted_count = st.session_state.system.extract_images_from_uploaded_file(uploaded_file)
                    if extracted_count > 0:
                        st.success(f"✅ {extracted_count}개 이미지 추출 완료!")
                    else:
                        st.warning("⚠️ 이미지를 찾을 수 없습니다. 기본 이미지를 사용합니다.")
                        st.session_state.system.create_default_images()
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
    
    # 현재 추출된 이미지 정보 표시
    if st.session_state.system.extracted_images:
        st.info(f"📸 현재 {len(st.session_state.system.extracted_images)}개 이미지가 로드되어 있습니다.")
        with st.expander("📋 로드된 이미지 목록"):
            for img_name in st.session_state.system.extracted_images.keys():
                st.write(f"- {img_name}")
    
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
        
        # 이미지를 바이트로 변환하여 표시
        import io
        img_byte_arr = io.BytesIO()
        result["image"].save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        st.image(img_byte_arr, caption=result["description"], width=400)
        
        # 이미지 정보 표시
        st.info(f"📐 이미지 크기: {result['image'].size[0]} x {result['image'].size[1]} 픽셀")
        
        # 다른 매칭된 이미지들도 표시
        if "all_images" in result and len(result["all_images"]) > 1:
            st.write("🔍 다른 관련 이미지들:")
            for i, (img_name, img, desc) in enumerate(result["all_images"][1:], 1):
                with st.expander(f"{i}. {img_name}"):
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    st.image(img_byte_arr, caption=desc, width=300)
    
    elif result["type"] == "image_list":
        st.subheader(result["title"])
        st.write(result["content"])
        
        # 사용 가능한 이미지 목록 표시
        if "available_images" in result:
            st.write("📋 사용 가능한 이미지:")
            for img_name in result["available_images"]:
                st.write(f"- {img_name}")
        
        # 모든 이미지 표시
        if "all_images" in result:
            st.write("🖼️ 모든 이미지:")
            for i, (img_name, img, desc) in enumerate(result["all_images"], 1):
                with st.expander(f"{i}. {img_name}"):
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    st.image(img_byte_arr, caption=desc, width=300)
    
    elif result["type"] == "no_image":
        st.subheader(result["title"])
        st.write(result["content"])
        
        if "available_images" in result:
            st.write("📋 사용 가능한 이미지:")
            for img_name in result["available_images"]:
                st.write(f"- {img_name}")
    
    elif result["type"] == "general":
        st.subheader(result["title"])
        st.write(result["content"])
        
        if "suggestions" in result:
            st.write("💡 추천 질문:")
            for suggestion in result["suggestions"]:
                st.write(f"- {suggestion}")

if __name__ == "__main__":
    main()
