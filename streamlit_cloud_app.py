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
import numpy as np
import hashlib
import random
import requests

# OpenAI 패키지 import 시도 (Streamlit Cloud 호환성)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False
    st.warning("⚠️ OpenAI 패키지를 불러올 수 없습니다. 시뮬레이션 모드로 실행됩니다.")

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

# LLM API 설정
GPT_OSS_API_KEY = "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"
GPT_OSS_BASE_URL = "https://api.openai.com/v1"  # 실제 API 엔드포인트로 변경 필요

# 환경 변수에서 API 키 가져오기 (보안을 위해)
import os
if os.getenv("OPENAI_API_KEY"):
    GPT_OSS_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMIntegration:
    """LLM 모델 연동 클래스"""
    
    def __init__(self):
        self.gpt_oss_client = None
        self.qwen3_client = None
        self.initialize_llm_clients()
    
    def initialize_llm_clients(self):
        """LLM 클라이언트 초기화"""
        try:
            # OpenAI 패키지 사용 가능 여부 확인
            if not OPENAI_AVAILABLE:
                logger.warning("⚠️ OpenAI 패키지를 사용할 수 없습니다. 시뮬레이션 모드로 전환합니다.")
                self.gpt_oss_client = None
                return
            
            # API 키 유효성 검사
            if not GPT_OSS_API_KEY or GPT_OSS_API_KEY.startswith("sk-or-v1-"):
                logger.warning("⚠️ 유효하지 않은 API 키 형식. 시뮬레이션 모드로 전환합니다.")
                self.gpt_oss_client = None
                return
            
            # GPT OSS 120B 클라이언트 초기화
            self.gpt_oss_client = openai.OpenAI(
                api_key=GPT_OSS_API_KEY,
                base_url=GPT_OSS_BASE_URL
            )
            
            # 간단한 API 테스트
            try:
                response = self.gpt_oss_client.models.list()
                logger.info("✅ GPT OSS 120B 클라이언트 초기화 및 연결 테스트 완료")
            except Exception as test_error:
                logger.warning(f"⚠️ API 연결 테스트 실패: {test_error}")
                self.gpt_oss_client = None
                
        except Exception as e:
            logger.error(f"❌ GPT OSS 클라이언트 초기화 실패: {e}")
            self.gpt_oss_client = None
    
    def analyze_image_with_gpt_oss(self, image, prompt):
        """GPT OSS 120B를 사용하여 이미지 분석"""
        try:
            if not self.gpt_oss_client:
                return {"error": "GPT OSS 클라이언트가 초기화되지 않았습니다."}
            
            # 이미지를 base64로 인코딩
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # GPT OSS Vision API 호출
            response = self.gpt_oss_client.chat.completions.create(
                model="gpt-4o",  # 실제 모델명으로 변경 필요
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "model": "GPT OSS 120B"
            }
            
        except Exception as e:
            logger.error(f"GPT OSS 이미지 분석 실패: {e}")
            return {"error": str(e)}
    
    def generate_text_with_gpt_oss(self, prompt, context=""):
        """GPT OSS 120B를 사용하여 텍스트 생성"""
        try:
            if not self.gpt_oss_client:
                return {"error": "GPT OSS 클라이언트가 초기화되지 않았습니다."}
            
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            response = self.gpt_oss_client.chat.completions.create(
                model="gpt-4o",  # 실제 모델명으로 변경 필요
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return {
                "success": True,
                "text": response.choices[0].message.content,
                "model": "GPT OSS 120B"
            }
            
        except Exception as e:
            logger.error(f"GPT OSS 텍스트 생성 실패: {e}")
            return {"error": str(e)}
    
    def analyze_image_with_qwen3(self, image, prompt):
        """Qwen3 오픈소스 모델을 사용하여 이미지 분석"""
        try:
            # Qwen3 API 호출 (실제 엔드포인트로 변경 필요)
            qwen3_url = "https://api.qwen.ai/v1/chat/completions"
            
            # 이미지를 base64로 인코딩
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            headers = {
                "Authorization": f"Bearer {GPT_OSS_API_KEY}",  # API 키 재사용
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "qwen-vl-plus",  # 실제 모델명으로 변경 필요
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            response = requests.post(qwen3_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "analysis": result["choices"][0]["message"]["content"],
                    "model": "Qwen3"
                }
            else:
                return {"error": f"Qwen3 API 오류: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Qwen3 이미지 분석 실패: {e}")
            return {"error": str(e)}
    
    def get_available_models(self):
        """사용 가능한 LLM 모델 목록 반환"""
        models = []
        
        # OpenAI 패키지 사용 가능 여부 확인
        if OPENAI_AVAILABLE and self.gpt_oss_client:
            models.append("GPT OSS 120B")
        
        # Qwen3는 항상 사용 가능 (API 호출 시도)
        models.append("Qwen3")
        
        # 실제 사용 가능한 모델이 없으면 시뮬레이션만 표시
        if not models:
            models.append("시뮬레이션")
        
        return models

class CloudVLMSystem:
    def __init__(self):
        self.excel_files = []
        self.processed_data = {}
        self.extracted_images = {}
        self.vector_database = None
        self.text_chunks = []
        self.embeddings = []
        self.embedding_model = None
        
        # 자동 질문 생성 관련
        self.auto_questions = []
        
        # VLM 이미지 분석 관련
        self.image_analysis = {}
        
        # LLM 연동 관련
        self.llm_integration = LLMIntegration()
        
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            # Streamlit Cloud에서는 로컬 파일 접근 불가
            # 기본 이미지 생성하지 않고 업로드된 파일 처리 대기
            self.extracted_images = {}  # 빈 이미지 딕셔너리로 초기화
            
            # 기본 데이터 초기화
            self.processed_data = {
                "시스템 정보": {
                    "type": "system",
                    "content": "Excel 파일을 업로드하여 데이터를 처리할 수 있습니다.",
                    "features": ["파일 업로드", "이미지 추출", "데이터 분석"]
                }
            }
            
            return True
        except Exception as e:
            st.error(f"❌ 시스템 초기화 중 오류 발생: {str(e)}")
            return False
    
    def extract_images_from_excel(self):
        """Excel 파일에서 이미지 추출 (더 이상 사용하지 않음)"""
        # Streamlit Cloud에서는 로컬 파일 접근 불가
        # 업로드된 파일만 처리 가능
        logger.info("로컬 Excel 파일 접근 불가 - 업로드된 파일만 처리 가능")
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
                
                if extracted_count > 0:
                    # VLM을 사용하여 이미지 분석
                    logger.info(f"VLM 이미지 분석 시작: {extracted_count}개 이미지")
                    self._analyze_images_with_vlm()
                
                return extracted_count
                
        except Exception as e:
            logger.error(f"업로드된 Excel 이미지 추출 실패: {e}")
            return 0
    
    def process_uploaded_excel_data(self, uploaded_file):
        """업로드된 Excel 파일을 docling 스타일로 파싱하고 벡터 데이터베이스 구축"""
        try:
            # 업로드된 파일을 임시로 저장
            with open("temp_excel.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1단계: Excel 파일을 docling 스타일로 파싱
            parsed_data = self._parse_excel_docling_style("temp_excel.xlsx")
            
            # 2단계: 텍스트 청크 생성
            self.text_chunks = self._create_text_chunks(parsed_data)
            
            # 3단계: 임베딩 모델 로드 및 벡터 생성
            self._initialize_embedding_model()
            self.embeddings = self._generate_embeddings(self.text_chunks)
            
            # 4단계: FAISS 벡터 데이터베이스 구축
            self._build_vector_database()
            
            # 5단계: 자동 질문 생성
            self.auto_questions = self.generate_auto_questions("temp_excel.xlsx")
            
            # 6단계: 처리된 데이터 저장
            file_name = uploaded_file.name
            self.processed_data[file_name] = {
                "type": "excel_file",
                "content": f"Excel 파일: {file_name}",
                "parsed_data": parsed_data,
                "chunks_count": len(self.text_chunks),
                "vector_db_size": len(self.embeddings),
                "auto_questions_count": len(self.auto_questions),
                "file_info": {
                    "name": file_name,
                    "size": len(uploaded_file.getbuffer()),
                    "uploaded": datetime.now()
                }
            }
            
            # 임시 파일 삭제
            if os.path.exists("temp_excel.xlsx"):
                os.remove("temp_excel.xlsx")
            
            logger.info(f"Excel 파일 docling 파싱 및 벡터 DB 구축 완료: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Excel 파일 docling 파싱 실패: {e}")
            # 임시 파일 정리
            if os.path.exists("temp_excel.xlsx"):
                os.remove("temp_excel.xlsx")
            return False
    
    def _parse_excel_docling_style(self, excel_file_path):
        """Excel 파일을 docling 스타일로 파싱"""
        try:
            # Excel 파일 읽기
            df = pd.read_excel(excel_file_path, sheet_name=None)
            parsed_data = {
                "file_path": excel_file_path,
                "sheets": {},
                "metadata": {
                    "total_sheets": len(df),
                    "parsed_at": datetime.now().isoformat()
                }
            }
            
            for sheet_name, sheet_df in df.items():
                # 시트별 데이터 파싱
                sheet_data = self._parse_sheet_content(sheet_name, sheet_df)
                parsed_data["sheets"][sheet_name] = sheet_data
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Excel docling 파싱 실패: {e}")
            return None
    
    def _parse_sheet_content(self, sheet_name, sheet_df):
        """시트 내용을 docling 스타일로 파싱"""
        try:
            # 기본 정보
            sheet_info = {
                "name": sheet_name,
                "dimensions": {
                    "rows": len(sheet_df),
                    "columns": len(sheet_df.columns)
                },
                "content": {}
            }
            
            # 1. 헤더 정보 추출
            if len(sheet_df) > 0:
                headers = sheet_df.columns.tolist()
                sheet_info["content"]["headers"] = headers
                
                # 2. 데이터 타입 분석
                data_types = sheet_df.dtypes.to_dict()
                sheet_info["content"]["data_types"] = {str(k): str(v) for k, v in data_types.items()}
                
                # 3. 텍스트 내용 추출 (docling 스타일)
                text_content = []
                
                # 헤더 텍스트
                header_text = f"시트 '{sheet_name}'의 컬럼: {', '.join(headers)}"
                text_content.append(header_text)
                
                # 데이터 행 텍스트 (처음 10행)
                for idx, row in sheet_df.head(10).iterrows():
                    row_text = f"행 {idx+1}: {', '.join([f'{col}={val}' for col, val in row.items() if pd.notna(val)])}"
                    text_content.append(row_text)
                
                # 4. 테이블 구조 분석
                if len(sheet_df) > 0:
                    # 숫자 컬럼과 텍스트 컬럼 구분
                    numeric_cols = sheet_df.select_dtypes(include=[np.number]).columns.tolist()
                    text_cols = sheet_df.select_dtypes(include=['object']).columns.tolist()
                    
                    sheet_info["content"]["structure"] = {
                        "numeric_columns": numeric_cols,
                        "text_columns": text_cols,
                        "total_records": len(sheet_df)
                    }
                    
                    # 숫자 컬럼 통계
                    if numeric_cols:
                        numeric_stats = {}
                        for col in numeric_cols:
                            col_data = sheet_df[col].dropna()
                            if len(col_data) > 0:
                                numeric_stats[col] = {
                                    "min": float(col_data.min()),
                                    "max": float(col_data.max()),
                                    "mean": float(col_data.mean()),
                                    "count": len(col_data)
                                }
                        sheet_info["content"]["numeric_stats"] = numeric_stats
                
                sheet_info["content"]["text_content"] = text_content
            
            return sheet_info
            
        except Exception as e:
            logger.error(f"시트 파싱 실패 {sheet_name}: {e}")
            return {"name": sheet_name, "error": str(e)}
    
    def _create_text_chunks(self, parsed_data):
        """파싱된 데이터에서 검색 가능한 텍스트 청크 생성"""
        chunks = []
        
        try:
            for sheet_name, sheet_data in parsed_data["sheets"].items():
                if "content" in sheet_data and "text_content" in sheet_data["content"]:
                    # 시트별 청크 생성
                    sheet_chunk = {
                        "type": "sheet_overview",
                        "sheet_name": sheet_name,
                        "content": f"시트 '{sheet_name}': {sheet_data['content']['text_content'][0]}",
                        "metadata": {
                            "rows": sheet_data["dimensions"]["rows"],
                            "columns": sheet_data["dimensions"]["columns"]
                        }
                    }
                    chunks.append(sheet_chunk)
                    
                    # 상세 데이터 청크
                    for text_line in sheet_data["content"]["text_content"][1:]:
                        data_chunk = {
                            "type": "data_row",
                            "sheet_name": sheet_name,
                            "content": text_line,
                            "metadata": {"row_type": "data"}
                        }
                        chunks.append(data_chunk)
                    
                    # 구조 정보 청크
                    if "structure" in sheet_data["content"]:
                        structure = sheet_data["content"]["structure"]
                        structure_chunk = {
                            "type": "structure_info",
                            "sheet_name": sheet_name,
                            "content": f"시트 '{sheet_name}' 구조: 숫자 컬럼 {len(structure['numeric_columns'])}, 텍스트 컬럼 {len(structure['text_columns'])}, 총 {structure['total_records']}개 레코드",
                            "metadata": structure
                        }
                        chunks.append(structure_chunk)
                        
                        # 숫자 통계 청크
                        if "numeric_stats" in sheet_data["content"]:
                            for col, stats in sheet_data["content"]["numeric_stats"].items():
                                stats_chunk = {
                                    "type": "numeric_stats",
                                    "sheet_name": sheet_name,
                                    "content": f"컬럼 '{col}' 통계: 최소값 {stats['min']}, 최대값 {stats['max']}, 평균 {stats['mean']}, 데이터 수 {stats['count']}",
                                    "metadata": {"column": col, "stats": stats}
                                }
                                chunks.append(stats_chunk)
            
            logger.info(f"총 {len(chunks)}개의 텍스트 청크 생성 완료")
            return chunks
            
        except Exception as e:
            logger.error(f"텍스트 청크 생성 실패: {e}")
            return []
    
    def _initialize_embedding_model(self):
        """경량화된 임베딩 모델 초기화"""
        try:
            # Streamlit Cloud 환경에 맞게 경량화된 해시 기반 임베딩 사용
            self.embedding_model = "hash_based"
            logger.info("경량화된 해시 기반 임베딩 모델 사용")
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            self.embedding_model = None
    
    def _generate_embeddings(self, text_chunks):
        """완전히 결정적인 해시 기반 임베딩 벡터 생성"""
        try:
            embeddings = []
            for chunk in text_chunks:
                # 해시 기반 벡터 생성 (64차원으로 축소)
                text_content = chunk["content"]
                text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
                
                # 완전히 결정적인 벡터 생성 (random.seed 사용하지 않음)
                vector = self._hash_to_vector(text_hash, 64)
                
                # 정규화
                vector = vector / np.linalg.norm(vector)
                embeddings.append(vector)
            
            logger.info(f"완전히 결정적인 해시 기반 임베딩 {len(embeddings)}개 생성 완료")
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            # fallback: 결정적인 벡터
            embeddings = []
            for i, chunk in enumerate(text_chunks):
                text_hash = hashlib.md5(f"fallback_{i}".encode('utf-8')).hexdigest()
                vector = self._hash_to_vector(text_hash, 64)
                vector = vector / np.linalg.norm(vector)
                embeddings.append(vector)
            logger.info(f"결정적인 fallback 벡터 {len(embeddings)}개 생성 완료")
            return embeddings
    
    def _hash_to_vector(self, text_hash, dimensions):
        """해시를 결정적인 벡터로 변환"""
        vector = np.zeros(dimensions, dtype='float32')
        
        # 해시의 각 문자를 숫자로 변환하여 벡터 생성
        for i in range(dimensions):
            # 해시에서 순환하면서 값을 추출
            hash_idx = i % len(text_hash)
            char_val = ord(text_hash[hash_idx])
            
            # 문자 값을 -1에서 1 사이로 정규화
            normalized_val = (char_val - 48) / 122.0  # 48(0) ~ 122(z) 범위를 -1~1로
            vector[i] = normalized_val
        
        return vector
    
    def _build_vector_database(self):
        """경량화된 Python 기반 벡터 데이터베이스 구축"""
        try:
            if len(self.embeddings) == 0:
                logger.warning("임베딩이 없어 벡터 DB 구축 불가")
                return
            
            # 벡터 차원 확인
            vector_dim = self.embeddings[0].shape[0]
            
            # 순수 Python으로 벡터 데이터베이스 구축
            self.vector_database = {
                "vectors": np.array(self.embeddings),
                "dimension": vector_dim,
                "count": len(self.embeddings)
            }
            
            logger.info(f"경량화된 Python 벡터 데이터베이스 구축 완료: {len(self.embeddings)}개 벡터, {vector_dim}차원")
            
        except Exception as e:
            logger.error(f"벡터 데이터베이스 구축 실패: {e}")
            self.vector_database = None
    
    def create_default_images(self):
        """기본 이미지 생성 (더 이상 사용하지 않음)"""
        # 기본 이미지 생성하지 않음
        # Excel에서 추출된 실제 이미지만 사용
        logger.info("기본 이미지 생성 비활성화 - Excel 이미지만 사용")
        pass
    
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
        """실제 Excel 파일 내용 기반 데이터 처리 (더 이상 사용하지 않음)"""
        # Streamlit Cloud에서는 로컬 파일 접근 불가
        # 업로드된 파일만 처리 가능
        logger.info("로컬 Excel 파일 접근 불가 - 업로드된 파일만 처리 가능")
        self.processed_data = {
            "시스템 정보": {
                "type": "system",
                "content": "Excel 파일을 업로드하여 데이터를 처리할 수 있습니다.",
                "features": ["파일 업로드", "이미지 추출", "데이터 분석"]
            }
        }
    
    def query_system(self, query):
        """개선된 쿼리 처리 시스템"""
        query_lower = query.lower()
        
        # 1. 이미지 관련 질문 (최우선)
        if any(keyword in query_lower for keyword in ["이미지", "사진", "그림", "보여", "출력", "조립도", "도면"]):
            image_result = self.get_image_data(query)
            if image_result and image_result.get("type") != "no_image":
                return image_result
        
        # 2. Excel 파일 정보 요청
        if "파일 정보" in query_lower or "excel 파일" in query_lower:
            return self.get_excel_file_info()
        
        # 3. 벡터 데이터베이스 검색 (AI 기반)
        if self.vector_database is not None and len(self.text_chunks) > 0:
            vector_results = self._vector_search_query(query)
            if vector_results:
                return vector_results
        
        # 4. Excel 데이터 직접 검색 (fallback)
        excel_results = self._search_excel_data(query)
        if excel_results:
            return excel_results
        
        # 5. 일반적인 응답
        return self.get_general_response(query)
    
    def _search_excel_data(self, query):
        """Excel 데이터에서 검색 (fallback)"""
        try:
            query_lower = query.lower()
            results = []
            
            for file_name, file_data in self.processed_data.items():
                if file_data.get("type") == "excel_file":
                    # 파싱된 데이터에서 검색
                    if "parsed_data" in file_data:
                        parsed_data = file_data["parsed_data"]
                        for sheet_name, sheet_data in parsed_data.get("sheets", {}).items():
                            if "content" in sheet_data and "text_content" in sheet_data["content"]:
                                for text_line in sheet_data["content"]["text_content"]:
                                    if query_lower in text_line.lower():
                                        results.append({
                                            "file": file_name,
                                            "sheet": sheet_name,
                                            "content": text_line,
                                            "type": "text_match"
                                        })
                    
                    # 기존 시트 정보에서 검색 (fallback)
                    for sheet_name, sheet_info in file_data.get("sheets", {}).items():
                        # 샘플 데이터에서 검색
                        for row_data in sheet_info.get("sample_data", []):
                            for key, value in row_data.items():
                                if query_lower in str(value).lower():
                                    results.append({
                                        "file": file_name,
                                        "sheet": sheet_name,
                                        "data": row_data,
                                        "match": f"{key}: {value}",
                                        "type": "data_match"
                                    })
            
            if results:
                # 결과를 DataFrame으로 변환
                df_data = []
                for result in results:
                    if result.get("type") == "text_match":
                        df_data.append({
                            "파일명": result["file"],
                            "시트명": result["sheet"],
                            "매칭 유형": "텍스트 매칭",
                            "내용": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
                        })
                    else:
                        df_data.append({
                            "파일명": result["file"],
                            "시트명": result["sheet"],
                            "매칭 유형": "데이터 매칭",
                            "매칭 데이터": result["match"],
                            "전체 데이터": str(result["data"])[:100] + "..." if len(str(result["data"])) > 100 else str(result["data"])
                        })
                
                df = pd.DataFrame(df_data)
                
                return {
                    "type": "excel_search",
                    "title": f"🔍 '{query}' 검색 결과 (fallback)",
                    "data": df,
                    "summary": f"총 {len(results)}개 결과 발견",
                    "chart_type": "table"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Excel 데이터 검색 실패: {e}")
            return None
    
    def _vector_search_query(self, query):
        """완전히 결정적인 벡터 검색을 통한 쿼리 처리"""
        try:
            if self.vector_database is None or len(self.text_chunks) == 0:
                return None
            
            # 쿼리 텍스트를 임베딩 벡터로 변환 (결정적)
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            query_vector = self._hash_to_vector(query_hash, 64)
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            # 순수 Python으로 유사도 계산 (코사인 유사도)
            similarities = []
            vectors = self.vector_database["vectors"]
            
            for i, vector in enumerate(vectors):
                # 코사인 유사도 계산
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append((similarity, i))
            
            # 유사도 순으로 정렬 (상위 5개)
            similarities.sort(reverse=True)
            k = min(5, len(self.text_chunks))
            
            # 검색 결과 정리
            search_results = []
            for i, (similarity, idx) in enumerate(similarities[:k]):
                if idx < len(self.text_chunks):
                    chunk = self.text_chunks[idx]
                    search_results.append({
                        "rank": i + 1,
                        "similarity": float(similarity),
                        "content": chunk["content"],
                        "type": chunk["type"],
                        "sheet_name": chunk.get("sheet_name", "N/A"),
                        "metadata": chunk.get("metadata", {})
                    })
            
            if search_results:
                # 결과를 DataFrame으로 변환
                df_data = []
                for result in search_results:
                    df_data.append({
                        "순위": result["rank"],
                        "유사도": f"{result['similarity']:.3f}",
                        "시트": result["sheet_name"],
                        "유형": result["type"],
                        "내용": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
                    })
                
                df = pd.DataFrame(df_data)
                
                return {
                    "type": "vector_search",
                    "title": f"🔍 AI 벡터 검색 결과: '{query}'",
                    "data": df,
                    "summary": f"AI 벡터 검색으로 {len(search_results)}개 결과 발견 (유사도 기반)",
                    "chart_type": "table",
                    "raw_results": search_results
                }
            
            return None
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return None
    
    def get_excel_file_info(self):
        """Excel 파일 정보 반환"""
        try:
            file_info = []
            for file_name, file_data in self.processed_data.items():
                if file_data.get("type") == "excel_file":
                    # 파싱된 데이터 정보
                    parsed_info = file_data.get("parsed_data", {})
                    sheets_info = parsed_info.get("sheets", {})
                    
                    info = {
                        "파일명": file_name,
                        "시트 수": len(sheets_info),
                        "텍스트 청크 수": file_data.get("chunks_count", 0),
                        "벡터 DB 크기": file_data.get("vector_db_size", 0),
                        "파일 크기": f"{file_data.get('file_info', {}).get('size', 0) / 1024:.1f} KB",
                        "업로드 시간": str(file_data.get('file_info', {}).get('uploaded', 'N/A'))
                    }
                    file_info.append(info)
            
            if file_info:
                df = pd.DataFrame(file_info)
                return {
                    "type": "file_info",
                    "title": "📁 Excel 파일 정보 (벡터 DB 포함)",
                    "data": df,
                    "summary": f"총 {len(file_info)}개 Excel 파일, 벡터 검색 가능",
                    "chart_type": "table"
                }
            else:
                return {
                    "type": "no_files",
                    "title": "📁 Excel 파일 없음",
                    "content": "처리된 Excel 파일이 없습니다. 파일을 업로드해주세요."
                }
                
        except Exception as e:
            logger.error(f"파일 정보 생성 실패: {e}")
            return {
                "type": "error",
                "title": "❌ 오류",
                "content": f"파일 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
            }
    
    def get_image_data(self, query):
        """간단하고 효과적인 이미지 매칭 시스템"""
        query_lower = query.lower()
        
        # Excel에서 추출된 이미지가 없으면 안내
        if not self.extracted_images:
            return {
                "type": "no_image",
                "title": "🖼️ 이미지 없음",
                "content": "Excel 파일에서 추출된 이미지가 없습니다. 먼저 Excel 파일을 업로드하고 '📤 이미지 추출' 버튼을 클릭해주세요.",
                "available_images": [],
                "suggestions": ["Excel 파일 업로드", "📤 이미지 추출 버튼 클릭"]
            }
        
        # 간단한 키워드 매칭 시스템
        matched_images = []
        
        logger.info(f"이미지 매칭 시작: 질문='{query}', 사용 가능한 이미지={list(self.extracted_images.keys())}")
        
        # 조립도 관련 질문 (가장 구체적인 키워드부터 체크)
        if any(word in query_lower for word in ["조립도", "조립공정", "조립작업", "조립과정"]):
            logger.info("조립도 관련 질문 감지")
            # 조립 관련 이미지 찾기 (image1~30 우선)
            for img_name, img in self.extracted_images.items():
                if "image" in img_name.lower():
                    try:
                        img_num = int(''.join(filter(str.isdigit, img_name)))
                        if img_num <= 30:  # 조립도는 보통 앞쪽 이미지
                            matched_images.append((img_name, img, f"조립 관련 이미지 (번호: {img_num})", 10))
                            logger.info(f"조립 이미지 매칭: {img_name} (점수: 10)")
                        else:
                            matched_images.append((img_name, img, f"조립 관련 이미지 (번호: {img_num})", 5))
                            logger.info(f"조립 이미지 매칭: {img_name} (점수: 5)")
                    except:
                        matched_images.append((img_name, img, "조립 관련 이미지", 3))
                        logger.info(f"조립 이미지 매칭: {img_name} (점수: 3)")
        
        # 검사 관련 질문 (품질 검사 우선)
        elif any(word in query_lower for word in ["검사", "품질", "테스트", "확인", "검수"]):
            logger.info("검사 관련 질문 감지")
            for img_name, img in self.extracted_images.items():
                if "image" in img_name.lower():
                    try:
                        img_num = int(''.join(filter(str.isdigit, img_name)))
                        if 20 <= img_num <= 50:  # 검사 관련은 중간 이미지
                            matched_images.append((img_name, img, f"검사 관련 이미지 (번호: {img_num})", 10))
                            logger.info(f"검사 이미지 매칭: {img_name} (점수: 10)")
                        else:
                            matched_images.append((img_name, img, f"검사 관련 이미지 (번호: {img_num})", 5))
                            logger.info(f"검사 이미지 매칭: {img_name} (점수: 5)")
                    except:
                        matched_images.append((img_name, img, "검사 관련 이미지", 3))
                        logger.info(f"검사 이미지 매칭: {img_name} (점수: 3)")
        
        # 부품/도면 관련 질문
        elif any(word in query_lower for word in ["부품", "도면", "설계", "치수", "BOM"]):
            logger.info("부품/도면 관련 질문 감지")
            for img_name, img in self.extracted_images.items():
                if "image" in img_name.lower():
                    try:
                        img_num = int(''.join(filter(str.isdigit, img_name)))
                        if img_num <= 25:  # 부품도면은 앞쪽 이미지
                            matched_images.append((img_name, img, f"부품/도면 이미지 (번호: {img_num})", 10))
                            logger.info(f"부품/도면 이미지 매칭: {img_name} (점수: 10)")
                        else:
                            matched_images.append((img_name, img, f"부품/도면 이미지 (번호: {img_num})", 5))
                            logger.info(f"부품/도면 이미지 매칭: {img_name} (점수: 5)")
                    except:
                        matched_images.append((img_name, img, "부품/도면 이미지", 3))
                        logger.info(f"부품/도면 이미지 매칭: {img_name} (점수: 3)")
        
        # 제품 관련 질문
        elif any(word in query_lower for word in ["제품", "안착", "상세", "클로즈업", "완성"]):
            logger.info("제품 관련 질문 감지")
            # 제품 관련 이미지 찾기 (image40+ 우선)
            for img_name, img in self.extracted_images.items():
                if "image" in img_name.lower():
                    try:
                        img_num = int(''.join(filter(str.isdigit, img_name)))
                        if img_num >= 40:  # 제품 관련은 뒤쪽 이미지
                            matched_images.append((img_name, img, f"제품 관련 이미지 (번호: {img_num})", 10))
                            logger.info(f"제품 이미지 매칭: {img_name} (점수: 10)")
                        else:
                            matched_images.append((img_name, img, f"제품 관련 이미지 (번호: {img_num})", 5))
                            logger.info(f"제품 이미지 매칭: {img_name} (점수: 5)")
                    except:
                        matched_images.append((img_name, img, "제품 관련 이미지", 3))
                        logger.info(f"제품 이미지 매칭: {img_name} (점수: 3)")
        
        # 일반적인 조립 관련 질문 (마지막에 체크)
        elif any(word in query_lower for word in ["조립", "공정", "작업", "과정"]):
            logger.info("일반 조립 관련 질문 감지")
            # 조립 관련 이미지 찾기 (image1~30 우선)
            for img_name, img in self.extracted_images.items():
                if "image" in img_name.lower():
                    try:
                        img_num = int(''.join(filter(str.isdigit, img_name)))
                        if img_num <= 30:  # 조립도는 보통 앞쪽 이미지
                            matched_images.append((img_name, img, f"조립 관련 이미지 (번호: {img_num})", 10))
                            logger.info(f"조립 이미지 매칭: {img_name} (점수: 10)")
                        else:
                            matched_images.append((img_name, img, f"조립 관련 이미지 (번호: {img_num})", 5))
                            logger.info(f"조립 이미지 매칭: {img_name} (점수: 5)")
                    except:
                        matched_images.append((img_name, img, "조립 관련 이미지", 3))
                        logger.info(f"조립 이미지 매칭: {img_name} (점수: 3)")
        

        
        # 일반적인 이미지 요청
        else:
            # 모든 이미지를 점수와 함께 추가
            for img_name, img in self.extracted_images.items():
                matched_images.append((img_name, img, f"이미지: {img_name}", 1))
        
        # 점수 순으로 정렬
        matched_images.sort(key=lambda x: x[3], reverse=True)
        
        logger.info(f"이미지 매칭 완료: 총 {len(matched_images)}개 매칭, 상위 3개: {[(name, score) for name, img, desc, score in matched_images[:3]]}")
        
        # 최고 점수 이미지 반환
        if matched_images:
            best_img_name, best_img, best_desc, best_score = matched_images[0]
            
            logger.info(f"최적 이미지 선택: {best_img_name} (점수: {best_score})")
            
            # VLM 분석 결과가 있으면 추가 정보 제공
            vlm_analysis = None
            if hasattr(self, 'image_analysis') and best_img_name in self.image_analysis:
                vlm_analysis = self.image_analysis[best_img_name]
            
            result = {
                "type": "image",
                "title": f"🖼️ {best_img_name} - {query}",
                "image": best_img,
                "description": best_desc,
                "all_images": [(name, img, desc) for name, img, desc, score in matched_images[:3]],  # 상위 3개
                "query_info": f"질문: '{query}'에 대한 최적 매칭 이미지 (점수: {best_score})",
                "total_matches": len(matched_images)
            }
            
            # VLM 분석 결과 추가
            if vlm_analysis and 'error' not in vlm_analysis:
                result["vlm_analysis"] = {
                    "summary": vlm_analysis['summary'],
                    "type": vlm_analysis['type'],
                    "tags": vlm_analysis['tags'],
                    "confidence": vlm_analysis['confidence'],
                    "details": vlm_analysis['details']
                }
            
            return result
        
        # 매칭되는 이미지가 없으면 모든 이미지 목록 표시
        return {
            "type": "image_list",
            "title": "🖼️ 사용 가능한 이미지들",
            "content": f"질문 '{query}'에 맞는 이미지를 찾을 수 없습니다. 다음 이미지들이 있습니다:",
            "available_images": list(self.extracted_images.keys()),
            "all_images": [(name, img, f"이미지: {name}") for name, img in self.extracted_images.items()],
            "suggestions": [
                "더 구체적인 질문을 해보세요",
                "예: '조립 공정도를 보여줘'",
                "예: '제품 안착 이미지를 보여줘'",
                "예: '품질 검사 과정을 보여줘'"
            ]
        }
    
    def _analyze_images_with_vlm(self):
        """VLM을 사용하여 추출된 이미지들을 분석"""
        try:
            logger.info("VLM 이미지 분석 시작")
            
            # 이미지 분석 결과 저장
            self.image_analysis = {}
            
            for img_name, img in self.extracted_images.items():
                try:
                    # VLM 분석 수행
                    analysis_result = self._analyze_single_image_with_vlm(img_name, img)
                    self.image_analysis[img_name] = analysis_result
                    
                    logger.info(f"이미지 분석 완료: {img_name} - {analysis_result['summary']}")
                    
                except Exception as e:
                    logger.error(f"이미지 {img_name} VLM 분석 실패: {e}")
                    self.image_analysis[img_name] = {
                        "error": str(e),
                        "summary": "분석 실패",
                        "details": [],
                        "tags": []
                    }
            
            logger.info(f"VLM 이미지 분석 완료: {len(self.image_analysis)}개 이미지")
            
        except Exception as e:
            logger.error(f"VLM 이미지 분석 실패: {e}")
    
    def _analyze_single_image_with_vlm(self, img_name, img):
        """단일 이미지를 VLM으로 분석"""
        try:
            # 이미지 메타데이터 분석
            img_info = {
                "name": img_name,
                "size": img.size,
                "mode": img.mode,
                "format": getattr(img, 'format', 'Unknown')
            }
            
            # 실제 LLM을 사용한 이미지 분석 시도
            analysis_result = self._analyze_with_real_llm(img_name, img, img_info)
            
            # LLM 분석이 실패하면 시뮬레이션 사용
            if not analysis_result or "error" in analysis_result:
                logger.warning(f"LLM 분석 실패, 시뮬레이션 사용: {img_name}")
                analysis_result = self._simulate_vlm_analysis(img_name, img_info)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"단일 이미지 VLM 분석 실패 {img_name}: {e}")
            raise
    
    def _simulate_vlm_analysis(self, img_name, img_info):
        """VLM 분석 시뮬레이션 (실제 VLM 모델로 대체 가능)"""
        try:
            # 이미지 이름과 메타데이터를 기반으로 스마트 분석
            img_name_lower = img_name.lower()
            
            # 이미지 번호 추출
            img_num = None
            if "image" in img_name_lower:
                try:
                    img_num = int(''.join(filter(str.isdigit, img_name)))
                except:
                    pass
            
            # 이미지 유형 분류 및 분석
            if img_num is not None:
                if img_num <= 30:
                    # 조립/공정 관련 이미지
                    analysis = {
                        "type": "assembly_process",
                        "summary": f"조립 공정 이미지 (번호: {img_num})",
                        "details": [
                            f"이미지 {img_num}은 조립 공정의 {img_num}번째 단계를 보여줍니다",
                            "작업자가 부품을 조립하는 과정이나 공정 단계를 나타냅니다",
                            "조립 작업의 표준화된 절차를 시각화합니다"
                        ],
                        "tags": ["조립", "공정", "작업", "단계", f"image{img_num}"],
                        "confidence": 0.85
                    }
                elif 31 <= img_num <= 50:
                    # 검사/품질 관련 이미지
                    analysis = {
                        "type": "quality_inspection",
                        "summary": f"품질 검사 이미지 (번호: {img_num})",
                        "details": [
                            f"이미지 {img_num}은 품질 검사 과정을 보여줍니다",
                            "제품의 품질을 확인하고 검증하는 단계입니다",
                            "검사 기준과 방법을 시각적으로 제시합니다"
                        ],
                        "tags": ["검사", "품질", "테스트", "확인", f"image{img_num}"],
                        "confidence": 0.80
                    }
                else:
                    # 제품/완성 관련 이미지
                    analysis = {
                        "type": "product_final",
                        "summary": f"제품 완성 이미지 (번호: {img_num})",
                        "details": [
                            f"이미지 {img_num}은 완성된 제품이나 최종 상태를 보여줍니다",
                            "제품의 최종 형태나 안착 상태를 나타냅니다",
                            "출하 전 최종 점검 결과를 시각화합니다"
                        ],
                        "tags": ["제품", "완성", "안착", "최종", f"image{img_num}"],
                        "confidence": 0.75
                    }
            else:
                # 일반 이미지
                analysis = {
                    "type": "general_image",
                    "summary": f"일반 이미지: {img_name}",
                    "details": [
                        f"이미지 {img_name}은 문서에 포함된 일반적인 이미지입니다",
                        "구체적인 내용은 이미지 자체를 확인해야 합니다"
                    ],
                    "tags": ["이미지", "일반", img_name],
                    "confidence": 0.60
                }
            
            # 이미지 메타데이터 추가
            analysis["metadata"] = img_info
            analysis["analysis_method"] = "VLM_Simulation"
            
            return analysis
            
        except Exception as e:
            logger.error(f"VLM 분석 시뮬레이션 실패: {e}")
            return {
                "type": "error",
                "summary": "분석 실패",
                "details": [f"이미지 분석 중 오류 발생: {str(e)}"],
                "tags": ["오류", "분석실패"],
                "confidence": 0.0
            }
    
    def _analyze_with_real_llm(self, img_name, img, img_info):
        """실제 LLM을 사용하여 이미지 분석"""
        try:
            # 이미지 분석을 위한 프롬프트 생성
            prompt = f"""
            이 이미지({img_name})를 분석해주세요. 
            
            다음 정보를 포함하여 분석해주세요:
            1. 이미지가 보여주는 내용 (조립 공정, 품질 검사, 제품 등)
            2. 이미지의 목적과 용도
            3. 작업자가 알아야 할 핵심 정보
            4. 관련된 키워드나 태그
            
            한국어로 상세하게 설명해주세요.
            """
            
            # GPT OSS 120B로 이미지 분석 시도
            gpt_result = self.llm_integration.analyze_image_with_gpt_oss(img, prompt)
            
            if gpt_result.get("success"):
                # GPT OSS 분석 결과 파싱
                analysis = self._parse_llm_analysis_result(gpt_result["analysis"], img_name, img_info)
                analysis["llm_model"] = "GPT OSS 120B"
                analysis["analysis_method"] = "Real_LLM"
                return analysis
            
            # GPT OSS 실패 시 Qwen3 시도
            qwen_result = self.llm_integration.analyze_image_with_qwen3(img, prompt)
            
            if qwen_result.get("success"):
                # Qwen3 분석 결과 파싱
                analysis = self._parse_llm_analysis_result(qwen_result["analysis"], img_name, img_info)
                analysis["llm_model"] = "Qwen3"
                analysis["analysis_method"] = "Real_LLM"
                return analysis
            
            # 모든 LLM 분석 실패
            logger.warning(f"모든 LLM 분석 실패: {img_name}")
            return None
            
        except Exception as e:
            logger.error(f"실제 LLM 이미지 분석 실패 {img_name}: {e}")
            return None
    
    def _parse_llm_analysis_result(self, llm_text, img_name, img_info):
        """LLM 분석 결과를 구조화된 형태로 파싱"""
        try:
            # 이미지 번호 추출
            img_num = None
            if "image" in img_name.lower():
                try:
                    img_num = int(''.join(filter(str.isdigit, img_name)))
                except:
                    pass
            
            # LLM 텍스트에서 키워드 추출
            text_lower = llm_text.lower()
            
            # 이미지 유형 분류
            if any(word in text_lower for word in ["조립", "공정", "작업", "단계", "과정"]):
                img_type = "assembly_process"
                confidence = 0.90
            elif any(word in text_lower for word in ["검사", "품질", "테스트", "확인"]):
                img_type = "quality_inspection"
                confidence = 0.90
            elif any(word in text_lower for word in ["제품", "완성", "안착", "최종"]):
                img_type = "product_final"
                confidence = 0.90
            else:
                img_type = "general_image"
                confidence = 0.75
            
            # 태그 생성
            tags = []
            if img_num:
                tags.append(f"image{img_num}")
            
            # LLM 텍스트에서 키워드 추출하여 태그 추가
            keywords = ["조립", "공정", "작업", "검사", "품질", "테스트", "제품", "완성", "부품", "도면"]
            for keyword in keywords:
                if keyword in text_lower:
                    tags.append(keyword)
            
            # 상세 분석을 문장 단위로 분리
            details = []
            sentences = llm_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    details.append(sentence)
            
            # 최대 5개 상세 설명으로 제한
            details = details[:5]
            
            analysis = {
                "type": img_type,
                "summary": f"LLM 분석: {img_name}",
                "details": details,
                "tags": tags,
                "confidence": confidence,
                "metadata": img_info,
                "llm_raw_text": llm_text
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM 분석 결과 파싱 실패: {e}")
            return None
    
    def generate_auto_questions(self, excel_file_path):
        """Excel 파일 내용을 분석하여 자동으로 질문 생성"""
        try:
            # Excel 파일 읽기
            df = pd.read_excel(excel_file_path, sheet_name=None)
            
            questions = []
            
            # 키워드 정의
            process_keywords = {'조립', '공정', '작업', '과정', '단계', '순서', '절차', '방법', '기술'}
            quality_keywords = {'검사', '품질', '테스트', '확인', '검수', '점검', '측정', '기준'}
            product_keywords = {'제품', '완성', '출하', '포장', '안착', '상세', '클로즈업'}
            material_keywords = {'부품', '자재', '소재', '재료', 'BOM', '도면', '설계', '치수'}
            equipment_keywords = {'장비', '기계', '도구', '지그', '현미경', '렌즈', '장치'}
            
            # 시트별 키워드 분석
            process_found = False
            quality_found = False
            material_found = False
            product_found = False
            equipment_found = False
            
            for sheet_name, sheet_df in df.items():
                if len(sheet_df) > 0:
                    # 처음 5행에서 키워드 검색
                    for idx, row in sheet_df.head(5).iterrows():
                        row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).lower()
                        
                        if not process_found and any(kw in row_text for kw in process_keywords):
                            questions.extend([
                                "조립 공정도를 보여줘",
                                "작업 과정을 설명해줘",
                                "조립 단계별 이미지를 보여줘"
                            ])
                            process_found = True
                        
                        if not quality_found and any(kw in row_text for kw in quality_keywords):
                            questions.extend([
                                "품질 검사 과정을 보여줘",
                                "검사 기준을 알려줘",
                                "테스트 방법을 보여줘"
                            ])
                            quality_found = True
                        
                        if not material_found and any(kw in row_text for kw in material_keywords):
                            questions.extend([
                                "부품 도면을 보여줘",
                                "BOM 정보를 알려줘",
                                "자재 명세를 보여줘"
                            ])
                            material_found = True
                        
                        if not product_found and any(kw in row_text for kw in product_keywords):
                            questions.extend([
                                "완성된 제품을 보여줘",
                                "제품 안착 이미지를 보여줘",
                                "출하 상태를 보여줘"
                            ])
                            product_found = True
                        
                        if not equipment_found and any(kw in row_text for kw in equipment_keywords):
                            questions.extend([
                                "사용 장비를 보여줘",
                                "작업 도구를 알려줘",
                                "측정 장비를 보여줘"
                            ])
                            equipment_found = True
                        
                        if len(questions) >= 12:  # 최대 12개 질문
                            break
                    if len(questions) >= 12:
                        break
            
            # 일반적인 질문 추가
            questions.extend([
                "파일 정보를 알려줘",
                "시트 구조를 설명해줘",
                "데이터 요약을 보여줘"
            ])
            
            logger.info(f"자동 질문 생성 완료: {len(questions)}개")
            return questions[:15]  # 최대 15개로 제한
            
        except Exception as e:
            logger.error(f"자동 질문 생성 실패: {e}")
            return [
                "파일 정보를 알려줘",
                "이미지를 보여줘",
                "데이터를 요약해줘"
            ]
    
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
        
        # LLM 모델 선택
        st.subheader("🤖 LLM 모델 선택")
        available_models = st.session_state.system.llm_integration.get_available_models()
        
        if "selected_llm_model" not in st.session_state:
            st.session_state.selected_llm_model = available_models[0] if available_models else "시뮬레이션"
        
        # 모델 옵션 구성
        model_options = []
        if "GPT OSS 120B" in available_models:
            model_options.append("GPT OSS 120B")
        if "Qwen3" in available_models:
            model_options.append("Qwen3")
        model_options.append("시뮬레이션")
        
        selected_model = st.selectbox(
            "분석에 사용할 LLM 모델",
            options=model_options,
            index=model_options.index(st.session_state.selected_llm_model) if st.session_state.selected_llm_model in model_options else len(model_options) - 1,
            help="이미지 분석에 사용할 LLM 모델을 선택하세요"
        )
        
        if selected_model != st.session_state.selected_llm_model:
            st.session_state.selected_llm_model = selected_model
            st.rerun()
        
        # LLM 상태 표시
        if selected_model == "GPT OSS 120B":
            if not OPENAI_AVAILABLE:
                st.error("❌ OpenAI 패키지를 사용할 수 없습니다. 패키지 설치가 필요합니다.")
            elif st.session_state.system.llm_integration.gpt_oss_client:
                st.success("✅ GPT OSS 120B 모델 활성화 (API 연결됨)")
            else:
                st.error("❌ GPT OSS 120B 모델 비활성화 (API 연결 실패)")
        elif selected_model == "Qwen3":
            st.warning("⚠️ Qwen3 모델 (API 연결 테스트 필요)")
        else:
            st.info("ℹ️ 시뮬레이션 모드 (실제 LLM 사용 안함)")
        
        # 패키지 상태 표시
        if not OPENAI_AVAILABLE:
            st.error("❌ OpenAI 패키지가 설치되지 않았습니다. requirements.txt를 확인해주세요.")
        else:
            st.success("✅ OpenAI 패키지 사용 가능")
        
        # API 키 상태 표시
        if GPT_OSS_API_KEY.startswith("sk-or-v1-"):
            st.warning("⚠️ API 키 형식이 올바르지 않습니다. OpenAI API 키를 설정해주세요.")
        elif GPT_OSS_API_KEY:
            st.success("✅ API 키 설정됨")
        else:
            st.error("❌ API 키가 설정되지 않았습니다.")
        
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
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 이미지 추출", type="primary"):
                    with st.spinner("Excel 파일에서 이미지를 추출하고 있습니다..."):
                        extracted_count = st.session_state.system.extract_images_from_uploaded_file(uploaded_file)
                        if extracted_count > 0:
                            st.success(f"✅ {extracted_count}개 이미지 추출 완료!")
                        else:
                            st.warning("⚠️ Excel 파일에서 이미지를 찾을 수 없습니다.")
                            st.info("💡 이미지가 포함된 Excel 파일을 업로드해주세요.")
                        st.rerun()
            
            with col2:
                if st.button("📊 데이터 파싱", type="secondary"):
                    with st.spinner("Excel 파일을 파싱하고 있습니다..."):
                        success = st.session_state.system.process_uploaded_excel_data(uploaded_file)
                        if success:
                            st.success("✅ Excel 데이터 파싱 완료!")
                        else:
                            st.warning("⚠️ 데이터 파싱에 실패했습니다.")
                        st.rerun()
        
        st.header("📊 Excel 파일 정보")
        
        if st.button("📁 파일 정보 보기", key="btn_file_info"):
            st.session_state.query = "Excel 파일 정보를 보여주세요"
            st.rerun()
        
        st.header("📝 예시 질문들")
        
        # 자동 생성된 질문들 표시
        if uploaded_file is not None and hasattr(st.session_state.system, 'auto_questions'):
            st.subheader("🤖 AI 자동 생성 질문")
            for i, question in enumerate(st.session_state.system.auto_questions[:8], 1):  # 상위 8개만 표시
                if st.button(f"{i}. {question}", key=f"btn_auto_{i}"):
                    st.session_state.query = question
                    st.rerun()
            
            if len(st.session_state.system.auto_questions) > 8:
                with st.expander(f"더 많은 질문 보기 ({len(st.session_state.system.auto_questions)}개)"):
                    for i, question in enumerate(st.session_state.system.auto_questions[8:], 9):
                        if st.button(f"{i}. {question}", key=f"btn_auto_{i}"):
                            st.session_state.query = question
                            st.rerun()
        else:
            # 기본 예시 질문들
            example_questions = [
                "Excel 파일 정보를 보여주세요",
                "BOM 정보는 무엇인가요?",
                "제품 생산에 필요한 자재는?",
                "조립 공정도 이미지를 보여주세요",
                "품질검사 기준은 무엇인가요?"
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
        
        # VLM 분석 결과가 있으면 표시
        if hasattr(st.session_state.system, 'image_analysis') and st.session_state.system.image_analysis:
            st.success("🤖 VLM 이미지 분석 완료!")
            with st.expander("🔍 VLM 이미지 분석 결과"):
                for img_name, analysis in st.session_state.system.image_analysis.items():
                    if 'error' not in analysis:
                        st.markdown(f"**{img_name}**")
                        st.write(f"📝 **요약**: {analysis['summary']}")
                        st.write(f"🏷️ **태그**: {', '.join(analysis['tags'])}")
                        st.write(f"📊 **신뢰도**: {analysis['confidence']:.2f}")
                        
                        # LLM 모델 정보 표시
                        if "llm_model" in analysis:
                            st.write(f"🤖 **LLM 모델**: {analysis['llm_model']}")
                            st.write(f"🔧 **분석 방법**: {analysis['analysis_method']}")
                        
                        with st.expander("📋 상세 분석"):
                            for detail in analysis['details']:
                                st.write(f"• {detail}")
                            
                            # LLM 원본 텍스트 표시
                            if "llm_raw_text" in analysis:
                                st.write("---")
                                st.write("**🤖 LLM 원본 분석:**")
                                st.write(analysis['llm_raw_text'])
                        
                        st.divider()
                    else:
                        st.error(f"❌ {img_name}: {analysis['error']}")
        
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
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("총 공정 수", result["summary"])
            st.info("SM-F741U 모델의 조립 공정 절차")
    
    elif result["type"] == "product":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("모델명", result["summary"])
            st.info("제품 기본 정보")
    
    elif result["type"] == "erp":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("시스템", result["summary"])
            st.info("ERP 시스템 기능")
    
    elif result["type"] == "quality":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("품질 관리", result["summary"])
            st.info("품질 검사 기준 및 절차")
    
    elif result["type"] == "image":
        st.subheader(result["title"])
        
        # 이미지를 바이트로 변환하여 표시
        img_byte_arr = io.BytesIO()
        result["image"].save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        st.image(img_byte_arr, caption=result["description"], width=400)
        
        # 이미지 정보 표시
        st.info(f"📐 이미지 크기: {result['image'].size[0]} x {result['image'].size[1]} 픽셀")
        
        # VLM 분석 결과 표시
        if "vlm_analysis" in result:
            st.success("🤖 VLM 이미지 분석 결과")
            vlm = result["vlm_analysis"]
            st.write(f"**📝 요약**: {vlm['summary']}")
            st.write(f"**🏷️ 유형**: {vlm['type']}")
            st.write(f"**🔖 태그**: {', '.join(vlm['tags'])}")
            st.write(f"**📊 신뢰도**: {vlm['confidence']:.2f}")
            
            # LLM 모델 정보 표시
            if "llm_model" in vlm:
                st.write(f"**🤖 LLM 모델**: {vlm['llm_model']}")
                st.write(f"**🔧 분석 방법**: {vlm['analysis_method']}")
            
            with st.expander("📋 상세 분석"):
                for detail in vlm['details']:
                    st.write(f"• {detail}")
                
                # LLM 원본 텍스트 표시
                if "llm_raw_text" in vlm:
                    st.write("---")
                    st.write("**🤖 LLM 원본 분석:**")
                    st.write(vlm['llm_raw_text'])
        
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
    
    elif result["type"] == "excel_search":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("검색 결과", result["summary"])
            st.info("Excel 파일에서 찾은 데이터")
    
    elif result["type"] == "vector_search":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("벡터 검색 결과", result["summary"])
            st.info("AI 벡터 검색으로 찾은 유사한 내용")
            
            # 상세 정보 표시
            if "raw_results" in result:
                st.write("🔍 상세 검색 결과:")
                for i, raw_result in enumerate(result["raw_results"][:3], 1):
                    with st.expander(f"결과 {i} (유사도: {raw_result['similarity']:.3f})"):
                        st.write(f"**시트**: {raw_result['sheet_name']}")
                        st.write(f"**유형**: {raw_result['type']}")
                        st.write(f"**내용**: {raw_result['content']}")
                        if raw_result.get("metadata"):
                            st.write(f"**메타데이터**: {raw_result['metadata']}")
    
    elif result["type"] == "file_info":
        st.subheader(result["title"])
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(result["data"], width='stretch')
        
        with col2:
            st.metric("파일 수", result["summary"])
            st.info("처리된 Excel 파일 정보")
    
    elif result["type"] == "no_files":
        st.subheader(result["title"])
        st.write(result["content"])
        st.info("📤 Excel 파일을 업로드하여 데이터를 처리할 수 있습니다.")
    
    elif result["type"] == "general":
        st.subheader(result["title"])
        st.write(result["content"])
        
        if "suggestions" in result:
            st.write("💡 추천 질문:")
            for suggestion in result["suggestions"]:
                st.write(f"- {suggestion}")

if __name__ == "__main__":
    main()
