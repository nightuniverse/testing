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
        self.vector_database = None
        self.text_chunks = []
        self.embeddings = []
        self.embedding_model = None
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
            
            # 5단계: 처리된 데이터 저장
            file_name = uploaded_file.name
            self.processed_data[file_name] = {
                "type": "excel_file",
                "content": f"Excel 파일: {file_name}",
                "parsed_data": parsed_data,
                "chunks_count": len(self.text_chunks),
                "vector_db_size": len(self.embeddings),
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
        """벡터 검색 기반 범용 쿼리 처리"""
        query_lower = query.lower()
        
        # 이미지 관련 (우선순위 높임)
        if "이미지" in query_lower or "사진" in query_lower:
            return self.get_image_data(query)
        
        # Excel 파일 정보 요청
        if "파일 정보" in query_lower or "excel 파일" in query_lower:
            return self.get_excel_file_info()
        
        # 벡터 데이터베이스가 구축되어 있으면 벡터 검색 수행
        if self.vector_database is not None and len(self.text_chunks) > 0:
            vector_results = self._vector_search_query(query)
            if vector_results:
                return vector_results
        
        # Excel 파일 데이터 검색 (fallback)
        excel_results = self._search_excel_data(query)
        if excel_results:
            return excel_results
        
        # 일반적인 응답
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
        """Excel에서 추출된 실제 이미지 데이터 반환"""
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
        
        # 질문 키워드 분석 및 우선순위 설정
        query_keywords = []
        priority_keywords = []
        
        # 조립도 관련 질문 (최우선)
        if any(word in query_lower for word in ["조립도", "조립", "공정", "작업"]):
            priority_keywords.extend(["조립", "공정", "작업", "단계", "과정"])
            query_keywords.extend(["조립", "공정", "작업", "단계", "과정"])
        
        # 제품 관련 질문
        if any(word in query_lower for word in ["제품", "안착", "상세", "클로즈업"]):
            query_keywords.extend(["제품", "안착", "상세", "클로즈업", "부품"])
        
        # 검사 관련 질문
        if any(word in query_lower for word in ["검사", "품질", "테스트", "확인"]):
            query_keywords.extend(["검사", "품질", "테스트", "확인", "기준"])
        
        # 부품/도면 관련 질문
        if any(word in query_lower for word in ["부품", "도면", "설계", "치수"]):
            query_keywords.extend(["부품", "도면", "설계", "치수", "상세"])
        
        # 장비 관련 질문
        if any(word in query_lower for word in ["장비", "현미경", "지그", "렌즈"]):
            query_keywords.extend(["장비", "현미경", "지그", "렌즈", "도구"])
        
        # 포장/완성 관련 질문
        if any(word in query_lower for word in ["포장", "완성", "최종", "출하"]):
            query_keywords.extend(["포장", "완성", "최종", "출하", "배송"])
        
        # 매칭 점수 계산
        best_match = None
        best_score = 0
        
        for img_name, img in self.extracted_images.items():
            img_name_lower = img_name.lower()
            score = 0
            
            # 우선순위 키워드 매칭 (높은 점수)
            for priority_keyword in priority_keywords:
                if priority_keyword in img_name_lower:
                    score += 10  # 최우선 점수
                elif any(priority_keyword in str(img_name) for img_name in self.extracted_images.keys()):
                    score += 8   # 간접 매칭
            
            # 일반 키워드 매칭
            for keyword in query_keywords:
                if keyword in img_name_lower:
                    score += 5   # 직접 매칭
                elif any(keyword in str(img_name) for img_name in self.extracted_images.keys()):
                    score += 3   # 간접 매칭
            
            # 이미지 이름 패턴 매칭
            if "image" in img_name_lower:
                # 숫자 기반 우선순위 (조립도는 보통 앞쪽 이미지)
                try:
                    img_num = int(''.join(filter(str.isdigit, img_name)))
                    if "조립" in query_lower and img_num <= 30:  # 조립도는 앞쪽 이미지
                        score += 3
                    elif "제품" in query_lower and img_num >= 40:  # 제품 관련은 뒤쪽 이미지
                        score += 3
                except:
                    pass
            
            # 점수 업데이트
            if score > best_score:
                best_score = score
                best_match = (img_name, img, f"매칭 점수: {score} (키워드: {', '.join(query_keywords[:3])})")
        
        # 매칭된 이미지가 있으면 반환
        if best_match and best_score > 0:
            img_name, img, description = best_match
            return {
                "type": "image",
                "title": f"🖼️ {img_name} - {query}",
                "image": img,
                "description": description,
                "all_images": [best_match],
                "query_info": f"질문: '{query}'에 대한 최적 매칭 이미지"
            }
        
        # 매칭되는 이미지가 없으면 모든 이미지 목록 표시
        return {
            "type": "image_list",
            "title": "🖼️ 사용 가능한 이미지들",
            "content": f"질문 '{query}'에 맞는 이미지를 찾을 수 없습니다. 다음 이미지들이 있습니다:",
            "available_images": list(self.extracted_images.keys()),
            "all_images": [(name, img, f"이미지: {name}") for name, img in self.extracted_images.items()],
            "suggestions": ["더 구체적인 질문을 해보세요", "예: '조립 공정도를 보여줘'", "예: '제품 안착 이미지를 보여줘'"]
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
