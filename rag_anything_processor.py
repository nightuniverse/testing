"""
RAG-Anything 문서 처리기
Docling 파서를 사용하여 문서를 파싱하고 RAG-Anything으로 처리
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional
import logging

# RAG-Anything 임포트
import raganything
from raganything import RAGAnything

# 설정 임포트
from rag_anything_config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAnythingProcessor:
    """RAG-Anything 문서 처리기"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 가져옴)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.rag = None
        self.output_dir = config.OUTPUT_DIR
        
    async def initialize(self):
        """RAG-Anything 초기화"""
        try:
            logger.info("RAG-Anything 초기화 중...")
            
            # RAG-Anything 인스턴스 생성
            self.rag = RAGAnything(config=config)
            
            logger.info("RAG-Anything 초기화 완료")
            
        except Exception as e:
            logger.error(f"RAG-Anything 초기화 실패: {e}")
            raise
    
    async def process_document(
        self, 
        file_path: str, 
        doc_id: Optional[str] = None,
        display_stats: bool = True
    ) -> dict:
        """
        문서 처리 (Docling 파서 사용)
        
        Args:
            file_path: 처리할 문서 경로
            doc_id: 문서 ID (None이면 파일명 사용)
            display_stats: 통계 표시 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        try:
            logger.info(f"문서 처리 시작: {file_path}")
            
            # 문서 ID 설정
            if doc_id is None:
                doc_id = Path(file_path).stem
            
            # Docling 파서로 문서 처리
            result = await self.rag.process_document_complete(
                file_path=file_path,
                doc_id=doc_id
            )
            
            logger.info(f"문서 처리 완료: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"문서 처리 실패: {file_path}, 오류: {e}")
            raise
    
    async def process_multimodal_content(
        self, 
        content_list: List[dict],
        doc_id: str
    ) -> dict:
        """
        멀티모달 콘텐츠 직접 처리
        
        Args:
            content_list: 콘텐츠 리스트
            doc_id: 문서 ID
            
        Returns:
            처리 결과
        """
        try:
            logger.info(f"멀티모달 콘텐츠 처리 시작: {doc_id}")
            
            result = await self.rag.process_multimodal_content(
                content_list=content_list,
                doc_id=doc_id
            )
            
            logger.info(f"멀티모달 콘텐츠 처리 완료: {doc_id}")
            return result
            
        except Exception as e:
            logger.error(f"멀티모달 콘텐츠 처리 실패: {doc_id}, 오류: {e}")
            raise
    
    async def query_documents(
        self, 
        query: str, 
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> dict:
        """
        문서 쿼리 (VLM 지원)
        
        Args:
            query: 쿼리 텍스트
            doc_ids: 검색할 문서 ID 리스트 (None이면 모든 문서)
            top_k: 반환할 결과 수
            
        Returns:
            쿼리 결과
        """
        try:
            logger.info(f"문서 쿼리 시작: {query}")
            
            result = await self.rag.query_documents(
                query=query,
                doc_ids=doc_ids,
                top_k=top_k
            )
            
            logger.info(f"문서 쿼리 완료: {query}")
            return result
            
        except Exception as e:
            logger.error(f"문서 쿼리 실패: {query}, 오류: {e}")
            raise
    
    def get_processed_documents(self) -> List[str]:
        """처리된 문서 목록 반환"""
        try:
            # 출력 디렉토리에서 처리된 문서 찾기
            processed_files = []
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.json', '.txt']:
                    processed_files.append(str(file_path))
            
            return processed_files
            
        except Exception as e:
            logger.error(f"처리된 문서 목록 조회 실패: {e}")
            return []

async def main():
    """메인 실행 함수"""
    # 프로세서 초기화
    processor = RAGAnythingProcessor()
    await processor.initialize()
    
    # 테스트 문서 처리 (data 폴더의 문서들)
    data_dir = Path("data")
    if data_dir.exists():
        for file_path in data_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.xlsx', '.txt']:
                try:
                    logger.info(f"문서 처리: {file_path}")
                    result = await processor.process_document(str(file_path))
                    logger.info(f"처리 결과: {result}")
                except Exception as e:
                    logger.error(f"문서 처리 실패: {file_path}, 오류: {e}")
    
    # 쿼리 테스트
    test_queries = [
        "제조 공정에 대해 설명해주세요",
        "현재고 현황은 어떻게 되나요?",
        "수율 현황을 보여주세요"
    ]
    
    for query in test_queries:
        try:
            result = await processor.query_documents(query)
            logger.info(f"쿼리: {query}")
            logger.info(f"결과: {result}")
        except Exception as e:
            logger.error(f"쿼리 실패: {query}, 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())
