"""
VLM 지원 쿼리 엔진
RAG-Anything을 사용하여 시각적 요소가 포함된 문서에 대한 고급 쿼리 처리
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from rag_anything_processor import RAGAnythingProcessor
from rag_anything_config import config

logger = logging.getLogger(__name__)

class VLMQueryEngine:
    """VLM 지원 쿼리 엔진"""
    
    def __init__(self, processor: RAGAnythingProcessor):
        """
        초기화
        
        Args:
            processor: RAG-Anything 프로세서 인스턴스
        """
        self.processor = processor
        self.knowledge_base = {}
        self.query_history = []
        
    async def build_knowledge_base(self, doc_ids: Optional[List[str]] = None):
        """
        지식 베이스 구축
        
        Args:
            doc_ids: 처리할 문서 ID 리스트 (None이면 모든 문서)
        """
        try:
            logger.info("지식 베이스 구축 시작...")
            
            # 처리된 문서 목록 가져오기
            processed_files = self.processor.get_processed_documents()
            
            for file_path in processed_files:
                try:
                    # JSON 파일에서 처리된 데이터 로드
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    doc_id = Path(file_path).stem
                    self.knowledge_base[doc_id] = data
                    
                    logger.info(f"지식 베이스에 추가: {doc_id}")
                    
                except Exception as e:
                    logger.error(f"지식 베이스 구축 중 오류: {file_path}, {e}")
            
            logger.info(f"지식 베이스 구축 완료: {len(self.knowledge_base)}개 문서")
            
        except Exception as e:
            logger.error(f"지식 베이스 구축 실패: {e}")
            raise
    
    async def enhanced_query(
        self, 
        query: str, 
        include_images: bool = True,
        include_tables: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        향상된 쿼리 처리 (VLM 지원)
        
        Args:
            query: 쿼리 텍스트
            include_images: 이미지 포함 여부
            include_tables: 테이블 포함 여부
            top_k: 반환할 결과 수
            
        Returns:
            쿼리 결과
        """
        try:
            logger.info(f"향상된 쿼리 시작: {query}")
            
            # 쿼리 히스토리에 추가
            self.query_history.append({
                "query": query,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # 기본 쿼리 실행
            basic_result = await self.processor.query_documents(
                query=query,
                top_k=top_k
            )
            
            # VLM 지원 결과 처리
            enhanced_result = {
                "query": query,
                "basic_results": basic_result,
                "vlm_enhanced": include_images,
                "multimodal_elements": {
                    "images": [],
                    "tables": [],
                    "equations": []
                },
                "knowledge_graph": self._extract_knowledge_graph(basic_result),
                "confidence_score": self._calculate_confidence(basic_result)
            }
            
            # 멀티모달 요소 추출
            if include_images or include_tables:
                enhanced_result["multimodal_elements"] = await self._extract_multimodal_elements(
                    basic_result, include_images, include_tables
                )
            
            logger.info(f"향상된 쿼리 완료: {query}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"향상된 쿼리 실패: {query}, 오류: {e}")
            raise
    
    async def _extract_multimodal_elements(
        self, 
        basic_result: Dict[str, Any],
        include_images: bool,
        include_tables: bool
    ) -> Dict[str, List]:
        """
        멀티모달 요소 추출
        
        Args:
            basic_result: 기본 쿼리 결과
            include_images: 이미지 포함 여부
            include_tables: 테이블 포함 여부
            
        Returns:
            멀티모달 요소 딕셔너리
        """
        multimodal_elements = {
            "images": [],
            "tables": [],
            "equations": []
        }
        
        try:
            # 기본 결과에서 멀티모달 요소 추출
            if "results" in basic_result:
                for result in basic_result["results"]:
                    # 이미지 요소 추출
                    if include_images and "images" in result:
                        multimodal_elements["images"].extend(result["images"])
                    
                    # 테이블 요소 추출
                    if include_tables and "tables" in result:
                        multimodal_elements["tables"].extend(result["tables"])
                    
                    # 수식 요소 추출
                    if "equations" in result:
                        multimodal_elements["equations"].extend(result["equations"])
            
            return multimodal_elements
            
        except Exception as e:
            logger.error(f"멀티모달 요소 추출 실패: {e}")
            return multimodal_elements
    
    def _extract_knowledge_graph(self, basic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        지식 그래프 추출
        
        Args:
            basic_result: 기본 쿼리 결과
            
        Returns:
            지식 그래프 딕셔너리
        """
        try:
            knowledge_graph = {
                "entities": [],
                "relationships": [],
                "concepts": []
            }
            
            # 결과에서 엔티티와 관계 추출
            if "results" in basic_result:
                for result in basic_result["results"]:
                    # 엔티티 추출
                    if "entities" in result:
                        knowledge_graph["entities"].extend(result["entities"])
                    
                    # 관계 추출
                    if "relationships" in result:
                        knowledge_graph["relationships"].extend(result["relationships"])
                    
                    # 개념 추출
                    if "concepts" in result:
                        knowledge_graph["concepts"].extend(result["concepts"])
            
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"지식 그래프 추출 실패: {e}")
            return {"entities": [], "relationships": [], "concepts": []}
    
    def _calculate_confidence(self, basic_result: Dict[str, Any]) -> float:
        """
        신뢰도 점수 계산
        
        Args:
            basic_result: 기본 쿼리 결과
            
        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        try:
            confidence = 0.0
            
            if "results" in basic_result and basic_result["results"]:
                # 결과 개수에 따른 기본 점수
                result_count = len(basic_result["results"])
                confidence += min(result_count * 0.1, 0.5)
                
                # 스코어가 있는 경우 평균 계산
                scores = []
                for result in basic_result["results"]:
                    if "score" in result:
                        scores.append(result["score"])
                
                if scores:
                    confidence += sum(scores) / len(scores) * 0.5
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {e}")
            return 0.0
    
    async def batch_query(
        self, 
        queries: List[str],
        include_images: bool = True,
        include_tables: bool = True
    ) -> List[Dict[str, Any]]:
        """
        배치 쿼리 처리
        
        Args:
            queries: 쿼리 리스트
            include_images: 이미지 포함 여부
            include_tables: 테이블 포함 여부
            
        Returns:
            쿼리 결과 리스트
        """
        try:
            logger.info(f"배치 쿼리 시작: {len(queries)}개 쿼리")
            
            results = []
            for query in queries:
                try:
                    result = await self.enhanced_query(
                        query=query,
                        include_images=include_images,
                        include_tables=include_tables
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"배치 쿼리 중 오류: {query}, {e}")
                    results.append({
                        "query": query,
                        "error": str(e),
                        "basic_results": {},
                        "vlm_enhanced": False,
                        "multimodal_elements": {"images": [], "tables": [], "equations": []},
                        "knowledge_graph": {"entities": [], "relationships": [], "concepts": []},
                        "confidence_score": 0.0
                    })
            
            logger.info(f"배치 쿼리 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"배치 쿼리 실패: {e}")
            raise
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        쿼리 통계 반환
        
        Returns:
            쿼리 통계 딕셔너리
        """
        try:
            total_queries = len(self.query_history)
            
            if total_queries == 0:
                return {
                    "total_queries": 0,
                    "average_confidence": 0.0,
                    "most_common_queries": [],
                    "query_timeline": []
                }
            
            # 평균 신뢰도 계산
            confidence_scores = []
            query_counts = {}
            
            for query_info in self.query_history:
                query = query_info["query"]
                query_counts[query] = query_counts.get(query, 0) + 1
            
            # 가장 많이 사용된 쿼리
            most_common = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total_queries": total_queries,
                "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                "most_common_queries": most_common,
                "query_timeline": self.query_history[-10:]  # 최근 10개 쿼리
            }
            
        except Exception as e:
            logger.error(f"쿼리 통계 계산 실패: {e}")
            return {
                "total_queries": 0,
                "average_confidence": 0.0,
                "most_common_queries": [],
                "query_timeline": []
            }

async def main():
    """VLM 쿼리 엔진 테스트"""
    # 프로세서 초기화
    processor = RAGAnythingProcessor()
    await processor.initialize()
    
    # VLM 쿼리 엔진 초기화
    vlm_engine = VLMQueryEngine(processor)
    
    # 지식 베이스 구축
    await vlm_engine.build_knowledge_base()
    
    # 테스트 쿼리
    test_queries = [
        "제조 공정의 흐름도를 보여주세요",
        "현재고 현황 테이블을 분석해주세요",
        "수율 현황 차트에서 어떤 패턴이 보이나요?",
        "품질 관리 프로세스를 설명해주세요"
    ]
    
    # 배치 쿼리 실행
    results = await vlm_engine.batch_query(test_queries)
    
    # 결과 출력
    for i, result in enumerate(results):
        print(f"\n=== 쿼리 {i+1}: {result['query']} ===")
        print(f"신뢰도: {result['confidence_score']:.2f}")
        print(f"VLM 지원: {result['vlm_enhanced']}")
        print(f"멀티모달 요소: {result['multimodal_elements']}")
    
    # 통계 출력
    stats = vlm_engine.get_query_statistics()
    print(f"\n=== 쿼리 통계 ===")
    print(f"총 쿼리 수: {stats['total_queries']}")
    print(f"평균 신뢰도: {stats['average_confidence']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
