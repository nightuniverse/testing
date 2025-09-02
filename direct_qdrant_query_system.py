#!/usr/bin/env python3
"""
직접 Qdrant를 사용하는 완전한 쿼리 시스템
- 기존 파이프라인 결과 활용
- rag-anything 우회
- 강력한 검색 및 응답 생성
"""

import asyncio
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    id: int
    score: float
    text: str
    source: str
    file_name: str
    metadata: Dict[str, Any]

class MockEmbeddings:
    """Mock 임베딩 클래스 - 해시 기반"""
    
    def __init__(self, dimension=1536):
        self.dimension = dimension
    
    def embed_query(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        vector = []
        for i in range(self.dimension):
            start_idx = (i * 4) % len(hash_hex)
            end_idx = start_idx + 4
            if end_idx > len(hash_hex):
                end_idx = len(hash_hex)
            
            hex_part = hash_hex[start_idx:end_idx]
            value = float(int(hex_part, 16)) / (16 ** len(hex_part))
            vector.append(value)
        
        return vector

class MockLLM:
    """Mock LLM 클래스"""
    
    def __call__(self, prompt: str, context: str = "") -> str:
        if context:
            return f"Mock LLM 응답 (컨텍스트 포함):\n\n컨텍스트: {context[:200]}...\n\n질문: {prompt}\n\n답변: 이는 Mock LLM의 응답입니다. 실제 OpenAI API를 사용하면 더 정확한 답변을 받을 수 있습니다."
        else:
            return f"Mock LLM 응답: {prompt[:100]}... (컨텍스트 없음)"

class DirectQdrantQuerySystem:
    """직접 Qdrant를 사용하는 쿼리 시스템"""
    
    def __init__(self):
        self.qdrant_url = "http://localhost:6333"
        self.collection_name = "manufacturing_docs"
        self.embeddings = MockEmbeddings()
        self.llm = MockLLM()
        
        # 검색 설정
        self.default_limit = 10
        self.default_score_threshold = 0.0
        
        # 결과 저장 디렉토리
        self.output_dir = Path("direct_qdrant_query_results")
        self.output_dir.mkdir(exist_ok=True)
    
    async def search_documents(self, query: str, limit: int = None, score_threshold: float = None) -> List[SearchResult]:
        """문서 검색"""
        try:
            from qdrant_client import QdrantClient
            
            client = QdrantClient(url=self.qdrant_url)
            
            # 검색 파라미터 설정
            limit = limit or self.default_limit
            score_threshold = score_threshold or self.default_score_threshold
            
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # Qdrant 검색
            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # 결과 변환
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=result.id,
                    score=result.score,
                    text=result.payload.get('text', ''),
                    source=result.payload.get('source', 'unknown'),
                    file_name=result.payload.get('file_name', 'unknown'),
                    metadata={k: v for k, v in result.payload.items() if k not in ['text', 'source', 'file_name']}
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """검색 결과를 바탕으로 응답 생성"""
        if not search_results:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        # 컨텍스트 구성
        context_parts = []
        for i, result in enumerate(search_results[:5]):  # 상위 5개 결과만 사용
            context_parts.append(f"[{i+1}] {result.text}")
        
        context = "\n\n".join(context_parts)
        
        # LLM을 사용하여 응답 생성
        response = self.llm(query, context)
        
        return response
    
    def analyze_search_results(self, search_results: List[SearchResult]) -> Dict[str, Any]:
        """검색 결과 분석"""
        if not search_results:
            return {"error": "검색 결과가 없습니다."}
        
        # 소스별 통계
        source_stats = {}
        for result in search_results:
            source = result.source
            if source not in source_stats:
                source_stats[source] = {
                    "count": 0,
                    "avg_score": 0.0,
                    "files": set()
                }
            
            source_stats[source]["count"] += 1
            source_stats[source]["avg_score"] += result.score
            source_stats[source]["files"].add(result.file_name)
        
        # 평균 점수 계산
        for source in source_stats:
            count = source_stats[source]["count"]
            source_stats[source]["avg_score"] /= count
            source_stats[source]["files"] = list(source_stats[source]["files"])
        
        # 전체 통계
        total_results = len(search_results)
        avg_score = sum(r.score for r in search_results) / total_results
        score_range = (min(r.score for r in search_results), max(r.score for r in search_results))
        
        return {
            "total_results": total_results,
            "average_score": avg_score,
            "score_range": score_range,
            "source_statistics": source_stats,
            "top_results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "source": r.source,
                    "file_name": r.file_name,
                    "text_preview": r.text[:100] + "..." if len(r.text) > 100 else r.text
                }
                for r in search_results[:5]
            ]
        }
    
    async def query(self, query: str, limit: int = None, score_threshold: float = None, 
                   include_analysis: bool = True) -> Dict[str, Any]:
        """완전한 쿼리 실행"""
        print(f"🔍 **쿼리: '{query}'**")
        print("=" * 60)
        
        # 1. 문서 검색
        search_results = await self.search_documents(query, limit, score_threshold)
        
        if not search_results:
            return {
                "query": query,
                "response": "죄송합니다. 관련 정보를 찾을 수 없습니다.",
                "search_results": [],
                "analysis": {"error": "검색 결과가 없습니다."}
            }
        
        # 2. 응답 생성
        response = self.generate_response(query, search_results)
        
        # 3. 결과 분석
        analysis = self.analyze_search_results(search_results) if include_analysis else {}
        
        # 4. 결과 출력
        print(f"📊 검색 결과: {len(search_results)}개")
        print(f"📈 평균 점수: {analysis.get('average_score', 0):.4f}")
        print(f"📁 소스별 통계:")
        for source, stats in analysis.get('source_statistics', {}).items():
            print(f"   - {source}: {stats['count']}개 (평균 점수: {stats['avg_score']:.4f})")
        
        print(f"\n💬 **응답:**")
        print(response)
        
        # 5. 결과 반환
        result = {
            "query": query,
            "response": response,
            "search_results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "text": r.text,
                    "source": r.source,
                    "file_name": r.file_name,
                    "metadata": r.metadata
                }
                for r in search_results
            ],
            "analysis": analysis
        }
        
        return result
    
    async def batch_query(self, queries: List[str], save_results: bool = True) -> List[Dict[str, Any]]:
        """배치 쿼리 실행"""
        print(f"🚀 **배치 쿼리 실행 ({len(queries)}개)**")
        print("=" * 60)
        
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] 쿼리 처리 중...")
            result = await self.query(query)
            results.append(result)
        
        # 결과 저장
        if save_results:
            timestamp = asyncio.get_event_loop().time()
            output_file = self.output_dir / f"batch_query_results_{int(timestamp)}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 배치 쿼리 결과 저장: {output_file}")
        
        return results
    
    async def interactive_query(self):
        """대화형 쿼리 모드"""
        print("🎯 **대화형 쿼리 모드**")
        print("=" * 60)
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print()
        
        while True:
            try:
                query = input("질문을 입력하세요: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print("대화형 모드를 종료합니다.")
                    break
                
                if not query:
                    continue
                
                await self.query(query)
                print("\n" + "-" * 60)
                
            except KeyboardInterrupt:
                print("\n대화형 모드를 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")

# 테스트 쿼리들
TEST_QUERIES = [
    "조립 작업 과정에 대해 설명해주세요",
    "수입검사는 어떻게 진행되나요?",
    "제품 품질 관리 방법은 무엇인가요?",
    "생산 공정에서 주의해야 할 점은?",
    "검사 기준과 방법을 알려주세요",
    "부품 관리 시스템은 어떻게 운영되나요?",
    "안전 관리 규정은 무엇인가요?",
    "불량품 처리 절차는 어떻게 되나요?",
    "생산성 향상을 위한 방법은?",
    "품질 개선 방안을 제시해주세요"
]

async def main():
    """메인 함수"""
    try:
        # 쿼리 시스템 초기화
        query_system = DirectQdrantQuerySystem()
        
        print("🚀 **직접 Qdrant 쿼리 시스템 시작**")
        print("=" * 60)
        
        # 사용자 선택
        print("실행 모드를 선택하세요:")
        print("1. 단일 쿼리 테스트")
        print("2. 배치 쿼리 테스트")
        print("3. 대화형 모드")
        print("4. 테스트 쿼리 실행")
        
        choice = input("선택 (1-4): ").strip()
        
        if choice == "1":
            query = input("쿼리를 입력하세요: ").strip()
            if query:
                await query_system.query(query)
        
        elif choice == "2":
            queries = []
            print("쿼리들을 입력하세요 (빈 줄로 종료):")
            while True:
                query = input().strip()
                if not query:
                    break
                queries.append(query)
            
            if queries:
                await query_system.batch_query(queries)
        
        elif choice == "3":
            await query_system.interactive_query()
        
        elif choice == "4":
            print(f"테스트 쿼리 {len(TEST_QUERIES)}개 실행...")
            await query_system.batch_query(TEST_QUERIES)
        
        else:
            print("잘못된 선택입니다.")
        
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
