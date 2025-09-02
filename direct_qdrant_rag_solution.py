#!/usr/bin/env python3
"""
직접 Qdrant 사용으로 rag-anything 우회 - 완전한 RAG 솔루션
"""

import asyncio
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectQdrantRAG:
    """직접 Qdrant 사용 RAG 시스템"""
    
    def __init__(self, collection_name: str = "manufacturing_docs"):
        self.collection_name = collection_name
        
        # 환경 변수 설정
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # Qdrant 클라이언트 초기화
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=self.qdrant_url)
            print(f"✅ Qdrant 연결 성공: {self.qdrant_url}")
        except Exception as e:
            print(f"❌ Qdrant 연결 실패: {e}")
            raise
        
        # OpenAI 임베딩 모델
        try:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key,
                openai_api_base=self.openai_base_url
            )
            print("✅ OpenAI 임베딩 모델 초기화 완료")
        except Exception as e:
            print(f"❌ OpenAI 임베딩 모델 초기화 실패: {e}")
            raise
        
        # Docling 파서
        try:
            from docling import DoclingParser
            self.parser = DoclingParser()
            print("✅ Docling 파서 초기화 완료")
        except Exception as e:
            print(f"❌ Docling 파서 초기화 실패: {e}")
            raise
        
        # 컬렉션 설정
        self._setup_collection()
    
    def _setup_collection(self):
        """컬렉션 설정"""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"✅ 컬렉션 생성: {self.collection_name}")
        except Exception as e:
            print(f"ℹ️ 컬렉션이 이미 존재합니다: {e}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """텍스트를 청크로 분할 (오버랩 포함)"""
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 공백이나 문장 부호를 찾아서 자르기
                last_space = text.rfind(' ', start, end)
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                cut_point = max(last_space, last_period, last_newline)
                if cut_point > start:
                    end = cut_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 오버랩을 고려한 다음 시작점
            start = max(start + 1, end - overlap)
        
        return chunks
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """문서 처리 및 Qdrant 저장"""
        try:
            print(f"📄 문서 처리: {file_path}")
            
            # 파일 존재 확인
            if not Path(file_path).exists():
                return {"status": "error", "message": f"파일이 존재하지 않습니다: {file_path}"}
            
            # 1. Docling으로 파싱
            print("   🔄 Docling 파싱 중...")
            result = await self.parser.parse_document(file_path)
            
            if not result or "content" not in result:
                return {"status": "error", "message": "파싱 실패 - content가 없습니다"}
            
            # 2. 텍스트 청킹
            content = result["content"]
            chunks = self._split_text_into_chunks(content)
            
            print(f"   청크 수: {len(chunks)}")
            
            if not chunks:
                return {"status": "error", "message": "청크가 생성되지 않았습니다"}
            
            # 3. 임베딩 생성 및 저장
            print("   🔄 임베딩 생성 중...")
            points = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # 임베딩 생성
                    embedding = self.embeddings.embed_query(chunk)
                    
                    # 포인트 생성
                    from qdrant_client.models import PointStruct
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,  # 통일된 필드명
                            "doc_id": Path(file_path).stem,
                            "chunk_index": i,
                            "file_path": file_path,
                            "source": "docling",
                            "chunk_size": len(chunk)
                        }
                    )
                    points.append(point)
                    
                    if (i + 1) % 10 == 0:
                        print(f"     {i + 1}/{len(chunks)} 청크 처리 완료")
                        
                except Exception as e:
                    print(f"     청크 {i} 처리 실패: {e}")
                    continue
            
            if not points:
                return {"status": "error", "message": "저장할 포인트가 없습니다"}
            
            # 4. Qdrant에 저장
            print("   🔄 Qdrant에 저장 중...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✅ {len(points)}개 청크 저장 완료")
            
            return {
                "status": "success",
                "chunks": len(chunks),
                "points_stored": len(points),
                "file_path": file_path,
                "doc_id": Path(file_path).stem
            }
            
        except Exception as e:
            print(f"❌ 문서 처리 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def query(self, question: str, top_k: int = 5, score_threshold: float = 0.0) -> Dict[str, Any]:
        """쿼리 실행"""
        try:
            print(f"🔍 쿼리: {question}")
            
            # 1. 쿼리 임베딩 생성
            print("   🔄 쿼리 임베딩 생성 중...")
            query_embedding = self.embeddings.embed_query(question)
            
            # 2. Qdrant에서 검색
            print("   🔄 Qdrant 검색 중...")
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            print(f"   검색 결과 수: {len(search_results)}")
            
            # 3. 컨텍스트 생성
            contexts = []
            for i, result in enumerate(search_results):
                context = {
                    "rank": i + 1,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "doc_id": result.payload.get("doc_id", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "file_path": result.payload.get("file_path", "")
                }
                contexts.append(context)
                print(f"   결과 {i+1}: 점수={result.score:.4f}, 텍스트={result.payload.get('text', '')[:50]}...")
            
            # 4. LLM으로 답변 생성
            if contexts:
                print("   🔄 LLM 답변 생성 중...")
                context_text = "\n\n".join([ctx["text"] for ctx in contexts])
                
                import openai
                client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url
                )
                
                prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context_text}

질문: {question}

답변:"""
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0
                )
                
                answer = response.choices[0].message.content
            else:
                answer = "(no-context)"
                print("   ⚠️ 검색 결과가 없어 (no-context) 반환")
            
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "total_results": len(search_results),
                "top_k": top_k,
                "score_threshold": score_threshold
            }
            
        except Exception as e:
            print(f"❌ 쿼리 실패: {e}")
            return {
                "question": question,
                "answer": f"오류 발생: {e}",
                "contexts": [],
                "total_results": 0,
                "error": str(e)
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self) -> Dict[str, Any]:
        """컬렉션 초기화"""
        try:
            self.client.delete_collection(self.collection_name)
            self._setup_collection()
            return {"status": "success", "message": "컬렉션이 초기화되었습니다"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# 사용 예시
async def main():
    """메인 함수 - 사용 예시"""
    print("🚀 **직접 Qdrant RAG 시스템 테스트**")
    print("=" * 60)
    
    try:
        # RAG 시스템 초기화
        rag = DirectQdrantRAG()
        
        # 컬렉션 정보 확인
        collection_info = rag.get_collection_info()
        print(f"📊 컬렉션 정보: {collection_info}")
        
        # 문서 처리 (선택적)
        data_dir = Path("data")
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            if pdf_files:
                test_file = pdf_files[0]
                print(f"\n📄 테스트 파일: {test_file}")
                
                # 문서 처리
                result = await rag.process_document(str(test_file))
                print(f"문서 처리 결과: {result}")
                
                if result["status"] == "success":
                    # 쿼리 테스트
                    test_queries = [
                        "매출액은 얼마인가요?",
                        "주요 내용을 요약해주세요",
                        "제조업 관련 정보는 무엇인가요?"
                    ]
                    
                    for query in test_queries:
                        print(f"\n🔍 쿼리 테스트: {query}")
                        query_result = await rag.query(query, top_k=5)
                        print(f"답변: {query_result['answer']}")
                        print(f"검색 결과 수: {query_result['total_results']}")
        else:
            print("📁 data 디렉토리가 없습니다. 쿼리만 테스트합니다.")
        
        # 빈 컬렉션에서 쿼리 테스트
        print(f"\n🔍 빈 컬렉션 쿼리 테스트")
        query_result = await rag.query("테스트 질문", top_k=5)
        print(f"결과: {query_result['answer']}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
