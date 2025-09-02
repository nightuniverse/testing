#!/usr/bin/env python3
"""
RAG-Anything과 Qdrant 연동 없이 기존 결과를 활용한 쿼리 솔루션들
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlternativeQuerySolutions:
    """기존 결과를 활용한 다양한 쿼리 솔루션"""
    
    def __init__(self, results_dir: str = "test_excels/test_results"):
        self.results_dir = Path(results_dir)
        self.knowledge_graphs_dir = self.results_dir / "knowledge_graphs"
        self.docling_dir = self.results_dir / "docling_parsing"
        self.image_modal_dir = self.results_dir / "image_modal_results"
        self.rag_processing_dir = self.results_dir / "rag_processing"
        
        # OpenAI API 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    async def solution1_direct_json_query(self, query: str, file_name: str = None) -> Dict[str, Any]:
        """솔루션 1: 직접 JSON 결과에서 검색"""
        print("🔍 **솔루션 1: 직접 JSON 결과에서 검색**")
        print("=" * 60)
        
        try:
            # 모든 결과 파일 로드
            all_data = {}
            
            # Knowledge Graph 로드
            if self.knowledge_graphs_dir.exists():
                for kg_file in self.knowledge_graphs_dir.glob("*.json"):
                    if file_name is None or file_name in kg_file.name:
                        with open(kg_file, 'r', encoding='utf-8') as f:
                            kg_data = json.load(f)
                            all_data[f"kg_{kg_file.stem}"] = kg_data
            
            # Docling 파싱 결과 로드
            if self.docling_dir.exists():
                for docling_file in self.docling_dir.glob("*.json"):
                    if file_name is None or file_name in docling_file.name:
                        with open(docling_file, 'r', encoding='utf-8') as f:
                            docling_data = json.load(f)
                            all_data[f"docling_{docling_file.stem}"] = docling_data
            
            # 이미지 모달 결과 로드
            if self.image_modal_dir.exists():
                for image_file in self.image_modal_dir.glob("*.json"):
                    if file_name is None or file_name in image_file.name:
                        with open(image_file, 'r', encoding='utf-8') as f:
                            image_data = json.load(f)
                            all_data[f"image_{image_file.stem}"] = image_data
            
            # 검색 실행
            search_results = []
            for data_type, data in all_data.items():
                matches = self._search_in_json(data, query)
                if matches:
                    search_results.extend([{
                        "source": data_type,
                        "match": match,
                        "context": self._extract_context(data, match)
                    } for match in matches])
            
            # LLM으로 답변 생성
            if search_results:
                answer = await self._generate_answer_from_results(query, search_results)
            else:
                answer = "(no-context)"
            
            return {
                "query": query,
                "answer": answer,
                "search_results": search_results,
                "total_matches": len(search_results)
            }
            
        except Exception as e:
            print(f"❌ 직접 JSON 검색 실패: {e}")
            return {"error": str(e)}
    
    def _search_in_json(self, data: Any, query: str) -> List[str]:
        """JSON 데이터에서 검색"""
        matches = []
        
        def search_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    search_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    search_recursive(item, current_path)
            elif isinstance(obj, str):
                if query.lower() in obj.lower():
                    matches.append(f"{path}: {obj[:200]}...")
        
        search_recursive(data)
        return matches
    
    def _extract_context(self, data: Any, match: str) -> str:
        """매치된 항목의 컨텍스트 추출"""
        try:
            # 간단한 컨텍스트 추출
            if ":" in match:
                content = match.split(":", 1)[1]
                return content[:500] + "..." if len(content) > 500 else content
            return match
        except:
            return match
    
    async def solution2_semantic_search(self, query: str, file_name: str = None) -> Dict[str, Any]:
        """솔루션 2: 의미적 검색 (임베딩 기반)"""
        print("🔍 **솔루션 2: 의미적 검색 (임베딩 기반)**")
        print("=" * 60)
        
        try:
            from langchain_openai import OpenAIEmbeddings
            import numpy as np
            
            # 임베딩 모델 초기화
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key,
                openai_api_base=self.openai_base_url
            )
            
            # 쿼리 임베딩 생성
            query_embedding = embeddings.embed_query(query)
            
            # 모든 텍스트 청크 수집
            text_chunks = []
            
            # Knowledge Graph에서 텍스트 추출
            if self.knowledge_graphs_dir.exists():
                for kg_file in self.knowledge_graphs_dir.glob("*.json"):
                    if file_name is None or file_name in kg_file.name:
                        with open(kg_file, 'r', encoding='utf-8') as f:
                            kg_data = json.load(f)
                            chunks = self._extract_text_from_kg(kg_data)
                            text_chunks.extend(chunks)
            
            # Docling 결과에서 텍스트 추출
            if self.docling_dir.exists():
                for docling_file in self.docling_dir.glob("*.json"):
                    if file_name is None or file_name in docling_file.name:
                        with open(docling_file, 'r', encoding='utf-8') as f:
                            docling_data = json.load(f)
                            chunks = self._extract_text_from_docling(docling_data)
                            text_chunks.extend(chunks)
            
            if not text_chunks:
                return {"query": query, "answer": "(no-context)", "search_results": []}
            
            # 각 청크의 임베딩 생성 및 유사도 계산
            similarities = []
            for i, chunk in enumerate(text_chunks):
                try:
                    chunk_embedding = embeddings.embed_query(chunk["text"])
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append({
                        "index": i,
                        "chunk": chunk,
                        "similarity": similarity
                    })
                except Exception as e:
                    print(f"청크 {i} 임베딩 실패: {e}")
                    continue
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 상위 결과 선택
            top_results = similarities[:5]
            
            # LLM으로 답변 생성
            if top_results:
                answer = await self._generate_answer_from_semantic_results(query, top_results)
            else:
                answer = "(no-context)"
            
            return {
                "query": query,
                "answer": answer,
                "search_results": top_results,
                "total_chunks": len(text_chunks),
                "top_similarity": top_results[0]["similarity"] if top_results else 0
            }
            
        except Exception as e:
            print(f"❌ 의미적 검색 실패: {e}")
            return {"error": str(e)}
    
    def _extract_text_from_kg(self, kg_data: Dict) -> List[Dict]:
        """Knowledge Graph에서 텍스트 추출"""
        chunks = []
        
        if "nodes" in kg_data:
            for node in kg_data["nodes"]:
                if "properties" in node:
                    props = node["properties"]
                    text_parts = []
                    
                    # 다양한 텍스트 필드 추출
                    for key, value in props.items():
                        if isinstance(value, str) and value.strip():
                            text_parts.append(f"{key}: {value}")
                    
                    if text_parts:
                        chunks.append({
                            "text": " | ".join(text_parts),
                            "source": "knowledge_graph",
                            "node_id": node.get("id", ""),
                            "node_type": node.get("type", "")
                        })
        
        return chunks
    
    def _extract_text_from_docling(self, docling_data: Dict) -> List[Dict]:
        """Docling 결과에서 텍스트 추출"""
        chunks = []
        
        # 테이블 데이터 추출
        if "tables" in docling_data:
            for table in docling_data["tables"]:
                if "rows" in table:
                    for row in table["rows"]:
                        if isinstance(row, list):
                            text = " | ".join([str(cell) for cell in row if cell])
                            if text.strip():
                                chunks.append({
                                    "text": text,
                                    "source": "docling_table",
                                    "table_index": table.get("table_index", 0)
                                })
        
        # 텍스트 콘텐츠 추출
        if "content" in docling_data:
            content = docling_data["content"]
            if isinstance(content, str) and content.strip():
                # 긴 텍스트를 청크로 분할
                words = content.split()
                chunk_size = 100
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i:i+chunk_size])
                    chunks.append({
                        "text": chunk_text,
                        "source": "docling_content",
                        "chunk_index": i // chunk_size
                    })
        
        return chunks
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    async def solution3_hybrid_search(self, query: str, file_name: str = None) -> Dict[str, Any]:
        """솔루션 3: 하이브리드 검색 (키워드 + 의미적)"""
        print("🔍 **솔루션 3: 하이브리드 검색 (키워드 + 의미적)**")
        print("=" * 60)
        
        try:
            # 1. 키워드 검색
            keyword_results = await self.solution1_direct_json_query(query, file_name)
            
            # 2. 의미적 검색
            semantic_results = await self.solution2_semantic_search(query, file_name)
            
            # 3. 결과 결합
            combined_results = {
                "keyword_matches": keyword_results.get("search_results", []),
                "semantic_matches": semantic_results.get("search_results", []),
                "keyword_answer": keyword_results.get("answer", ""),
                "semantic_answer": semantic_results.get("answer", "")
            }
            
            # 4. 최종 답변 생성
            final_answer = await self._generate_hybrid_answer(query, combined_results)
            
            return {
                "query": query,
                "answer": final_answer,
                "keyword_results": keyword_results,
                "semantic_results": semantic_results,
                "combined_results": combined_results
            }
            
        except Exception as e:
            print(f"❌ 하이브리드 검색 실패: {e}")
            return {"error": str(e)}
    
    async def solution4_knowledge_graph_query(self, query: str, file_name: str = None) -> Dict[str, Any]:
        """솔루션 4: Knowledge Graph 기반 쿼리"""
        print("🔍 **솔루션 4: Knowledge Graph 기반 쿼리**")
        print("=" * 60)
        
        try:
            # Knowledge Graph 로드
            kg_data = {}
            if self.knowledge_graphs_dir.exists():
                for kg_file in self.knowledge_graphs_dir.glob("*.json"):
                    if file_name is None or file_name in kg_file.name:
                        with open(kg_file, 'r', encoding='utf-8') as f:
                            kg_data[kg_file.stem] = json.load(f)
            
            if not kg_data:
                return {"query": query, "answer": "(no-context)", "kg_analysis": {}}
            
            # Knowledge Graph 분석
            kg_analysis = self._analyze_knowledge_graph(kg_data, query)
            
            # LLM으로 답변 생성
            answer = await self._generate_kg_answer(query, kg_analysis)
            
            return {
                "query": query,
                "answer": answer,
                "kg_analysis": kg_analysis,
                "total_nodes": sum(len(kg.get("nodes", [])) for kg in kg_data.values()),
                "total_edges": sum(len(kg.get("edges", [])) for kg in kg_data.values())
            }
            
        except Exception as e:
            print(f"❌ Knowledge Graph 쿼리 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_knowledge_graph(self, kg_data: Dict, query: str) -> Dict[str, Any]:
        """Knowledge Graph 분석"""
        analysis = {
            "relevant_nodes": [],
            "node_types": {},
            "image_references": [],
            "table_references": [],
            "text_content": []
        }
        
        for kg_name, kg in kg_data.items():
            if "nodes" in kg:
                for node in kg["nodes"]:
                    # 노드 타입 통계
                    node_type = node.get("type", "unknown")
                    analysis["node_types"][node_type] = analysis["node_types"].get(node_type, 0) + 1
                    
                    # 쿼리와 관련된 노드 찾기
                    if "properties" in node:
                        props = node["properties"]
                        node_text = str(props).lower()
                        if any(keyword in node_text for keyword in query.lower().split()):
                            analysis["relevant_nodes"].append({
                                "kg_name": kg_name,
                                "node_id": node.get("id", ""),
                                "node_type": node_type,
                                "properties": props
                            })
                    
                    # 이미지 참조
                    if node_type == "assembly_diagram" or "image" in node_type:
                        if "properties" in node and "image_ref" in node["properties"]:
                            analysis["image_references"].append(node["properties"]["image_ref"])
        
        return analysis
    
    async def _generate_answer_from_results(self, query: str, search_results: List[Dict]) -> str:
        """검색 결과로부터 답변 생성"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            
            # 컨텍스트 구성
            context_parts = []
            for result in search_results[:5]:  # 상위 5개 결과만 사용
                context_parts.append(f"출처: {result['source']}\n내용: {result['context']}")
            
            context_text = "\n\n".join(context_parts)
            
            prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context_text}

질문: {query}

답변:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"답변 생성 실패: {e}")
            return f"오류 발생: {e}"
    
    async def _generate_answer_from_semantic_results(self, query: str, semantic_results: List[Dict]) -> str:
        """의미적 검색 결과로부터 답변 생성"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            
            # 컨텍스트 구성
            context_parts = []
            for result in semantic_results:
                chunk = result["chunk"]
                context_parts.append(f"출처: {chunk['source']}\n유사도: {result['similarity']:.4f}\n내용: {chunk['text']}")
            
            context_text = "\n\n".join(context_parts)
            
            prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요. 유사도 점수가 높은 내용을 우선적으로 참고하세요.

컨텍스트:
{context_text}

질문: {query}

답변:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"답변 생성 실패: {e}")
            return f"오류 발생: {e}"
    
    async def _generate_hybrid_answer(self, query: str, combined_results: Dict) -> str:
        """하이브리드 검색 결과로부터 답변 생성"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            
            # 키워드와 의미적 검색 결과 결합
            keyword_context = combined_results.get("keyword_answer", "")
            semantic_context = combined_results.get("semantic_answer", "")
            
            prompt = f"""키워드 검색과 의미적 검색 결과를 종합하여 질문에 답변해주세요.

키워드 검색 결과:
{keyword_context}

의미적 검색 결과:
{semantic_context}

질문: {query}

통합 답변:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"하이브리드 답변 생성 실패: {e}")
            return f"오류 발생: {e}"
    
    async def _generate_kg_answer(self, query: str, kg_analysis: Dict) -> str:
        """Knowledge Graph 분석 결과로부터 답변 생성"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            
            # Knowledge Graph 분석 정보 구성
            kg_info = f"""
노드 타입 분포: {kg_analysis.get('node_types', {})}
관련 노드 수: {len(kg_analysis.get('relevant_nodes', []))}
이미지 참조 수: {len(kg_analysis.get('image_references', []))}
"""
            
            relevant_nodes_info = ""
            for node in kg_analysis.get('relevant_nodes', [])[:3]:  # 상위 3개만
                relevant_nodes_info += f"- {node['node_type']}: {str(node['properties'])[:200]}...\n"
            
            prompt = f"""Knowledge Graph 분석 결과를 바탕으로 질문에 답변해주세요.

Knowledge Graph 정보:
{kg_info}

관련 노드:
{relevant_nodes_info}

질문: {query}

답변:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Knowledge Graph 답변 생성 실패: {e}")
            return f"오류 발생: {e}"

# 사용 예시
async def main():
    """메인 함수 - 사용 예시"""
    print("🚀 **대안 쿼리 솔루션 테스트**")
    print("=" * 60)
    
    try:
        # 솔루션 초기화
        solutions = AlternativeQuerySolutions()
        
        # 테스트 쿼리들
        test_queries = [
            "조립 다이어그램은 무엇인가요?",
            "수입검사 관련 정보를 알려주세요",
            "이미지 파일의 위치는 어디인가요?",
            "테이블 데이터를 요약해주세요"
        ]
        
        for query in test_queries:
            print(f"\n🔍 쿼리: {query}")
            
            # 솔루션 1: 직접 JSON 검색
            print("\n1️⃣ 직접 JSON 검색:")
            result1 = await solutions.solution1_direct_json_query(query)
            print(f"   답변: {result1.get('answer', '오류')[:100]}...")
            
            # 솔루션 2: 의미적 검색
            print("\n2️⃣ 의미적 검색:")
            result2 = await solutions.solution2_semantic_search(query)
            print(f"   답변: {result2.get('answer', '오류')[:100]}...")
            
            # 솔루션 3: 하이브리드 검색
            print("\n3️⃣ 하이브리드 검색:")
            result3 = await solutions.solution3_hybrid_search(query)
            print(f"   답변: {result3.get('answer', '오류')[:100]}...")
            
            # 솔루션 4: Knowledge Graph 쿼리
            print("\n4️⃣ Knowledge Graph 쿼리:")
            result4 = await solutions.solution4_knowledge_graph_query(query)
            print(f"   답변: {result4.get('answer', '오류')[:100]}...")
            
            print("\n" + "-" * 60)
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())
