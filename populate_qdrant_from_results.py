#!/usr/bin/env python3
"""
기존 파이프라인 결과를 Qdrant에 저장 (Mock 임베딩 사용)
"""

import asyncio
import json
import os
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockEmbeddings:
    """Mock 임베딩 클래스 - 해시 기반"""
    
    def __init__(self, dimension=1536):
        self.dimension = dimension
    
    def embed_query(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환 (해시 기반)"""
        # 텍스트를 해시로 변환
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # 해시를 1536차원 벡터로 변환
        vector = []
        for i in range(self.dimension):
            # 해시의 각 부분을 사용하여 벡터 생성
            start_idx = (i * 4) % len(hash_hex)
            end_idx = start_idx + 4
            if end_idx > len(hash_hex):
                end_idx = len(hash_hex)
            
            hex_part = hash_hex[start_idx:end_idx]
            # 16진수를 0-1 범위의 실수로 변환
            value = float(int(hex_part, 16)) / (16 ** len(hex_part))
            vector.append(value)
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 임베딩 벡터로 변환"""
        return [self.embed_query(text) for text in texts]

class PopulateQdrantFromResults:
    """기존 파이프라인 결과를 Qdrant에 저장"""
    
    def __init__(self):
        self.results_dir = Path("test_excels/test_results")
        self.knowledge_graphs_dir = self.results_dir / "knowledge_graphs"
        self.docling_dir = self.results_dir / "docling_parsing"
        self.image_modal_dir = self.results_dir / "image_modal_results"
        
        # Qdrant 설정
        self.qdrant_url = "http://localhost:6333"
        self.collection_name = "manufacturing_docs"
        
        # Mock 임베딩 초기화
        self.embeddings = MockEmbeddings(dimension=1536)
        
        # 포인트 카운터
        self.point_counter = 0
    
    def generate_safe_id(self, prefix: str, filename: str, suffix: str = "") -> int:
        """안전한 포인트 ID 생성 (정수)"""
        self.point_counter += 1
        return self.point_counter
    
    async def populate_qdrant(self):
        """Qdrant에 데이터 저장"""
        print("🚀 **기존 결과를 Qdrant에 저장 (Mock 임베딩 사용)**")
        print("=" * 60)
        
        try:
            # 1. Qdrant 클라이언트 초기화
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            client = QdrantClient(url=self.qdrant_url)
            print(f"✅ Qdrant 연결 성공: {self.qdrant_url}")
            
            # 2. 컬렉션 생성
            try:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                print(f"✅ 컬렉션 생성: {self.collection_name}")
            except Exception as e:
                print(f"ℹ️ 컬렉션이 이미 존재합니다: {e}")
            
            # 3. Mock 임베딩 모델 초기화
            print("✅ Mock 임베딩 모델 초기화 완료")
            
            # 4. Knowledge Graph 데이터 처리
            await self.process_knowledge_graphs(client, PointStruct)
            
            # 5. Docling 파싱 결과 처리
            await self.process_docling_results(client, PointStruct)
            
            # 6. 이미지 모달 결과 처리
            await self.process_image_modal_results(client, PointStruct)
            
            # 7. 결과 요약
            await self.print_population_summary(client)
            
        except Exception as e:
            print(f"❌ Qdrant 저장 실패: {e}")
            import traceback
            traceback.print_exc()
    
    async def process_knowledge_graphs(self, client, PointStruct):
        """Knowledge Graph 데이터 처리"""
        print("\n📊 **Knowledge Graph 데이터 처리**")
        
        if not self.knowledge_graphs_dir.exists():
            print("   ⚠️ Knowledge Graph 디렉토리가 없습니다.")
            return
        
        points = []
        kg_files = list(self.knowledge_graphs_dir.glob("*.json"))
        
        for kg_file in kg_files:
            print(f"   📄 처리 중: {kg_file.name}")
            
            try:
                with open(kg_file, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)
                
                # 노드 데이터 추출
                if "nodes" in kg_data:
                    for node in kg_data["nodes"]:
                        if "properties" in node:
                            props = node["properties"]
                            
                            # 텍스트 데이터 추출
                            text_parts = []
                            for key, value in props.items():
                                if isinstance(value, str) and value.strip():
                                    text_parts.append(f"{key}: {value}")
                            
                            if text_parts:
                                text_content = " | ".join(text_parts)
                                
                                # Mock 임베딩 생성
                                try:
                                    embedding = self.embeddings.embed_query(text_content)
                                    
                                    # 안전한 ID 생성
                                    point_id = self.generate_safe_id("kg", kg_file.stem, node.get('id', 'node'))
                                    
                                    # 포인트 생성
                                    point = PointStruct(
                                        id=point_id,
                                        vector=embedding,
                                        payload={
                                            "text": text_content,
                                            "source": "knowledge_graph",
                                            "file_name": kg_file.name,
                                            "node_id": node.get("id", ""),
                                            "node_type": node.get("type", ""),
                                            "properties": props
                                        }
                                    )
                                    points.append(point)
                                    
                                except Exception as e:
                                    print(f"     임베딩 생성 실패: {e}")
                                    continue
                
                print(f"     ✅ {len([p for p in points if kg_file.name in p.payload.get('file_name', '')])}개 포인트 생성")
                
            except Exception as e:
                print(f"     ❌ 파일 처리 실패: {e}")
                continue
        
        # Qdrant에 저장
        if points:
            client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"   💾 {len(points)}개 Knowledge Graph 포인트 저장 완료")
    
    async def process_docling_results(self, client, PointStruct):
        """Docling 파싱 결과 처리"""
        print("\n📊 **Docling 파싱 결과 처리**")
        
        if not self.docling_dir.exists():
            print("   ⚠️ Docling 디렉토리가 없습니다.")
            return
        
        points = []
        docling_files = list(self.docling_dir.glob("*.json"))
        
        for docling_file in docling_files:
            print(f"   📄 처리 중: {docling_file.name}")
            
            try:
                with open(docling_file, 'r', encoding='utf-8') as f:
                    docling_data = json.load(f)
                
                # 테이블 데이터 추출
                if "tables" in docling_data:
                    for table in docling_data["tables"]:
                        if "rows" in table:
                            for row_idx, row in enumerate(table["rows"]):
                                if isinstance(row, list):
                                    text = " | ".join([str(cell) for cell in row if cell])
                                    if text.strip():
                                        try:
                                            embedding = self.embeddings.embed_query(text)
                                            
                                            # 안전한 ID 생성
                                            point_id = self.generate_safe_id("docling", docling_file.stem, f"table_{table.get('table_index', 0)}_row_{row_idx}")
                                            
                                            point = PointStruct(
                                                id=point_id,
                                                vector=embedding,
                                                payload={
                                                    "text": text,
                                                    "source": "docling_table",
                                                    "file_name": docling_file.name,
                                                    "table_index": table.get("table_index", 0),
                                                    "row_index": row_idx,
                                                    "row_data": row
                                                }
                                            )
                                            points.append(point)
                                            
                                        except Exception as e:
                                            print(f"     임베딩 생성 실패: {e}")
                                            continue
                
                # 텍스트 콘텐츠 추출
                if "content" in docling_data:
                    content = docling_data["content"]
                    if isinstance(content, str) and content.strip():
                        # 긴 텍스트를 청크로 분할
                        words = content.split()
                        chunk_size = 100
                        for i in range(0, len(words), chunk_size):
                            chunk_text = " ".join(words[i:i+chunk_size])
                            
                            try:
                                embedding = self.embeddings.embed_query(chunk_text)
                                
                                # 안전한 ID 생성
                                point_id = self.generate_safe_id("docling", docling_file.stem, f"content_{i//chunk_size}")
                                
                                point = PointStruct(
                                    id=point_id,
                                    vector=embedding,
                                    payload={
                                        "text": chunk_text,
                                        "source": "docling_content",
                                        "file_name": docling_file.name,
                                        "chunk_index": i // chunk_size
                                    }
                                )
                                points.append(point)
                                
                            except Exception as e:
                                print(f"     임베딩 생성 실패: {e}")
                                continue
                
                print(f"     ✅ {len([p for p in points if docling_file.name in p.payload.get('file_name', '')])}개 포인트 생성")
                
            except Exception as e:
                print(f"     ❌ 파일 처리 실패: {e}")
                continue
        
        # Qdrant에 저장
        if points:
            client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"   💾 {len(points)}개 Docling 포인트 저장 완료")
    
    async def process_image_modal_results(self, client, PointStruct):
        """이미지 모달 결과 처리"""
        print("\n📊 **이미지 모달 결과 처리**")
        
        if not self.image_modal_dir.exists():
            print("   ⚠️ 이미지 모달 디렉토리가 없습니다.")
            return
        
        points = []
        image_files = list(self.image_modal_dir.glob("*.json"))
        
        for image_file in image_files:
            print(f"   📄 처리 중: {image_file.name}")
            
            try:
                with open(image_file, 'r', encoding='utf-8') as f:
                    image_data = json.load(f)
                
                # 조립 다이어그램 정보 추출
                if "assembly_diagrams" in image_data:
                    for diagram_idx, diagram in enumerate(image_data["assembly_diagrams"]):
                        if isinstance(diagram, dict):
                            text_content = f"조립 다이어그램: {diagram.get('pattern', '')} - {diagram.get('context', '')}"
                            
                            try:
                                embedding = self.embeddings.embed_query(text_content)
                                
                                # 안전한 ID 생성
                                point_id = self.generate_safe_id("image", image_file.stem, f"diagram_{diagram_idx}")
                                
                                point = PointStruct(
                                    id=point_id,
                                    vector=embedding,
                                    payload={
                                        "text": text_content,
                                        "source": "image_modal",
                                        "file_name": image_file.name,
                                        "diagram_index": diagram_idx,
                                        "diagram_info": diagram
                                    }
                                )
                                points.append(point)
                                
                            except Exception as e:
                                print(f"     임베딩 생성 실패: {e}")
                                continue
                
                # 조립 단계 정보 추출
                if "modal_processing" in image_data and "assembly_steps" in image_data["modal_processing"]:
                    for step in image_data["modal_processing"]["assembly_steps"]:
                        if isinstance(step, dict):
                            text_content = f"조립 단계 {step.get('step_number', 0)}: {step.get('description', '')}"
                            
                            try:
                                embedding = self.embeddings.embed_query(text_content)
                                
                                # 안전한 ID 생성
                                point_id = self.generate_safe_id("image", image_file.stem, f"step_{step.get('step_number', 0)}")
                                
                                point = PointStruct(
                                    id=point_id,
                                    vector=embedding,
                                    payload={
                                        "text": text_content,
                                        "source": "image_modal",
                                        "file_name": image_file.name,
                                        "step_info": step
                                    }
                                )
                                points.append(point)
                                
                            except Exception as e:
                                print(f"     임베딩 생성 실패: {e}")
                                continue
                
                print(f"     ✅ {len([p for p in points if image_file.name in p.payload.get('file_name', '')])}개 포인트 생성")
                
            except Exception as e:
                print(f"     ❌ 파일 처리 실패: {e}")
                continue
        
        # Qdrant에 저장
        if points:
            client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"   💾 {len(points)}개 이미지 모달 포인트 저장 완료")
    
    async def print_population_summary(self, client):
        """저장 결과 요약"""
        print("\n" + "=" * 60)
        print("📊 **저장 결과 요약**")
        print("=" * 60)
        
        try:
            collection_info = client.get_collection(self.collection_name)
            print(f"📁 컬렉션: {self.collection_name}")
            print(f"📊 총 포인트 수: {collection_info.points_count}")
            print(f"🔢 벡터 크기: {collection_info.config.params.vectors.size}")
            print(f"📏 거리 메트릭: {collection_info.config.params.vectors.distance}")
            
            if collection_info.points_count > 0:
                # 샘플 데이터 확인
                sample_points = client.scroll(
                    collection_name=self.collection_name,
                    limit=3
                )[0]
                
                print(f"\n📋 **샘플 데이터**")
                for i, point in enumerate(sample_points):
                    print(f"   포인트 {i+1}:")
                    print(f"     ID: {point.id}")
                    print(f"     소스: {point.payload.get('source', 'unknown')}")
                    print(f"     파일: {point.payload.get('file_name', 'unknown')}")
                    print(f"     텍스트: {point.payload.get('text', '')[:100]}...")
            
            print(f"\n✅ **Qdrant 저장 완료 (Mock 임베딩)**")
            print(f"   이제 rag-anything 쿼리를 테스트할 수 있습니다.")
            
        except Exception as e:
            print(f"❌ 요약 생성 실패: {e}")

# 메인 함수
async def main():
    """메인 함수"""
    try:
        populator = PopulateQdrantFromResults()
        await populator.populate_qdrant()
        
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
