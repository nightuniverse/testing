#!/usr/bin/env python3
"""
Qdrant 데이터 확인 스크립트
"""

from qdrant_client import QdrantClient

def check_qdrant_data():
    """Qdrant에 저장된 데이터 확인"""
    client = QdrantClient(url="http://localhost:6333")
    
    # 컬렉션 정보 확인
    collections = client.get_collections()
    print("=== Qdrant Collections ===")
    for collection in collections.collections:
        print(f"Collection: {collection.name}")
        
        # 컬렉션 정보 가져오기
        info = client.get_collection(collection.name)
        print(f"  Points count: {info.points_count}")
        print(f"  Vectors count: {info.vectors_count}")
        print()
    
    # context_aware_manufacturing 컬렉션의 데이터 확인
    collection_name = "context_aware_manufacturing"
    
    try:
        # 전체 포인트 스크롤
        points = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False
        )[0]
        
        print(f"=== {collection_name} Collection Data ===")
        print(f"Total points: {len(points)}")
        print()
        
        for i, point in enumerate(points, 1):
            print(f"Point {i}:")
            print(f"  ID: {point.id}")
            print(f"  Doc Name: {point.payload.get('doc_name', 'N/A')}")
            print(f"  Content Type: {point.payload.get('content_type', 'N/A')}")
            print(f"  Source: {point.payload.get('source', 'N/A')}")
            print(f"  Text: {point.payload.get('text', 'N/A')[:100]}...")
            print(f"  Context: {point.payload.get('context', 'N/A')[:100]}...")
            print()
            
    except Exception as e:
        print(f"Error reading collection data: {e}")

if __name__ == "__main__":
    check_qdrant_data()
