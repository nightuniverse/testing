"""
RAG-Anything 설정 파일
Docling 파서를 사용하여 문서를 파싱하고 RAG-Anything으로 처리하는 설정
"""

import os
from pathlib import Path
from raganything.config import RAGAnythingConfig as BaseConfig

# 기본 설정
class RAGAnythingConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            working_dir='./rag_anything_output',
            parse_method='auto',
            parser_output_dir='./rag_anything_output',
            parser='docling',  # docling 또는 mineru
            display_content_stats=True,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=1,
            recursive_folder_processing=True,
            context_window=1,
            context_mode='page',
            max_context_tokens=2000,
            include_headers=True,
            include_captions=True,
            content_format='minerU'
        )
        
        # 추가 설정
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.LANG = "ko"  # 한국어 문서
        self.DEVICE = "cpu"  # cpu, cuda, cuda:0, npu, mps
        self.VLM_BACKEND = "pipeline"  # pipeline, vlm-transformers, vlm-sglang-engine, vlm-sglang-client
        
        # Vector Store 설정 (LightRAG 초기화를 위해)
        self.vector_store = "nano"  # nano, qdrant, chroma, faiss 등
        self.vector_store_config = {
            "collection_name": "manufacturing_docs",
            "embedding_model": "text-embedding-3-small",
            "distance_metric": "cosine"
        }
        
        # LightRAG 설정
        self.lightrag_config = {
            "vector_store": self.vector_store,
            "vector_store_config": self.vector_store_config,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o",
            "max_tokens": 2000,
            "temperature": 0
        }
        
        # 출력 디렉토리 생성
        self.OUTPUT_DIR = Path(self.working_dir)
        self.OUTPUT_DIR.mkdir(exist_ok=True)

# 설정 인스턴스
config = RAGAnythingConfig()

# LLM 모델 함수 생성 (RAG-Anything 초기화에 필요)
def create_llm_model_func():
    """LLM 모델 함수 생성"""
    
    def llm_model_func(prompt, **kwargs):
        """LLM 모델 함수 (실제 LLM 호출 없이 파싱만 수행)"""
        # 실제 LLM 호출 없이 파싱 결과만 반환
        return {
            "content": f"Parsed content from prompt: {prompt[:100]}...",
            "usage": {"total_tokens": 0}
        }
    
    return llm_model_func

# Vision 모델 함수 생성
def create_vision_model_func():
    """Vision 모델 함수 생성"""
    
    def vision_model_func(image_path, prompt=None, **kwargs):
        """Vision 모델 함수 (이미지 분석)"""
        # prompt가 None이면 기본값 사용
        if prompt is None:
            prompt = "Describe this image"
        
        return {
            "content": f"Image analysis result for {image_path}: {prompt[:50]}...",
            "usage": {"total_tokens": 0}
        }
    
    return vision_model_func

# Embedding 함수 생성
def create_embedding_func():
    """Embedding 함수 생성"""
    
    def embedding_func(text, **kwargs):
        """Embedding 함수 (텍스트를 벡터로 변환)"""
        # 리스트인 경우 문자열로 변환
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
        
        # 간단한 해시 기반 임베딩 (실제로는 OpenAI API 사용)
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        return [float(int(hash_obj.hexdigest()[:8], 16)) / 1e8 for _ in range(1536)]
    
    # embedding_dim 속성 추가
    embedding_func.embedding_dim = 1536
    
    return embedding_func
