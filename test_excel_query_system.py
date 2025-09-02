#!/usr/bin/env python3
"""
test_excels 폴더의 엑셀 파일들을 대상으로 대안 쿼리 시스템 테스트
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestExcelQuerySystem:
    """test_excels 폴더의 엑셀 파일들을 대상으로 쿼리 시스템 테스트"""
    
    def __init__(self):
        self.test_excels_dir = Path("test_excels")
        self.results_dir = Path("test_excels/test_results")
        self.query_results_dir = self.results_dir / "query_results"
        self.query_results_dir.mkdir(exist_ok=True)
        
        # OpenAI API 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 대안 쿼리 솔루션 임포트
        from alternative_query_solutions import AlternativeQuerySolutions
        self.query_solutions = AlternativeQuerySolutions(str(self.results_dir))
    
    async def run_complete_test(self):
        """완전한 테스트 실행"""
        print("🚀 **test_excels 폴더 쿼리 시스템 테스트**")
        print("=" * 80)
        
        # 1. 엑셀 파일 확인
        excel_files = await self.get_excel_files()
        print(f"📊 처리할 엑셀 파일: {len(excel_files)}개")
        for file in excel_files:
            print(f"   - {file.name}")
        
        # 2. 테스트 쿼리 정의
        test_queries = await self.define_test_queries()
        print(f"🔍 테스트 쿼리: {len(test_queries)}개")
        for i, query in enumerate(test_queries, 1):
            print(f"   {i}. {query}")
        
        # 3. 각 엑셀 파일별로 테스트 실행
        all_results = {}
        
        for excel_file in excel_files:
            print(f"\n📄 **엑셀 파일 테스트: {excel_file.name}**")
            print("=" * 60)
            
            file_results = await self.test_single_excel_file(excel_file, test_queries)
            all_results[excel_file.name] = file_results
            
            # 개별 파일 결과 저장
            await self.save_file_results(excel_file.name, file_results)
        
        # 4. 종합 결과 생성 및 저장
        await self.save_comprehensive_results(all_results, test_queries)
        
        # 5. 결과 요약
        await self.print_test_summary(all_results, test_queries)
    
    async def get_excel_files(self) -> List[Path]:
        """test_excels 폴더에서 엑셀 파일 찾기"""
        excel_files = []
        
        if self.test_excels_dir.exists():
            for file_path in self.test_excels_dir.glob("*.xlsx"):
                if not file_path.name.startswith('~$'):  # 임시 파일 제외
                    excel_files.append(file_path)
        
        return excel_files
    
    async def define_test_queries(self) -> List[str]:
        """테스트용 쿼리 정의"""
        return [
            "조립 다이어그램은 무엇인가요?",
            "수입검사 관련 정보를 알려주세요",
            "이미지 파일의 위치는 어디인가요?",
            "테이블 데이터를 요약해주세요",
            "조립 작업표준서의 주요 내용은?",
            "생성형 AI 연동 자료의 핵심 정보는?",
            "엑셀 파일에서 추출된 이미지 개수는?",
            "조립 파트 관련 정보를 찾아주세요",
            "품질 관리 프로세스는 어떻게 되어있나요?",
            "작업 순서나 단계별 정보가 있나요?"
        ]
    
    async def test_single_excel_file(self, excel_file: Path, test_queries: List[str]) -> Dict[str, Any]:
        """단일 엑셀 파일에 대한 쿼리 테스트"""
        
        file_results = {
            "file_name": excel_file.name,
            "file_size": excel_file.stat().st_size,
            "test_timestamp": datetime.now().isoformat(),
            "queries": {}
        }
        
        # 파일명에서 키워드 추출 (쿼리 필터링용)
        file_keywords = self.extract_file_keywords(excel_file.name)
        
        for i, query in enumerate(test_queries, 1):
            print(f"   🔍 쿼리 {i}/{len(test_queries)}: {query}")
            
            query_results = {}
            
            try:
                # 1. 직접 JSON 검색
                print("     1️⃣ 직접 JSON 검색...")
                result1 = await self.query_solutions.solution1_direct_json_query(
                    query, excel_file.stem
                )
                query_results["direct_json"] = {
                    "answer": result1.get("answer", ""),
                    "total_matches": result1.get("total_matches", 0),
                    "status": "success" if "error" not in result1 else "error",
                    "error": result1.get("error", None)
                }
                
                # 2. 의미적 검색
                print("     2️⃣ 의미적 검색...")
                result2 = await self.query_solutions.solution2_semantic_search(
                    query, excel_file.stem
                )
                query_results["semantic_search"] = {
                    "answer": result2.get("answer", ""),
                    "total_chunks": result2.get("total_chunks", 0),
                    "top_similarity": result2.get("top_similarity", 0),
                    "status": "success" if "error" not in result2 else "error",
                    "error": result2.get("error", None)
                }
                
                # 3. 하이브리드 검색
                print("     3️⃣ 하이브리드 검색...")
                result3 = await self.query_solutions.solution3_hybrid_search(
                    query, excel_file.stem
                )
                query_results["hybrid_search"] = {
                    "answer": result3.get("answer", ""),
                    "keyword_matches": len(result3.get("keyword_results", {}).get("search_results", [])),
                    "semantic_matches": len(result3.get("semantic_results", {}).get("search_results", [])),
                    "status": "success" if "error" not in result3 else "error",
                    "error": result3.get("error", None)
                }
                
                # 4. Knowledge Graph 쿼리
                print("     4️⃣ Knowledge Graph 쿼리...")
                result4 = await self.query_solutions.solution4_knowledge_graph_query(
                    query, excel_file.stem
                )
                query_results["knowledge_graph"] = {
                    "answer": result4.get("answer", ""),
                    "total_nodes": result4.get("total_nodes", 0),
                    "total_edges": result4.get("total_edges", 0),
                    "relevant_nodes": len(result4.get("kg_analysis", {}).get("relevant_nodes", [])),
                    "status": "success" if "error" not in result4 else "error",
                    "error": result4.get("error", None)
                }
                
                # 5. 최적 답변 선택
                best_answer = self.select_best_answer(query_results)
                query_results["best_answer"] = best_answer
                
            except Exception as e:
                print(f"     ❌ 쿼리 {i} 실패: {e}")
                query_results["error"] = str(e)
            
            file_results["queries"][f"query_{i}"] = {
                "question": query,
                "results": query_results
            }
            
            print(f"     ✅ 쿼리 {i} 완료")
        
        return file_results
    
    def extract_file_keywords(self, file_name: str) -> List[str]:
        """파일명에서 키워드 추출"""
        keywords = []
        
        # 파일명을 소문자로 변환하고 특수문자 제거
        clean_name = file_name.lower().replace('.xlsx', '').replace('(', ' ').replace(')', ' ')
        
        # 주요 키워드들
        key_terms = [
            "조립", "작업표준서", "수입검사", "생성형", "AI", "연동", "파트", "자료",
            "FRONT", "DECO", "SUB", "SM-F741U", "B6"
        ]
        
        for term in key_terms:
            if term.lower() in clean_name:
                keywords.append(term)
        
        return keywords
    
    def select_best_answer(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """가장 좋은 답변 선택"""
        best_answer = {
            "method": "none",
            "answer": "(no-context)",
            "confidence": 0.0,
            "reason": "모든 방법이 실패"
        }
        
        # 각 방법별 점수 계산
        scores = {}
        
        # 1. 직접 JSON 검색
        if "direct_json" in query_results:
            direct_result = query_results["direct_json"]
            if direct_result["status"] == "success" and direct_result["total_matches"] > 0:
                scores["direct_json"] = min(direct_result["total_matches"] * 0.1, 1.0)
        
        # 2. 의미적 검색
        if "semantic_search" in query_results:
            semantic_result = query_results["semantic_search"]
            if semantic_result["status"] == "success" and semantic_result["top_similarity"] > 0:
                scores["semantic_search"] = semantic_result["top_similarity"]
        
        # 3. 하이브리드 검색
        if "hybrid_search" in query_results:
            hybrid_result = query_results["hybrid_search"]
            if hybrid_result["status"] == "success":
                total_matches = hybrid_result["keyword_matches"] + hybrid_result["semantic_matches"]
                scores["hybrid_search"] = min(total_matches * 0.05, 1.0)
        
        # 4. Knowledge Graph 쿼리
        if "knowledge_graph" in query_results:
            kg_result = query_results["knowledge_graph"]
            if kg_result["status"] == "success" and kg_result["relevant_nodes"] > 0:
                scores["knowledge_graph"] = min(kg_result["relevant_nodes"] * 0.2, 1.0)
        
        # 최고 점수 방법 선택
        if scores:
            best_method = max(scores, key=scores.get)
            best_score = scores[best_method]
            
            if best_score > 0:
                best_answer = {
                    "method": best_method,
                    "answer": query_results[best_method]["answer"],
                    "confidence": best_score,
                    "reason": f"{best_method}에서 가장 높은 점수 ({best_score:.3f})"
                }
        
        return best_answer
    
    async def save_file_results(self, file_name: str, file_results: Dict[str, Any]):
        """개별 파일 결과 저장"""
        output_file = self.query_results_dir / f"{file_name}_query_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(file_results, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 파일 결과 저장: {output_file}")
    
    async def save_comprehensive_results(self, all_results: Dict[str, Any], test_queries: List[str]):
        """종합 결과 저장"""
        
        # 통계 계산
        stats = {
            "total_files": len(all_results),
            "total_queries": len(test_queries),
            "test_timestamp": datetime.now().isoformat(),
            "file_statistics": {},
            "query_statistics": {},
            "method_performance": {
                "direct_json": {"success": 0, "total": 0},
                "semantic_search": {"success": 0, "total": 0},
                "hybrid_search": {"success": 0, "total": 0},
                "knowledge_graph": {"success": 0, "total": 0}
            }
        }
        
        # 파일별 통계
        for file_name, file_result in all_results.items():
            file_stats = {
                "file_size": file_result["file_size"],
                "queries_processed": len(file_result["queries"]),
                "best_answers": 0,
                "no_context_answers": 0
            }
            
            for query_key, query_data in file_result["queries"].items():
                results = query_data["results"]
                
                # 방법별 성공률 계산
                for method in stats["method_performance"]:
                    if method in results:
                        stats["method_performance"][method]["total"] += 1
                        if results[method]["status"] == "success":
                            stats["method_performance"][method]["success"] += 1
                
                # 최적 답변 분석
                if "best_answer" in results:
                    best_answer = results["best_answer"]
                    if best_answer["method"] != "none":
                        file_stats["best_answers"] += 1
                    else:
                        file_stats["no_context_answers"] += 1
            
            stats["file_statistics"][file_name] = file_stats
        
        # 쿼리별 통계
        for i, query in enumerate(test_queries, 1):
            query_key = f"query_{i}"
            query_stats = {
                "question": query,
                "files_processed": 0,
                "successful_answers": 0,
                "no_context_answers": 0,
                "best_methods": {}
            }
            
            for file_name, file_result in all_results.items():
                if query_key in file_result["queries"]:
                    query_stats["files_processed"] += 1
                    
                    best_answer = file_result["queries"][query_key]["results"].get("best_answer", {})
                    if best_answer["method"] != "none":
                        query_stats["successful_answers"] += 1
                        method = best_answer["method"]
                        query_stats["best_methods"][method] = query_stats["best_methods"].get(method, 0) + 1
                    else:
                        query_stats["no_context_answers"] += 1
            
            stats["query_statistics"][query_key] = query_stats
        
        # 성공률 계산
        for method, performance in stats["method_performance"].items():
            if performance["total"] > 0:
                performance["success_rate"] = performance["success"] / performance["total"]
            else:
                performance["success_rate"] = 0
        
        # 종합 결과 저장
        comprehensive_results = {
            "test_summary": stats,
            "detailed_results": all_results,
            "test_queries": test_queries
        }
        
        output_file = self.query_results_dir / "comprehensive_query_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 종합 결과 저장: {output_file}")
    
    async def print_test_summary(self, all_results: Dict[str, Any], test_queries: List[str]):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 **테스트 결과 요약**")
        print("=" * 80)
        
        print(f"📁 처리된 파일: {len(all_results)}개")
        print(f"🔍 테스트 쿼리: {len(test_queries)}개")
        
        # 파일별 요약
        print(f"\n📄 **파일별 결과**")
        for file_name, file_result in all_results.items():
            file_size_mb = file_result["file_size"] / (1024 * 1024)
            print(f"   {file_name} ({file_size_mb:.1f}MB)")
            
            # 쿼리 성공률 계산
            total_queries = len(file_result["queries"])
            successful_queries = sum(
                1 for query_data in file_result["queries"].values()
                if query_data["results"].get("best_answer", {}).get("method") != "none"
            )
            success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
            
            print(f"     성공률: {success_rate:.1f}% ({successful_queries}/{total_queries})")
        
        # 방법별 성능
        print(f"\n🔧 **방법별 성능**")
        method_counts = {"direct_json": 0, "semantic_search": 0, "hybrid_search": 0, "knowledge_graph": 0}
        
        for file_result in all_results.values():
            for query_data in file_result["queries"].values():
                best_method = query_data["results"].get("best_answer", {}).get("method", "none")
                if best_method in method_counts:
                    method_counts[best_method] += 1
        
        total_queries = sum(method_counts.values())
        for method, count in method_counts.items():
            percentage = (count / total_queries) * 100 if total_queries > 0 else 0
            print(f"   {method}: {count}회 ({percentage:.1f}%)")
        
        print(f"\n✅ **테스트 완료**")
        print(f"   결과 저장 위치: {self.query_results_dir}")

# 메인 함수
async def main():
    """메인 함수"""
    try:
        # 테스트 시스템 초기화
        test_system = TestExcelQuerySystem()
        
        # 완전한 테스트 실행
        await test_system.run_complete_test()
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
