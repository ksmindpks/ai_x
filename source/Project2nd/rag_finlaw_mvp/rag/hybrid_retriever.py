"""
hybrid_retriever.py - 로그 시스템 통합 버전
"""
import os
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from rag.embedder_upstage import get_embedder
from rag.utils import SearchResult

def log_message(log_type, message, module="RETRIEVER"):
    """통합된 로그 함수 - 3단계 분류"""
    print(f"[{module}-{log_type.upper()}] {message}")
    
    # 웹 인터페이스로 전달 시도
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'global_log_callback'):
            callback = st.session_state.global_log_callback
            if callable(callback):
                callback(log_type, message, module, "evaluation")
    except Exception:
        pass

class HybridRetriever:
    """로그 시스템 통합된 하이브리드 검색기"""
    
    def __init__(self, config):
        log_message("INFO", "하이브리드 검색기 초기화 중...")
        self.config = config
        
        self.embedder = get_embedder()
        
        self.bm25_index = None
        self.corpus = None
        self._load_bm25()
        
        self.pinecone_index = None
        self._init_pinecone()
        
        # 검색 통계
        self.search_stats = {
            'bm25_failures': 0,
            'vector_fallbacks': 0,
            'total_queries': 0
        }
        
        success_msg = f"초기화 완료 (BM25: {self.bm25_index is not None}, Vector: {self.pinecone_index is not None})"
        log_message("SUCCESS", success_msg)

    def _load_bm25(self):
        """BM25 인덱스 로드"""
        if not os.path.exists(self.config.bm25_index_path):
            log_message("FAILURE", "BM25 인덱스 파일이 존재하지 않음")
            return
        
        try:
            with open(self.config.bm25_index_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and "bm25" in data and "corpus" in data:
                self.bm25_index = data["bm25"]
                self.corpus = data["corpus"]
                log_message("SUCCESS", f"BM25 로드 성공 ({len(self.corpus)}개 문서)")
            else:
                log_message("FAILURE", "BM25 데이터 형식이 올바르지 않음")
        except Exception as e:
            log_message("FAILURE", f"BM25 로드 실패: {e}")

    def _init_pinecone(self):
        """Pinecone 초기화"""
        if not self.config.pinecone_api_key:
            log_message("INFO", "Pinecone API 키가 설정되지 않음")
            return
        
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            self.pinecone_index = pc.Index(self.config.pinecone_index_name)
            log_message("SUCCESS", "Pinecone 연결 완료")
        except Exception as e:
            log_message("FAILURE", f"Pinecone 연결 실패: {e}")

    def _legal_tokenize(self, text: str) -> List[str]:
        """법령 토크나이저"""
        tokens = []
        
        # 조문 패턴
        articles = re.findall(r'제\s*\d+\s*조(?:제\s*\d+\s*항)?(?:제\s*\d+\s*호)?', text)
        tokens.extend(articles * 3)  # 가중치
        
        # 기관명 동의어 매핑
        synonyms = {
            "중소벤처기업부": ["중기부", "중소기업부"],
            "금융위원회": ["금위", "금융위"],
            "금융감독원": ["금감원"],
            "공정거래위원회": ["공정위", "공거위"],
        }
        
        for standard, variants in synonyms.items():
            if standard in text:
                tokens.extend(variants)
            else:
                for variant in variants:
                    if variant in text:
                        tokens.extend([standard] + [v for v in variants if v != variant])
                        break
        
        # 기본 토큰
        korean_words = re.findall(r'[가-힣]{2,}', text)
        tokens.extend(korean_words)
        
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)
        
        # 법령 패턴
        patterns = [
            r'\d+(?:년|개월|일)',
            r'\d+(?:억|만)?원',
            r'별지\s*(?:제\s*)?\d+\s*(?:호|번)(?:서식|양식)?',
            r'[가-힣]+(?:위원회|청|부|처|원)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            tokens.extend(matches)
        
        # 중복 제거 (순서 보존)
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        return unique_tokens

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 검색"""
        if not self.bm25_index:
            log_message("FAILURE", "BM25 인덱스를 사용할 수 없음")
            return []
        
        query_tokens = self._legal_tokenize(query)
        
        if not query_tokens:
            query_tokens = query.split()
        
        if not query_tokens:
            log_message("FAILURE", "검색 토큰이 없음")
            return []
        
        try:
            scores = self.bm25_index.get_scores(query_tokens)
        except Exception as e:
            log_message("FAILURE", f"BM25 점수 계산 오류: {e}")
            return []
        
        if len(scores) == 0:
            log_message("FAILURE", "BM25 점수 결과 없음")
            return []
        
        max_score = max(scores)
        
        if max_score == 0.0:
            self.search_stats['bm25_failures'] += 1
            log_message("FAILURE", f"BM25 매칭 실패 (누적: {self.search_stats['bm25_failures']}회)")
        else:
            log_message("SUCCESS", f"BM25 매칭 성공 (최고:{max_score:.2f})")
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.corpus) and scores[idx] > 0:
                results.append(SearchResult(
                    content=self.corpus[idx].get('text', ''),
                    score=float(scores[idx]),
                    metadata={
                        'search_method': 'bm25',
                        'index': idx,
                        'query_tokens_used': len(query_tokens),
                        **self.corpus[idx].get('metadata', {})
                    }
                ))
        
        return results

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Vector 검색"""
        if not self.pinecone_index:
            log_message("FAILURE", "Vector 검색 불가 (Pinecone 없음)")
            return []
        
        try:
            # 1차: 원본 쿼리로 검색
            results = self._vector_search_single(query, top_k)
            max_score = 0
            for r in results:
                if hasattr(r, 'score'):
                    max_score = max(max_score, r.score)
            
            log_message("SUCCESS", f"Vector 검색 완료 (최고점수: {max_score:.1f})")
            
            # 2차: 점수가 낮으면 키워드만으로 재검색
            if max_score < 45:
                keywords = re.findall(r'[가-힣]{3,}|\d+(?:년|개월|일)|\d+(?:억|만)?원|제\d+조', query)
                if keywords:
                    keyword_query = " ".join(keywords[:4])
                    fallback_results = self._vector_search_single(keyword_query, top_k * 2)
                    
                    fallback_max = 0
                    for r in fallback_results:
                        if hasattr(r, 'score'):
                            fallback_max = max(fallback_max, r.score)
                    
                    if fallback_max > max_score:
                        results = fallback_results
                        log_message("INFO", "키워드 폴백 적용")
            
            return results[:top_k]
            
        except Exception as e:
            log_message("FAILURE", f"Vector 검색 오류: {e}")
            return []

    def _vector_search_single(self, query: str, top_k: int) -> List[SearchResult]:
        """단일 Vector 검색 실행"""
        query_embedding = self.embedder.embed_query(query)
        if not query_embedding:
            log_message("FAILURE", "임베딩 생성 실패")
            return []
        
        try:
            response = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            log_message("FAILURE", f"Pinecone 쿼리 오류: {e}")
            return []
        
        results = []
        for match in response['matches']:
            metadata = match['metadata'].copy()
            
            results.append(SearchResult(
                content=metadata.get('text', ''),
                score=float(match['score']) * 100,
                metadata={
                    'search_method': 'vector',
                    'vector_score': float(match['score']),
                    **metadata
                }
            ))
        
        return results

    def search(self, query: str, question_type: str = "general", top_k: int = None) -> List[SearchResult]:
        """메인 검색 함수"""
        
        self.search_stats['total_queries'] += 1
        query_preview = query[:50] + "..." if len(query) > 50 else query
        log_message("INFO", f"검색 시작 (쿼리: '{query_preview}' 유형: {question_type})")
        
        if question_type == "MCQ":
            bm25_count = 3
            vector_count = 9
            bm25_weight = 0.2
            vector_weight = 0.8
            target_results = 10
        elif question_type == "short":
            bm25_count = 5
            vector_count = 7
            bm25_weight = 0.3
            vector_weight = 0.7
            target_results = 10
        else:
            bm25_count = 4
            vector_count = 8
            bm25_weight = 0.2
            vector_weight = 0.8
            target_results = 10
        
        # BM25 검색
        bm25_results = self._bm25_search(query, bm25_count * 2)
        bm25_max_score = 0
        bm25_valid_results = 0
        
        for result in bm25_results:
            if hasattr(result, 'score'):
                bm25_max_score = max(bm25_max_score, result.score)
                bm25_valid_results += 1
            else:
                log_message("FAILURE", f"BM25 결과에 score 속성 없음")
        
        # BM25 결과 품질 검증
        if len(bm25_results) > 0 and bm25_valid_results == 0:
            log_message("FAILURE", f"심각한 문제: BM25 {len(bm25_results)}개 결과 모두 score 없음")
        elif bm25_valid_results < len(bm25_results):
            log_message("FAILURE", f"BM25 품질 문제: {len(bm25_results)}개 중 {bm25_valid_results}개만 유효")
        
        # BM25 실패 시 Vector 강화 모드
        if question_type == "short" and bm25_max_score < 0.3:
            log_message("INFO", f"BM25 실패 (최고:{bm25_max_score:.1f}), Vector 강화 모드 진입")
            
            vector_results = self._vector_search(query, vector_count * 3)
            self.search_stats['vector_fallbacks'] += 1
            
            final_results = self._diversify_by_source(vector_results, max_per_source=4)
            log_message("SUCCESS", f"Vector 강화 완료: {len(final_results)}개 결과")
            return final_results
        else:
            vector_results = self._vector_search(query, vector_count * 2)
            log_message("INFO", f"하이브리드 모드: BM25({bm25_count}) + Vector({vector_count})")
        
        # 결과 병합
        merged_results = self._merge_results(
            bm25_results[:bm25_count], 
            vector_results[:vector_count],
            bm25_weight, 
            vector_weight, 
            target_results
        )
        
        # 다양성 확보
        final_results = self._diversify_by_source(merged_results, max_per_source=4)
        
        log_message("SUCCESS", f"검색 완료: {len(final_results)}개 결과")
        
        # 주기적 통계 출력
        if self.search_stats['total_queries'] % 15 == 0:
            self._print_search_stats()
        
        return final_results

    def _diversify_by_source(self, results: List[SearchResult], max_per_source: int = 4) -> List[SearchResult]:
        """소스별 다양성 확보"""
        source_counts = {}
        diversified = []
        
        for result in results:
            source = result.metadata.get('source_file', 'unknown')
            current_count = source_counts.get(source, 0)
            
            if current_count < max_per_source:
                diversified.append(result)
                source_counts[source] = current_count + 1
            
            if len(diversified) >= 12:
                break
        
        # 결과가 부족하면 소스 제한 완화
        if len(diversified) < 8:
            for result in results:
                if result not in diversified:
                    diversified.append(result)
                    if len(diversified) >= 12:
                        break
        
        return diversified

    def _merge_results(self, bm25_results: List[SearchResult], vector_results: List[SearchResult], 
                       bm25_weight: float, vector_weight: float, target_count: int) -> List[SearchResult]:
        """결과 병합"""
        all_results = []
        seen_content = set()
        
        log_message("INFO", f"결과 병합 시작: BM25 {len(bm25_results)}개, Vector {len(vector_results)}개")
        
        # Vector 결과를 먼저 추가 (우선순위)
        for result in vector_results:
            content_key = self._get_content_key(result.content)
            if content_key not in seen_content:
                result.metadata['final_score'] = result.score * vector_weight
                result.metadata['bm25_contribution'] = 0.0
                result.metadata['vector_contribution'] = result.score * vector_weight
                all_results.append(result)
                seen_content.add(content_key)
        
        # BM25 결과 추가
        for result in bm25_results:
            content_key = self._get_content_key(result.content)
            if content_key not in seen_content:
                result.metadata['final_score'] = result.score * bm25_weight
                result.metadata['bm25_contribution'] = result.score * bm25_weight
                result.metadata['vector_contribution'] = 0.0
                all_results.append(result)
                seen_content.add(content_key)
            else:
                # 중복 시 점수 합산
                for existing in all_results:
                    if self._get_content_key(existing.content) == content_key:
                        existing.metadata['final_score'] += result.score * bm25_weight
                        existing.metadata['bm25_contribution'] = result.score * bm25_weight
                        break
        
        # final_score로 정렬
        all_results.sort(key=lambda x: x.metadata.get('final_score', 0), reverse=True)
        
        final_results = all_results[:target_count]
        
        # 점수 업데이트
        for result in final_results:
            result.score = result.metadata.get('final_score', result.score)
        
        log_message("SUCCESS", f"병합 완료: {len(final_results)}개 최종 결과")
        return final_results

    def _get_content_key(self, content: str) -> str:
        """컨텐츠 중복 판단용 키 생성"""
        return content[:150].strip()

    def _print_search_stats(self):
        """검색 통계 출력"""
        total = self.search_stats['total_queries']
        failures = self.search_stats['bm25_failures']
        vector_fallbacks = self.search_stats['vector_fallbacks']
        
        failure_rate = (failures / total * 100) if total > 0 else 0
        vector_rate = (vector_fallbacks / total * 100) if total > 0 else 0
        
        log_message("INFO", f"검색 통계 - 총:{total}회")
        log_message("INFO", f"  - BM25실패:{failures}회({failure_rate:.1f}%)")
        log_message("INFO", f"  - Vector강화:{vector_fallbacks}회({vector_rate:.1f}%)")