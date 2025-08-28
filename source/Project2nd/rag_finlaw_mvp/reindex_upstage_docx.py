# -*- coding: utf-8 -*-
"""
법령 DOCX → Upstage 임베딩 → Pinecone 벡터DB 구축 (개선된 버전)

핵심 개선사항:
- 명확한 에러 발생 (개발용)
- 정규식 패턴 최적화
- 청킹 로직 개선
- 메타데이터 생성 최적화
- 비동기 처리 지원
"""

import os
import re
import hashlib
import time
import json
import logging
from typing import List, Dict, Tuple, Optional, Generator
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

from dotenv import load_dotenv
import docx
import requests
from pinecone import Pinecone

# =========================
# 로깅 설정
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# 예외 클래스 정의
# =========================
class DocumentProcessingError(Exception):
    """문서 처리 관련 예외"""
    pass

class EmbeddingError(Exception):
    """임베딩 관련 예외"""
    pass

class VectorDBError(Exception):
    """벡터DB 관련 예외"""
    pass

# =========================
# 설정 클래스
# =========================
@dataclass
class ProcessingConfig:
    # 실행 모드
    retry_failed_only: bool = True
    processed_files_log: str = "processed_files.txt"
    
    # 디렉토리 설정
    input_dirs: List[str] = None
    
    # 청킹 설정
    chunk_max_size: int = 1000
    chunk_min_size: int = 200
    overlap_paragraphs: int = 2
    
    # 단락 처리 설정
    min_paragraph_size: int = 30
    max_paragraph_size: int = 600
    
    # API 설정 (속도 최적화)
    upstage_models: List[str] = None
    batch_size: int = 50  # 20 -> 50 (2.5배 향상)
    max_workers: int = 3  # 4 -> 3 (안정성 유지하면서 속도 향상)
    timeout_s: int = 15   # 30 -> 15 (더 빠른 응답)
    
    def __post_init__(self):
        if self.input_dirs is None:
            self.input_dirs = [
                r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/금융법령",
                r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/기업법령",
                r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/보험법령",
                r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/상법투자자산증권주식법령",
                r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/은행법령",
                r"C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/조합법령",
            ]
        
        if self.upstage_models is None:
            self.upstage_models = [
                "solar-embedding-1-large-passage",  # 문서용 최신 모델
                "solar-embedding-1-large",          # 범용 최신 모델  
                "embedding-passage",                # fallback용
            ]

# =========================
# 정규식 패턴 최적화
# =========================
class LegalTextPatterns:
    """법령 텍스트 처리용 컴파일된 정규식 패턴들"""
    
    # 기본 정규화 패턴
    WHITESPACE = re.compile(r'\s+')
    LAW_NAME = re.compile(r'「\s*([^」]+)\s*」')
    ARTICLE_NUM = re.compile(r'제\s*(\d+)\s*조(?:의\s*(\d+))?')
    ITEM_NUM = re.compile(r'(\d+)\s*\.\s*')
    KOREAN_ITEM = re.compile(r'([가-힣])\s*\.\s*')
    
    # 조문 관련 패턴
    ARTICLE_START = re.compile(r'^제(\d+)조(?:의(\d+))?')
    ARTICLE_REF = re.compile(r'제(\d+(?:의\d+)?)조')
    
    # 분할 패턴
    ITEM_SPLIT = re.compile(r'(?=(?:\d+\.|[가-힣]\.|[①②③④⑤⑥⑦⑧⑨⑩]|[ㄱ-ㅎ]\.))')
    SENTENCE_SPLIT = re.compile(r'(?<=[.!?。．])\s+')
    
    # 메타데이터 추출 패턴
    DEFINITION = re.compile(r'"[^"]+"\s*(?:이란|라고\s*한다)')
    LAW_NUMBER = re.compile(r'(?:법률\s*)?제(\d+)호')
    DATE_8DIGIT = re.compile(r'\((\d{8})\)')
    DATE_KOREAN = re.compile(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일')
    
    # 이해관계자 패턴
    AUTHORITY = re.compile(r'(?:금융위원회|금융감독원|한국은행)')
    MINISTER = re.compile(r'(?:장관|청장)')
    INSTITUTION = re.compile(r'(?:금융기관|은행|보험회사)')
    APPLICANT = re.compile(r'(?:신청인|허가받은\s*자)')
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """텍스트 정규화"""
        text = cls.WHITESPACE.sub(' ', text)
        text = cls.LAW_NAME.sub(r'「\1」', text)
        text = cls.ARTICLE_NUM.sub(
            lambda m: f'제{m.group(1)}조' + (f'의{m.group(2)}' if m.group(2) else ''), 
            text
        )
        text = cls.ITEM_NUM.sub(r'\1. ', text)
        text = cls.KOREAN_ITEM.sub(r'\1. ', text)
        return text.strip()
    
    @classmethod
    def extract_article_number(cls, text: str) -> Optional[str]:
        """조문 번호 추출"""
        match = cls.ARTICLE_START.search(text.strip())
        if match:
            article_num = match.group(1)
            sub_num = match.group(2)
            return f"{article_num}{f'의{sub_num}' if sub_num else ''}"
        
        match = cls.ARTICLE_REF.search(text.strip())
        if match:
            return match.group(1)
        
        return None
    
    @classmethod
    def find_article_references(cls, text: str, exclude_self: str = None) -> List[str]:
        """조문 참조 찾기"""
        refs = [match.group(1) for match in cls.ARTICLE_REF.finditer(text)]
        if exclude_self:
            refs = [ref for ref in refs if ref != exclude_self]
        return list(set(refs))

# =========================
# 외부 메타데이터 추출기 Import
# =========================
try:
    from rag.metadata_extractor import LegalMetadataExtractor
    logger.info("기존 metadata_extractor 모듈 사용")
except ImportError as e:
    logger.error(f"metadata_extractor 모듈 import 실패: {e}")
    logger.info("fallback 메타데이터 추출기 사용")
    
    class LegalMetadataExtractor:
        """Fallback 법령 메타데이터 추출기"""
        
        FIELD_KEYWORDS = {
            "금융": ["금융", "은행", "신용", "여신", "대출", "예금", "저축", "금융위원회"],
            "증권": ["증권", "자본시장", "투자", "상장", "주식", "채권", "파생상품", "펀드"],
            "보험": ["보험", "연금", "공제", "생명보험", "손해보험"],
            "기업": ["상법", "회사", "법인", "기업", "상업", "경영"],
            "조합": ["조합", "협동조합", "단체", "협회"],
            "부동산": ["부동산", "건설", "주택", "토지", "임대"],
            "세무": ["세법", "국세", "지방세", "관세", "부가가치세"],
            "노동": ["근로", "노동", "고용", "임금", "근로기준"],
            "환경": ["환경", "오염", "폐기물", "대기", "수질"],
            "통신": ["통신", "정보통신", "방송", "인터넷"],
            "의료": ["의료", "보건", "의약품", "병원"]
        }
        
        @classmethod
        def extract_law_metadata(cls, title: str) -> Dict:
            """법령 메타데이터 추출"""
            info = {
                "law_name": title,
                "law_type": "기타",
                "law_level": 0,
                "category": "기타",
                "law_field": "기타",
                "law_number": "",
                "date": "",
                "law_scope": "일반"
            }
            
            # 법령 레벨 및 유형 분류
            if "법률" in title or title.endswith("법"):
                info["law_type"] = "법률"
                info["law_level"] = 1
            elif "시행령" in title:
                info["law_type"] = "시행령"
                info["law_level"] = 2
            elif "시행규칙" in title:
                info["law_type"] = "시행규칙"
                info["law_level"] = 3
            elif any(x in title for x in ["고시", "훈령", "지침", "규정"]):
                for type_name in ["고시", "훈령", "지침", "규정"]:
                    if type_name in title:
                        info["law_type"] = type_name
                        break
                info["law_level"] = 4
            
            # 분야 분류
            for field, keywords in cls.FIELD_KEYWORDS.items():
                if any(keyword in title for keyword in keywords):
                    info["law_field"] = field
                    info["category"] = field
                    break
            
            # 법령 범위
            if "특별법" in title or "특례" in title:
                info["law_scope"] = "특별"
            elif "기본법" in title:
                info["law_scope"] = "기본"
            
            # 법령 번호 추출
            match = LegalTextPatterns.LAW_NUMBER.search(title)
            if match:
                info["law_number"] = match.group(1)
            
            # 날짜 추출
            match = LegalTextPatterns.DATE_8DIGIT.search(title)
            if match:
                date_str = match.group(1)
                info["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                match = LegalTextPatterns.DATE_KOREAN.search(title)
                if match:
                    year, month, day = match.groups()
                    info["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            return info

# =========================
# 향상된 청킹 시스템
# =========================
class LegalDocumentChunker:
    """법령 문서 청킹 처리기"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def chunk_document(self, paragraphs: List[str]) -> List[Dict]:
        """문서 청킹 메인 함수"""
        processed_paragraphs = self._preprocess_paragraphs(paragraphs)
        chunks = self._create_chunks_with_overlap(processed_paragraphs)
        return self._validate_chunks(chunks)
    
    def _preprocess_paragraphs(self, paragraphs: List[str]) -> List[Dict]:
        """단락 전처리"""
        processed = []
        i = 0
        
        while i < len(paragraphs):
            current = paragraphs[i]
            current_size = len(current)
            article_num = LegalTextPatterns.extract_article_number(current)
            
            # 단락 병합 로직
            if current_size < self.config.min_paragraph_size and i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                next_article = LegalTextPatterns.extract_article_number(next_para)
                
                # 병합 조건: 같은 조문이거나 다음이 새로운 조문이 아님 + 크기 제한
                can_merge = (
                    (not next_article or article_num == next_article) and
                    (current_size + len(next_para)) <= self.config.max_paragraph_size
                )
                
                if can_merge:
                    combined_text = current + "\n" + next_para
                    processed.append({
                        "text": combined_text,
                        "size": len(combined_text),
                        "article": article_num or next_article,
                        "type": "merged"
                    })
                    i += 2
                    continue
            
            # 긴 단락 분할
            if current_size > self.config.max_paragraph_size:
                split_parts = self._split_long_paragraph(current, article_num)
                processed.extend(split_parts)
            else:
                processed.append({
                    "text": current,
                    "size": current_size,
                    "article": article_num,
                    "type": "original"
                })
            
            i += 1
        
        return processed
    
    def _split_long_paragraph(self, text: str, article_num: Optional[str]) -> List[Dict]:
        """긴 단락 분할"""
        # 항목 번호로 분할 시도
        item_splits = LegalTextPatterns.ITEM_SPLIT.split(text)
        
        if len(item_splits) > 1:
            splits = []
            for split in item_splits:
                split = split.strip()
                if not split:
                    continue
                
                if len(split) <= self.config.max_paragraph_size:
                    splits.append({
                        "text": split,
                        "size": len(split),
                        "article": article_num,
                        "type": "item_split"
                    })
                else:
                    # 문장 단위 분할
                    sentence_splits = self._split_by_sentences(split, article_num)
                    splits.extend(sentence_splits)
            
            return splits
        
        # 문장 단위 분할
        return self._split_by_sentences(text, article_num)
    
    def _split_by_sentences(self, text: str, article_num: Optional[str]) -> List[Dict]:
        """문장 단위 분할"""
        sentences = LegalTextPatterns.SENTENCE_SPLIT.split(text)
        
        splits = []
        current_text = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_text = current_text + " " + sentence if current_text else sentence
            
            if len(test_text) > self.config.max_paragraph_size and current_text:
                splits.append({
                    "text": current_text.strip(),
                    "size": len(current_text),
                    "article": article_num,
                    "type": "sentence_split"
                })
                current_text = sentence
            else:
                current_text = test_text
        
        if current_text:
            splits.append({
                "text": current_text.strip(),
                "size": len(current_text),
                "article": article_num,
                "type": "sentence_split"
            })
        
        return splits
    
    def _create_chunks_with_overlap(self, paragraphs: List[Dict]) -> List[Dict]:
        """오버랩 청킹"""
        chunks = []
        i = 0
        
        while i < len(paragraphs):
            chunk_text = ""
            chunk_size = 0
            chunk_articles = set()
            chunk_paragraphs = []
            chunk_types = set()
            
            # 청크 생성
            while i < len(paragraphs) and chunk_size < self.config.chunk_max_size:
                para = paragraphs[i]
                
                if chunk_size + para["size"] > self.config.chunk_max_size and chunk_text:
                    break
                
                chunk_text += "\n" + para["text"] if chunk_text else para["text"]
                chunk_size += para["size"] + (1 if chunk_text else 0)
                
                if para["article"]:
                    chunk_articles.add(para["article"])
                
                chunk_paragraphs.append(i)
                chunk_types.add(para.get("type", "original"))
                i += 1
            
            # 청크 저장
            if chunk_size >= self.config.chunk_min_size and chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "size": chunk_size,
                    "articles": sorted(list(chunk_articles)),
                    "paragraph_indices": chunk_paragraphs,
                    "types": list(chunk_types),
                    "article_count": len(chunk_articles)
                })
                
                # 오버랩 처리
                if i < len(paragraphs) and len(chunk_paragraphs) > self.config.overlap_paragraphs:
                    overlap_size = min(self.config.overlap_paragraphs, len(chunk_paragraphs) // 2)
                    i -= overlap_size
            elif not chunk_text:
                i += 1
        
        return chunks
    
    def _validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """청크 품질 검증"""
        quality_chunks = []
        
        for chunk in chunks:
            # 기본 크기 검증
            if chunk["size"] < self.config.chunk_min_size:
                continue
            
            # 의미있는 내용 검증
            text = chunk["text"]
            word_count = len(text.split())
            if word_count < 10:
                continue
            
            # 조문 정보 보강
            chunk["has_articles"] = bool(chunk["articles"])
            chunk["primary_article"] = chunk["articles"][0] if chunk["articles"] else None
            
            quality_chunks.append(chunk)
        
        return quality_chunks

# =========================
# 최적화된 메타데이터 생성기
# =========================
class OptimizedMetadataGenerator:
    """최적화된 메타데이터 생성기"""
    
    def __init__(self):
        self.extractor = LegalMetadataExtractor()
    
    def create_vector_metadata(self, doc_id: str, title: str, law_info: Dict, 
                            chunk: Dict, chunk_index: int, context_paragraphs: List[str] = None) -> Dict:
        """벡터별 메타데이터 생성"""
        text = chunk["text"]
        
        # 청크 정보 준비
        chunk_info = {
            "chunk_index": chunk_index,
            "chunk_size": len(text),
            "articles": chunk.get("articles", []),
            "article_count": chunk.get("article_count", 0),
            "has_articles": chunk.get("has_articles", False)
        }
        
        # 고도화된 메타데이터 생성 시도 (구체적 예외만 처리)
        try:
            enhanced_metadata = self.extractor.create_enhanced_metadata(
                doc_id, title, text, chunk_info, context_paragraphs
            )
            enhanced_metadata = self._clean_metadata_for_pinecone(enhanced_metadata)
            
            # 기본 법령 정보 병합
            enhanced_metadata.update({
                "law_name": law_info.get("law_name", title),
                "law_number": law_info.get("law_number", ""),
                "date": law_info.get("date", ""),
                "law_scope": law_info.get("law_scope", "일반")
            })
            
            return enhanced_metadata
            
        except (AttributeError, KeyError, TypeError) as e:
            # 예상되는 구체적 오류만 처리
            logger.warning(f"메타데이터 구조 오류, 기본 메타데이터 사용: {e}")
        except re.error as e:
            # 정규식 오류
            logger.warning(f"정규식 처리 오류, 기본 메타데이터 사용: {e}")
        
        # Fallback: 기본 메타데이터 생성
        basic_metadata = self._create_basic_metadata(doc_id, title, law_info, chunk, chunk_index, text)
        return self._clean_metadata_for_pinecone(basic_metadata)
    
    def _clean_metadata_for_pinecone(self, metadata: Dict) -> Dict:
        """Pinecone용 메타데이터 정리 (null 값 제거)"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # null 값을 빈 문자열로 변환
                cleaned[key] = ""
            elif isinstance(value, list):
                # 리스트에서 None 값 제거
                cleaned_list = [str(item) for item in value if item is not None]
                cleaned[key] = cleaned_list
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _create_basic_metadata(self, doc_id: str, title: str, law_info: Dict, 
                             chunk: Dict, chunk_index: int, text: str) -> Dict:
        """기본 메타데이터 생성 (fallback)"""
        metadata = {
            "doc_id": doc_id,
            "title": title[:200],
            "text": text[:3000] if len(text) > 3000 else text,
            "chunk_index": chunk_index,
            "chunk_size": len(text),
            "doc_type": "docx"
        }
        
        # 법령 분류 정보
        metadata.update({
            "law_type": law_info["law_type"],
            "law_field": law_info["law_field"]
        })
        
        # 조문 정보 (기본)
        current_article = LegalTextPatterns.extract_article_number(text)
        article_refs = LegalTextPatterns.find_article_references(text, current_article)
        
        metadata.update({
            "current_article": current_article,
            "article_references": article_refs[:5]
        })
        
        # 내용 유형 분석 (기본)
        content_types = []
        if LegalTextPatterns.DEFINITION.search(text):
            content_types.append("정의")
        if any(word in text for word in ["신청", "승인", "허가", "절차"]):
            content_types.append("절차")
        if any(word in text for word in ["벌금", "과태료", "제재"]):
            content_types.append("제재")
        
        metadata.update({
            "content_types": content_types[:3],
            "has_definitions": "정의" in content_types,
            "has_procedures": "절차" in content_types,
            "has_penalties": "제재" in content_types
        })
        
        # 이해관계자 추출 (기본)
        stakeholders = []
        authority_matches = LegalTextPatterns.AUTHORITY.findall(text)
        minister_matches = LegalTextPatterns.MINISTER.findall(text)
        institution_matches = LegalTextPatterns.INSTITUTION.findall(text)
        applicant_matches = LegalTextPatterns.APPLICANT.findall(text)
        
        stakeholders.extend(authority_matches)
        stakeholders.extend(minister_matches)
        stakeholders.extend(institution_matches)
        stakeholders.extend(applicant_matches)
        
        metadata["stakeholders"] = list(set(stakeholders))[:3]
        
        # 중요도 점수 (기본)
        importance = 0.5
        if text.strip().startswith('제') and '조' in text[:20]:
            importance += 0.2
        if "정의" in content_types:
            importance += 0.15
        if len(article_refs) > 0:
            importance += 0.1
        
        metadata["importance_score"] = min(round(importance, 2), 1.0)
        
        # 청크 특화 정보
        metadata.update({
            "articles": ",".join(chunk.get("articles", [])),
            "article_count": chunk.get("article_count", 0),
            "has_articles": chunk.get("has_articles", False)
        })
        
        return metadata

# =========================
# 향상된 Upstage 임베더
# =========================
class EnhancedUpstageEmbedder:
    """향상된 Upstage 임베더"""
    
    def __init__(self, api_key: str, config: ProcessingConfig):
        self.api_key = api_key
        self.config = config
        self.model = None
        self.dimension = None
        
        # 정규식 패턴 미리 컴파일 (성능 최적화)
        self._control_chars_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]')
        self._backslash_pattern = re.compile(r'\\(?!["\\/bfnrt])')
        self._quote_pattern = re.compile(r'"')
        self._newline_pattern = re.compile(r'[\n\r\t]')
        self._whitespace_pattern = re.compile(r'\s+')
        self._safe_chars_pattern = re.compile(r'[\w\s가-힣.,!?()\-]')
    
    def _find_working_model(self):
        """사용 가능한 모델 찾기"""
        for model in self.config.upstage_models:
            for attempt in range(3):
                try:
                    resp = requests.post(
                        "https://api.upstage.ai/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={"input": ["테스트"], "model": model},
                        timeout=self.config.timeout_s
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        self.dimension = len(data["data"][0]["embedding"])
                        self.model = model
                        logger.info(f"모델 연결 성공: {model} ({self.dimension}차원)")
                        return
                    elif resp.status_code in [429, 500, 502, 503, 504]:
                        wait_time = (2 ** attempt) + 1
                        logger.warning(f"모델 {model} 재시도 {attempt+1}/3 (대기: {wait_time}초)")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"모델 {model} 연결 실패: HTTP {resp.status_code}")
                        break
                        
                except requests.exceptions.RequestException as e:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"모델 {model} 네트워크 오류 (재시도 {attempt+1}/3): {e}")
                    if attempt < 2:
                        time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"모델 {model} 예상치 못한 오류: {e}")
                    break
        
        raise EmbeddingError("사용 가능한 Upstage 모델이 없습니다")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 (적응형 배치 크기)"""
        if not self.model:
            self._find_working_model()
        
        # 적응형 배치 크기 계산
        avg_text_length = sum(len(text) for text in texts) / len(texts)
        if avg_text_length > 2000:
            MAX_BATCH_SIZE = 25  # 긴 텍스트: 작은 배치
        elif avg_text_length > 1000:
            MAX_BATCH_SIZE = 50  # 중간 텍스트: 중간 배치
        else:
            MAX_BATCH_SIZE = 100  # 짧은 텍스트: 큰 배치
        
        logger.debug(f"평균 텍스트 길이: {avg_text_length:.0f}, 배치 크기: {MAX_BATCH_SIZE}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch_texts = texts[i:i + MAX_BATCH_SIZE]
            cleaned_texts = self._clean_texts(batch_texts)
            
            # 요청 크기 체크
            total_size = sum(len(text.encode('utf-8')) for text in cleaned_texts)
            if total_size > 1024 * 1024:  # 1MB 초과시
                logger.warning(f"큰 요청 크기 감지: {total_size/1024/1024:.1f}MB")
                # 배치를 더 작게 분할
                smaller_batches = [cleaned_texts[j:j+10] for j in range(0, len(cleaned_texts), 10)]
                batch_embeddings = []
                for small_batch in smaller_batches:
                    small_embeddings = self._process_embedding_batch(small_batch)
                    batch_embeddings.extend(small_embeddings)
            else:
                batch_embeddings = self._process_embedding_batch(cleaned_texts)
            
            all_embeddings.extend(batch_embeddings)
            
            if i + MAX_BATCH_SIZE < len(texts):
                time.sleep(0.01)
        
        return all_embeddings
    
    def _process_embedding_batch(self, cleaned_texts: List[str]) -> List[List[float]]:
        """단일 배치 임베딩 처리"""
        try:
            resp = requests.post(
                "https://api.upstage.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": cleaned_texts, "model": self.model},
                timeout=min(self.config.timeout_s * 2, 60)  # 대용량 처리시 더 긴 타임아웃
            )
            
            if resp.status_code != 200:
                error_msg = f"임베딩 API 오류: HTTP {resp.status_code}"
                try:
                    error_detail = resp.json().get('error', {}).get('message', '')
                    if error_detail:
                        error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - Response: {resp.text[:200]}"
                
                # JSON 오류인 경우 텍스트 재정제 시도
                if resp.status_code == 400 and "JSON" in resp.text:
                    logger.warning("JSON 파싱 오류 감지, 텍스트 재정제 시도")
                    retry_texts = []
                    for text in cleaned_texts:
                        # 더 강력한 정제
                        safe_text = ''.join(char for char in text if (ord(char) >= 32 and ord(char) < 127) or (ord(char) >= 0xAC00 and ord(char) <= 0xD7A3))
                        safe_text = safe_text[:500]  # 더 짧게
                        retry_texts.append(safe_text if safe_text else "안전텍스트")
                    
                    # 재시도
                    retry_resp = requests.post(
                        "https://api.upstage.ai/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={"input": retry_texts, "model": self.model},
                        timeout=self.config.timeout_s
                    )
                    
                    if retry_resp.status_code == 200:
                        resp = retry_resp
                        logger.info("재정제 후 성공")
                    else:
                        raise EmbeddingError(error_msg)
                else:
                    raise EmbeddingError(error_msg)
            
            data = resp.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            batch_embeddings = [[float(x) for x in embed] for embed in batch_embeddings]
            
            return batch_embeddings
            
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(f"네트워크 오류: {e}")
        except Exception as e:
            raise EmbeddingError(f"임베딩 처리 오류: {e}")
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
        """텍스트 정제 (최적화된 버전)"""
        cleaned_texts = []
        
        for i, text in enumerate(texts):
            try:
                # 기본 정규화
                cleaned = LegalTextPatterns.normalize_text(text)
                
                # JSON 안전 문자 변환
                cleaned = self._make_json_safe(cleaned)
                
                # 길이 제한
                if len(cleaned) > 4000:
                    cleaned = cleaned[:4000] + "..."
                if not cleaned:
                    cleaned = "내용 없음"
                
                # JSON 유효성 검증
                json.dumps({"test": cleaned})  # 검증용
                
                cleaned_texts.append(cleaned)
                
            except (json.JSONEncodeError, UnicodeEncodeError):
                logger.warning(f"텍스트 {i+1}: 인코딩 실패, 안전 모드 적용")
                safe_text = self._create_safe_text(text)
                cleaned_texts.append(safe_text)
            except re.error as e:
                logger.warning(f"텍스트 {i+1}: 정규식 오류 {e}, 원본 사용")
                cleaned_texts.append(text[:1000] if len(text) > 1000 else text)
            except Exception as e:
                logger.warning(f"텍스트 {i+1}: 예상치 못한 오류 {e}, 안전 모드 적용")
                safe_text = self._create_safe_text(text)
                cleaned_texts.append(safe_text)
        
        return cleaned_texts
    
    def _make_json_safe(self, text: str) -> str:
        """JSON 안전 텍스트 생성 (최적화된 버전)"""
        # 미리 컴파일된 패턴 사용
        text = self._control_chars_pattern.sub('', text)
        text = self._backslash_pattern.sub('\\\\', text)
        text = self._quote_pattern.sub('\\"', text)
        text = self._newline_pattern.sub(' ', text)
        text = self._whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def _create_safe_text(self, original_text: str) -> str:
        """안전한 대체 텍스트 생성 (최적화된 버전)"""
        # 미리 컴파일된 패턴 사용
        safe_chars = self._safe_chars_pattern.findall(original_text)
        safe_text = ''.join(safe_chars)
        safe_text = self._whitespace_pattern.sub(' ', safe_text).strip()
        
        return safe_text[:500] if safe_text else "처리불가텍스트"

# =========================
# 유틸리티 함수들
# =========================
def load_env() -> Tuple[str, str, str, str]:
    """환경변수 로드"""
    load_dotenv()
    keys = {
        "upstage": os.getenv("UPSTAGE_API_KEY", "").strip(),
        "pinecone": os.getenv("PINECONE_API_KEY", "").strip(),
        "index": os.getenv("PINECONE_INDEX", "").strip(),
        "namespace": os.getenv("PINECONE_NAMESPACE", "").strip()
    }
    
    missing = [k for k, v in keys.items() if not v and k != "namespace"]
    if missing:
        missing_keys = ', '.join(f"{k.upper()}_API_KEY" for k in missing)
        raise RuntimeError(f"환경변수 누락: {missing_keys}")
    
    return keys["upstage"], keys["pinecone"], keys["index"], keys["namespace"]

def get_docx_files(dirs: List[str]) -> List[str]:
    """DOCX 파일 목록 수집"""
    files = []
    for d in dirs:
        if os.path.isdir(d):
            files.extend(Path(d).rglob("*.docx"))
    return [str(f) for f in files if not f.name.startswith("~")]

def make_doc_id(path: str) -> str:
    """문서 ID 생성"""
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:12]

def load_processed_files(log_file: str) -> set:
    """처리된 파일 목록 로드"""
    if not os.path.exists(log_file):
        return set()
    
    processed = set()
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                processed.add(line)
    return processed

def save_processed_file(file_path: str, log_file: str):
    """처리된 파일 기록"""
    normalized_path = os.path.abspath(file_path)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{normalized_path}\n")

def filter_files_to_process(files: List[str], config: ProcessingConfig) -> Tuple[List[str], int]:
    """처리할 파일 필터링"""
    if not config.retry_failed_only:
        return files, 0
    
    processed_files = load_processed_files(config.processed_files_log)
    remaining_files = [f for f in files if os.path.abspath(f) not in processed_files]
    skipped_count = len(files) - len(remaining_files)
    
    return remaining_files, skipped_count

def read_docx_content(path: str) -> Tuple[str, List[str], Dict]:
    """DOCX 문서 읽기"""
    if not os.path.exists(path):
        raise DocumentProcessingError(f"파일이 존재하지 않습니다: {path}")
    
    try:
        doc = docx.Document(path)
    except Exception as e:
        raise DocumentProcessingError(f"DOCX 파일 읽기 실패: {e}")
    
    title = doc.core_properties.title or Path(path).stem
    
    # 메타데이터 추출 (구체적 예외 처리)
    try:
        law_info = LegalMetadataExtractor.extract_law_metadata(title)
    except (AttributeError, ImportError) as e:
        logger.warning(f"메타데이터 추출기 오류: {e}")
        law_info = _get_default_law_info(title)
    
    # 단락 추출
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text and len(text) > 10:
            try:
                normalized_text = LegalTextPatterns.normalize_text(text)
                paragraphs.append(normalized_text)
            except re.error as e:
                # 정규식 오류시 원본 텍스트 사용
                logger.debug(f"텍스트 정규화 실패: {e}")
                paragraphs.append(text)
    
    if not paragraphs:
        raise DocumentProcessingError("문서에 유효한 내용이 없습니다")
    
    return title, paragraphs, law_info

def _get_default_law_info(title: str) -> Dict:
    """기본 법령 정보 생성"""
    return {
        "law_name": title,
        "law_type": "기타",
        "law_level": 0,
        "category": "기타",
        "law_field": "기타",
        "law_number": "",
        "date": "",
        "law_scope": "일반"
    }

def connect_pinecone(api_key: str, index_name: str):
    """Pinecone 연결"""
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        logger.info(f"Pinecone 연결 성공: {index_name} (벡터 수: {vector_count})")
        
        return index
    except Exception as e:
        masked_key = f"{api_key[:8]}***{api_key[-4:]}" if len(api_key) > 12 else "***"
        raise VectorDBError(f"Pinecone 연결 실패 - 인덱스: {index_name}, 키: {masked_key}, 오류: {e}")

def delete_existing_vectors(index, doc_id: str, namespace: str = ""):
    """기존 벡터 삭제"""
    try:
        index.delete(filter={"doc_id": doc_id}, namespace=namespace or "")
        logger.info(f"기존 벡터 삭제 완료: {doc_id}")
    except Exception as e:
        logger.warning(f"벡터 삭제 실패 (무시): {e}")

def upsert_vectors(index, items: List[Dict], namespace: str = "", batch_size: int = 20):
    """벡터 업서트"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            logger.info(f"업서트 완료: {i+1}-{i+len(batch)}/{len(items)}")
        except Exception as e:
            error_msg = str(e).lower()
            if "request too large" in error_msg or "payload too large" in error_msg:
                # 배치 크기 줄이기
                smaller_batch_size = max(1, batch_size // 3)
                logger.warning(f"배치 크기 초과, {smaller_batch_size}개씩 재시도")
                
                for j in range(i, min(i + batch_size, len(items)), smaller_batch_size):
                    small_batch = items[j:j + smaller_batch_size]
                    try:
                        index.upsert(vectors=small_batch, namespace=namespace)
                        logger.info(f"소배치 업서트 완료: {j+1}-{j+len(small_batch)}/{len(items)}")
                    except Exception as e2:
                        raise VectorDBError(f"소배치 업서트 실패 {j+1}-{j+len(small_batch)}: {e2}")
            else:
                raise VectorDBError(f"업서트 실패: {e}")

def print_metadata_statistics(items: List[Dict]):
    """메타데이터 통계 출력"""
    if not items:
        return
    
    logger.info(f"메타데이터 통계: 총 벡터 {len(items)}개")
    
    # 내용 유형 분포
    all_types = []
    for item in items:
        all_types.extend(item["metadata"]["content_types"])
    
    if all_types:
        type_counts = Counter(all_types)
        logger.info(f"내용 유형 분포: {dict(type_counts)}")
    
    # 중요도 분포
    importance_scores = [item["metadata"]["importance_score"] for item in items]
    avg_importance = sum(importance_scores) / len(importance_scores)
    high_importance = sum(1 for score in importance_scores if score >= 0.7)
    
    logger.info(f"평균 중요도: {avg_importance:.2f}")
    logger.info(f"고중요도 청크: {high_importance}/{len(items)} ({high_importance/len(items):.1%})")

def process_document(file_path: str, embedder: EnhancedUpstageEmbedder, 
                    chunker: LegalDocumentChunker, 
                    metadata_gen: OptimizedMetadataGenerator) -> Tuple[str, any]:
    """문서 처리"""
    try:
        title, paragraphs, law_info = read_docx_content(file_path)
        chunks = chunker.chunk_document(paragraphs)
        
        if not chunks:
            logger.warning(f"청크 생성 실패: {Path(file_path).name}")
            return file_path, 0
        
        # 임베딩 생성
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_batch(texts)
        
        doc_id = make_doc_id(file_path)
        
        return file_path, (doc_id, title, law_info, chunks, embeddings, paragraphs)
        
    except (DocumentProcessingError, EmbeddingError) as e:
        logger.error(f"문서 처리 실패 {Path(file_path).name}: {e}")
        raise  # 개발 중이므로 에러 발생시키기
    except Exception as e:
        logger.error(f"예상치 못한 오류 {Path(file_path).name}: {e}")
        raise DocumentProcessingError(f"문서 처리 중 예상치 못한 오류: {e}")

def main():
    """메인 함수"""
    logger.info("법령 벡터DB 구축 시작")
    
    # 설정 로드
    config = ProcessingConfig()
    upstage_key, pinecone_key, index_name, namespace = load_env()
    
    # 컴포넌트 초기화
    embedder = EnhancedUpstageEmbedder(upstage_key, config)
    chunker = LegalDocumentChunker(config)
    metadata_gen = OptimizedMetadataGenerator()
    index = connect_pinecone(pinecone_key, index_name)
    
    # 파일 목록 수집
    all_files = get_docx_files(config.input_dirs)
    files_to_process, skipped_count = filter_files_to_process(all_files, config)
    
    logger.info(f"처리 대상: {len(files_to_process)}개")
    if config.retry_failed_only and skipped_count > 0:
        logger.info(f"건너뜀: {skipped_count}개")
    
    if not files_to_process:
        logger.info("모든 파일이 이미 처리되었습니다")
        return
    
    success_count = 0
    
    # 파일 처리
    for i, file_path in enumerate(files_to_process, 1):
        filename = Path(file_path).name
        progress = (i / len(files_to_process)) * 100
        
        logger.info(f"[{progress:5.1f}%] {i}/{len(files_to_process)} - {filename[:50]}...")
        
        try:
            result = process_document(file_path, embedder, chunker, metadata_gen)
            
            if result[1] and isinstance(result[1], tuple):
                doc_id, title, law_info, chunks, embeddings, paragraphs = result[1]
                
                # 기존 벡터 삭제
                if not config.retry_failed_only:
                    delete_existing_vectors(index, doc_id, namespace)
                
                # 메타데이터 생성 (컨텍스트 포함)
                items = []
                for j, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
                    vector_id = f"{doc_id}-{j:05d}"
                    
                    # 컨텍스트 단락 준비 (현재 청크 주변 단락들)
                    context_start = max(0, j - 3)
                    context_end = min(len(paragraphs), j + 2)
                    context_paragraphs = paragraphs[context_start:context_end]
                    
                    metadata = metadata_gen.create_vector_metadata(
                        doc_id, title, law_info, chunk, j, context_paragraphs
                    )
                    
                    items.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                
                # 업서트
                upsert_vectors(index, items, namespace, config.batch_size)
                
                # 처리 완료 기록
                save_processed_file(file_path, config.processed_files_log)
                success_count += 1
                
                logger.info(f"성공: {len(chunks)}개 청크 처리 완료")
                
                # 통계 출력 (첫 번째 파일만)
                if i == 1:
                    print_metadata_statistics(items)
            else:
                logger.warning("스킵: 내용 없음")
                
        except KeyboardInterrupt:
            logger.info(f"사용자 중단 (처리된 파일: {success_count}개)")
            return
        except Exception as e:
            logger.error(f"처리 실패: {e}")
            logger.info(f"처리 완료: {success_count}개")
            raise  # 개발 중이므로 에러 발생시키기
    
    logger.info(f"처리 완료: {success_count}/{len(files_to_process)} 성공")
    logger.info("최적화된 메타데이터로 검색 품질 향상 완료")

if __name__ == "__main__":
    main()