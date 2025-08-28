# -*- coding: utf-8 -*-
"""
rag/metadata_extractor.py - 법령 메타데이터 추출 모듈 (수정됨)
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

@dataclass
class LawStructure:
    """법령 구조 정보"""
    part: Optional[str] = None      # 편
    chapter: Optional[str] = None   # 장  
    section: Optional[str] = None   # 절
    subsection: Optional[str] = None # 관
    article: Optional[str] = None   # 조
    paragraph: Optional[str] = None # 항
    item: Optional[str] = None      # 호
    subitem: Optional[str] = None   # 목

class LegalMetadataExtractor:
    """법령 메타데이터 추출기 (기존 인터페이스 호환)"""
    
    # 기존 인터페이스를 위한 분야 키워드
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
    
    def __init__(self):
        self.legal_term_cache = {}
        self.stakeholder_patterns = self._compile_stakeholder_patterns()
        self.content_type_patterns = self._compile_content_patterns()
    
    def _compile_stakeholder_patterns(self) -> List[Tuple[str, str]]:
        """이해관계자 패턴 컴파일"""
        return [
            (r'(?:금융위원회|금융감독원|한국은행)', "금융당국"),
            (r'(?:기획재정부|국세청|관세청)', "정부기관"),
            (r'(?:장관|청장|시장|도지사|구청장|군수)', "행정기관"),
            (r'(?:금융기관|은행|신용협동조합)', "금융기관"),
            (r'(?:보험회사|생명보험|손해보험)', "보험기관"),
            (r'(?:증권회사|투자회사|자산운용회사)', "투자기관"),
            (r'(?:신청인|허가받은\s*자|등록한\s*자|신고한\s*자)', "신청자"),
            (r'(?:조합|협회|기관|단체|법인)', "단체")
        ]
    
    def _compile_content_patterns(self) -> Dict[str, List[str]]:
        """내용 유형 패턴 컴파일"""
        return {
            "정의": [
                r'"([^"]+)"\s*(?:이란|라고\s*한다)',
                r'이\s*(?:법|령|규칙)에서\s*"([^"]+)"',
                r'다음\s*각\s*호의\s*(?:용어|어휘)의\s*(?:뜻|의미)'
            ],
            "절차": [
                r'(?:신청|접수|심사|승인|허가|등록|신고).*?(?:절차|방법|순서)',
                r'다음\s*(?:절차|순서|방법)에\s*따라',
                r'(?:\d+일?\s*이내|기간\s*내)에.*?(?:하여야|해야)'
            ],
            "제재": [
                r'(?:벌금|과태료|징역|금고|제재|처벌)',
                r'(?:위반|불법).*?(?:처벌|제재)',
                r'\d+(?:만원|억원)\s*이하의\s*(?:벌금|과태료)'
            ],
            "예외": [
                r'다만,?\s*',
                r'(?:예외|제외).*?(?:한다|있다)',
                r'적용하지\s*(?:아니한다|않는다)'
            ],
            "요건": [
                r'(?:요건|조건|기준|자격).*?(?:갖추어야|구비하여야|충족하여야)',
                r'다음\s*각\s*호의\s*(?:요건|조건)',
                r'(?:자격|요건)을\s*(?:갖춘|구비한)'
            ],
            "기한": [
                r'\d+일?\s*(?:이내|내)',
                r'\d+(?:개월|년)\s*(?:이내|내|이하|미만)',
                r'(?:즉시|지체없이|신속히)'
            ],
            "목적": [
                r'이\s*(?:법|령|규칙)은.*?목적으로\s*한다',
                r'목적.*?(?:증진|향상|보호|발전)',
                r'기본\s*목적'
            ]
        }

    @classmethod
    def extract_law_metadata(cls, title: str) -> Dict:
        """기존 인터페이스 호환 메서드"""
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
        number_patterns = [
            r'제(\d+)호',
            r'법률\s*제(\d+)호',
            r'대통령령\s*제(\d+)호',
            r'부령\s*제(\d+)호'
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, title)
            if match:
                info["law_number"] = match.group(1)
                break
        
        # 날짜 추출
        date_patterns = [
            r'\((\d{8})\)',
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            r'(\d{4}\.\d{1,2}\.\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, title)
            if match:
                if len(match.groups()) == 1:
                    date_str = match.group(1)
                    if len(date_str) == 8:
                        info["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        info["date"] = date_str.replace('.', '-')
                else:
                    year, month, day = match.groups()
                    info["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                break
        
        return info

    def extract_law_structure(self, text: str, context_paragraphs: List[str] = None) -> LawStructure:
        """법령 구조 정보 추출"""
        structure = LawStructure()
        
        # 현재 텍스트에서 구조 요소 추출
        patterns = {
            'part': r'제(\d+)편\s*([^\n]*)',
            'chapter': r'제(\d+)장\s*([^\n]*)',
            'section': r'제(\d+)절\s*([^\n]*)',
            'subsection': r'제(\d+)관\s*([^\n]*)',
            'article': r'제(\d+)조(?:의(\d+))?\s*([^\n]*)',
            'paragraph': r'①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩',
            'item': r'(\d+)\.',
            'subitem': r'([가-힣])\.'
        }
        
        for struct_type, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                try:
                    if struct_type == 'article':
                        article_num = match.group(1)
                        sub_num = match.group(2) if len(match.groups()) >= 2 and match.group(2) else ""
                        structure.article = f"{article_num}{f'의{sub_num}' if sub_num else ''}"
                    elif struct_type in ['paragraph']:
                        # 특수 문자는 매치 여부만 확인
                        setattr(structure, struct_type, "1")
                    else:
                        if len(match.groups()) >= 1:
                            setattr(structure, struct_type, match.group(1))
                except IndexError:
                    # 그룹이 없는 경우 무시
                    pass
        
        # 컨텍스트에서 상위 구조 정보 추론
        if context_paragraphs:
            structure = self._infer_higher_structure(structure, context_paragraphs)
        
        return structure

    def _infer_higher_structure(self, structure: LawStructure, context: List[str]) -> LawStructure:
        """컨텍스트에서 상위 구조 정보 추론"""
        for para in reversed(context):
            if not structure.section and '절' in para:
                match = re.search(r'제(\d+)절', para)
                if match:
                    structure.section = match.group(1)
            
            if not structure.chapter and '장' in para:
                match = re.search(r'제(\d+)장', para)
                if match:
                    structure.chapter = match.group(1)
            
            if not structure.part and '편' in para:
                match = re.search(r'제(\d+)편', para)
                if match:
                    structure.part = match.group(1)
        
        return structure

    def extract_article_relationships(self, text: str, current_article: str = None) -> Dict:
        """조문 간 관계 정보 추출"""
        relationships = {
            "references": [],
            "has_exceptions": False,
            "has_procedures": False,
            "has_penalties": False,
            "has_definitions": False,
            "reference_count": 0
        }
        
        # 조문 참조 추출
        reference_patterns = [
            r'제(\d+(?:의\d+)?)조',
            r'(?:이|그|당해)\s*조',
            r'(?:전|다음)\s*조',
        ]
        
        all_refs = []
        for pattern in reference_patterns:
            try:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        # 튜플인 경우 첫 번째 요소만 사용
                        ref = match[0] if match[0] else ""
                    else:
                        ref = match
                    
                    if ref and (ref.replace('의', '').replace('조', '').isdigit() or ref in ['이', '그', '당해', '전', '다음']):
                        all_refs.append(ref)
            except Exception:
                # 정규식 오류 시 무시
                continue
        
        # 자기 참조 제거
        if current_article:
            all_refs = [ref for ref in all_refs if ref != current_article]
        
        relationships["references"] = list(set(all_refs))[:10]  # 최대 10개
        relationships["reference_count"] = len(all_refs)
        
        # 특수 내용 탐지 (안전한 방식)
        try:
            relationships["has_exceptions"] = any(
                keyword in text for keyword in ["다만", "단서", "예외", "제외"]
            )
            
            relationships["has_procedures"] = any(
                keyword in text for keyword in ["신청", "승인", "허가", "등록", "신고", "절차"]
            )
            
            relationships["has_penalties"] = any(
                keyword in text for keyword in ["벌금", "과태료", "징역", "금고", "제재", "처벌"]
            )
            
            relationships["has_definitions"] = bool(
                re.search(r'"[^"]+"\s*(?:이란|라고\s*한다)', text)
            )
        except Exception:
            # 에러 시 기본값 유지
            pass
        
        return relationships

    def extract_content_types(self, text: str) -> List[str]:
        """내용 유형 분류"""
        content_types = []
        
        for content_type, patterns in self.content_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    content_types.append(content_type)
                    break
        
        return content_types

    def extract_stakeholders(self, text: str) -> List[str]:
        """이해관계자 추출"""
        stakeholders = []
        
        for pattern, category in self.stakeholder_patterns:
            matches = re.findall(pattern, text)
            if matches:
                stakeholders.extend(matches)
        
        return list(set(stakeholders))[:5]  # 최대 5개

    def extract_legal_concepts(self, text: str) -> List[str]:
        """법적 개념 추출"""
        concepts = []
        
        # 정의 구문에서 개념 추출
        definition_matches = re.findall(r'"([^"]*(?:권|의무|책임|자격|면허|허가|승인|신고|등록|신청)[^"]*)"', text)
        concepts.extend(definition_matches)
        
        # 일반적인 법적 개념
        legal_keywords = [
            "권리", "의무", "책임", "자격", "면허", "허가", "승인", "신고", "등록", "신청",
            "계약", "약정", "합의", "협정", "거래", "투자", "융자", "대출", "예금", "적금"
        ]
        
        for keyword in legal_keywords:
            if keyword in text:
                concepts.append(keyword)
        
        return list(set(concepts))[:10]  # 최대 10개

    def extract_temporal_info(self, title: str, text: str) -> Dict:
        """시간 정보 추출"""
        temporal_info = {
            "has_deadlines": False,
            "deadline_periods": [],
            "enforcement_terms": []
        }
        
        # 기한 정보
        deadline_patterns = [
            r'(\d+)일?\s*(?:이내|내)',
            r'(\d+)(?:개월|년)\s*(?:이내|내|이하|미만)',
            r'(?:즉시|지체없이|신속히)'
        ]
        
        for pattern in deadline_patterns:
            matches = re.findall(pattern, text)
            if matches:
                temporal_info["has_deadlines"] = True
                temporal_info["deadline_periods"].extend(matches)
        
        # 시행 관련 용어
        enforcement_terms = ["시행", "적용", "발효", "개시", "종료", "만료"]
        found_terms = [term for term in enforcement_terms if term in text]
        temporal_info["enforcement_terms"] = found_terms
        
        return temporal_info

    def calculate_importance_score(self, text: str, structure: LawStructure, 
                                 relationships: Dict, content_types: List[str]) -> float:
        """중요도 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 구조적 중요도
        if structure.article:
            score += 0.15
            if structure.article in ["1", "2", "3"]:  # 초기 조문들
                score += 0.1
        
        if "목적" in content_types:
            score += 0.2
        if "정의" in content_types:
            score += 0.15
        
        # 관계적 중요도
        ref_count = relationships.get("reference_count", 0)
        if ref_count > 0:
            score += min(ref_count * 0.02, 0.1)
        
        # 내용적 중요도
        important_keywords = ["기본", "원칙", "목적", "정의", "중요", "필수", "의무"]
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        score += min(keyword_count * 0.05, 0.15)
        
        # 길이 보정 (너무 짧거나 긴 것은 중요도 하락)
        text_length = len(text)
        if text_length < 100:
            score *= 0.8
        elif text_length > 2000:
            score *= 0.9
        
        return min(score, 1.0)

    def calculate_complexity_score(self, text: str) -> str:
        """내용 복잡도 계산"""
        factors = {
            "length": len(text),
            "sentences": len(re.findall(r'[.!?。]', text)),
            "legal_terms": len(re.findall(r'(?:따라|의하여|규정|명시|준용|적용)', text)),
            "references": len(re.findall(r'제\d+조', text)),
            "exceptions": len(re.findall(r'다만|단서|예외', text)),
            "numbers": len(re.findall(r'\d+', text))
        }
        
        # 정규화된 복잡도 점수 계산
        complexity_score = (
            min(factors["length"] / 1000, 1) * 0.25 +
            min(factors["sentences"] / 15, 1) * 0.15 +
            min(factors["legal_terms"] / 8, 1) * 0.25 +
            min(factors["references"] / 5, 1) * 0.15 +
            min(factors["exceptions"] / 3, 1) * 0.1 +
            min(factors["numbers"] / 10, 1) * 0.1
        )
        
        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.6:
            return "medium"
        else:
            return "complex"

    def create_enhanced_metadata(self, doc_id: str, title: str, text: str, 
                               chunk_info: Dict, context_paragraphs: List[str] = None) -> Dict:
        """통합 메타데이터 생성"""
        # 기본 메타데이터
        metadata = {
            "doc_id": doc_id,
            "title": title[:200],
            "chunk_index": chunk_info.get("chunk_index", 0),
            "chunk_size": len(text),
            "text": text[:3000] if len(text) > 3000 else text,
            "doc_type": "docx"
        }
        
        # 기존 인터페이스 호환 메타데이터 추가
        law_info = self.extract_law_metadata(title)
        metadata.update({
            "law_type": law_info["law_type"],
            "law_field": law_info["law_field"],
            "law_level": law_info["law_level"],
            "category": law_info["category"]
        })
        
        # 법령 구조 분석
        structure = self.extract_law_structure(text, context_paragraphs)
        metadata.update({
            "law_part": structure.part,
            "law_chapter": structure.chapter,
            "law_section": structure.section,
            "current_article": structure.article,
            "law_paragraph": structure.paragraph
        })
        
        # 조문 관계 분석
        relationships = self.extract_article_relationships(text, structure.article)
        metadata.update({
            "article_references": relationships["references"],
            "reference_count": relationships["reference_count"],
            "has_exceptions": relationships["has_exceptions"],
            "has_procedures": relationships["has_procedures"],
            "has_penalties": relationships["has_penalties"],
            "has_definitions": relationships["has_definitions"]
        })
        
        # 내용 분류
        content_types = self.extract_content_types(text)
        metadata["content_types"] = content_types
        
        # 이해관계자
        stakeholders = self.extract_stakeholders(text)
        metadata["stakeholders"] = stakeholders
        
        # 법적 개념
        legal_concepts = self.extract_legal_concepts(text)
        metadata["legal_concepts"] = legal_concepts
        
        # 시간 정보
        temporal_info = self.extract_temporal_info(title, text)
        metadata.update({
            "has_deadlines": temporal_info["has_deadlines"],
            "deadline_periods": temporal_info["deadline_periods"][:3]
        })
        
        # 점수 계산
        metadata["importance_score"] = self.calculate_importance_score(
            text, structure, relationships, content_types
        )
        metadata["content_complexity"] = self.calculate_complexity_score(text)
        
        return metadata