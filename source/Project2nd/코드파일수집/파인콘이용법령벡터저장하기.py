import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

import tiktoken
from docx import Document
from openai import OpenAI
import pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocxToPineconeProcessor:
    def __init__(self, index_name: str = "docx-embeddings"):
        """
        DOCX 파일을 Pinecone 벡터 DB에 저장하는 프로세서
        
        Args:
            index_name: Pinecone 인덱스 이름
        """
        self.index_name = index_name
        self.chunk_size = 2000  # 토큰 단위
        self.overlap_size = 100  # 오버랩 토큰
        self.batch_size = 75
        self.min_paragraph_tokens = 300  # 최소 단락 토큰 수
        self.max_paragraph_tokens = 1500  # 최대 단락 토큰 수
        
        # API 클라이언트 초기화
        self._init_clients()
        
        # 토큰 인코더 초기화
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # 처리 상태 추적 파일
        self.status_file = "processing_status.json"
        self.processed_files = self._load_processed_files()
        
    def _init_clients(self):
        """API 클라이언트들 초기화"""
        try:
            # OpenAI 클라이언트
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Pinecone 클라이언트
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # 인덱스 존재 확인 및 생성
            if self.index_name not in [idx.name for idx in pc.list_indexes()]:
                logger.info(f"인덱스 '{self.index_name}' 생성 중...")
                pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large 차원
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                logger.info(f"인덱스 '{self.index_name}' 생성 완료")
            
            self.index = pc.Index(self.index_name)
            logger.info("Pinecone 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"API 클라이언트 초기화 실패: {e}")
            raise
    
    def _load_processed_files(self) -> Dict:
        """처리된 파일 상태 로드"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"상태 파일 로드 실패: {e}")
        return {}
    
    def _save_processed_files(self):
        """처리된 파일 상태 저장"""
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"상태 파일 저장 실패: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """파일 해시값 계산"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"파일 해시 계산 실패 {file_path}: {e}")
            return ""
    
    def _should_process_file(self, file_path: Path) -> bool:
        """파일 처리 필요 여부 확인"""
        file_key = str(file_path)
        current_hash = self._get_file_hash(file_path)
        current_mtime = file_path.stat().st_mtime
        
        if file_key in self.processed_files:
            stored_info = self.processed_files[file_key]
            if (stored_info.get('hash') == current_hash and 
                stored_info.get('mtime') == current_mtime):
                return False
        
        return True
    
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoder.encode(text))
    
    def _extract_text_from_docx(self, file_path: Path) -> List[str]:
        """DOCX 파일에서 단락별로 텍스트 추출"""
        try:
            doc = Document(file_path)
            paragraphs = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # 빈 단락 제외
                    paragraphs.append(text)
            
            logger.info(f"{file_path.name}: {len(paragraphs)}개 단락 추출")
            return paragraphs
            
        except Exception as e:
            logger.error(f"텍스트 추출 실패 {file_path}: {e}")
            return []
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """긴 단락을 문장 단위로 분할"""
        sentences = paragraph.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + sentence + "."
            if self._count_tokens(test_chunk) <= self.max_paragraph_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _process_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """단락들을 토큰 기준으로 적절히 분할/병합"""
        processed = []
        
        i = 0
        while i < len(paragraphs):
            current_para = paragraphs[i]
            current_tokens = self._count_tokens(current_para)
            
            # 너무 긴 단락은 분할
            if current_tokens > self.max_paragraph_tokens:
                split_paras = self._split_long_paragraph(current_para)
                processed.extend(split_paras)
                i += 1
                continue
            
            # 너무 짧은 단락은 다음 단락과 병합 시도
            if current_tokens < self.min_paragraph_tokens and i + 1 < len(paragraphs):
                combined = current_para + "\n\n" + paragraphs[i + 1]
                if self._count_tokens(combined) <= self.max_paragraph_tokens:
                    processed.append(combined)
                    i += 2
                    continue
            
            processed.append(current_para)
            i += 1
        
        return processed
    
    def _create_chunks(self, paragraphs: List[str]) -> List[str]:
        """단락들을 청크로 분할 (오버랩 포함)"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        overlap_text = ""
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            # 현재 청크에 추가 가능한지 확인
            if current_tokens + para_tokens <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
            else:
                # 현재 청크 완성
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 오버랩 텍스트 계산
                overlap_text = self._get_overlap_text(current_chunk)
                
                # 새 청크 시작
                if overlap_text:
                    current_chunk = overlap_text + "\n\n" + para
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = para
                    current_tokens = para_tokens
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """오버랩할 텍스트 추출 (뒤에서부터 100토큰 정도)"""
        sentences = text.split('.')
        overlap = ""
        
        # 뒤에서부터 문장을 추가하며 오버랩 크기 맞추기
        for sentence in reversed(sentences):
            test_overlap = sentence + "." + overlap
            if self._count_tokens(test_overlap) <= self.overlap_size:
                overlap = test_overlap
            else:
                break
        
        return overlap.strip()
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트들의 임베딩 생성"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return []
    
    def _upsert_to_pinecone(self, vectors: List[Tuple], file_path: Path):
        """Pinecone에 벡터 업로드"""
        try:
            # 배치 단위로 업로드
            for i in range(0, len(vectors), self.batch_size):
                batch = vectors[i:i + self.batch_size]
                self.index.upsert(vectors=batch)
                
            logger.info(f"{file_path.name}: {len(vectors)}개 벡터 업로드 완료")
            
        except Exception as e:
            logger.error(f"Pinecone 업로드 실패 {file_path}: {e}")
            raise
    
    def process_file(self, file_path: Path) -> bool:
        """단일 파일 처리"""
        try:
            logger.info(f"파일 처리 시작: {file_path.name}")
            
            # 1. 텍스트 추출
            paragraphs = self._extract_text_from_docx(file_path)
            if not paragraphs:
                logger.warning(f"추출할 텍스트가 없음: {file_path.name}")
                return False
            
            # 2. 단락 전처리
            processed_paragraphs = self._process_paragraphs(paragraphs)
            
            # 3. 청킹
            chunks = self._create_chunks(processed_paragraphs)
            logger.info(f"{file_path.name}: {len(chunks)}개 청크 생성")
            
            if not chunks:
                logger.warning(f"생성된 청크가 없음: {file_path.name}")
                return False
            
            # 4. 임베딩 생성
            embeddings = self._create_embeddings(chunks)
            if not embeddings:
                return False
            
            # 5. Pinecone 업로드용 벡터 준비
            vectors = []
            # 파일명을 ASCII 안전한 ID로 변환
            safe_filename = hashlib.md5(file_path.stem.encode('utf-8')).hexdigest()[:16]
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{safe_filename}_{i}"
                metadata = {
                    "filename": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "token_count": self._count_tokens(chunk),
                    "text": chunk[:1000]  # 텍스트 일부만 메타데이터에 저장
                }
                vectors.append((vector_id, embedding, metadata))
            
            # 6. Pinecone에 업로드
            self._upsert_to_pinecone(vectors, file_path)
            
            # 7. 처리 완료 상태 저장
            self.processed_files[str(file_path)] = {
                "hash": self._get_file_hash(file_path),
                "mtime": file_path.stat().st_mtime,
                "processed_at": datetime.now().isoformat(),
                "chunks_count": len(chunks)
            }
            self._save_processed_files()
            
            logger.info(f"파일 처리 완료: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"파일 처리 실패 {file_path}: {e}")
            return False
    
    def process_folders(self, folder_paths: List[str]):
        """여러 폴더의 DOCX 파일들을 처리"""
        all_files = []
        
        # 모든 폴더에서 DOCX 파일 수집
        for folder_path in folder_paths:
            folder = Path(folder_path)
            if not folder.exists():
                logger.warning(f"폴더가 존재하지 않음: {folder_path}")
                continue
            
            docx_files = list(folder.glob("**/*.docx"))
            logger.info(f"{folder_path}: {len(docx_files)}개 DOCX 파일 발견")
            all_files.extend(docx_files)
        
        if not all_files:
            logger.warning("처리할 DOCX 파일이 없습니다.")
            return
        
        # 중복 제거 필요한 파일만 필터링
        files_to_process = []
        for file_path in all_files:
            if self._should_process_file(file_path):
                files_to_process.append(file_path)
            else:
                logger.info(f"이미 처리됨 (건너뜀): {file_path.name}")
        
        if not files_to_process:
            logger.info("처리할 새로운 파일이 없습니다.")
            return
        
        logger.info(f"총 {len(files_to_process)}개 파일 처리 시작")
        
        # 진행률 표시와 함께 파일 처리
        successful = 0
        failed = 0
        
        for file_path in tqdm(files_to_process, desc="파일 처리 진행"):
            if self.process_file(file_path):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"처리 완료 - 성공: {successful}, 실패: {failed}")


def main():
    """메인 실행 함수"""
    # 처리할 폴더들 지정 (여러 개 가능)
    folders_to_process = [
        "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/금융법령",
        "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/기업법령", 
        "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/보험법령",
        "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/상법투자자산증권주식법령",
        "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/은행법령", 
        "C:/ai_x/source/Project2nd/금융법령수집/금융문제수집/조합법령"
    ]
    
    # 프로세서 생성 및 실행
    processor = DocxToPineconeProcessor(index_name="codedoc-law-index")
    processor.process_folders(folders_to_process)


if __name__ == "__main__":
    main()