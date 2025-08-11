import os
import requests
import ast
from openai import OpenAI
from pinecone import Pinecone
from time import sleep
from dotenv import load_dotenv
import logging
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_code_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# ---- 사용자 설정 ----
GITHUB_USER = "python"
GITHUB_REPO = "cpython"
GITHUB_BRANCH = "3.11"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "code-doc-index"
TARGET_EXT = ".py"
# NUM_FILES 제거 - 전체 파일 처리

# 임베딩 모델 설정
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072

logger.info(f"설정: {GITHUB_USER}/{GITHUB_REPO}:{GITHUB_BRANCH}, 모델: {EMBEDDING_MODEL}")
logger.info("전체 Python 파일 처리 모드")

# API 키 확인
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API 키가 .env 파일에 설정되지 않았습니다.")

# ---- 초기화 ----
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스 연결
existing_indexes = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX in existing_indexes:
    logger.info(f"기존 인덱스 '{PINECONE_INDEX}' 삭제 중...")
    pc.delete_index(PINECONE_INDEX)
    sleep(10)  # 삭제 완료 대기
    logger.info(f"기존 인덱스 '{PINECONE_INDEX}' 삭제 완료")

logger.info(f"새 인덱스 '{PINECONE_INDEX}' 생성 중...")
pc.create_index(
    name=PINECONE_INDEX,
    dimension=EMBEDDING_DIMENSION,
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
)
sleep(30)  # 인덱스 초기화 대기
logger.info(f"새 인덱스 '{PINECONE_INDEX}' 생성 완료")

index = pc.Index(PINECONE_INDEX)
HEADERS = {"Accept": "application/vnd.github.v3+json"}

# ---- 1. GitHub에서 파일 탐색 ----
def get_python_files():
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
    
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        tree = r.json().get("tree", [])
        py_files = [f for f in tree if f["path"].endswith(TARGET_EXT)]
        
        # 디렉토리별 파일 수 분석
        dir_stats = {}
        for file in py_files:
            path_parts = file["path"].split('/')
            if len(path_parts) > 1:
                main_dir = path_parts[0]
            else:
                main_dir = "root"
            
            dir_stats[main_dir] = dir_stats.get(main_dir, 0) + 1
        
        logger.info(f"전체 Python 파일: {len(py_files)}개")
        logger.info("디렉토리별 파일 분포:")
        for dir_name, count in sorted(dir_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {dir_name}: {count}개")
        
        return py_files
    except requests.exceptions.RequestException as e:
        logger.error(f"GitHub API 요청 오류: {e}")
        return []

# ---- 2. 파일 다운로드 ----
def download_raw_file(file_path):
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{file_path}"
    
    try:
        r = requests.get(raw_url)
        r.raise_for_status()
        return r.text
    except requests.exceptions.RequestException as e:
        logger.error(f"파일 다운로드 오류 ({file_path}): {e}")
        return None

# ---- 3. AST로 docstring과 코드 분리 ----
def extract_docs_and_code(code):
    pairs = []  # (docstring, code) 쌍으로 저장
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                
                if doc:  # docstring이 있는 경우만
                    try:
                        code_snip = ast.unparse(node)
                        pairs.append((doc, code_snip, node.name, type(node).__name__))
                    except AttributeError:
                        logger.warning("ast.unparse를 사용할 수 없습니다. Python 3.9 이상이 필요합니다.")
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"AST 파싱 오류: {e}")
    
    return pairs

# ---- 4. OpenAI 임베딩 ----
def embed_text_pairs(pairs):
    if not pairs:
        return []
    
    # docstring과 code를 결합한 텍스트 생성
    combined_texts = []
    for doc, code, name, node_type in pairs:
        combined_text = f"# {node_type}: {name}\n\n## Documentation:\n{doc}\n\n## Code:\n{code}"
        combined_texts.append(combined_text)
    
    results = []
    batch_size = 10
    total_batches = (len(combined_texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(combined_texts), batch_size):
        batch = combined_texts[batch_idx:batch_idx+batch_size]
        current_batch = (batch_idx // batch_size) + 1
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [d.embedding for d in response.data]
            results.extend(batch_embeddings)
            logger.info(f"  임베딩 배치 {current_batch}/{total_batches} 완료")
            
        except Exception as e:
            logger.error(f"임베딩 오류: {e}")
            results.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch))
        
        sleep(0.1)
    
    return results

# ---- 5. Pinecone에 저장 ----
def store_pairs_to_pinecone(vectors, pairs, origin):
    if not vectors or not pairs:
        return
    
    items = []
    for i, ((doc, code, name, node_type), vec) in enumerate(zip(pairs, vectors)):
        # 메타데이터에 docstring과 code를 모두 포함
        truncated_doc = doc[:4000] if len(doc) > 4000 else doc
        truncated_code = code[:4000] if len(code) > 4000 else code
        
        items.append({
            "id": f"{origin.replace('/', '-')}-{node_type.lower()}-{name}-{i}",
            "values": vec,
            "metadata": {
                "type": node_type.lower(),
                "name": name,
                "source": origin,
                "docstring": truncated_doc,
                "code": truncated_code
            }
        })
    
    # 재시도 로직
    max_retries = 3
    for attempt in range(max_retries):
        try:
            index.upsert(vectors=items)
            logger.info(f"  {len(items)}개 함수/클래스 쌍 저장 완료")
            return
        except Exception as e:
            logger.warning(f"저장 시도 {attempt + 1} 실패: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                sleep(wait_time)
            else:
                # 백업 파일 생성
                backup_filename = f"backup_pairs_{origin.replace('/', '_')}.json"
                try:
                    with open(backup_filename, 'w', encoding='utf-8') as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)
                    logger.info(f"백업 파일 생성: {backup_filename}")
                except Exception:
                    logger.error("백업 파일 생성 실패")

# ---- 실행 ----
def main():
    start_time = datetime.now()
    logger.info("GitHub 코드 문서화 시작")
    
    try:
        py_files = get_python_files()
        if not py_files:
            logger.error("처리할 파이썬 파일을 찾을 수 없습니다.")
            return
        
        processed_files = 0
        skipped_files = 0
        total_pairs = 0
        
        for idx, file in enumerate(py_files, 1):
            path = file["path"]
            
            # 파일 경로에서 디렉토리 정보 추출
            path_parts = path.split('/')
            main_dir = path_parts[0] if len(path_parts) > 1 else "root"
            
            logger.info(f"[{idx}/{len(py_files)}] {path} ({main_dir})")
            
            raw_code = download_raw_file(path)
            if not raw_code:
                skipped_files += 1
                continue
            
            pairs = extract_docs_and_code(raw_code)
            
            # docstring-code 쌍이 있는 경우만 처리
            if pairs:
                logger.info(f"  처리: {len(pairs)}개 함수/클래스 쌍")
                processed_files += 1
                total_pairs += len(pairs)
                
                # 쌍으로 임베딩 및 저장
                vectors = embed_text_pairs(pairs)
                store_pairs_to_pinecone(vectors, pairs, path)
                
                sleep(0.5)  # API 제한 완화
            else:
                skipped_files += 1
                logger.info("  스킵: docstring이 있는 함수/클래스 없음")
            
            # 진행상황 요약 (100파일마다)
            if idx % 100 == 0:
                logger.info(f"진행상황: {idx}/{len(py_files)} ({idx/len(py_files)*100:.1f}%) - "
                          f"처리됨: {processed_files}, 스킵: {skipped_files}, 총 쌍: {total_pairs}")
        
        # 완료 통계
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 50)
        logger.info("처리 완료")
        logger.info(f"소요 시간: {duration}")
        logger.info(f"전체 파일: {len(py_files)}, 처리: {processed_files}, 스킵: {skipped_files}")
        logger.info(f"총 함수/클래스 쌍: {total_pairs}")
        logger.info(f"총 임베딩: {total_pairs}")
        logger.info(f"처리 비율: {processed_files/len(py_files)*100:.1f}%")
        
        # 처리된 파일들의 디렉토리별 통계
        processed_dirs = {}
        for idx, file in enumerate(py_files):
            if idx < processed_files:  # 대략적인 처리된 파일 추정
                path_parts = file["path"].split('/')
                main_dir = path_parts[0] if len(path_parts) > 1 else "root"
                processed_dirs[main_dir] = processed_dirs.get(main_dir, 0) + 1
        
        if processed_dirs:
            logger.info("처리된 파일의 디렉토리별 분포:")
            for dir_name, count in sorted(processed_dirs.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {dir_name}: {count}개")
        
        logger.info("=" * 50)
        
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()