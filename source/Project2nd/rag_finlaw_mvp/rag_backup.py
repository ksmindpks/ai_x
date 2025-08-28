import os
import zipfile
from datetime import datetime

def backup_project_files(output_dir="backup"):
    print("[START] 백업 스크립트 실행 시작", flush=True)

    # root에 있는 모든 .py 파일
    root_files = [f for f in os.listdir(".") if f.endswith(".py")]

    # rag 폴더 안의 모든 .py 파일
    rag_files = []
    rag_dir = "rag"
    if os.path.exists(rag_dir):
        for f in os.listdir(rag_dir):
            if f.endswith(".py"):
                rag_files.append(os.path.join(rag_dir, f))

    # 백업 대상 전체
    files_to_backup = root_files + rag_files

    # 백업 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # zip 파일 이름 (시간 스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(output_dir, f"rag_backup_{timestamp}.zip")

    # zip 생성
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file in files_to_backup:
            if os.path.exists(file):
                zipf.write(file, arcname=file)  # 구조 보존
                print(f"[✓] {file} 백업 완료", flush=True)
            else:
                print(f"[!] {file} 없음 → 건너뜀", flush=True)

    print(f"\n[END] 백업 완료: {os.path.abspath(zip_filename)}", flush=True)

if __name__ == "__main__":
    backup_project_files()
