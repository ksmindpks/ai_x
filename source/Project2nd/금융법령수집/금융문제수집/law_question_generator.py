
"""
law_question_generator.py

Generate exam-quality questions from a collection of law documents using the OpenAI API.

Features
- Reads .docx files from folders or from unzipped folders.
- Creates Multiple-Choice (4 options) and Short-Answer questions.
- Difficulty distribution per law: High:Medium:Low = 3:4:3
- Type ratio per law: MCQ:Short = 3:1
- ~5% cross-law comparative questions (if enabled).
- Robust batching, incremental autosave, and resume support.
- Writes to a single Excel file with two sheets: "사지선다형", "단답형".
- Deterministic-ish via seed and temperature.

Usage
1) Install dependencies: `pip install -r requirements.txt`
2) Set your OpenAI API key (environment variable): 
   - Linux/macOS: `export OPENAI_API_KEY="sk-..."`
   - Windows (PowerShell): `$env:OPENAI_API_KEY="sk-..."`
3) Edit config.yaml (paths, model, limits, etc.)
4) Run: `python law_question_generator.py --config config.yaml`

Note: Requires official OpenAI Python SDK (>=1.0).
"""

import argparse
import os
import sys
import json
import time
import math
import random
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from docx import Document

# OpenAI SDK (official)
try:
    from openai import OpenAI
except Exception:
    print("ERROR: OpenAI SDK not found. Please install with: pip install openai", file=sys.stderr)
    sys.exit(1)

# ----------------------------
# Utilities
# ----------------------------

def read_docx_text(path: str) -> str:
    doc = Document(path)
    texts = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            texts.append(t)
    return "\n".join(texts)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def chunk_text(text: str, max_chars: int = 12000) -> str:
    """Return the first max_chars characters. Keep it simple to avoid token overflows."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

def load_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(path: str, state: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# ----------------------------
# Config dataclass
# ----------------------------

@dataclass
class Config:
    input_dirs: List[str]
    output_excel: str = "법령문제_통합.xlsx"
    processed_state: str = "processed_state.json"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    seed: int = 42
    per_law_target: int = 10
    mcq_ratio: float = 0.75
    difficulty_ratio: Tuple[float, float, float] = (0.3, 0.4, 0.3)  # (High, Medium, Low)
    crosslaw_ratio: float = 0.05
    batch_size: int = 5                     # number of laws per API batch
    flush_every_batches: int = 1            # flush to disk after this many batches
    resume: bool = True
    max_retries: int = 3
    request_timeout: int = 120
    include_explanations: bool = True

# ----------------------------
# Prompting
# ----------------------------

SYSTEM_PROMPT = """당신은 금융/보험/은행 법령 기반의 출제 전문가입니다.
요구 사항:
- 애매한 정답을 허용하지 말고, 조문에 근거한 명확한 정답만 출제할 것.
- 문제 유형은 4지선다형(MCQ)과 단답형(Short)입니다.
- 4지선다형은 오답 보기를 그럴듯하지만 명확히 틀리게 구성하고, 중복 답/모호한 표현 금지.
- 단답형은 한두 단어 또는 짧은 구로 답이 명확해야 함(문장형 금지).
- 각 문항에는 간결한 해설을 포함.
- 요구한 비율(유형, 난이도)을 최대한 만족하도록 구성.
- 반환 형식은 JSON ONLY.
- 출제된 문제는 자체만으로 다른 것 참조 없이 완성된 문장의 문제가 되게 정리(잘못된 예시1: 법 제4조에서 '대통령령으로 정하는 금액'은 얼마인가?
에서 무슨법인지 모름, 예시2: 개인금융채무자의 정의는 무엇인가? 에서는 질문의 근거 없이 모호한 질문임).
"""

USER_PROMPT_TEMPLATE = """다음은 특정 법령의 축약 본문입니다. 이 내용을 기준으로 문제를 출제하세요.

[법령명]: {law_name}

[법령 본문 (필요 일부만 발췌/요약됨)]
{law_excerpt}

요구사항:
- 총 문항 수(target): {total_questions}
- 유형 비율: MCQ(4지선다) {mcq_pct}% / Short {short_pct}%
- 난이도 비율: 상/중/하 = {diff_high}:{diff_mid}:{diff_low}
- 연계 문제 허용: {enable_crosslaw} (이 요청에서는 타 법령 내용을 임의 창작하지 말고, 주어진 본문 범위 내에서만 출제. 연계형은 동일 문서 내 정의/시행령 차이 등 "명확히 본문에 근거 가능한" 비교만 허용. 불가하면 생략.)
- 반환 형식(JSON) 스키마:
{json_schema}

주의:
- JSON 외 텍스트 절대 출력 금지.
- 각 문항의 "정답"은 보기의 인덱스(1~4) 또는 단답형 정답 문자열로 정확히 채워라.
"""

JSON_SCHEMA = json.dumps(
    {
        "questions": [
            {
                "type": "MCQ or SHORT",
                "difficulty": "HIGH or MEDIUM or LOW",
                "question": "string",
                "choices": ["A", "B", "C", "D"],  # only for MCQ
                "answer": "number for MCQ (1-4) OR short string for SHORT",
                "explanation": "string (brief, cite the clause/definition)"
            }
        ]
    },
    ensure_ascii=False,
    indent=2
)

def build_user_prompt(law_name: str, law_excerpt: str, total_questions: int, cfg: Config, enable_crosslaw: bool) -> str:
    mcq_pct = int(cfg.mcq_ratio * 100)
    short_pct = 100 - mcq_pct
    diff_high = int(cfg.difficulty_ratio[0] * 10)
    diff_mid  = int(cfg.difficulty_ratio[1] * 10)
    diff_low  = int(cfg.difficulty_ratio[2] * 10)

    return USER_PROMPT_TEMPLATE.format(
        law_name=law_name,
        law_excerpt=law_excerpt,
        total_questions=total_questions,
        mcq_pct=mcq_pct,
        short_pct=short_pct,
        diff_high=diff_high,
        diff_mid=diff_mid,
        diff_low=diff_low,
        enable_crosslaw="허용(단, 본문 근거 필수)" if enable_crosslaw else "비활성화",
        json_schema=JSON_SCHEMA
    )

# ----------------------------
# OpenAI client
# ----------------------------

def get_client() -> OpenAI:
    # OPENAI_API_KEY is read from env var
    return OpenAI()

def call_openai(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float, seed: int, timeout: int) -> Optional[dict]:
    """
    Calls Chat Completions API and expects a JSON object in the first choice.
    """
    for attempt in range(1, 6):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                seed=seed,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                timeout=timeout
            )
            content = resp.choices[0].message.content
            data = safe_json_loads(content)
            if not data or "questions" not in data:
                raise ValueError("Invalid JSON or missing 'questions'")
            return data
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[WARN] OpenAI call failed (attempt {attempt}): {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    return None

# ----------------------------
# Generation logic
# ----------------------------

def compute_counts(total: int, mcq_ratio: float, diff_ratio: Tuple[float, float, float]) -> Dict[str, Any]:
    mcq = int(round(total * mcq_ratio))
    short = total - mcq
    # difficulty per total
    d_high = int(round(total * diff_ratio[0]))
    d_mid  = int(round(total * diff_ratio[1]))
    d_low  = max(0, total - d_high - d_mid)
    return {"mcq": mcq, "short": short, "high": d_high, "mid": d_mid, "low": d_low}

def normalize_and_split(questions: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Split questions into MCQ and SHORT, apply minimal validation."""
    mcq, short = [], []
    for q in questions:
        qtype = q.get("type", "").upper()
        diff = q.get("difficulty", "").upper()
        if diff not in {"HIGH", "MEDIUM", "LOW"}:
            continue
        base = {
            "난이도": {"HIGH":"상","MEDIUM":"중","LOW":"하"}[diff],
            "문제내용": q.get("question", "").strip(),
            "해설": q.get("explanation", "").strip()
        }
        if qtype == "MCQ":
            choices = q.get("choices", [])
            ans = q.get("answer", None)
            if not (isinstance(choices, list) and len(choices) == 4):
                continue
            if not (isinstance(ans, int) and 1 <= ans <= 4):
                continue
            row = dict(base)
            row.update({
                "문제유형": "사지선다형",
                "보기1": choices[0], "보기2": choices[1],
                "보기3": choices[2], "보기4": choices[3],
                "정답": ans
            })
            mcq.append(row)
        elif qtype == "SHORT":
            ans = q.get("answer", "")
            if not isinstance(ans, str) or len(ans.strip()) == 0:
                continue
            row = dict(base)
            row.update({
                "문제유형": "단답형",
                "정답": ans.strip()
            })
            short.append(row)
    return mcq, short

def process_law(client: OpenAI, cfg: Config, law_name: str, law_text: str, crosslaw_flag: bool) -> Tuple[List[dict], List[dict]]:
    excerpt = chunk_text(law_text, 12000)
    user_prompt = build_user_prompt(law_name, excerpt, cfg.per_law_target, cfg, enable_crosslaw=crosslaw_flag)
    data = call_openai(client, cfg.model, SYSTEM_PROMPT, user_prompt, cfg.temperature, cfg.seed, cfg.request_timeout)
    if not data:
        return [], []
    mcq, short = normalize_and_split(data.get("questions", []))
    # add law name column
    for row in mcq:
        row["법령명"] = law_name
    for row in short:
        row["법령명"] = law_name
    return mcq, short

# ----------------------------
# Main pipeline
# ----------------------------

def discover_docx(input_dirs: List[str]) -> List[str]:
    paths = []
    for d in input_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".docx"):
                    paths.append(os.path.join(root, f))
    paths.sort()
    return paths

def unzip_if_needed(zip_paths: List[str], dest_root: str) -> List[str]:
    out_dirs = []
    for zp in zip_paths:
        name = os.path.splitext(os.path.basename(zp))[0]
        out_dir = os.path.join(dest_root, name)
        ensure_dir(out_dir)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(out_dir)
        out_dirs.append(out_dir)
    return out_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # Simple YAML loader (avoid PyYAML dependency)
    import re
    def simple_yaml_load(path: str) -> dict:
        content = open(path, "r", encoding="utf-8").read()
        data = {}
        for line in content.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            m = re.match(r'^([A-Za-z0-9_]+)\s*:\s*(.*)$', line.strip())
            if m:
                key, val = m.group(1), m.group(2)
                # try to parse bool/int/float/list
                if val.lower() in ("true", "false"):
                    data[key] = val.lower() == "true"
                elif re.match(r'^\d+$', val):
                    data[key] = int(val)
                elif re.match(r'^\d+\.\d+$', val):
                    data[key] = float(val)
                elif val.startswith("[") and val.endswith("]"):
                    try:
                        data[key] = json.loads(val)
                    except:
                        data[key] = val
                else:
                    data[key] = val
        return data

    y = simple_yaml_load(args.config)

    # Build Config
    cfg = Config(
        input_dirs=[p.strip() for p in y.get("input_dirs", "").split(",") if p.strip()],
        output_excel=y.get("output_excel", "법령문제_통합.xlsx"),
        processed_state=y.get("processed_state", "processed_state.json"),
        model=y.get("model", "gpt-4o-mini"),
        temperature=float(y.get("temperature", 0.2)),
        seed=int(y.get("seed", 42)),
        per_law_target=int(y.get("per_law_target", 10)),
        mcq_ratio=float(y.get("mcq_ratio", 0.75)),
        difficulty_ratio=(
            float(y.get("difficulty_high", 0.3)),
            float(y.get("difficulty_mid", 0.4)),
            float(y.get("difficulty_low", 0.3)),
        ),
        crosslaw_ratio=float(y.get("crosslaw_ratio", 0.05)),
        batch_size=int(y.get("batch_size", 5)),
        flush_every_batches=int(y.get("flush_every_batches", 1)),
        resume=bool(y.get("resume", True)),
        max_retries=int(y.get("max_retries", 3)),
        request_timeout=int(y.get("request_timeout", 120)),
        include_explanations=bool(y.get("include_explanations", True)),
    )

    random.seed(cfg.seed)

    from dotenv import load_dotenv
    # .env 파일 로드
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    client = get_client()

    # Discover .docx
    law_files = discover_docx(cfg.input_dirs)
    if not law_files:
        print("No .docx files found in input_dirs. Please check config.", file=sys.stderr)
        sys.exit(2)

    print(f"Discovered {len(law_files)} law files.")

    # Resume state
    state = load_state(cfg.processed_state) if cfg.resume else {}
    done_set = set(state.get("processed_files", [])) if state else set()

    all_mcq_rows, all_short_rows = [], []
    processed = list(done_set)
    batch_count = 0

    for batch in batched(law_files, cfg.batch_size):
        batch_count += 1
        print(f"\n[Batch {batch_count}] Processing {len(batch)} files...", flush=True)

        for path in batch:
            if path in done_set:
                print(f" - SKIP (already processed): {os.path.basename(path)}")
                continue

            law_name = os.path.splitext(os.path.basename(path))[0]
            try:
                text = read_docx_text(path)
                excerpt = chunk_text(text, 12000)
                # stochastic cross-law activation based on ratio
                crosslaw_flag = (random.random() < cfg.crosslaw_ratio)
                mcq_rows, short_rows = process_law(client, cfg, law_name, excerpt, crosslaw_flag)
                for r in mcq_rows:
                    r["법령명"] = law_name
                for r in short_rows:
                    r["법령명"] = law_name
                all_mcq_rows.extend(mcq_rows)
                all_short_rows.extend(short_rows)
                processed.append(path)
                done_set.add(path)
                # save state per file
                save_state(cfg.processed_state, {"processed_files": processed})
                print(f" - OK: {law_name} (MCQ {len(mcq_rows)}, SHORT {len(short_rows)})")
            except Exception as e:
                print(f" - ERROR processing {law_name}: {e}", file=sys.stderr)

        # flush every N batches
        if batch_count % cfg.flush_every_batches == 0:
            if all_mcq_rows or all_short_rows:
                df_mcq = pd.DataFrame(all_mcq_rows)
                df_short = pd.DataFrame(all_short_rows)
                with pd.ExcelWriter(cfg.output_excel, mode="w") as writer:
                    if not df_mcq.empty:
                        df_mcq[["법령명","문제유형","난이도","문제내용","보기1","보기2","보기3","보기4","정답","해설"]].to_excel(writer, sheet_name="사지선다형", index=False)
                    else:
                        pd.DataFrame(columns=["법령명","문제유형","난이도","문제내용","보기1","보기2","보기3","보기4","정답","해설"]).to_excel(writer, sheet_name="사지선다형", index=False)
                    if not df_short.empty:
                        df_short[["법령명","문제유형","난이도","문제내용","정답","해설"]].to_excel(writer, sheet_name="단답형", index=False)
                    else:
                        pd.DataFrame(columns=["법령명","문제유형","난이도","문제내용","정답","해설"]).to_excel(writer, sheet_name="단답형", index=False)
                print(f"Saved interim results to: {cfg.output_excel}")

    # final save (ensure results are written)
    df_mcq = pd.DataFrame(all_mcq_rows)
    df_short = pd.DataFrame(all_short_rows)
    with pd.ExcelWriter(cfg.output_excel, mode="w") as writer:
        if not df_mcq.empty:
            df_mcq[["법령명","문제유형","난이도","문제내용","보기1","보기2","보기3","보기4","정답","해설"]].to_excel(writer, sheet_name="사지선다형", index=False)
        else:
            pd.DataFrame(columns=["법령명","문제유형","난이도","문제내용","보기1","보기2","보기3","보기4","정답","해설"]).to_excel(writer, sheet_name="사지선다형", index=False)
        if not df_short.empty:
            df_short[["법령명","문제유형","난이도","문제내용","정답","해설"]].to_excel(writer, sheet_name="단답형", index=False)
        else:
            pd.DataFrame(columns=["법령명","문제유형","난이도","문제내용","정답","해설"]).to_excel(writer, sheet_name="단답형", index=False)
    print(f"\nDone. Results saved to: {cfg.output_excel}")
    print(f"Processed files: {len(processed)} / {len(law_files)}")
    print("You can re-run to resume from where you left off.")

if __name__ == "__main__":
    main()
