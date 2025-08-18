# -*- coding: utf-8 -*-
"""
rag/llm_bridge.py
- LLM 클라이언트 초기화 및 JSON 응답 헬퍼
- OpenAI / Upstage 자동 감지. 키가 없으면 None(비활성)
"""

from __future__ import annotations
import os, re, json
from typing import Optional, Dict, Any, List, Tuple

_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
_MODEL    = (os.getenv("LLM_MODEL")    or ("gpt-4o-mini" if _PROVIDER=="openai" else "solar-pro")).strip()
_TEMP     = float(os.getenv("LLM_TEMPERATURE") or 0.0)
_MAXTOK   = int(os.getenv("LLM_MAX_TOKENS") or 80)

_client = None
_client_mode = None  # "openai" | "upstage" | None


def init_llm_client():
    global _client, _client_mode
    if _client is not None:
        return _client

    try:
        if _PROVIDER == "openai":
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                _client = None
                _client_mode = None
                return None
            _client = OpenAI(api_key=key)
            _client_mode = "openai"
            return _client

        elif _PROVIDER == "upstage":
            # langchain-upstage의 ChatUpstage 사용 (채팅 인터페이스)
            from langchain_upstage import ChatUpstage
            key = os.getenv("UPSTAGE_API_KEY")
            if not key:
                _client = None
                _client_mode = None
                return None
            _client = ChatUpstage(model=_MODEL)
            _client_mode = "upstage"
            return _client

        else:
            _client = None
            _client_mode = None
            return None

    except Exception:
        _client = None
        _client_mode = None
        return None


def llm_available() -> bool:
    return init_llm_client() is not None


def ask_json(system_prompt: str, user_prompt: str, schema: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    간단 JSON 스키마(키: str)로 응답을 강제.
    실패/비활성 시 None.
    """
    cli = init_llm_client()
    if not cli:
        return None

    try:
        if _client_mode == "openai":
            res = cli.responses.create(
                model=_MODEL,
                temperature=_TEMP,
                max_output_tokens=_MAXTOK,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            text = ""
            # responses.create: SDK 버전에 따라 output_text 또는 choices 경로
            if hasattr(res, "output_text"):
                text = res.output_text
            elif hasattr(res, "choices") and res.choices and hasattr(res.choices[0], "message"):
                text = res.choices[0].message.content or ""
            else:
                text = str(res)

        elif _client_mode == "upstage":
            # langchain ChatUpstage
            msg = [("system", system_prompt), ("user", user_prompt)]
            out = cli.invoke(msg)
            text = getattr(out, "content", str(out))

        else:
            return None

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        clean: Dict[str, str] = {}
        for k in schema.keys():
            v = data.get(k, "")
            if v is None:
                v = ""
            clean[k] = str(v).strip()
        return clean
    except Exception:
        return None


def ctx_join_for_llm(contexts: List[Dict[str, Any]], max_n: int = 6) -> str:
    parts = []
    for i, c in enumerate((contexts or [])[:max_n], 1):
        t = (c.get("text", "") or "").strip().replace("\n", " ")
        parts.append(f"[{i}] {t}")
    return "\n".join(parts)
