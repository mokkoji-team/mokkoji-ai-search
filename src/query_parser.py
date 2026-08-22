"""자연어 검색어 → 구조화된 검색 의도 파서 (Ollama 로컬 LLM 사용)

Ollama가 로컬에서 실행 중이어야 합니다.
  brew install ollama
  ollama serve
  ollama pull qwen2.5:7b   # 또는 exaone3.5:7.8b (한국어 특화)

사용 모델은 OLLAMA_MODEL 환경변수로 변경 가능합니다.
"""

import json
import os
from functools import lru_cache

import ollama
from pydantic import BaseModel, ValidationError

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


class ParsedIntent(BaseModel):
    semantic_query: str
    sort_by: str | None = None           # "recruitment_count" | "member_count" | null
    sort_order: str = "desc"             # "desc" | "asc"
    category_filter: str | None = None  # "SPORTS" | "ACADEMIC_CULTURAL" | "CULTURAL_ART" | "RELIGIOUS" | null
    recruiting_now: bool | None = None  # 현재 모집 중인 동아리만 필터


_SYSTEM = """\
당신은 한양대학교 동아리 검색 서비스(모꼬지)의 쿼리 파서입니다.
사용자의 자연어 검색어를 구조화된 검색 의도로 변환하세요.

동아리 분류 (category_filter):
- SPORTS: 스포츠, 체육, 운동 (축구, 농구, 야구, 수영, 테니스, 배드민턴 등)
- ACADEMIC_CULTURAL: 학술, 학문, 공부, 스터디, IT, 코딩 관련
- CULTURAL_ART: 문화예술, 공연, 음악, 미술, 댄스, 영상 관련
- RELIGIOUS: 종교, 신앙 관련
- 분류가 불명확하거나 여러 분류에 걸치면 null

정렬 기준 (sort_by):
- "recruitment_count": 모집글 수 ("모집글 많이", "가장 많이 모집하는", "활발한", "모집 자주 하는")
- "member_count": 회원 수 ("인기있는", "큰 동아리", "회원 많은", "사람 많은")
- null: 의미 유사도 기준 (기본값, 명시적 정렬 조건 없을 때)

규칙:
- semantic_query: SBERT 임베딩 검색에 쓸 핵심 키워드만 (한국어, 간결하게)
  예) "축구동아리 중에 모집글 많이 하는 곳" → "축구"
  예) "AI 관련 학술 동아리 추천해줘" → "AI 인공지능"
  예) "가장 인기있는 동아리 추천해줘" → ""
- sort_order: 기본 "desc", "적게"/"낮은" 표현이 있으면 "asc"
- recruiting_now: "지금 모집", "현재 모집 중", "바로 지원", "요즘 모집" 표현이 있으면 true, 그 외 null\
"""

_FALLBACK = {"semantic_query": "", "sort_by": None, "sort_order": "desc", "category_filter": None}


@lru_cache(maxsize=512)
def parse_query_intent(query: str) -> dict:
    """자연어 쿼리를 구조화된 검색 의도로 변환. 동일 쿼리는 캐시에서 반환."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ],
        format=ParsedIntent.model_json_schema(),
        options={"temperature": 0},
    )
    try:
        data = json.loads(response.message.content)
        return ParsedIntent(**data).model_dump()
    except (json.JSONDecodeError, ValidationError):
        return {**_FALLBACK, "semantic_query": query}
