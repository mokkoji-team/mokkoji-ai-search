"""FastAPI 검색 서버 — Spring Boot 백엔드에서 AI 검색 결과를 가져오기 위한 마이크로서비스.

실행: uvicorn src.serve:app --host 0.0.0.0 --port 8000
      (mokkoji-ai/ 루트에서 실행)

Spring Boot 연동: ClubService에서 AI 검색 결과(club_id 목록)를 받아 DB 조회
  POST http://localhost:8000/search
  Body: {"query": "스마클", "top_k": 10}
  Response: {"results": [{"club_id": 1, "score": 0.92}, ...], "query": "스마클"}
"""

import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from typing import Annotated
from pydantic import BaseModel

import sys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from search import ClubSearchEngine
from query_parser import parse_query_intent
from db_loader import db_available, load_clubs_from_db

BASE_DIR = Path(__file__).parent.parent

# bge-m3: 다국어 검색 특화 임베딩 모델 (HuggingFace에서 자동 다운로드)
# 파인튜닝된 모델이 있으면 models/finetuned/ 경로로 변경 가능
MODEL_PATH = "models/finetuned"

# bge-reranker-v2-m3: 임베딩 검색 후 정밀 재정렬
RERANKER_PATH = "BAAI/bge-reranker-v2-m3"

INDEX_PATH = str(BASE_DIR / "index")
CLUBS_PATH = str(BASE_DIR / "data" / "clubs.json")

engine: ClubSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ClubSearchEngine(MODEL_PATH, reranker_path=RERANKER_PATH)
    index_dir = Path(INDEX_PATH)
    if (index_dir / "vectors.npy").exists():
        engine.load_index(INDEX_PATH)
    elif db_available():
        print("인덱스 없음 → DB에서 클럽 데이터 로드 후 빌드")
        clubs = load_clubs_from_db()
        engine.index(clubs)
        engine.save_index(INDEX_PATH)
    else:
        print(f"인덱스 없음 → {CLUBS_PATH}에서 새로 빌드")
        engine.index_from_file(CLUBS_PATH)
        engine.save_index(INDEX_PATH)
    yield


app = FastAPI(title="모꼬지 AI 검색", version="2.0.0", lifespan=lifespan)


def _is_recruiting(club: dict) -> bool:
    """상시모집이거나 오늘이 모집기간 안에 있으면 True."""
    today = date.today()
    for r in club.get("recruitments", []):
        if r.get("is_always_open"):
            return True
        start = r.get("start_date")
        end = r.get("end_date")
        if start and end:
            try:
                if date.fromisoformat(start) <= today <= date.fromisoformat(end):
                    return True
            except ValueError:
                pass
    return False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    threshold: float = 0.4  # 이 점수 미만은 결과에서 제외
    university_code: str | None = None  # 특정 학교 동아리만 검색 (예: "SEJONG", "HANYANG")


class SearchResult(BaseModel):
    club_id: int
    club_name: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str


class NaturalSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    threshold: float = 0.3
    university_code: str | None = None  # 특정 학교 동아리만 검색 (예: "SEJONG", "HANYANG")
    clubs: list[dict] | None = None  # Spring이 recruitment_count 등 DB 데이터를 함께 넘길 때 사용


class NaturalSearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    parsed_intent: dict  # Spring이 sort_by 힌트를 받아 DB 레벨 정렬에 활용


@app.post("/natural-search", response_model=NaturalSearchResponse)
async def natural_search(req: NaturalSearchRequest):
    """자연어 쿼리 처리 엔드포인트.

    1. Ollama(로컬 LLM)로 쿼리 파싱 → {semantic_query, sort_by, category_filter}
    2. bge-m3로 semantic_query 임베딩 검색
    3. category_filter 적용
    4. threshold 필터 적용
    5. bge-reranker로 재정렬
    6. clubs 데이터가 있으면 sort_by 필드로 직접 정렬, 없으면 parsed_intent를 Spring에 반환해 DB 정렬 위임
    """
    if engine is None or engine.vectors is None:
        raise HTTPException(status_code=503, detail="검색 엔진이 준비되지 않았습니다")

    try:
        intent = parse_query_intent(req.query)
    except Exception as e:
        print(f"[query_parser] 파싱 실패, 원본 쿼리로 폴백: {e}")
        intent = {"semantic_query": req.query, "sort_by": None, "sort_order": "desc", "category_filter": None}

    semantic_query = intent["semantic_query"] or req.query

    # 필터가 있으면 후보를 더 뽑되, 리랭커 부담을 줄이기 위해 상한 30개로 제한
    needs_post_filter = bool(intent.get("sort_by") or intent.get("category_filter"))
    candidate_k = min(req.top_k * 3 if needs_post_filter else req.top_k * 2, 30)
    filter_fn = (lambda c: c.get("university_code") == req.university_code) if req.university_code else None
    raw = engine.search(semantic_query, top_k=candidate_k, filter_fn=filter_fn)

    if intent.get("category_filter"):
        raw = [r for r in raw if r["club"].get("club_category") == intent["category_filter"]]

    if intent.get("recruiting_now"):
        raw = [r for r in raw if _is_recruiting(r["club"])]

    raw = [r for r in raw if r["score"] >= req.threshold]

    # 리랭커로 재정렬
    raw = engine.rerank(semantic_query, raw)

    # Spring이 enriched clubs 데이터를 넘겨줬으면 AI 서버에서 직접 정렬
    if intent.get("sort_by") and req.clubs:
        club_map = {c["id"]: c for c in req.clubs}
        reverse = intent.get("sort_order", "desc") == "desc"
        raw.sort(
            key=lambda r: club_map.get(r["club"]["id"], {}).get(intent["sort_by"], 0),
            reverse=reverse,
        )

    results = [
        SearchResult(club_id=r["club"]["id"], club_name=r["club"]["name"], score=r["score"])
        for r in raw[: req.top_k]
    ]
    return NaturalSearchResponse(results=results, query=req.query, parsed_intent=intent)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if engine is None or engine.vectors is None:
        raise HTTPException(status_code=503, detail="검색 엔진이 준비되지 않았습니다")
    filter_fn = (lambda c: c.get("university_code") == req.university_code) if req.university_code else None
    raw = engine.search(req.query, top_k=req.top_k, filter_fn=filter_fn)
    results = [
        SearchResult(club_id=r["club"]["id"], club_name=r["club"]["name"], score=r["score"])
        for r in raw
        if r["score"] >= req.threshold
    ]
    return SearchResponse(results=results, query=req.query)


class RebuildRequest(BaseModel):
    clubs: list[dict] | None = None  # recruitments 포함한 전체 동아리 데이터


@app.post("/index/club")
async def upsert_club(club: Annotated[dict, Body()]):
    """동아리 최초 등록 또는 전체 데이터 업서트. recruitments 포함해서 전송."""
    if engine is None or engine.vectors is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    action = engine.upsert_club(club)
    engine.save_index(INDEX_PATH)
    return {"status": action, "club_id": club.get("id")}


@app.patch("/index/club/{club_id}/fields")
async def update_club_fields(club_id: int, fields: Annotated[dict, Body()]):
    """동아리 기본 정보만 업데이트 (이름, 설명, 카테고리 등). 모집글은 유지."""
    if engine is None or engine.vectors is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    updated = engine.update_club_fields(club_id, fields)
    if not updated:
        raise HTTPException(status_code=404, detail=f"club_id {club_id} 없음")
    engine.save_index(INDEX_PATH)
    return {"status": "updated", "club_id": club_id}


@app.patch("/index/club/{club_id}/recruitments")
async def update_club_recruitments(club_id: int, recruitments: Annotated[list, Body()]):
    """모집글만 업데이트. 동아리 기본 정보는 유지.

    Spring Boot에서 해당 동아리의 전체 모집글 목록을 보내면 됩니다.
    Body 예시: [{"title": "...", "content": "...", "is_always_open": false, "start_date": "2024-03-01", "end_date": "2024-03-31"}]
    """
    if engine is None or engine.vectors is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    updated = engine.update_club_recruitments(club_id, recruitments)
    if not updated:
        raise HTTPException(status_code=404, detail=f"club_id {club_id} 없음")
    engine.save_index(INDEX_PATH)
    return {"status": "updated", "club_id": club_id}


@app.delete("/index/club/{club_id}")
async def delete_club(club_id: int):
    """동아리 삭제. 관련 모집글도 함께 인덱스에서 제거됨."""
    if engine is None or engine.vectors is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    removed = engine.remove_club(club_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"club_id {club_id} 없음")
    engine.save_index(INDEX_PATH)
    return {"status": "removed", "club_id": club_id}


@app.post("/index/rebuild")
async def rebuild_index(req: RebuildRequest | None = None):
    """동아리/모집글 DB 변경 후 인덱스 재빌드.

    Spring Boot에서 recruitments 포함한 clubs 데이터를 body로 전송하면
    모집글 내용까지 임베딩에 포함된다. body 없이 호출하면 clubs.json 파일 사용.

    Spring Boot 연동 예시:
      POST /index/rebuild
      Body: {
        "clubs": [
          {
            "id": 1, "name": "스마클", "description": "...",
            "club_category": "ACADEMIC_CULTURAL",
            "recruitments": [
              {"title": "2024 신입 모집", "content": "..."}
            ]
          }
        ]
      }
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    if req and req.clubs:
        engine.index(req.clubs)
    elif db_available():
        print("DB에서 전체 클럽 로드 후 재빌드")
        engine.index(load_clubs_from_db())
    else:
        engine.index_from_file(CLUBS_PATH)
    engine.save_index(INDEX_PATH)
    return {"status": "ok", "indexed": len(engine.clubs)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedding_model": MODEL_PATH,
        "reranker_model": RERANKER_PATH,
        "indexed_clubs": len(engine.clubs) if engine else 0,
    }
