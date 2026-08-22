"""동아리 검색 엔진 — ClubSearchEngine 클래스와 club_to_text 유틸"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

_kiwi = Kiwi()

_CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
_JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
# 복합 종성 → 구성 단자음 (IME가 두 자음을 하나로 합칠 때 발생하는 오타 처리)
_COMPOUND_JONG = {
    'ㄳ': ['ㄱ', 'ㅅ'], 'ㄵ': ['ㄴ', 'ㅈ'], 'ㄶ': ['ㄴ', 'ㅎ'],
    'ㄺ': ['ㄹ', 'ㄱ'], 'ㄻ': ['ㄹ', 'ㅁ'], 'ㄼ': ['ㄹ', 'ㅂ'],
    'ㄽ': ['ㄹ', 'ㅅ'], 'ㄾ': ['ㄹ', 'ㅌ'], 'ㄿ': ['ㄹ', 'ㅍ'],
    'ㅀ': ['ㄹ', 'ㅎ'], 'ㅄ': ['ㅂ', 'ㅅ'],
}


def _decompose(ch: str):
    code = ord(ch) - 0xAC00
    if not (0 <= code <= 11171):
        return None
    return _CHO[code // 588], _JUNG[(code // 28) % 21], _JONG[code % 28]


def _compose(cho: str, jung: str, jong: str) -> str:
    try:
        return chr(0xAC00 + _CHO.index(cho) * 588 + _JUNG.index(jung) * 28 + _JONG.index(jong))
    except ValueError:
        return cho + jung + jong


@lru_cache(maxsize=512)
def correct_spelling(query: str) -> str:
    """IME 복합 종성 오타 교정 (ㄳ→ㄱ 등). 외부 API 불필요.

    한국어 IME가 ㄱ+ㅅ를 ㄳ으로 합칠 때 생기는 오타만 처리한다.
    3음절 이상 OOV는 고유명사로 판단해 교정하지 않는다.
    """
    tokens = _kiwi.tokenize(query)
    oov_tokens = [t for t in tokens if t.oov]
    if not oov_tokens:
        return query
    # 3음절 이상 OOV는 고유명사로 판단, 교정 건너뜀
    if any(len(t.form) >= 3 for t in oov_tokens):
        return query

    chars = list(query)
    for i, ch in enumerate(chars):
        jamo = _decompose(ch)
        if jamo is None:
            continue
        cho, jung, jong = jamo
        # 복합 종성 분해 시도 (ㄳ→ㄱ/ㅅ 등 IME 오타 — 가장 먼저 시도)
        for simple_jong in _COMPOUND_JONG.get(jong, []):
            trial = ''.join(chars[:i] + [_compose(cho, jung, simple_jong)] + chars[i+1:])
            if not any(t.oov for t in _kiwi.tokenize(trial)):
                print(f"[spell] '{query}' → '{trial}'")
                return trial

    return query


_CATEGORY_KO = {
    "ACADEMIC_CULTURAL": "학술 교양",
    "CULTURAL_ART": "문화 예술",
    "RELIGIOUS": "종교",
    "SPORTS": "스포츠 체육",
    "VOLUNTEER_SOCIAL": "봉사 사회",
}
_AFFILIATION_KO = {
    "CENTRAL_CLUB": "중앙동아리",
    "DEPARTMENT_CLUB": "학과동아리 과동아리",
    "SMALL_GROUP": "소모임",
}

# DB 컬럼 최대 길이. configure_field_limits()로 서버 시작 시 주입된다.
# None = 제약 없음 (text 타입). varchar(N)이면 N.
_FIELD_LIMITS: dict = {}


def configure_field_limits(limits: dict) -> None:
    """DB 컬럼 제약 조건을 주입. serve.py lifespan에서 load_field_limits() 결과를 넘긴다."""
    global _FIELD_LIMITS
    _FIELD_LIMITS = limits


def _lim(key: str, value: str) -> str:
    n = _FIELD_LIMITS.get(key)
    return value[:n] if n else value


def club_to_text(club: dict) -> str:
    """동아리 dict → 검색 인덱싱용 텍스트. train.py와 serve.py 양쪽에서 공유."""
    parts = [_lim("club.name", club["name"])]

    category_ko = _CATEGORY_KO.get(club.get("club_category", ""), "")
    affiliation_ko = _AFFILIATION_KO.get(club.get("club_affiliation", ""), "")
    if category_ko:
        parts.append(category_ko)
    if affiliation_ko:
        parts.append(affiliation_ko)

    if club.get("description"):
        parts.append(_lim("club.description", club["description"]))
    if club.get("tags"):
        parts.append(" ".join(club["tags"]))

    # 인스타그램 핸들 추출 (URL에서 @username 부분만)
    instagram = club.get("instagram") or ""
    if instagram:
        handle = instagram.rstrip("/").split("/")[-1]
        if handle:
            parts.append(handle)

    for r in club.get("recruitments", []):
        if r.get("is_always_open"):
            parts.append("상시모집 언제든지 지원 가능")
        if r.get("title"):
            parts.append(_lim("recruitment.title", r["title"]))
        if r.get("content"):
            parts.append(_lim("recruitment.content", r["content"]))
        # 모집 기간 텍스트
        start = r.get("recruit_start")
        end = r.get("recruit_end")
        if start and end:
            parts.append(f"모집기간 {str(start)[:10]} ~ {str(end)[:10]}")
        elif start:
            parts.append(f"모집시작 {str(start)[:10]}")

    return " ".join(p for p in parts if p).strip()


def _tokenize(text: str) -> list[str]:
    """형태소 분석 기반 토크나이저 (BM25용). 명사·동사·형용사·영문·숫자만 추출."""
    tokens = []
    for token in _kiwi.tokenize(text):
        # NNG/NNP(명사), VV(동사), VA(형용사), SL(영문), SN(숫자)
        if token.tag.startswith(("NN", "VV", "VA", "SL", "SN")):
            tokens.append(token.form)
    return tokens if tokens else re.findall(r'\w+', text)


class ClubSearchEngine:
    def __init__(self, model_path: str, reranker_path: str | None = None):
        print(f"임베딩 모델 로드: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.reranker: CrossEncoder | None = None
        if reranker_path:
            print(f"리랭커 로드: {reranker_path}")
            self.reranker = CrossEncoder(reranker_path)
        self.clubs: list[dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    def _build_bm25(self) -> None:
        tokenized = [_tokenize(club_to_text(c)) for c in self.clubs]
        self.bm25 = BM25Okapi(tokenized)

    def index(self, clubs: list[dict]) -> None:
        self.clubs = clubs
        texts = [club_to_text(c) for c in clubs]
        print(f"  {len(texts)}개 동아리 임베딩 중...")
        self.vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        self._build_bm25()

    def index_from_file(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            clubs = json.load(f)
        self.index(clubs)

    def search(self, query: str, top_k: int | None = None, filter_fn=None, alpha: float = 0.7) -> list[dict]:
        """Hybrid Search: 임베딩(alpha) + BM25(1-alpha) 점수를 결합.

        top_k=None이면 전체 풀을 점수 순으로 반환 (threshold 필터는 호출부에서 처리).
        """
        if self.vectors is None:
            raise RuntimeError("index() 또는 load_index()를 먼저 실행하세요.")

        if filter_fn:
            mask = [i for i, c in enumerate(self.clubs) if filter_fn(c)]
            if not mask:
                return []
            pool_vectors = self.vectors[mask]
            pool_clubs = [self.clubs[i] for i in mask]
        else:
            mask = None
            pool_vectors = self.vectors
            pool_clubs = self.clubs

        # 임베딩 점수 (정규화된 코사인 유사도, -1~1)
        query_vec = self.model.encode(query, normalize_embeddings=True)
        embed_scores = query_vec @ pool_vectors.T
        norm_embed = (embed_scores + 1) / 2  # 0~1로 스케일

        # BM25 점수
        tokens = _tokenize(query)
        all_bm25 = np.array(self.bm25.get_scores(tokens))
        bm25_scores = all_bm25[mask] if mask is not None else all_bm25
        max_bm25 = bm25_scores.max()
        norm_bm25 = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

        # 합산
        hybrid_scores = alpha * norm_embed + (1 - alpha) * norm_bm25

        sorted_indices = np.argsort(hybrid_scores)[::-1]
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        return [
            {"club": pool_clubs[i], "score": float(hybrid_scores[i])}
            for i in sorted_indices
        ]

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """리랭커로 후보 목록을 재정렬. 리랭커가 없으면 원본 순서 유지."""
        if self.reranker is None or not candidates:
            return candidates
        pairs = [(query, club_to_text(r["club"])) for r in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked = [
            {"club": r["club"], "score": float(s), "embed_score": r["score"]}
            for r, s in zip(candidates, rerank_scores)
        ]
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    def update_club_fields(self, club_id: int, fields: dict) -> bool:
        """동아리 기본 정보만 업데이트. 기존 모집글은 유지."""
        for i, c in enumerate(self.clubs):
            if c["id"] == club_id:
                updated = {**c, **fields, "id": club_id, "recruitments": c.get("recruitments", [])}
                self.clubs[i] = updated
                self.vectors[i] = self.model.encode(club_to_text(updated), normalize_embeddings=True)
                self._build_bm25()
                return True
        return False

    def update_club_recruitments(self, club_id: int, recruitments: list) -> bool:
        """모집글만 업데이트. 동아리 기본 정보는 유지."""
        for i, c in enumerate(self.clubs):
            if c["id"] == club_id:
                updated = {**c, "recruitments": recruitments}
                self.clubs[i] = updated
                self.vectors[i] = self.model.encode(club_to_text(updated), normalize_embeddings=True)
                self._build_bm25()
                return True
        return False

    def upsert_club(self, club: dict) -> str:
        """동아리 1개를 추가하거나 업데이트. 반환값: 'added' | 'updated'"""
        club_id = club["id"]
        text = club_to_text(club)
        vector = self.model.encode(text, normalize_embeddings=True)

        for i, c in enumerate(self.clubs):
            if c["id"] == club_id:
                self.clubs[i] = club
                self.vectors[i] = vector
                self._build_bm25()
                return "updated"

        self.clubs.append(club)
        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
        self._build_bm25()
        return "added"

    def remove_club(self, club_id: int) -> bool:
        """동아리 1개를 인덱스에서 제거. 없으면 False 반환."""
        for i, c in enumerate(self.clubs):
            if c["id"] == club_id:
                self.clubs.pop(i)
                self.vectors = np.delete(self.vectors, i, axis=0)
                self._build_bm25()
                return True
        return False

    def save_index(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vectors)
        with open(p / "clubs.json", "w", encoding="utf-8") as f:
            json.dump(self.clubs, f, ensure_ascii=False, indent=2)
        print(f"인덱스 저장 완료: {p} ({len(self.clubs)}개)")

    def load_index(self, path: str) -> None:
        p = Path(path)
        self.vectors = np.load(p / "vectors.npy")
        with open(p / "clubs.json", encoding="utf-8") as f:
            self.clubs = json.load(f)
        self._build_bm25()
        print(f"인덱스 로드 완료: {len(self.clubs)}개 동아리")
