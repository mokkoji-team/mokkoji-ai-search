"""동아리 검색 엔진 — ClubSearchEngine 클래스와 club_to_text 유틸"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

_kiwi = Kiwi()


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


def club_to_text(club: dict) -> str:
    """동아리 dict → 검색 인덱싱용 텍스트. train.py와 serve.py 양쪽에서 공유."""
    parts = [club["name"]]

    category_ko = _CATEGORY_KO.get(club.get("club_category", ""), "")
    affiliation_ko = _AFFILIATION_KO.get(club.get("club_affiliation", ""), "")
    if category_ko:
        parts.append(category_ko)
    if affiliation_ko:
        parts.append(affiliation_ko)

    if club.get("description"):
        parts.append(club["description"])
    if club.get("tags"):
        parts.append(" ".join(club["tags"]))

    for r in club.get("recruitments", []):
        if r.get("is_always_open"):
            parts.append("상시모집 언제든지 지원 가능")
        if r.get("title"):
            parts.append(r["title"])
        if r.get("content"):
            parts.append(r["content"][:500])
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

    def search(self, query: str, top_k: int = 5, filter_fn=None, alpha: float = 0.7) -> list[dict]:
        """Hybrid Search: 임베딩(alpha) + BM25(1-alpha) 점수를 결합."""
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

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            {"club": pool_clubs[i], "score": float(hybrid_scores[i])}
            for i in top_indices
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
