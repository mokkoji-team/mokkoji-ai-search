"""학습 데이터(train_pairs.json) 자동 생성 스크립트.

clubs.json의 실제 동아리 데이터에서 (query, club_id) 쌍을 자동으로 만든다.
새 동아리가 추가되거나 설명이 수정되면 이 스크립트를 다시 실행하면 된다.

사용:
  python src/generate_pairs.py           # 이름/설명 기반 자동 생성만
  python src/generate_pairs.py --claude  # Claude API로 사용자 쿼리 추가 생성 (ANTHROPIC_API_KEY 필요)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent.parent
CLUBS_PATH = BASE_DIR / "data" / "clubs.json"
OUTPUT_PATH = BASE_DIR / "data" / "train_pairs.json"

# 실제 사용자 검색 패턴 기반으로 직접 작성한 고품질 쌍
# (자동 생성만으로는 커버 못하는 별칭·오타·의도 기반 검색어)
CURATED_PAIRS = [
    # SMARCLE (id=36, 세종대 AI 학술동아리)
    {"query": "스마클", "club_id": 36},
    {"query": "SMARCLE", "club_id": 36},
    {"query": "AI 동아리", "club_id": 36},
    {"query": "인공지능 동아리", "club_id": 36},
    {"query": "머신러닝 공부하는 동아리", "club_id": 36},
    {"query": "딥러닝 스터디", "club_id": 36},
    {"query": "데이터사이언스 동아리", "club_id": 36},
    {"query": "자연어처리 NLP 배우고 싶어", "club_id": 36},
    {"query": "캐글 대회 참가하는 동아리", "club_id": 36},
    {"query": "AI 프로젝트 해보고 싶어", "club_id": 36},
    {"query": "논문 리뷰 모임", "club_id": 36},
    {"query": "파이썬 딥러닝 공부", "club_id": 36},
    {"query": "AI 경진대회 나가는 동아리", "club_id": 36},
]


def split_sentences(text: str) -> list[str]:
    """설명에서 문장을 분리한다."""
    # 이모지·특수문자 제거 후 문장 분리
    text = re.sub(r'[😊🙌🏻💪🔥✨🎉🎵🎶🎸🎤🎷🎺🏃🏋️⚽🎨📸📚✈️🌏]+', '', text)
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# 노이즈가 많아 제거됨 — 첫 문장 방식으로 대체


def is_english_name(name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9\s\-_\.]+$', name))


def generate_auto_pairs(clubs: list[dict]) -> list[dict]:
    pairs = []
    seen = set()

    def add(query: str, club_id: int):
        key = (query.strip(), club_id)
        if key not in seen and len(query.strip()) > 1:
            seen.add(key)
            pairs.append({"query": query.strip(), "club_id": club_id})

    for club in clubs:
        cid = club["id"]
        name = club["name"]
        desc = club.get("description") or ""

        # 1. 클럽 이름 그대로 (띄어쓰기 정규화)
        add(name, cid)

        # 2. 영어 이름이면 "XX 동아리" 형태 추가
        if is_english_name(name):
            add(f"{name} 동아리", cid)

        # 3. 설명 첫 문장 (사용자가 설명 읽고 검색하는 경우)
        if desc:
            sentences = split_sentences(desc)
            if sentences:
                first = sentences[0]
                if len(first) < 120:
                    add(first, cid)

        # 4. 설명 두 번째 문장 (첫 문장이 동아리 소개, 두 번째가 활동 설명인 경우가 많음)
        if desc:
            sentences = split_sentences(desc)
            if len(sentences) >= 2:
                second = sentences[1]
                if len(second) < 100:
                    add(second, cid)

    return pairs


def generate_claude_pairs(clubs: list[dict]) -> list[dict]:
    """Claude API로 동아리별 사용자 검색 쿼리 생성 (--claude 플래그 시 호출)."""
    try:
        import anthropic
        from pydantic import BaseModel
    except ImportError:
        print("anthropic 패키지 필요: pip install anthropic")
        return []

    class ClubQueries(BaseModel):
        queries: list[str]

    _SYSTEM = """\
당신은 대학교 동아리 검색 서비스 학습 데이터 생성기입니다.
주어진 동아리 정보를 보고, 실제 대학생이 이 동아리를 찾기 위해 검색창에 입력할 법한 검색어 6개를 생성하세요.

규칙:
- 2~20자의 자연스러운 한국어 (필요 시 영어 약어 포함)
- 동아리 이름 그대로는 제외 (별도로 추가됨)
- 활동 종류, 목적, 분위기, 관련 키워드 중심
- 다양한 표현: "~동아리", "~하는 곳", "~모임", "~스터디", "~배우는" 등
- 실제 학생이 검색창에 치는 구어체 포함
- 정확히 6개 반환\
"""

    client = anthropic.Anthropic()
    pairs = []
    seen: set[tuple] = set()

    print(f"\nClaude로 {len(clubs)}개 동아리 쿼리 생성 중...")
    for i, club in enumerate(clubs):
        desc = (club.get("description") or "")[:400]
        content = f"""동아리명: {club['name']}
카테고리: {club.get('club_category', '')}
설명: {desc}"""
        try:
            response = client.messages.parse(
                model="claude-opus-4-8",
                max_tokens=300,
                system=_SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_format=ClubQueries,
            )
            queries = response.parsed_output.queries[:6]
            added = 0
            for q in queries:
                key = (q.strip(), club["id"])
                if key not in seen and len(q.strip()) > 1:
                    seen.add(key)
                    pairs.append({"query": q.strip(), "club_id": club["id"]})
                    added += 1
            print(f"  [{i+1:3d}/{len(clubs)}] {club['name'][:18]:<18} +{added}개: {queries}")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(clubs)}] {club['name'][:18]:<18} 실패: {e}")

    return pairs


def main():
    use_claude = "--claude" in sys.argv

    with open(CLUBS_PATH, encoding="utf-8") as f:
        clubs = json.load(f)

    print(f"동아리 수: {len(clubs)}개")

    # 자동 생성 쌍
    auto_pairs = generate_auto_pairs(clubs)
    print(f"자동 생성 쌍: {len(auto_pairs)}개")

    # 수동 curated 쌍 (club_id 유효성 확인)
    valid_ids = {c["id"] for c in clubs}
    curated = []
    for p in CURATED_PAIRS:
        if p["club_id"] in valid_ids:
            curated.append(p)
        else:
            print(f"  경고: club_id {p['club_id']} 없음 — '{p['query']}' 건너뜀")

    print(f"수동 curated 쌍: {len(curated)}개")

    # 중복 제거 후 합치기 (curated 우선)
    seen = {(p["query"], p["club_id"]) for p in curated}
    combined = curated[:]
    for p in auto_pairs:
        key = (p["query"], p["club_id"])
        if key not in seen:
            seen.add(key)
            combined.append(p)

    # Claude 쌍 추가
    if use_claude:
        claude_pairs = generate_claude_pairs(clubs)
        for p in claude_pairs:
            key = (p["query"], p["club_id"])
            if key not in seen:
                seen.add(key)
                combined.append(p)
        print(f"Claude 생성 쌍: {len(claude_pairs)}개")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(combined)}개 쌍 저장 → {OUTPUT_PATH}")
    if use_claude:
        print("다음 단계: python src/train.py")


if __name__ == "__main__":
    main()
