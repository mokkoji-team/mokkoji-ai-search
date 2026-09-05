"""학습 데이터(train_pairs.json) 자동 생성 스크립트.

동아리/모집글의 모든 필드를 활용해 (query, club_id) 쌍을 생성한다.
새 동아리가 추가되거나 설명이 수정되면 이 스크립트를 다시 실행하면 된다.

사용:
  python src/generate_pairs.py           # 전체 필드 기반 자동 생성
  python src/generate_pairs.py --claude  # Claude API로 사용자 쿼리 추가 생성 (ANTHROPIC_API_KEY 필요)
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent.parent
CLUBS_PATH = BASE_DIR / "data" / "clubs.json"
OUTPUT_PATH = BASE_DIR / "data" / "train_pairs.json"

# 카테고리 → 대표 검색어
CATEGORY_QUERIES = {
    "SPORTS":            ["스포츠", "운동", "체육", "sports"],
    "ACADEMIC_CULTURAL": ["학술", "스터디", "공부", "IT", "코딩", "학문"],
    "CULTURAL_ART":      ["문화예술", "공연", "예술", "문화"],
    "RELIGIOUS":         ["종교", "신앙", "기도"],
    "VOLUNTEER_SOCIAL":  ["봉사", "사회", "나눔", "봉사활동"],
    "OTHER":             [],
}

# 소속 → 대표 검색어
AFFILIATION_QUERIES = {
    "CENTRAL_CLUB":    ["중앙동아리", "중앙 동아리"],
    "DEPARTMENT_CLUB": ["과 동아리", "학과 동아리"],
    "SMALL_GROUP":     ["소모임"],
}

# 학교 → 대표 검색어
UNIVERSITY_QUERIES = {
    "HANYANG": ["한양대", "한양대학교", "HY"],
    "KONKUK":  ["건국대", "건국대학교"],
    "SEJONG":  ["세종대", "세종대학교"],
}


# 두벌식 기준 오타 augmentation
_KB_ADJ_AUG = {
    'ㄱ': ['ㄷ', 'ㅅ'], 'ㄴ': ['ㅇ', 'ㄹ'], 'ㄷ': ['ㄱ', 'ㄴ'],
    'ㄹ': ['ㄴ', 'ㅎ'], 'ㅁ': ['ㄴ'], 'ㅂ': ['ㅈ'],
    'ㅅ': ['ㄱ', 'ㅎ'], 'ㅇ': ['ㄹ', 'ㄴ'], 'ㅈ': ['ㄷ'],
    'ㅎ': ['ㄹ', 'ㅅ'],
}
_COMPOUND_AUG = {
    'ㄳ': 'ㄱ', 'ㄵ': 'ㄴ', 'ㄶ': 'ㄴ', 'ㄺ': 'ㄹ',
    'ㄻ': 'ㄹ', 'ㄼ': 'ㄹ', 'ㄽ': 'ㄹ', 'ㄺ': 'ㄹ',
    'ㅀ': 'ㄹ', 'ㅄ': 'ㅂ',
}
_CHO_AUG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_JUNG_AUG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
_JONG_AUG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']


def _aug_decompose(ch: str):
    code = ord(ch) - 0xAC00
    if not (0 <= code <= 11171):
        return None
    return _CHO_AUG[code // 588], _JUNG_AUG[(code // 28) % 21], _JONG_AUG[code % 28]


def _aug_compose(cho: str, jung: str, jong: str) -> str:
    try:
        return chr(0xAC00 + _CHO_AUG.index(cho) * 588 + _JUNG_AUG.index(jung) * 28 + _JONG_AUG.index(jong))
    except ValueError:
        return cho + jung + jong


def make_typo(query: str) -> str | None:
    """쿼리에서 랜덤한 한 음절의 자음을 인접 키로 교체해 오타 버전 생성."""
    chars = list(query)
    candidates = []
    for i, ch in enumerate(chars):
        jamo = _aug_decompose(ch)
        if jamo is None:
            continue
        cho, jung, jong = jamo
        # 복합 종성 생성 (ㄱ→ㄳ: 실제 IME 오타 재현)
        if jong in ('ㄱ', 'ㄴ', 'ㄹ', 'ㅂ'):
            compound = {'ㄱ': 'ㄳ', 'ㄴ': 'ㄵ', 'ㄹ': 'ㄺ', 'ㅂ': 'ㅄ'}[jong]
            candidates.append((i, _aug_compose(cho, jung, compound)))
        # 인접 종성 교체
        for new_jong in _KB_ADJ_AUG.get(jong, []):
            candidates.append((i, _aug_compose(cho, jung, new_jong)))

    if not candidates:
        return None
    i, new_ch = random.choice(candidates)
    result = ''.join(chars[:i] + [new_ch] + chars[i+1:])
    return result if result != query else None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r'[^\w\s.,!?()·\-]', ' ', text)  # 이모지·특수문자 제거
    sentences = re.split(r'[.!?\n▶▷•·]+', text)
    return [s.strip() for s in sentences if 10 < len(s.strip()) < 100]


def is_english_name(name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9\s\-_\.]+$', name))


def generate_auto_pairs(clubs: list[dict]) -> list[dict]:
    pairs = []
    seen: set[tuple] = set()

    def add(query: str, club_id: int):
        q = query.strip()
        if len(q) > 1 and (q, club_id) not in seen:
            seen.add((q, club_id))
            pairs.append({"query": q, "club_id": club_id})

    for club in clubs:
        cid       = club["id"]
        name      = club["name"]
        desc      = club.get("description") or ""
        category  = club.get("club_category", "")
        affil     = club.get("club_affiliation", "")
        univ      = club.get("university_code", "")
        recs      = club.get("recruitments", [])

        # ── 1. 동아리 이름 ───────────────────────────────────────
        add(name, cid)
        if is_english_name(name):
            add(f"{name} 동아리", cid)

        # ── 2. 설명 문장 (최대 4문장) ────────────────────────────
        for s in split_sentences(desc)[:4]:
            add(s, cid)

        # ── 3. 카테고리 기반 ─────────────────────────────────────
        for q in CATEGORY_QUERIES.get(category, []):
            add(f"{q} 동아리", cid)
            # 학교 + 카테고리 조합
            for uq in UNIVERSITY_QUERIES.get(univ, []):
                add(f"{uq} {q} 동아리", cid)

        # ── 4. 소속 기반 ─────────────────────────────────────────
        for q in AFFILIATION_QUERIES.get(affil, []):
            add(q, cid)
            add(f"{name} {q}", cid)

        # ── 5. 학교 기반 ─────────────────────────────────────────
        for uq in UNIVERSITY_QUERIES.get(univ, []):
            add(f"{uq} {name}", cid)

        # ── 6. 모집글 제목 ───────────────────────────────────────
        for r in recs:
            title = (r.get("title") or "").strip()
            if 2 < len(title) < 80:
                add(title, cid)

        # ── 7. 모집글 내용 문장 (모집글당 최대 3문장) ───────────
        for r in recs:
            content = r.get("content") or ""
            for s in split_sentences(content)[:3]:
                add(s, cid)

        # ── 8. 상시모집 ──────────────────────────────────────────
        if any(r.get("is_always_open") for r in recs):
            add(f"{name} 상시모집", cid)
            add("상시모집 동아리", cid)
            add("언제든지 지원 가능한 동아리", cid)

    # ── 9. 오타 augmentation (동아리 이름 기반, 쌍당 1개) ────────
    random.seed(42)
    for club in clubs:
        cid = club["id"]
        for base_query in [club["name"]] + [
            q for q in CATEGORY_QUERIES.get(club.get("club_category", ""), [])[:2]
        ]:
            typo = make_typo(base_query)
            if typo:
                add(typo, cid)

    return pairs


def generate_claude_pairs(clubs: list[dict]) -> list[dict]:
    """Ollama로 동아리별 사용자 검색 쿼리 생성 (--claude 플래그 시 호출)."""
    import json as _json
    import os
    try:
        import ollama
    except ImportError:
        print("ollama 패키지 필요: pip install ollama")
        return []

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    _SYSTEM = """\
당신은 대학교 동아리 검색 서비스 학습 데이터 생성기입니다.
주어진 동아리 정보를 보고, 실제 대학생이 이 동아리를 찾기 위해 검색창에 입력할 법한 검색어 6개를 JSON 배열로만 반환하세요.

규칙:
- 2~20자의 자연스러운 한국어 (필요 시 영어 약어 포함)
- 동아리 이름 그대로는 제외
- 활동 종류, 목적, 분위기, 관련 키워드 중심
- 다양한 표현: "~동아리", "~하는 곳", "~모임", "~스터디", "~배우는" 등
- 실제 학생이 검색창에 치는 구어체 포함
- 반드시 JSON 배열 형식으로만 출력: ["쿼리1", "쿼리2", ...]"""

    pairs = []
    seen: set[tuple] = set()

    print(f"\nOllama({model})로 {len(clubs)}개 동아리 쿼리 생성 중...")
    for i, club in enumerate(clubs):
        desc = (club.get("description") or "")[:400]
        rec_titles = [r.get("title", "") for r in club.get("recruitments", [])[:3]]
        content = f"""동아리명: {club['name']}
카테고리: {club.get('club_category', '')}
소속: {club.get('club_affiliation', '')}
학교: {club.get('university_code', '')}
설명: {desc}
모집글 제목: {', '.join(rec_titles)}"""
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": content},
                ],
                options={"temperature": 0.7},
            )
            raw = response["message"]["content"].strip()
            # JSON 배열 파싱
            start, end = raw.find("["), raw.rfind("]")
            queries = _json.loads(raw[start:end+1]) if start != -1 else []
            queries = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 1][:6]
            added = 0
            for q in queries:
                key = (q, club["id"])
                if key not in seen:
                    seen.add(key)
                    pairs.append({"query": q, "club_id": club["id"]})
                    added += 1
            print(f"  [{i+1:3d}/{len(clubs)}] {club['name'][:18]:<18} +{added}개: {queries}")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(clubs)}] {club['name'][:18]:<18} 실패: {e}")

    return pairs


# 수동 고품질 쌍 (자동 생성으로 커버 안 되는 별칭·오타·의도 기반 검색어)
CURATED_PAIRS = [
    {"query": "스마클",               "club_id": 36},
    {"query": "SMARCLE",             "club_id": 36},
    {"query": "AI 동아리",            "club_id": 36},
    {"query": "인공지능 동아리",        "club_id": 36},
    {"query": "머신러닝 공부하는 동아리", "club_id": 36},
    {"query": "딥러닝 스터디",          "club_id": 36},
    {"query": "데이터사이언스 동아리",   "club_id": 36},
    {"query": "자연어처리 NLP 배우고 싶어", "club_id": 36},
    {"query": "캐글 대회 참가하는 동아리", "club_id": 36},
    {"query": "AI 프로젝트 해보고 싶어", "club_id": 36},
    {"query": "논문 리뷰 모임",         "club_id": 36},
    {"query": "파이썬 딥러닝 공부",      "club_id": 36},
    {"query": "AI 경진대회 나가는 동아리", "club_id": 36},
    # 그리디 (코드리뷰가 주요 활동으로 명시된 유일한 동아리)
    {"query": "코드리뷰",               "club_id": 60},
    {"query": "코드 리뷰 하는 동아리",   "club_id": 60},
    {"query": "그리디",                 "club_id": 60},
    {"query": "Greedy",                "club_id": 60},
    # 설명/모집글에 한국어 별칭이 명시된 동아리
    {"query": "인트로",      "club_id": 9},    # INTRO
    {"query": "알스아망디",  "club_id": 298},  # ARS AMANDI
    {"query": "하이포",      "club_id": 310},  # HYPO
    {"query": "아엠슈",      "club_id": 328},  # IMSH
    {"query": "뮤트",        "club_id": 329},  # MUTE
    {"query": "나인식스",    "club_id": 13},   # NinesiX
    {"query": "지음",        "club_id": 14},   # ZIUM
    {"query": "라이프러리",  "club_id": 32},   # LIFrary
    {"query": "아퀴",        "club_id": 271},  # AQUI
    {"query": "쿠필",        "club_id": 272},  # Kuphil
    {"query": "후세문화회",  "club_id": 345},  # HUSE-C
    {"query": "오파츠",      "club_id": 350},  # OOPArts
    {"query": "푸글",        "club_id": 34},   # Foogle
    {"query": "지지",        "club_id": 86},   # ZIZZY
    {"query": "하이볼",      "club_id": 335},  # HY-BALL
    {"query": "건대를 달리다", "club_id": 262}, # RIKU
    # 영어 이름 한국어 발음
    {"query": "싸이",        "club_id": 27},   # SAI
    {"query": "사이",        "club_id": 27},   # SAI
    {"query": "페이지세븐",  "club_id": 11},   # PAGE7
    {"query": "노드",        "club_id": 26},   # NODE
    {"query": "메쉬",        "club_id": 25},   # MESH
    {"query": "러더",        "club_id": 29},   # RUDDER
    {"query": "프롬프트",    "club_id": 33},   # Prompt
    {"query": "백스트릿",    "club_id": 74},   # BACKSTREET
    {"query": "세짐",        "club_id": 99},   # SEGYM
    {"query": "글리",        "club_id": 299},  # GLEE
    {"query": "매직아이",    "club_id": 300},  # Magic-i
    {"query": "쇼다운",      "club_id": 301},  # SHOW DOWN
    {"query": "랙트",        "club_id": 303},  # RACT
    {"query": "하마",        "club_id": 340},  # HAMA
    {"query": "하스라",      "club_id": 341},  # HASRA
    {"query": "포리프",      "club_id": 343},  # FORIF
    {"query": "허스",        "club_id": 344},  # HUHS
    {"query": "쿠사",        "club_id": 346},  # KUSA
    {"query": "엣지",        "club_id": 222},  # EDGE
    {"query": "쿠니멀",      "club_id": 236},  # Kunimal
    {"query": "뮤즈",        "club_id": 273},  # MUSE
    {"query": "점프",        "club_id": 51},   # JUMP
    {"query": "이스케이퍼",  "club_id": 55},   # ESCAPER
    {"query": "더아이즈",    "club_id": 61},   # THE EYES
    {"query": "케미스피어",  "club_id": 80},   # CHEMISPHERE
    {"query": "비쿠",        "club_id": 254},  # BiKU
    {"query": "리쿠",        "club_id": 262},  # RIKU
    {"query": "헥",          "club_id": 342},  # HECC
    # 나머지 영문 동아리 한국어 발음
    {"query": "타임",        "club_id": 348},  # TIME (한양대)
    {"query": "타임",        "club_id": 230},  # TIME (건국대)
    {"query": "씨씨씨",      "club_id": 315},  # CCC (한양대)
    {"query": "씨씨씨",      "club_id": 42},   # CCC (세종대)
    {"query": "아이브이에프", "club_id": 317},  # IVF (한양대)
    {"query": "아이브이에프", "club_id": 43},   # IVF (세종대)
    {"query": "이에스에프",  "club_id": 316},  # ESF
    {"query": "햄",          "club_id": 220},  # HAM
    {"query": "하이티씨",    "club_id": 327},  # HYTC
    {"query": "피프틴어스",  "club_id": 330},  # 15ers
    {"query": "에스에스씨",  "club_id": 347},  # SSC
    {"query": "디에스엠",    "club_id": 349},  # DSM
    {"query": "씨이에스",    "club_id": 219},  # CES
    {"query": "이목",        "club_id": 221},  # IMOK
    {"query": "이프",        "club_id": 229},  # IF
    {"query": "운사",        "club_id": 240},  # UNSA
    {"query": "유엔에스에이", "club_id": 240},  # UNSA
    {"query": "케이티씨",    "club_id": 250},  # KTC
    {"query": "에스에스지",  "club_id": 31},   # S.S.G
    {"query": "제이와이엠",  "club_id": 44},   # JYM
    {"query": "에스티씨",    "club_id": 48},   # STC
    {"query": "티에스피",    "club_id": 52},   # TSP
    {"query": "탑",          "club_id": 76},   # TOP
    {"query": "에스유에이브이", "club_id": 77}, # s.UAV
    {"query": "에프씨지지",  "club_id": 79},   # FCGG
    {"query": "에프씨쿠니브", "club_id": 249}, # FC KUNIV
]


def main():
    use_claude = "--claude" in sys.argv

    with open(CLUBS_PATH, encoding="utf-8") as f:
        clubs = json.load(f)

    print(f"동아리 수: {len(clubs)}개")

    auto_pairs = generate_auto_pairs(clubs)
    print(f"자동 생성 쌍: {len(auto_pairs)}개")

    valid_ids = {c["id"] for c in clubs}
    curated = [p for p in CURATED_PAIRS if p["club_id"] in valid_ids]
    print(f"수동 curated 쌍: {len(curated)}개")

    seen = {(p["query"], p["club_id"]) for p in curated}
    combined = curated[:]
    for p in auto_pairs:
        key = (p["query"], p["club_id"])
        if key not in seen:
            seen.add(key)
            combined.append(p)

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


if __name__ == "__main__":
    main()
