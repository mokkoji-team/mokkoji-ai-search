"""대화형 동아리 검색 CLI. 단어를 입력하면 관련 동아리 목록을 출력한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from search import ClubSearchEngine

BASE_DIR = Path(__file__).parent.parent
CLUBS_PATH = str(BASE_DIR / "data" / "clubs.json")
FINETUNED_MODEL = str(BASE_DIR / "models" / "finetuned")

CATEGORY_KO = {
    "ACADEMIC_CULTURAL": "학술/교양",
    "CULTURAL_ART": "문화/예술",
    "SPORTS": "체육",
    "VOLUNTEER_SOCIAL": "봉사/사회",
    "RELIGIOUS": "종교",
    "OTHER": "기타",
}

AFFILIATION_KO = {
    "CENTRAL_CLUB": "중앙",
    "DEPARTMENT_CLUB": "학과",
    "SMALL_GROUP": "소모임",
}


def main():
    print("모꼬지 동아리 검색 (파인튜닝 모델)")
    print("=" * 50)

    engine = ClubSearchEngine(FINETUNED_MODEL)
    engine.index_from_file(CLUBS_PATH)

    print("\n검색어를 입력하세요. (종료: q 또는 Ctrl+C)\n")

    while True:
        try:
            query = input("검색 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        results = engine.search(query, top_k=10)

        print()
        for i, r in enumerate(results, 1):
            club = r["club"]
            score = r["score"]
            category = CATEGORY_KO.get(club.get("club_category", ""), "")
            affiliation = AFFILIATION_KO.get(club.get("club_affiliation", ""), "")
            uni = club.get("university_code", "")
            desc = (club.get("description") or "")[:60]
            desc = desc + "..." if len(club.get("description") or "") > 60 else desc

            print(f"  {i:2}. [{score:.3f}] {club['name']}")
            print(f"       {uni} | {category} | {affiliation}")
            if desc:
                print(f"       {desc}")
        print()


if __name__ == "__main__":
    main()
