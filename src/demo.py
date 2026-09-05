"""베이스 모델 vs 파인튜닝 모델 검색 성능 비교 데모."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from search import ClubSearchEngine

BASE_DIR = Path(__file__).parent.parent
CLUBS_PATH = str(BASE_DIR / "data" / "clubs.json")
BGE_MODEL = str(BASE_DIR / "models" / "finetuned")
KO_MODEL = str(BASE_DIR / "models" / "finetuned-ko-sroberta")

TEST_QUERIES = [
    "스마클",
    "AI 동아리",
    "인공지능 배우고 싶어",
    "머신러닝 프로젝트",
    "음악 하고 싶어",
    "사진 배우고 싶은데 카메라 없어도 돼?",
    "취업 준비에 도움되는 동아리",
    "코딩 못해도 되는 개발 동아리 있어?",
    "같이 해외여행 갈 사람",
    "그림이나 디자인 배우고 싶어",
]


def run(engine: ClubSearchEngine, queries: list[str], top_k: int = 3):
    for query in queries:
        results = engine.search(query, top_k=top_k)
        print(f"\n  [{query}]")
        for r in results:
            print(f"    {r['score']:.3f}  {r['club']['name']}")


def main():
    print("=" * 60)
    print("bge-m3 파인튜닝 검색 결과")
    print("=" * 60)
    bge = ClubSearchEngine(BGE_MODEL)
    bge.index_from_file(CLUBS_PATH)
    run(bge, TEST_QUERIES)

    print("\n\n" + "=" * 60)
    print("ko-sroberta 파인튜닝 검색 결과")
    print("=" * 60)
    ko = ClubSearchEngine(KO_MODEL)
    ko.index_from_file(CLUBS_PATH)
    run(ko, TEST_QUERIES)


if __name__ == "__main__":
    main()
