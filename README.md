# mokkoji-ai

모꼬지 동아리 검색 AI — Hybrid Search(임베딩 + BM25) 기반 의미 검색 마이크로서비스.

"SMARCLE" 이름을 몰라도 "AI 동아리", "스마클", "머신러닝 공부" 같은 검색어로 찾을 수 있게 한다.

## 구조

```
mokkoji-ai/
├── data/
│   ├── clubs.json          # 동아리 데이터 (초기 시드용, 실제 서비스는 DB에서 로드)
│   └── train_pairs.json    # 학습 데이터 (query → club_id 쌍)
├── src/
│   ├── search.py           # ClubSearchEngine (Hybrid Search + 리랭커)
│   ├── query_parser.py     # 자연어 쿼리 파서 (Ollama)
│   ├── db_loader.py        # prod DB에서 동아리+모집글 로드
│   ├── serve.py            # FastAPI 검색 서버 (Spring Boot 연동)
│   ├── train.py            # bge-m3 파인튜닝
│   ├── demo.py             # 파인튜닝 전/후 비교 데모
│   └── search_cli.py       # 로컬 테스트용 대화형 CLI
├── models/finetuned/       # 파인튜닝된 모델 (학습 후 생성)
└── index/                  # 벡터 인덱스 (서버 시작 시 자동 생성)
```

## 설치

```bash
pip install -r requirements.txt
```

## 서버 실행

### 1. SSH 터널 열기 (prod DB 연결)

```bash
cd ~/mokkoji-auto-ingest && ./tunnel.sh prod
```

### 2. `.env` 설정

```bash
cp .env.example .env
# .env에 실제 DB 접속 정보 입력
```

### 3. 서버 시작

```bash
python3 -m uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

- 인덱스(`index/`)가 없으면 DB에서 자동 로드 후 빌드
- DB 환경변수 없으면 `data/clubs.json` 파일로 폴백

### Ollama 설정 (`/natural-search` 사용 시 필요)

```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b
```

## API

Swagger UI: **http://localhost:8000/docs**

### POST /search — Hybrid Search

```json
// Request
{
  "query": "스마클",
  "top_k": 10,
  "threshold": 0.4,
  "university_code": "HANYANG"  // 선택: HANYANG | KONKUK | SEJONG
}

// Response
{
  "query": "스마클",
  "results": [
    {"club_id": 36, "club_name": "SMARCLE", "score": 0.91},
    {"club_id": 2,  "club_name": "인터페이스", "score": 0.54}
  ]
}
```

### POST /natural-search — 자연어 검색

```json
// Request
{
  "query": "지금 모집 중인 축구 동아리 추천해줘",
  "top_k": 10,
  "threshold": 0.3,
  "university_code": "SEJONG"
}

// Response
{
  "query": "지금 모집 중인 축구 동아리 추천해줘",
  "results": [
    {"club_id": 63, "club_name": "FC해례본", "score": 0.85}
  ],
  "parsed_intent": {
    "semantic_query": "축구",
    "sort_by": null,
    "sort_order": "desc",
    "category_filter": "SPORTS",
    "recruiting_now": true
  }
}
```

### 인덱스 관리 API

| 메서드 | 경로 | 역할 |
|--------|------|------|
| `POST` | `/index/club` | 동아리 추가/업서트 |
| `PATCH` | `/index/club/{id}/fields` | 기본 정보 수정 |
| `PATCH` | `/index/club/{id}/recruitments` | 모집글 교체 |
| `DELETE` | `/index/club/{id}` | 동아리 삭제 |
| `POST` | `/index/rebuild` | 전체 재빌드 |
| `GET` | `/health` | 서버 상태 확인 |

### Spring Boot 연동 예시

```java
// AI 검색 서비스 호출 (club_id 목록 반환)
List<Long> aiClubIds = aiSearchClient.search(keyword);

// 기존 DB에서 해당 club_id들만 조회
List<Club> aiResults = clubRepository.findAllById(aiClubIds);
```

## 파인튜닝 (선택)

```bash
# 학습 데이터 생성
python3 src/generate_pairs.py
python3 src/generate_pairs.py --claude  # Claude API 활용 (ANTHROPIC_API_KEY 필요)

# 파인튜닝 실행 (bge-m3 베이스, data/train_pairs.json 사용)
python3 src/train.py

# 파인튜닝 전/후 검색 품질 비교
python3 src/demo.py
```

파인튜닝 결과는 `models/finetuned/`에 저장된다. `serve.py`의 `MODEL_PATH`를 이 경로로 변경하면 적용된다.
