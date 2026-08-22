# mokkoji-ai

모꼬지 동아리 검색 AI — 의미 기반 검색(Semantic Search)을 위한 파인튜닝 파이프라인.

"SMARCLE" 이름을 몰라도 "AI 동아리", "스마클", "머신러닝 공부" 같은 검색어로 찾을 수 있게 한다.

## 구조

```
mokkoji-ai/
├── data/
│   ├── clubs.json          # 동아리 데이터 (name, description, tags 등)
│   └── train_pairs.json    # 학습 데이터 (query → club_id 쌍)
├── src/
│   ├── search.py           # ClubSearchEngine 클래스
│   ├── train.py            # 파인튜닝 실행
│   ├── demo.py             # Before/After 비교 데모
│   └── serve.py            # FastAPI 검색 서버 (Spring Boot 연동)
├── models/finetuned/       # 파인튜닝된 모델 (학습 후 생성)
└── index/                  # 벡터 인덱스 (serve 시 생성)
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행 순서

### 1단계: 파인튜닝

```bash
python src/train.py
```

- 베이스 모델 `jhgan/ko-sroberta-multitask` 을 `data/train_pairs.json` 으로 파인튜닝
- 결과는 `models/finetuned/` 에 저장
- 소요 시간: 약 5~10분 (M1 Mac 기준)

### 2단계: 성능 비교

```bash
python src/demo.py
```

베이스 모델과 파인튜닝 모델의 검색 결과를 나란히 출력한다.

### 3단계: 서버 실행

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Spring Boot 백엔드에서 `POST http://localhost:8000/search` 로 AI 검색 결과를 가져온다.

## API

### POST /search

```json
// Request
{"query": "스마클", "top_k": 10, "threshold": 0.3}

// Response
{
  "query": "스마클",
  "results": [
    {"club_id": 1, "score": 0.91},
    {"club_id": 2, "score": 0.54}
  ]
}
```

Spring Boot는 이 `club_id` 목록으로 DB를 조회해 실제 동아리 데이터를 반환한다.

### POST /index/rebuild

동아리 추가/수정 후 인덱스를 갱신한다. Spring Boot의 클럽 등록 API에서 호출하면 된다.

## 학습 데이터 추가 방법

`data/train_pairs.json` 에 항목을 추가하면 된다:

```json
{"query": "새로운 검색어", "club_id": 1}
```

실제 사용자의 검색 로그가 쌓이면 그걸 학습 데이터로 써서 재학습하면 성능이 크게 오른다.

## Spring Boot 연동 예시

`ClubService.java` 에서 AI 검색 결과와 기존 DB 결과를 합치는 방식:

```java
// AI 검색 서비스 호출 (club_id 목록 반환)
List<Long> aiClubIds = aiSearchClient.search(keyword);

// 기존 DB에서 해당 club_id들만 조회
List<Club> aiResults = clubRepository.findAllById(aiClubIds);
```

또는 AI 결과만 사용하거나, 기존 LIKE 검색과 병합해서 사용할 수 있다.
