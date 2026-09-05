# mokkoji-ai — 대학 동아리 AI 검색 마이크로서비스

## 프로젝트 개요

Spring Boot 기반 대학 동아리 플랫폼 **모꼬지**의 AI 검색 백엔드. 기존 MySQL LIKE 쿼리 방식이 의미 기반 검색을 전혀 지원하지 못하는 문제를 해결하기 위해 처음부터 설계·구현했다.

- **기존 방식의 한계**: "인공지능 배우고 싶어" → AI 동아리 0건 / "풋볼" → 축구 동아리 0건
- **해결 후**: 211개 동아리 + 1,351개 모집글을 의미 기반으로 검색. 유사어·별칭·오타도 처리

**기술 스택**: Python · FastAPI · bge-m3 · bge-reranker-v2-m3 · BM25 · kiwipiepy · Ollama · MySQL

---

## 아키텍처

```
Spring Boot (Java)
       │
       ├─ POST /search          → Hybrid Search (빠른 경로)
       └─ POST /natural-search  → 자연어 파싱 → Hybrid Search → Rerank
              │
      ┌───────▼────────┐
      │  FastAPI 서버  │  (Python, port 8000)
      │   serve.py     │
      └───────┬────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
search.py  query_parser  db_loader
(엔진)     (Ollama LLM)  (MySQL)
```

### 데이터 로드 우선순위 (서버 시작 시)

```
index/ 폴더 있음 → 저장된 벡터 인덱스 로드  (가장 빠름)
없음 + DB 환경변수 있음 → DB에서 로드 후 빌드
없음 + DB 없음 → data/clubs.json으로 폴백
```

---

## 핵심 구현

### 1. Hybrid Search (`src/search.py`)

임베딩과 BM25의 단점을 서로 보완하는 2-track 점수 합산.

```
hybrid_score = 0.7 × 임베딩점수 + 0.3 × BM25점수
```

**임베딩(bge-m3)**: 의미 유사도. "풋볼" → 축구 동아리, "인공지능" → AI 동아리 등 유사어 처리.

**BM25**: 키워드 정확 매칭. 동아리 이름·별칭("스마클", "SMARCLE") 같은 고유명사는 임베딩보다 BM25가 더 정확하게 잡음.

두 점수를 [0,1]로 정규화 후 가중 합산해 두 장점을 결합했다.

**형태소 분석 BM25**: 단순 공백 분리 대신 `kiwipiepy`로 명사(NN)·동사(VV)·형용사(VA)·영문(SL)·숫자(SN)만 추출. 조사·어미가 BM25 점수를 희석하는 문제 해결.

```python
"인공지능 배우고 싶은 동아리" → ["인공지능", "배우", "싶", "동아리"]
```

### 2. 2단계 검색 파이프라인 (`/natural-search`)

```
쿼리 → LLM 파싱 → Hybrid Search → 필터 → CrossEncoder 리랭커 → 정렬
```

1. **쿼리 파싱** (Ollama `qwen2.5:7b`): 자연어를 구조화된 인텐트로 분해
   ```json
   "지금 모집 중인 축구 동아리" →
   { "semantic_query": "축구", "category_filter": "SPORTS", "recruiting_now": true }
   ```

2. **Hybrid Search**: `semantic_query`로 후보 최대 30개 추출

3. **필터**: `category_filter` / `recruiting_now` / `threshold`

4. **CrossEncoder 리랭커** (`bge-reranker-v2-m3`): 임베딩 검색의 순위 정밀도를 보완. 쿼리-문서 쌍을 직접 비교해 재정렬

5. **정렬**: Spring이 enriched 데이터를 넘기면 AI 서버에서 직접 정렬, 아니면 `parsed_intent`를 반환해 DB 레벨 정렬 위임

### 3. Pre-filter 설계 (`university_code`)

학교 필터를 **검색 전**에 적용해 `top_k`를 항상 보장.

- 기존: 전체 211개에서 10개 추출 → 학교 필터 → 3~4개 반환
- 수정: 해당 학교 서브셋(~70개)에서만 유사도 계산 → 10개 보장

`search()` 내부에서 `filter_fn`으로 마스크를 만들어 NumPy 인덱싱으로 서브셋만 계산:

```python
mask = [i for i, c in enumerate(self.clubs) if filter_fn(c)]
pool_vectors = self.vectors[mask]  # 해당 학교 벡터만
```

### 4. IME 복합 종성 오타 교정 (`src/search.py`)

외부 API 없이 로컬에서 한국어 IME 오타를 교정.

한국어 IME는 자음 두 개가 연속 입력되면 복합 종성으로 합친다:
```
"축구" 타이핑 중 ㄱ 다음 실수로 ㅅ 추가
→ IME: ㄱ+ㅅ = ㄳ(복합 종성)으로 합침 → "춗구"
```

교정 흐름:
```
"춗구" 입력
  → kiwipiepy: oov=True 감지 (사전에 없는 단어)
  → '춗' 자모 분해: (ㅊ, ㅜ, ㄳ)
  → ㄳ = 복합 종성 → ㄱ으로 분해 시도
  → "축구" → kiwi oov 없음 ✓ → 교정 완료
```

**설계 원칙**: 복합 종성 분해만 수행. 초성/종성 인접키 치환은 오교정(정상 단어를 다른 단어로 교정)을 유발해 제거했다. 3음절 이상 OOV는 고유명사로 판단해 교정하지 않는다.

이미 BM25용으로 `kiwipiepy`를 사용 중이라 추가 의존성 없이 구현했다.

### 5. 파인튜닝 파이프라인

#### 학습 데이터 생성 (`src/generate_pairs.py`)

(쿼리, 동아리) 쌍을 자동으로 대량 생성:

| 생성 방식 | 내용 |
|---------|------|
| 자동 생성 | 동아리 이름, 설명 문장, 카테고리/소속/학교 조합, 모집글 제목·내용 |
| 수동 curated | 별칭("스마클" ↔ SMARCLE), 의도 기반 쿼리 |
| 오타 augmentation | IME 복합 종성·인접 키 오타 버전 자동 생성 |
| `--claude` 플래그 | Ollama LLM으로 동아리당 6개 추가 쿼리 생성 |

결과: **671개 → 9,686개** (14.4배), 211개 전 동아리 커버, 불일치 0개

#### 학습 (`src/train.py`)

`MultipleNegativesRankingLoss` — 같은 배치 내 다른 쌍이 자동으로 negative sample이 되어 별도의 hard negative 구축 없이 효율적으로 학습.

```python
# (anchor=쿼리, positive=club_to_text 결과) 쌍
{"anchor": "AI 동아리", "positive": "SMARCLE 학술 교양 인공지능..."}
```

파인튜닝 효과 (9,686쌍, 3 epochs, `ko-sroberta-multitask`):

| 쿼리 | 파인튜닝 전 | 파인튜닝 후 |
|------|-----------|-----------|
| "스마클" | 0.737 | 0.769 |
| "머신러닝 프로젝트" | 0.807 | 0.863 |

### 6. 인덱스 실시간 CRUD

동아리/모집글 변경 시 전체 재빌드 없이 부분 업데이트:

| 엔드포인트 | 역할 |
|-----------|------|
| `POST /index/club` | 동아리 추가/업서트 |
| `PATCH /index/club/{id}/fields` | 기본 정보만 수정 (모집글 유지) |
| `PATCH /index/club/{id}/recruitments` | 모집글만 교체 (기본 정보 유지) |
| `DELETE /index/club/{id}` | 동아리 삭제 |
| `POST /index/rebuild` | 전체 재빌드 |

모집글과 기본 정보를 분리한 이유: Spring Boot에서 동아리 정보 수정과 모집글 수정이 다른 이벤트로 발생하므로, 두 업데이트 경로를 별도로 두어 불필요한 임베딩 재계산을 방지한다.

---

## 최적화 포인트

| 문제 | 해결책 | 효과 |
|-----|--------|------|
| top_k 부족 (필터 후 결과 감소) | 검색 전 pre-filter | top_k 항상 보장 |
| 고유명사 임베딩 검색 약함 | BM25 0.3 가중치 병합 | "스마클" 1위 |
| BM25 한국어 단어 분리 부정확 | kiwipiepy 형태소 분석 | 조사·어미 노이즈 제거 |
| 임베딩 순위 정밀도 부족 | CrossEncoder 2단계 리랭커 | 재정렬 정확도 향상 |
| 리랭커 후보 과다 → 응답 느림 | `min(top_k×2~3, 30)` 상한 | 응답 속도 개선 |
| 서버 재시작마다 임베딩 재계산 | `vectors.npy` 인덱스 저장 | 재시작 수 분 → 수 초 |
| 동일 쿼리 반복 LLM 호출 | `@lru_cache(maxsize=512)` | ~1초 단축 |
| 런타임 LLM API 비용 | Ollama 로컬 LLM | 비용 $0 |
| 오타 쿼리 검색 실패 | IME 복합 종성 분해 교정 | 외부 API 없이 처리 |
| 학습/서빙 텍스트 불일치 위험 | `club_to_text()` 양쪽 공유 | train-serve 일관성 보장 |
| 관련 없는 결과까지 반환 | threshold 기반 가변 반환 | 관련 없으면 빈 결과 반환 |

---

## Spring Boot 연동

```java
// AI 검색 → club_id 목록 수신
List<Long> aiClubIds = aiSearchClient.search(keyword);

// 기존 DB에서 해당 club_id들만 조회
List<Club> results = clubRepository.findAllById(aiClubIds);
```

AI 서버는 `club_id`와 점수만 반환하고, 실제 데이터는 Spring Boot가 DB에서 조회한다. AI 마이크로서비스가 DB 구조 변경에 영향을 받지 않도록 역할을 명확히 분리했다.

---

## 주요 설계 결정

**`club_to_text()` 공유**: `search.py`의 이 함수를 `train.py`와 `serve.py` 양쪽에서 임포트해 사용한다. 학습 시 텍스트 표현과 서빙 시 인덱스 텍스트가 반드시 동일해야 하기 때문. 함수가 두 곳에 분리되면 한쪽만 수정했을 때 오류를 감지하기 어렵다.

**Ollama 로컬 LLM 선택**: 사용자 검색 요청마다 외부 LLM API를 호출하면 응답 지연과 비용이 발생한다. `qwen2.5:7b`를 로컬에서 실행해 런타임 비용 $0, Claude API는 학습 데이터 생성(개발 시 1회성 작업)에만 사용.

**벡터 인덱스 파일 저장**: `vectors.npy` + `clubs.json`으로 디스크에 저장. bge-m3으로 211개 동아리를 임베딩하면 수 분이 소요되므로, 서버 재시작 시 파일에서 로드해 수 초로 단축.
