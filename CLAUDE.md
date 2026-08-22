# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# 의존성 설치
pip3 install -r requirements.txt

# SSH 터널 열기 (prod DB 접속 — 서버 실행 전 필요)
cd ~/mokkoji-auto-ingest && ./tunnel.sh prod

# FastAPI 서버 실행 (Spring Boot 연동용)
python3 -m uvicorn src.serve:app --host 0.0.0.0 --port 8000

# 서버 재시작
pkill -f "uvicorn src.serve" ; python3 -m uvicorn src.serve:app --host 0.0.0.0 --port 8000

# 인덱스 재빌드 (DB 데이터로 새로 빌드하려면)
rm -rf index/
# 이후 서버 시작 시 자동으로 DB에서 로드 후 빌드

# 대화형 검색 CLI (로컬 테스트)
python3 src/search_cli.py

# 학습 데이터 생성
python3 src/generate_pairs.py            # 이름/설명 기반 자동 생성
python3 src/generate_pairs.py --claude   # Claude API로 추가 쿼리 생성 (ANTHROPIC_API_KEY 필요)

# 파인튜닝 (bge-m3 베이스, data/train_pairs.json 사용)
python3 src/train.py

# 파인튜닝 전/후 검색 품질 비교
python3 src/demo.py
```

### Ollama 설정 (/natural-search 사용 시 필수)
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b   # 기본 모델. OLLAMA_MODEL 환경변수로 변경 가능
```

## Architecture

Python FastAPI 마이크로서비스로, Spring Boot 백엔드에서 `POST /search` 또는 `POST /natural-search`를 호출해 club_id 목록을 받아간다.

### 검색 파이프라인 (/search)

```
사용자 쿼리
  → ClubSearchEngine.search()  (Hybrid Search)
      → bge-m3 임베딩 유사도 (alpha=0.7)
      → BM25 키워드 점수 (1-alpha=0.3)
      → 두 점수 합산 후 top_k 반환
  → threshold 필터
  → 결과 반환
```

### 검색 파이프라인 (/natural-search)

```
사용자 쿼리
  → query_parser.py  (Ollama 로컬 LLM, LRU 캐시 512개)
      → {semantic_query, sort_by, category_filter, sort_order, recruiting_now}
  → ClubSearchEngine.search()  (Hybrid Search, 후보 min(top_k*2~3, 30)개)
  → category_filter / recruiting_now / threshold 필터
  → ClubSearchEngine.rerank()  (bge-reranker-v2-m3 CrossEncoder)
      → 정밀 재정렬
  → sort_by 적용 (Spring이 clubs 데이터를 넘긴 경우 AI 서버에서 직접 정렬,
                   아니면 parsed_intent를 Spring에 반환해 DB 레벨 정렬 위임)
  → top_k 결과 반환
```

### 데이터 로드 순서 (서버 시작 시)

```
index/ 폴더 존재? → Yes: 저장된 인덱스 로드 (빠름)
                 → No:  DB 환경변수 있음? → Yes: DB에서 로드 후 빌드 (db_loader.py)
                                          → No:  data/clubs.json 파일로 빌드
```

### 핵심 설계 결정

**`club_to_text()`** (`src/search.py`)는 `train.py`와 `serve.py` 양쪽에서 공유된다. 이 함수를 수정하면 학습 시 텍스트 표현과 서빙 시 인덱스 텍스트가 달라지므로, 변경 후 반드시 재학습 + 인덱스 재빌드가 필요하다.

**Hybrid Search**: `search()`는 bge-m3 임베딩(alpha=0.7)과 BM25(0.3)를 합산한다. 고유명사(동아리 이름)는 BM25가, 의미 검색은 임베딩이 담당해 상호 보완한다.

**university_code 필터**: 검색 후 필터가 아니라 검색 전 pre-filter로 적용해 top_k를 항상 보장한다.

**인덱스 호환성**: 임베딩 모델을 교체하면 벡터 차원이 달라져 기존 `index/` 디렉토리와 호환되지 않는다. 모델 변경 시 `rm -rf index/`로 삭제하면 서버 시작 시 자동 재빌드된다.

**파인튜닝 적용**: `python3 src/train.py` 결과는 `models/finetuned/`에 저장된다. serve.py의 `MODEL_PATH`를 이 경로로 변경하면 파인튜닝 모델을 사용할 수 있다.

**Claude 사용 범위**: Claude API는 `generate_pairs.py --claude` (학습 데이터 생성, 개발 시)에만 사용된다. 런타임(사용자 검색)에서는 Ollama 로컬 LLM을 사용하므로 토큰 비용이 없다.

### 모델 구성 (serve.py)

| 역할 | 모델 | 비고 |
|------|------|------|
| 임베딩 | `BAAI/bge-m3` | 다국어 검색 특화, 1024차원 |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | CrossEncoder, 정밀 재정렬 |
| 쿼리 파서 | `qwen2.5:7b` (Ollama) | 환경변수 `OLLAMA_MODEL`로 변경 |

### 환경변수 (.env)

| 변수 | 설명 |
|------|------|
| `MOKKOJI_DB_HOST` | DB 호스트 (SSH 터널 시 127.0.0.1) |
| `MOKKOJI_DB_PORT` | DB 포트 (SSH 터널 시 3308) |
| `MOKKOJI_DB_NAME` | DB 이름 |
| `MOKKOJI_DB_USER` | DB 유저 |
| `MOKKOJI_DB_PASSWORD` | DB 비밀번호 |
| `OLLAMA_MODEL` | Ollama 모델명 (기본: qwen2.5:7b) |
| `ANTHROPIC_API_KEY` | Claude API 키 (generate_pairs.py --claude 시 필요) |
