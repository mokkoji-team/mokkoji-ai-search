# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# 의존성 설치
pip3 install -r requirements.txt

# FastAPI 서버 실행 (Spring Boot 연동용)
python3 -m uvicorn src.serve:app --host 0.0.0.0 --port 8000

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

### Ollama 설정 (서버 실행 전 필수)
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b   # 기본 모델. OLLAMA_MODEL 환경변수로 변경 가능
```

## Architecture

Python FastAPI 마이크로서비스로, Spring Boot 백엔드에서 `POST /search` 또는 `POST /natural-search`를 호출해 club_id 목록을 받아간다.

### 검색 파이프라인 (/natural-search)

```
사용자 쿼리
  → query_parser.py  (Ollama 로컬 LLM)
      → {semantic_query, sort_by, category_filter, sort_order}
  → ClubSearchEngine.search()  (bge-m3 임베딩 유사도)
      → 후보 목록 (top_k * 5~10개)
  → category_filter / threshold 필터
  → ClubSearchEngine.rerank()  (bge-reranker-v2-m3 CrossEncoder)
      → 정밀 재정렬
  → sort_by 적용 (Spring이 clubs 데이터를 넘긴 경우 AI 서버에서 직접 정렬,
                   아니면 parsed_intent를 Spring에 반환해 DB 레벨 정렬 위임)
  → top_k 결과 반환
```

단순 키워드 검색(`/search`)은 Ollama 없이 임베딩 유사도만 사용한다.

### 핵심 설계 결정

**`club_to_text()`** (`src/search.py`)는 `train.py`와 `serve.py` 양쪽에서 공유된다. 이 함수를 수정하면 학습 시 텍스트 표현과 서빙 시 인덱스 텍스트가 달라지므로, 변경 후 반드시 재학습 + 인덱스 재빌드가 필요하다.

**인덱스 호환성**: 임베딩 모델을 교체하면 벡터 차원이 달라져 기존 `index/` 디렉토리와 호환되지 않는다. 모델 변경 시 `rm -rf index/`로 삭제하면 서버 시작 시 자동 재빌드된다.

**파인튜닝 적용**: `python3 src/train.py` 결과는 `models/finetuned/`에 저장된다. serve.py의 `MODEL_PATH`를 이 경로로 변경하면 파인튜닝 모델을 사용할 수 있다. train.py의 베이스 모델도 `BAAI/bge-m3`이므로 파인튜닝 모델도 bge-m3 기반이다.

**Claude 사용 범위**: Claude API는 `generate_pairs.py --claude` (학습 데이터 생성, 개발 시)에만 사용된다. 런타임(사용자 검색)에서는 Ollama 로컬 LLM을 사용하므로 토큰 비용이 없다.

### 모델 구성 (serve.py)

| 역할 | 모델 | 비고 |
|------|------|------|
| 임베딩 | `BAAI/bge-m3` | 다국어 검색 특화, 1024차원 |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | CrossEncoder, 정밀 재정렬 |
| 쿼리 파서 | `qwen2.5:7b` (Ollama) | 환경변수 `OLLAMA_MODEL`로 변경 |
