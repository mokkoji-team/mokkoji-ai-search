"""파인튜닝 실행 스크립트. data/train_pairs.json + data/clubs.json 을 읽어 모델을 학습시킨다."""

import json
import sys
from pathlib import Path

# src/ 디렉토리를 path에 추가 (search.py import용)
sys.path.insert(0, str(Path(__file__).parent))

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

from search import club_to_text

BASE_DIR = Path(__file__).parent.parent
BASE_MODEL = "BAAI/bge-m3"
OUTPUT_DIR = str(BASE_DIR / "models" / "finetuned")
CLUBS_PATH = str(BASE_DIR / "data" / "clubs.json")
TRAIN_PAIRS_PATH = str(BASE_DIR / "data" / "train_pairs.json")


def load_train_dataset() -> Dataset:
    with open(TRAIN_PAIRS_PATH, encoding="utf-8") as f:
        pairs = json.load(f)
    with open(CLUBS_PATH, encoding="utf-8") as f:
        clubs = json.load(f)

    club_lookup = {c["id"]: c for c in clubs}

    anchors, positives = [], []
    for pair in pairs:
        club = club_lookup.get(pair["club_id"])
        if club is None:
            print(f"  경고: club_id {pair['club_id']} 없음 — 건너뜀")
            continue
        anchors.append(pair["query"])
        positives.append(club_to_text(club))

    print(f"  학습 쌍: {len(anchors)}개")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def main():
    print("=== 모꼬지 AI 검색 파인튜닝 ===")
    print(f"베이스 모델: {BASE_MODEL}\n")

    model = SentenceTransformer(BASE_MODEL)

    print(f"학습 데이터 로드: {TRAIN_PAIRS_PATH}")
    train_dataset = load_train_dataset()

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0.1,
        fp16=False,   # Mac MPS 호환
        bf16=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print("\n파인튜닝 시작...")
    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    print(f"\n모델 저장 완료: {OUTPUT_DIR}")
    print("다음: python src/demo.py 로 검색 성능을 비교해보세요.")


if __name__ == "__main__":
    main()
