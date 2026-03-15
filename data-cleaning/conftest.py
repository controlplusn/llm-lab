import os
import json
import pytest
from datasets import load_dataset
 
from config import CONFIG
from data_cleaning import DataCleaning



@pytest.fixture(scope="session")
def cleaner():
    return DataCleaning(CONFIG)


@pytest.fixture(scope="session")
def sample_texts():
    """
        Behaviour:
        First run  → downloads CONFIG["test_sample_size"] rows
                     and writes them to CONFIG["test_cache_path"]
        Every run after → loads from cache in <1 second

        Used by test_metrics.py to measure real-data pass rates
        without loading the full 1M row subset.
    """
    cache_path = CONFIG["test_cache_path"]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
 
    # ── Load from cache if available ─────────────────────────
    if os.path.exists(cache_path):
        print(f"\n[fixture] Loading cached sample → {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            texts = json.load(f)
        print(f"[fixture] {len(texts):,} samples loaded from cache\n")
        return texts
 
    # ── First run: download and cache ────────────────────────
    print(f"\n[fixture] Cache not found — downloading "
          f"{CONFIG['test_sample_size']:,} rows from FineWeb (one-time)...")
 
    fw = load_dataset(
        CONFIG["dataset_name"],
        name=CONFIG["dataset_split"],
        split="train",
        streaming=True
    )
 
    texts = [
        row["text"]
        for row in fw.select_columns(["text"]).take(CONFIG["test_sample_size"])
    ]
 
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
 
    print(f"[fixture] Cached {len(texts):,} samples → {cache_path}\n")
    return texts