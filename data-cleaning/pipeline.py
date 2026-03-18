import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

from config import CONFIG
from data_cleaning import DataCleaning

# ----- Config Evaluation -----
def print_config(config: dict):
    print("\n" + "=" * 60)
    print("  PIPELINE CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key:<35} {value}")
    print("=" * 60 + "\n")


# ----- Dataset Loader -----
def load_fw(config: dict):
    print("[1/6] Loading dataset...")

    raw_cache = config.get("raw_cache_path", "cache/fineweb_raw.jsonl")
    os.makedirs("cache", exist_ok=True)
    
    if os.path.exists(raw_cache):
        print(f"Found local cache → {raw_cache}")
        print("Loading from disk...")

        texts = []
        with open(raw_cache, encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading cache", ncols=80):
                texts.append(json.loads(line)["text"])

        print(f"Loaded {len(texts):,} rows from cache\n")
        return texts

    print(f"No cache found — downloading {config['subset_size']:,} rows...")
    print(f"Saving to {raw_cache}...")

    fw = load_dataset(
        config["dataset_name"],
        name=config["dataset_split"],
        split="train",
        streaming=True
    )

    already_saved = 0
    if os.path.exists(raw_cache):
        with open(raw_cache, encoding="utf-8") as f:
            already_saved = sum(1 for _ in f)

    if already_saved > 0:
        print(f"Resuming from row {already_saved:,}...")

    texts = []
    target = config["subset_size"]
    stream = fw.select_columns(["text"]).skip(already_saved).take(
        target - already_saved
    )

    # Load existing rows into memory first
    if already_saved > 0:
        with open(raw_cache, encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])

    with open(raw_cache, "a", encoding="utf-8") as f:
        for row in tqdm(
            stream,
            total = target - already_saved,
            desc = "Downloading",
            unit = "docs",
            ncols = 80
        ):
            text = row["text"]
            if isinstance(text, str) and text.strip():
                texts.append(text)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"{len(texts):,} rows cached to {raw_cache}\n")
    return texts


# ----- TTR Computation -----
def compute_ttr(source, config: dict, tokenizer, label: str = "raw") -> np.ndarray:
    print("[2/6] Computing TTR scores...")

    ttr_scores = []
    batch      = []
    batch_size = config["ttr_batch_size"]
    skipped    = 0

    if isinstance(source, list):
        iterable = iter(source)
        total = len(source)
        def get_text(item):
            return item

    else:
        iterable = fw.select_columns(["text"]).take(config["subset_size"])
        total = config["subset_size"]
        def get_text(item):
            return item["text"]

    progress = tqdm(
        total=total,
        desc=f"TTR ({label})",
        unit="docs",
        ncols=80
    )

    for item in iterable:
        text = get_text(item)

        if not isinstance(text, str) or not text.strip():
            skipped += 1
            progress.update(1)
            continue

        batch.append(text)

        if len(batch) == batch_size:
            tokenized = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=8192,
            )
            for ids in tokenized["input_ids"]:
                if len(ids) > 0:
                    ttr_scores.append(len(set(ids)) / len(ids))
            progress.update(batch_size)
            batch = []

    # Flush remaining rows
    if batch:
        tokenized = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=8192,
        )
        for ids in tokenized["input_ids"]:
            if len(ids) > 0:
                ttr_scores.append(len(set(ids)) / len(ids))
        progress.update(len(batch))

    progress.close()

    scores = np.array(ttr_scores)
    print(f"\n  {'Label':<30} {label}")
    print(f"  {'Documents processed':<30} {len(scores):,}")
    print(f"  {'Skipped (None/empty)':<30} {skipped:,}")
    print(f"  {'Average TTR':<30} {scores.mean():.4f}")
    print(f"  {'Min TTR':<30} {scores.min():.4f}")
    print(f"  {'Max TTR':<30} {scores.max():.4f}")

    return scores


# ----- Length threshold -----
def compute_length_stats(texts: list[str], config: dict) -> np.ndarray:
    print("[3/6] Computing length distribution...")

    lengths = [
        len(t.split())
        for t in tqdm(texts, desc="  Length stats", unit="docs", ncols=80)
    ]

    arr = np.array(lengths)
    print(f"\n{'Mean':<30} {arr.mean():.0f} words")
    print(f"{'Median':<30} {np.median(arr):.0f} words")
    print(f"{'P5':<30} {np.percentile(arr, 5):.0f} words")
    print(f"{'P95':<30} {np.percentile(arr, 95):.0f} words")
    print(f"{'Min':<30} {arr.min():.0f} words")
    print(f"{'Max':<30} {arr.max():.0f} words")

    return arr


# ----- Data Cleaner -----
def run_cleaning(texts: list[str], config: dict) -> tuple[list[str], dict]:
    print("[4/6] Running data cleaning pipeline...")

    cleaner = DataCleaning(config)
    subset  = [{"text": t} for t in texts]

    results, stats = cleaner.apply(subset)
    return results, stats

# ----- Save output -----
def save_output(results: list[str], config: dict):
    print("[5/6] Saving output...")

    os.makedirs("output", exist_ok=True)
    out_path = config["output_path"]

    with open(out_path, "w", encoding="utf-8") as f:
        for text in results:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Path    : {out_path}")
    print(f"Docs    : {len(results):,}")
    print(f"Size    : {size_mb:.1f} MB")


# ------ TTR Comparison -----
def print_ttr_comparison(raw_scores: np.ndarray, cleaned_scores: np.ndarray):
    print("\n" + "=" * 60)
    print("  TTR COMPARISON — RAW vs CLEANED")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Raw':>12} {'Cleaned':>12} {'Δ':>10}")
    print(f"  {'-' * 59}")

    metrics = [
        ("Docs processed", len(raw_scores),                   len(cleaned_scores),              False),
        ("Average TTR",    raw_scores.mean(),                  cleaned_scores.mean(),            True),
        ("Min TTR",        raw_scores.min(),                   cleaned_scores.min(),             True),
        ("Max TTR",        raw_scores.max(),                   cleaned_scores.max(),             True),
        ("Std TTR",        raw_scores.std(),                   cleaned_scores.std(),             True),
        ("P25 TTR",        np.percentile(raw_scores, 25),      np.percentile(cleaned_scores, 25), True),
        ("P75 TTR",        np.percentile(raw_scores, 75),      np.percentile(cleaned_scores, 75), True),
    ]

    for label, raw_val, clean_val, is_float in metrics:
        if is_float:
            delta = clean_val - raw_val
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<25} {raw_val:>12.4f} {clean_val:>12.4f} "
                  f"{sign}{delta:>9.4f}")
        else:
            print(f"  {label:<25} {int(raw_val):>12,} {int(clean_val):>12,}")

    print("=" * 60 + "\n")



if __name__ == "__main__":
    load_dotenv()
 
    # 0 — Config
    print_config(CONFIG)
 
    # 1 — Load
    fw = load_fw(CONFIG)
 
    # 2 — TTR
    tokenizer  = AutoTokenizer.from_pretrained(
        CONFIG["tokenizer_id"],
        token=os.getenv("HF_TOKEN")
    )

    raw_ttr = compute_ttr(fw, CONFIG, tokenizer, label="raw")
 
    # 3 — Length stats  (comment out after thresholds are set)
    length_arr = compute_length_stats(fw, CONFIG)
 
    # 4 — Clean
    cleaned, stats = run_cleaning(fw, CONFIG)
 
    # 5 — Save
    save_output(cleaned, CONFIG)

    # 6 - TTR Calculation for cleaned data
    cleaned_ttr = compute_ttr(cleaned, CONFIG, tokenizer, label="cleaned")
    print_ttr_comparison(raw_ttr, cleaned_ttr)