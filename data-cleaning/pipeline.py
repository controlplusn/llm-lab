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
    print("[1/5] Loading dataset...")

    fw = load_dataset(
        config["dataset_name"],
        name=config["dataset_split"],
        split="train",
        streaming=True
    )

    subset = fw.select_columns(["text"]).take(config["subset_size"])


    # Preview of first row
    first_row = next(iter(
        fw.select_columns(["text"]).take(1)
    ))

    print(f"      Columns : {list(first_row.keys())}")
    print(f"      Preview : {first_row['text'][:80]}...")
    print(f"      ✓ Subset of {config['subset_size']:,} rows ready\n")

    return fw, subset


# ----- TTR Computation -----
def compute_ttr(subset, config: dict, tokenizer) -> np.ndarray:
    print("[2/5] Computing TTR scores...")

    ttr_scores = []
    batch      = []
    batch_size = config["ttr_batch_size"]

    progress = tqdm(
        total=config["subset_size"],
        desc="  TTR",
        unit="docs",
        ncols=80
    )

    for row in subset:
        batch.append(row["text"])

        if len(batch) == batch_size:
            tokenized = tokenizer(batch, add_special_tokens=False)
            for ids in tokenized["input_ids"]:
                if len(ids) > 0:
                    ttr_scores.append(len(set(ids)) / len(ids))
            progress.update(batch_size)
            batch = []

    if batch:
        tokenized = tokenizer(batch, add_special_tokens=False)
        for ids in tokenized["input_ids"]:
            if len(ids) > 0:
                ttr_scores.append(len(set(ids)) / len(ids))
        progress.update(len(batch))

    progress.close()

    scores = np.array(ttr_scores)
    print(f"\n  {'Documents processed':<30} {len(scores):,}")
    print(f"  {'Average TTR':<30} {scores.mean():.4f}")
    print(f"  {'Min TTR':<30} {scores.min():.4f}")
    print(f"  {'Max TTR':<30} {scores.max():.4f}")
    print("  ✓ TTR complete\n")

    return scores


# ----- Length threshold -----
def compute_length_stats(fw, config: dict):
    print("[3/5] Computing length distribution...")

    lengths = []
    stream  = fw.select_columns(["text"]).take(config["subset_size"])

    for batch in tqdm(
        stream.iter(batch_size=10_000),
        desc="  Length stats",
        unit="batch",
        ncols=80
    ):
        lengths.extend(len(t.split()) for t in batch["text"])

    arr = np.array(lengths)
    print(f"\n  {'Mean':<30} {arr.mean():.0f} words")
    print(f"  {'Median':<30} {np.median(arr):.0f} words")
    print(f"  {'P5':<30} {np.percentile(arr, 5):.0f} words")
    print(f"  {'P95':<30} {np.percentile(arr, 95):.0f} words")
    print("  ✓ Length stats complete\n")

    return arr


# ----- Data Cleaner -----
def run_cleaning(fw, config: dict) -> tuple[list[str], dict]:
    print("[4/5] Running data cleaning pipeline...")

    cleaner = DataCleaning(config)
    stream  = fw.select_columns(["text"]).take(config["subset_size"])
 
    results, stats = cleaner.apply(stream)
    return results, stats


# ----- Save output -----
def save_output(results: list[str], config: dict):
    """
    Write cleaned documents to a JSONL file — one JSON object
    per line with a "text" key, UTF-8 encoded.
    """
    print("[5/5] Saving output...")
 
    os.makedirs("output", exist_ok=True)
    out_path = config["output_path"]
 
    with open(out_path, "w", encoding="utf-8") as f:
        for text in results:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
 
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Path: {out_path}")
    print(f"Docs: {len(results):,}")
    print(f"Size: {size_mb:.1f} MB")
    print("✓ Saved\n")



if __name__ == "__main__":
    load_dotenv()
 
    # 0 — Config
    print_config(CONFIG)
 
    # 1 — Load
    fw = load_fw(CONFIG)
 
    # 2 — TTR  (comment out if not needed)
    tokenizer  = AutoTokenizer.from_pretrained(
        CONFIG["tokenizer_id"],
        token=os.getenv("HF_TOKEN")
    )
    ttr_scores = compute_ttr(fw, CONFIG, tokenizer)
 
    # 3 — Length stats  (comment out after thresholds are set)
    length_arr = compute_length_stats(fw, CONFIG)
 
    # 4 — Clean
    cleaned, stats = run_cleaning(fw, CONFIG)
 
    # 5 — Save
    save_output(cleaned, CONFIG)