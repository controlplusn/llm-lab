import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

from config import CONFIG

# ----- Config Evaluation -----
def print_config(config: dict):
    print("\n" + "=" * 60)
    print("  PIPELINE CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key:<35} {value}")
    print("=" * 60 + "\n")


# ----- Dataset Loader -----
def load_subset(config: dict):
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

