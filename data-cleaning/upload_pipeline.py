"""
    stream -> clean -> token count -> upload
"""

import math
import os
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoTokenizer
from dotenv import load_dotenv

from config import CONFIG
from data_cleaning import DataCleaning


# Global Variables
CHUNK_SIZE = 50_000
TARGET_TOKEN_COUNT = 20_000_000_000
OVER_FETCH_RATIO = 3
MAX_UPLOAD_RETRIES = 5


@dataclass
class TokenStats:
    total_tokens: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    p25_tokens: float
    p75_tokens: float
 
 
@dataclass
class Checkpoint:
    chunks_uploaded: int = 0
    docs_uploaded: int = 0
    docs_processed: int = 0
    total_tokens: int = 0
    chunk_token_log: list = field(default_factory=list)
 
    def to_dict(self) -> dict:
        return asdict(self)
 
    @staticmethod
    def from_dict(data: dict) -> "Checkpoint":
        return Checkpoint(**data)


class TokenCounter:
    def __init__(self, tokenizer, batch_size: int = 1_000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    
    def _tokenize_batched(self, texts: list[str]) -> list[list[int]]:
        all_ids = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            tokenized = self.tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=8192,
           )
            all_ids.extend(tokenized["input_ids"])
        
        return all_ids


    def count(self, texts: list[str]) -> int:
        return sum(len(ids) for ids in self._tokenize_batched(texts))


    def stats(self, texts: list[str]) -> TokenStats:
        lengths = np.array([
            len(ids) for ids in self._tokenize_batched(texts)
        ])
        return TokenStats(
            total_tokens = int(lengths.sum()),
            avg_tokens = float(lengths.mean()),
            min_tokens = int(lengths.min()),
            max_tokens = int(lengths.max()),
            p25_tokens = float(np.percentile(lengths, 25)),
            p75_tokens = float(np.percentile(lengths, 75)),
        )
    

class CheckpointManager:
    def __init__(self, path: str = CONFIG["checkpoint_path"]):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
 
    def load(self) -> Checkpoint:
        if not os.path.exists(self.path):
            return Checkpoint()
 
        with open(self.path) as f:
            cp = Checkpoint.from_dict(json.load(f))
 
        remaining = TARGET_TOKEN_COUNT - cp.total_tokens
        print(f"\n  [checkpoint] Resuming from chunk {cp.chunks_uploaded}")
        print(f"  Docs uploaded : {cp.docs_uploaded:,}")
        print(f"  Tokens so far : {cp.total_tokens / 1e9:.4f}B / 20B")
        print(f"  Remaining : {remaining / 1e9:.4f}B tokens\n")
        return cp
 
    def save(self, cp: Checkpoint):
        with open(self.path, "w") as f:
            json.dump(cp.to_dict(), f, indent=2)
 
    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            print("  [checkpoint] Cleared — next run starts fresh")


class HFUploader:
    def __init__(self, api: HfApi, repo_id: str, token: str):
        self.api = api
        self.repo_id = repo_id
        self.token = token
 
    @staticmethod
    def resolve_repo_id(api: HfApi, config: dict) -> str:
        username = api.whoami()["name"]
        return f"{username}/{config['hf_repo_id']}"
 
    @staticmethod
    def setup_repo(api: HfApi, repo_id: str, private: bool):
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"  Repo : https://huggingface.co/datasets/{repo_id}")
 
    def upload(self, texts: list[str], chunk_idx: int):
        tmp_path   = os.path.join(CONFIG["tmp_parquet_dir"], f"tmp_chunk_{chunk_idx:05d}.parquet")
        shard_name = f"data/train/chunk-{chunk_idx:05d}.parquet"
 
        Dataset.from_dict({"text": texts}).to_parquet(tmp_path)
 
        try:
            self._upload_with_retry(tmp_path, shard_name)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
 
    def _upload_with_retry(self, tmp_path: str, shard_name: str):
        for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
            try:
                self.api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=shard_name,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
                return
 
            except HfHubHTTPError as e:
                wait = 2 ** attempt
                print(f"\n  ✗ Upload attempt {attempt}/{MAX_UPLOAD_RETRIES}: {e}")
                if attempt < MAX_UPLOAD_RETRIES:
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


# Display

def print_pipeline_header(config: dict, repo_id: str):
    print(f"\n{'='*60}")
    print("  PRODUCTION PIPELINE — 20B TOKEN TARGET")
    print(f"{'='*60}")
    print(f"  Dataset : {config['dataset_name']} ({config['dataset_split']})")
    print(f"  Chunk size : {CHUNK_SIZE:,} docs")
    print(f"  Target : {TARGET_TOKEN_COUNT / 1e9:.0f}B tokens")
    print(f"  Repo : https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}\n")
 
 
def print_chunk_summary(
    chunk_idx: int,
    stats: TokenStats,
    total_tokens: int,
    docs_cleaned: int,
    docs_raw: int,
    elapsed: float,
):
    pct_done = total_tokens / TARGET_TOKEN_COUNT * 100
    remaining = TARGET_TOKEN_COUNT - total_tokens
    filter_rate = (1 - docs_cleaned / max(docs_raw, 1)) * 100
    chunks_left = math.ceil(remaining / max(stats.total_tokens, 1))
 
    print(f"\n  {'─'*56}")
    print(f"  CHUNK {chunk_idx} SUMMARY")
    print(f"  {'─'*56}")
    print(f"  {'Raw docs fetched':<30} {docs_raw:>10,}")
    print(f"  {'Docs after cleaning':<30} {docs_cleaned:>10,}")
    print(f"  {'Filter rate':<30} {filter_rate:>9.1f}%")
    print(f"  {'─'*56}")
    print(f"  {'Tokens this chunk':<30} {stats.total_tokens:>10,}")
    print(f"  {'Avg tokens / doc':<30} {stats.avg_tokens:>10.1f}")
    print(f"  {'Min tokens / doc':<30} {stats.min_tokens:>10,}")
    print(f"  {'Max tokens / doc':<30} {stats.max_tokens:>10,}")
    print(f"  {'P25 tokens / doc':<30} {stats.p25_tokens:>10.1f}")
    print(f"  {'P75 tokens / doc':<30} {stats.p75_tokens:>10.1f}")
    print(f"  {'─'*56}")
    print(f"  {'Total tokens so far':<30} {total_tokens / 1e9:>9.4f}B")
    print(f"  {'Target':<30} {'20.0000B':>10}")
    print(f"  {'Progress':<30} {pct_done:>9.2f}%")
    print(f"  {'Remaining':<30} {remaining / 1e9:>9.4f}B")
    print(f"  {'Est. chunks left':<30} {chunks_left:>10,}")
    print(f"  {'Upload time':<30} {elapsed:>9.1f}s")
    print(f"  {'─'*56}\n")
 
 
def print_final_summary(cp: Checkpoint, repo_id: str):
    target_hit = cp.total_tokens >= TARGET_TOKEN_COUNT
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Chunks uploaded : {cp.chunks_uploaded:,}")
    print(f"  Docs uploaded : {cp.docs_uploaded:,}")
    print(f"  Total tokens : {cp.total_tokens / 1e9:.4f}B")
    print(f"  Target hit : {'✅ Yes' if target_hit else '⚠️ Stream exhausted early'}")
    print(f"  Dataset URL : https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}\n")
 
 
def save_token_log(chunk_token_log: list, path: str = CONFIG["token_log_path"]):
    with open(path, "w") as f:
        f.write("chunk,docs,tokens,avg_tokens\n")
        for row in chunk_token_log:
            f.write(
                f"{row['chunk']},{row['docs']},"
                f"{row['tokens']},{row['avg_tokens']:.1f}\n"
            )
    print(f"  Token log saved → {path}\n")



# Stream Helpers

def build_stream(config: dict, skip: int):
    fw = load_dataset(
        config["dataset_name"],
        name=config["dataset_split"],
        split="train",
        streaming=True,
    )
    stream = fw.select_columns(["text"])
 
    if skip > 0:
        print(f"  Skipping {skip:,} already-processed docs...")
        stream = stream.skip(skip)
 
    return iter(stream)



def fetch_raw_chunk(stream_iter, chunk_idx: int) -> tuple[list[str], bool]:
    raw, exhausted = [], False
 
    pbar = tqdm(
        total=CHUNK_SIZE * OVER_FETCH_RATIO,
        desc=f"  Fetching chunk {chunk_idx + 1}",
        unit="docs",
        ncols=80,
    )
 
    try:
        while len(raw) < CHUNK_SIZE * OVER_FETCH_RATIO:
            row  = next(stream_iter)
            text = row["text"] if isinstance(row, dict) else row
            if isinstance(text, str) and text.strip():
                raw.append(text)
            pbar.update(1)
    except StopIteration:
        exhausted = True
 
    pbar.close()
    return raw, exhausted


def clean_chunk(cleaner: DataCleaning, raw: list[str]) -> list[str]:
    cleaned = []
    for text in raw:
        result = cleaner.clean(text)
        if result is not None:
            cleaned.append(result)
        if len(cleaned) >= CHUNK_SIZE:
            break
    return cleaned


def trim_to_budget(
    cleaned:          list[str],
    remaining_tokens: int,
    counter:          TokenCounter,
    avg_tokens:       float,
) -> tuple[list[str], TokenStats]:
    docs_to_keep = math.ceil(remaining_tokens / avg_tokens)
    trimmed = cleaned[:docs_to_keep]
    stats = counter.stats(trimmed)
    print(f"  Trimmed to {len(trimmed):,} docs to hit 20B target exactly")
    return trimmed, stats 


# Main Pipeline

def run(config: dict):
    load_dotenv()
 
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env")
    
    # Setup
    api = HfApi(token=token)
    repo_id = HFUploader.resolve_repo_id(api, config)
    HFUploader.setup_repo(api, repo_id, config.get("hf_private", False))
 
    uploader= HFUploader(api, repo_id, token)
    cleaner= DataCleaning(config)
    tokenizer= AutoTokenizer.from_pretrained(config["tokenizer_id"], token=token)
    counter= TokenCounter(tokenizer)
    cp_manager = CheckpointManager()
    cp = cp_manager.load()
 
    print_pipeline_header(config, repo_id)

    # Stream
    print("[1/3] Loading FineWeb stream...")
    stream_iter = build_stream(config, skip=cp.docs_processed)

    
    while cp.total_tokens < TARGET_TOKEN_COUNT:
        raw, exhausted = fetch_raw_chunk(stream_iter, cp.chunks_uploaded)
        cp.docs_processed += len(raw)
 
        if not raw:
            print("  Stream exhausted — no more docs available")
            break
 
        print(f"  Cleaning {len(raw):,} raw docs...")
        cleaned = clean_chunk(cleaner, raw)
 
        if not cleaned:
            print(f"  No docs passed cleaning — skipping chunk {cp.chunks_uploaded + 1}")
            if exhausted:
                break
            continue
 
        # Count tokens
        print(f"  Counting tokens for {len(cleaned):,} docs...")
        stats        = counter.stats(cleaned)
        chunk_tokens = stats.total_tokens
 
        # Trim if overshooting budget
        remaining = TARGET_TOKEN_COUNT - cp.total_tokens
        if chunk_tokens > remaining:
            cleaned, stats = trim_to_budget(cleaned, remaining, counter, stats.avg_tokens)
            chunk_tokens   = stats.total_tokens
 
        cp.total_tokens  += chunk_tokens
        cp.docs_uploaded += len(cleaned)
 
        # Upload
        print(f"  Uploading chunk {cp.chunks_uploaded + 1} ({len(cleaned):,} docs)...")
        start = time.time()
        uploader.upload(cleaned, cp.chunks_uploaded)
        elapsed = time.time() - start
 
        # Display
        print_chunk_summary(
            chunk_idx = cp.chunks_uploaded + 1,
            stats = stats,
            total_tokens = cp.total_tokens,
            docs_cleaned = len(cleaned),
            docs_raw = len(raw),
            elapsed = elapsed,
        )
 
        # Checkpoint
        cp.chunk_token_log.append({
            "chunk": cp.chunks_uploaded + 1,
            "docs": len(cleaned),
            "tokens": chunk_tokens,
            "avg_tokens": stats.avg_tokens,
        })
        cp.chunks_uploaded += 1
        cp_manager.save(cp)
 
        if exhausted:
            print("  Stream fully exhausted")
            break
 
    print("[3/3] Finalizing...")
    print_final_summary(cp, repo_id)
    save_token_log(cp.chunk_token_log)



if __name__ == "__main__":
    run(CONFIG)