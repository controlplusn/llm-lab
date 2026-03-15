CONFIG = {
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_split": "sample-10BT",
    "subset_size": 1_000_000,
    "tokenizer_id": "meta-llama/Meta-Llama-3-8B",
    "ttr_batch_size": 1_000,

    # ── Test config ──────────────────────────────
    "test_sample_size": 1_000,
    "test_cache_path": "cache/test_sample.json",
}