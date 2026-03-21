CONFIG = {
    # Dataset
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_split": "sample-100BT",
    "subset_size": 100_000_000,

    # Tokenizer
    "tokenizer_id": "meta-llama/Meta-Llama-3-8B",
    "ttr_batch_size": 1_000,

    # Paths
    "output_path": "output/fineweb_cleaned_20B.jsonl",

    # Checkpoint paths
    "checkpoint_path": "cache/upload_checkpoint.json",
    "token_log_path": "cache/chunk_token_log.csv",
    "tmp_parquet_dir": "cache",

    # HuggingFace
    "hf_repo_id": "fineweb-cleaned-20B",
    "hf_private": False,

    # Test Config
    "test_sample_size": 1_000,
    "test_cache_path": "cache/test_sample.json",


    # Length Filter
    # Word count  Adjust based on the word length calculation
    "min_words": 75,
    "max_words": 10_000,

    # Character-level
    "min_chars": 300,
    "max_chars": 100_000,
    "min_avg_word_len": 3,
    "max_avg_word_len": 15,

    # Line-level
    "min_lines": 3,
    "max_lines": 10_000,
    "min_avg_words_per_line": 5,
    "max_avg_words_per_line": 300,


    # Repetition filter
    # Line-level
    "max_line_duplicate_ratio": 0.40,
    "max_line_chars_duplicate_ratio": 0.25,

    # Paragraph-level
    "max_para_duplicate_ratio": 0.30,

    # N-gram level
    "max_top_ngram_ratio_2": 0.20,
    "max_top_ngram_ratio_3": 0.25,
    "max_top_ngram_ratio_4": 0.20,

    # Word-level
    "max_word_duplicate_ratio": 0.25,


    # ── New filters ───────────────────────────────────────────
    "max_pipe_line_ratio":          0.15,


    # ── PII Scrubbing ─────────────────────────────────────────
    "pii_replace_email": True,
    "pii_replace_phone": True,
    "pii_replace_ip": True,
    "pii_replace_ssn": True,
    "pii_replace_creditcard": True,


    # ── Final Normalization ───────────────────────────────────
    "norm_lowercase": False,
    "norm_remove_urls": False,
    "norm_collapse_puncts": True,
    "norm_collapse_newlines": True,
}