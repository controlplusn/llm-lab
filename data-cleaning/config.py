CONFIG = {
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_split": "sample-10BT",
    "subset_size": 1_000_000,
    "tokenizer_id": "meta-llama/Meta-Llama-3-8B",
    "ttr_batch_size": 1_000,
    "output_path": "output/fineweb_cleaned.jsonl",
    "raw_cache_path": "cache/fineweb_raw.jsonl",


    # ── Test config ──────────────────────────────
    "test_sample_size": 1_000,
    "test_cache_path": "cache/test_sample.json",



    # ── Length filter ─────────────────────────────────────────
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
    "max_avg_words_per_line": 500,



    # ── Repetition filter ─────────────────────────────────────
    # Line-level
    "max_line_duplicate_ratio": 0.5,
    "max_line_chars_duplicate_ratio": 0.3,

    # Paragraph-level
    "max_para_duplicate_ratio": 0.3,

    # N-gram level
    "max_top_ngram_ratio_2": 0.2,
    "max_top_ngram_ratio_3": 0.30,
    "max_top_ngram_ratio_4": 0.25,

    # Word-level
    "max_word_duplicate_ratio": 0.3,



    # ── PII Scrubbing ─────────────────────────────────────────
    "pii_replace_email": True,
    "pii_replace_phone": True,
    "pii_replace_ip": True,
    "pii_replace_ssn": True,
    "pii_replace_creditcard": True,



    # ── MinHash LSH Deduplication ─────────────────────────────
    "minhash_num_perm": 128,
    "minhash_threshold": 0.85,
    "minhash_ngram_size": 5,



    # ── Final Normalization ───────────────────────────────────
    "norm_lowercase":False,
    "norm_remove_urls":False,
    "norm_collapse_puncts": True,
    "norm_collapse_newlines": True,
}