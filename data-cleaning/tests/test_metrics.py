import numpy as np


# --- length_filter pass rate

def test_length_filter_pass_rate(cleaner, sample_texts):
    passed = [t for t in sample_texts if cleaner.length_filter(t)]
    rate   = len(passed) / len(sample_texts)
 
    lengths = np.array([len(t.split()) for t in passed])
    print(f"\n  --- length_filter ---")
    print(f"  Pass rate        : {rate:.1%}")
    print(f"  Passed           : {len(passed):,} / {len(sample_texts):,}")
    print(f"  Word count mean  : {lengths.mean():.0f}")
    print(f"  Word count min   : {lengths.min()}")
    print(f"  Word count max   : {lengths.max()}")
 
    # FineWeb is pre-filtered — expect most docs to pass
    assert 0.5 <= rate <= 1.0, (
        f"Length filter pass rate {rate:.1%} is suspiciously low — "
        f"check min_words / max_words thresholds in config.py"
    )



# --- repetition_filter pass rate

def test_repetition_filter_pass_rate(cleaner, sample_texts):
    passed = [t for t in sample_texts if cleaner.repetition_filter(t)]
    rate   = len(passed) / len(sample_texts)
 
    print("\n  --- repetition_filter ---")
    print(f"  Pass rate        : {rate:.1%}")
    print(f"  Passed           : {len(passed):,} / {len(sample_texts):,}")
    print(f"  Filtered out     : {len(sample_texts) - len(passed):,}")
 
    # FineWeb is already fairly clean — high pass rate expected
    assert 0.7 <= rate <= 1.0, (
        f"Repetition filter pass rate {rate:.1%} — "
        f"thresholds may be too aggressive for FineWeb"
    )


# --- pii_scrubbing hit rate

def test_pii_scrubbing_hit_rate(cleaner, sample_texts):
    hits = sum(
        1 for t in sample_texts
        if any(tag in cleaner.pii_scrubbing(t)
               for tag in ["<EMAIL>", "<PHONE>", "<IP>", "<SSN>", "<CREDITCARD>"])
    )
    rate = hits / len(sample_texts)
 
    print("\n  --- pii_scrubbing ---")
    print(f"  Docs with PII found : {hits:,} / {len(sample_texts):,}")
    print(f"  PII hit rate        : {rate:.1%}")
 
    # FineWeb contains some PII — expect a small but non-zero hit rate
    # No hard assert here — just report so you can tune patterns
    assert rate >= 0.0, "PII hit rate should be non-negative"


# --- deduplication removal rate

def test_deduplication_removal_rate(cleaner, sample_texts):
    results = cleaner.deduplicate(list(sample_texts))
    removed = len(sample_texts) - len(results)
    rate    = removed / len(sample_texts)
 
    print("\n  --- deduplicate ---")
    print(f"  Input docs       : {len(sample_texts):,}")
    print(f"  Unique docs kept : {len(results):,}")
    print(f"  Duplicates found : {removed:,}")
    print(f"  Removal rate     : {rate:.1%}")
 
    # FineWeb is pre-deduped — expect very few duplicates in 1k sample
    assert rate < 0.2, (
        f"Dedup removal rate {rate:.1%} is unexpectedly high — "
        f"check minhash_threshold in config.py"
    )


# --- final_normalization side effects
def test_final_normalization_does_not_empty_docs(cleaner, sample_texts):
    emptied = [
        t for t in sample_texts
        if cleaner.final_normalization(t) == ""
    ]
    rate = len(emptied) / len(sample_texts)
 
    print("\n  --- final_normalization ---")
    print(f"  Docs emptied after norm : {len(emptied):,} / {len(sample_texts):,}")
    print(f"  Emptied rate            : {rate:.1%}")
 
    assert rate < 0.01, (
        f"final_normalization is emptying {rate:.1%} of docs — "
        f"check norm_remove_urls or other aggressive settings"
    )


# --- full pipeline pass rate

def test_full_pipeline_pass_rate(cleaner, sample_texts):
    results = [cleaner.clean(t) for t in sample_texts]
    passed  = [r for r in results if r is not None]
    rate    = len(passed) / len(sample_texts)
 
    # Word count distribution of passed docs
    lengths = np.array([len(t.split()) for t in passed])
 
    print(f"\n  --- full clean() pipeline ---")
    print(f"  Input docs       : {len(sample_texts):,}")
    print(f"  Passed           : {len(passed):,}")
    print(f"  Filtered out     : {len(sample_texts) - len(passed):,}")
    print(f"  Pass rate        : {rate:.1%}")
    print(f"  Word count mean  : {lengths.mean():.0f}")
    print(f"  Word count P5    : {np.percentile(lengths, 5):.0f}")
    print(f"  Word count P95   : {np.percentile(lengths, 95):.0f}")
 
    # Pipeline should not be so aggressive it removes most of FineWeb
    assert rate > 0.3, (
        f"Full pipeline pass rate {rate:.1%} is too low — "
        f"one or more filters may be misconfigured"
    )
 
    # Passed docs should all be non-empty strings
    assert all(isinstance(t, str) and len(t) > 0 for t in passed), \
        "Some passed docs are empty strings — check final_normalization"
    

# --- Word count distribution of cleaned data

def test_word_count_distribution_after_cleaning(cleaner, sample_texts):
    passed  = [cleaner.clean(t) for t in sample_texts]
    passed  = [t for t in passed if t is not None]
    lengths = np.array([len(t.split()) for t in passed])
 
    print("\n  --- word count distribution (post-clean) ---")
    print(f"  Mean             : {lengths.mean():.0f} words")
    print(f"  Median           : {np.median(lengths):.0f} words")
    print(f"  P5               : {np.percentile(lengths, 5):.0f} words")
    print(f"  P95              : {np.percentile(lengths, 95):.0f} words")
    print(f"  Min              : {lengths.min()} words")
    print(f"  Max              : {lengths.max()} words")
 
    # After cleaning, average doc should be a reasonable length
    assert lengths.mean() > 50, \
        "Average cleaned doc is too short — length filter may be too loose"
 
    assert lengths.min() >= 50, \
        "Some docs shorter than min_words slipped through length_filter"