import unicodedata


# --- Unicode fix

def test_unicode_clean_text(cleaner):
    assert cleaner.unicode_fix("Hello world") == "Hello world"
 
def test_unicode_mojibake(cleaner):
    assert cleaner.unicode_fix("CafÃ©") == "Café"
 
def test_unicode_empty_string(cleaner):
    assert cleaner.unicode_fix("") == ""
 
def test_unicode_already_valid_accents(cleaner):
    assert cleaner.unicode_fix("naïve résumé") == "naïve résumé"


# --- Whitespace normalization

def test_whitespace_extra_spaces(cleaner):
    assert cleaner.whitespace_normalization("hello   world") == "hello world"
 
def test_whitespace_newlines_and_tabs(cleaner):
    assert cleaner.whitespace_normalization("hello\n\tworld") == "hello world"
 
def test_whitespace_already_clean(cleaner):
    assert cleaner.whitespace_normalization("hello world") == "hello world"
 
def test_whitespace_empty_string(cleaner):
    assert cleaner.whitespace_normalization("") == ""
 
def test_whitespace_only_spaces(cleaner):
    assert cleaner.whitespace_normalization("     ") == ""


# --- Length filter
# Word count
def test_length_valid_median_range(cleaner):
    assert cleaner.length_filter("word " * 300) is True
 
def test_length_at_min_boundary(cleaner):
    assert cleaner.length_filter("word " * 75) is True
 
def test_length_below_min_boundary(cleaner):
    assert cleaner.length_filter("word " * 74) is False
 
def test_length_too_short(cleaner):
    assert cleaner.length_filter("hi there") is False
 
def test_length_at_p95(cleaner):
    assert cleaner.length_filter("word " * 1492) is True
 
def test_length_exceeds_upper_bound(cleaner):
    assert cleaner.length_filter("word " * 15_000) is False
 
# Character-level
def test_length_low_char_count(cleaner):
    assert cleaner.length_filter("ab " * 75) is False
 
def test_length_garbled_encoding(cleaner):
    # avg_word_len < 3 — single/double char tokens
    assert cleaner.length_filter("a b c d e f g h " * 20) is False
 
def test_length_url_heavy(cleaner):
    # avg_word_len > 15 — hash or URL tokens
    assert cleaner.length_filter("averylongurltokenwithnospacesatall " * 100) is False
 
# Line-level
def test_length_too_few_lines(cleaner):
    # All words on one line — wall of text
    text = "word " * 100 + "\n" + "word " * 100
    assert cleaner.length_filter(text) is False

def test_length_valid_multiline(cleaner):
    text = "\n".join(
        ["This is a normal sentence with enough words here."] * 20
    )
    assert cleaner.length_filter(text) is True
 
def test_length_bullet_spam(cleaner):
    # One word per line — nav/menu remnant
    text = "\n".join(["word"] * 200)
    assert cleaner.length_filter(text) is False


# --- Repetition filter
# Line-level
def test_repetition_clean_text(cleaner):
    text = "\n".join([
        "The quick brown fox jumps over the lazy dog.",
        "A completely different sentence about something else.",
        "Yet another unique line that adds new information.",
        "This document has varied and diverse content throughout.",
        "Each line here contributes something genuinely new.",
    ] * 4)
    assert cleaner.repetition_filter(text) is True
 
def test_repetition_duplicate_lines(cleaner):
    text = "Buy now and get a free discount on all products today.\n" * 50
    assert cleaner.repetition_filter(text) is False
 
def test_repetition_duplicate_line_chars(cleaner):
    repeated = "This is a very long repeated boilerplate line. " * 5 + "\n"
    unique   = "Unique sentence.\n" * 5
    text     = repeated * 20 + unique
    assert cleaner.repetition_filter(text) is False
 
# Paragraph-level
def test_repetition_duplicate_paragraphs(cleaner):
    para = "This paragraph is copy pasted across the document.\n\n"
    text = para * 15
    assert cleaner.repetition_filter(text) is False
 
def test_repetition_unique_paragraphs(cleaner):
    text = "\n\n".join([
        f"This is paragraph {i} with unique content about topic {i}."
        for i in range(10)
    ])
    assert cleaner.repetition_filter(text) is True
 
# N-gram level
def test_repetition_bigram_spam(cleaner):
    assert cleaner.repetition_filter("buy now " * 100) is False
 
def test_repetition_trigram_spam(cleaner):
    assert cleaner.repetition_filter("best cheap deals " * 80) is False
 
def test_repetition_fourgram_spam(cleaner):
    assert cleaner.repetition_filter("click here to buy " * 60) is False
 
def test_repetition_normal_ngrams(cleaner):
    text = " ".join([
        "The quick brown fox jumps over the lazy dog and runs fast.",
        "Scientists discovered a new species in the Amazon rainforest.",
        "The economy grew by three percent in the last quarter.",
        "Researchers published findings on climate change and its impact.",
    ] * 10)
    assert cleaner.repetition_filter(text) is True
 
# Word-level
def test_repetition_single_word_spam(cleaner):
    assert cleaner.repetition_filter("discount " * 200) is False
 
def test_repetition_normal_word_distribution(cleaner):
    text = (
        "The researchers found that climate change significantly affects "
        "biodiversity across tropical regions and temperate zones alike. "
        "Several studies published this year confirm earlier predictions "
        "about rising sea levels and temperature increases worldwide. "
    ) * 10
    assert cleaner.repetition_filter(text) is True
 
# Edge cases
def test_repetition_empty_string(cleaner):
    assert cleaner.repetition_filter("") is True
 
def test_repetition_single_line(cleaner):
    assert cleaner.repetition_filter("Just one line of text here.") is True


# --- PII Scrubbing
def test_pii_email_replaced(cleaner):
    result = cleaner.pii_scrubbing("Contact john@example.com today")
    assert "<EMAIL>" in result
    assert "john@example.com" not in result
 
def test_pii_phone_replaced(cleaner):
    result = cleaner.pii_scrubbing("Call us at 555-867-5309 now")
    assert "<PHONE>" in result
 
def test_pii_ip_replaced(cleaner):
    result = cleaner.pii_scrubbing("Server is at 192.168.1.1")
    assert "<IP>" in result
 
def test_pii_ssn_replaced(cleaner):
    result = cleaner.pii_scrubbing("SSN: 123-45-6789")
    assert "<SSN>" in result
 
def test_pii_multiple_in_one_doc(cleaner):
    text   = "Email john@example.com or call 555-867-5309"
    result = cleaner.pii_scrubbing(text)
    assert "<EMAIL>" in result
    assert "<PHONE>" in result
 
def test_pii_no_false_positive(cleaner):
    text = "The quick brown fox jumps over the lazy dog"
    assert cleaner.pii_scrubbing(text) == text
 
def test_pii_empty_string(cleaner):
    assert cleaner.pii_scrubbing("") == ""


# --- deduplicate
def test_dedup_removes_exact_duplicates(cleaner):
    texts   = ["word " * 100] * 5 + ["unique text content " * 50]
    results = cleaner.deduplicate(texts)
    assert len(results) == 2
 
def test_dedup_keeps_all_unique(cleaner):
    texts   = [f"This is unique document number {i}. " * 20 for i in range(10)]
    results = cleaner.deduplicate(texts)
    assert len(results) == 10
 
def test_dedup_preserves_first_occurrence(cleaner):
    texts   = [f"Document {i} contains unique text. " * 30 for i in range(5)]
    results = cleaner.deduplicate(texts)
    assert results[0].startswith("Document 0")
 
def test_dedup_single_doc(cleaner):
    texts   = ["only one document here with some content " * 20]
    results = cleaner.deduplicate(texts)
    assert len(results) == 1
 
def test_dedup_empty_list(cleaner):
    assert cleaner.deduplicate([]) == []


# Final normalization
def test_norm_collapse_repeated_exclamation(cleaner):
    assert cleaner.final_normalization("Wow!!!") == "Wow!"
 
def test_norm_collapse_repeated_question(cleaner):
    assert cleaner.final_normalization("Really???") == "Really?"
 
def test_norm_collapse_excess_ellipsis(cleaner):
    assert cleaner.final_normalization("Wait.....") == "Wait..."
 
def test_norm_collapse_excess_newlines(cleaner):
    result = cleaner.final_normalization("a\n\n\n\nb")
    assert result == "a\n\nb"
 
def test_norm_strips_whitespace(cleaner):
    assert cleaner.final_normalization("  hello  ") == "hello"
 
def test_norm_nfc_unicode(cleaner):
    # Decomposed "é" (e + combining accent) → composed NFC "é"
    decomposed = "cafe\u0301"
    result     = cleaner.final_normalization(decomposed)
    assert unicodedata.is_normalized("NFC", result)
 
def test_norm_empty_string(cleaner):
    assert cleaner.final_normalization("") == ""


# clean() integration
def test_clean_valid_doc_returns_string(cleaner):
    text = "\n".join([
        "This is a well-formed document with enough content.",
        "It has multiple lines and varied vocabulary throughout.",
        "The text covers several topics without repetition here.",
        "Scientists and researchers often publish findings on topics.",
        "This document should pass all cleaning stages cleanly.",
    ] * 10)
    result = cleaner.clean(text)
    assert isinstance(result, str)
    assert len(result) > 0
 
def test_clean_too_short_returns_none(cleaner):
    assert cleaner.clean("hi") is None
 
def test_clean_spam_returns_none(cleaner):
    assert cleaner.clean("buy now " * 200) is None
 
def test_clean_pii_is_scrubbed(cleaner):
    base = "\n".join([
        "Contact our team at support@example.com for help.",
        "Our offices are open Monday through Friday daily.",
        "We provide support across multiple time zones worldwide.",
        "Customers can also reach us via phone during business hours.",
        "Our team is dedicated to resolving issues efficiently always.",
    ] * 10)
    result = cleaner.clean(base)
    if result is not None:
        assert "support@example.com" not in result
        assert "<EMAIL>" in result