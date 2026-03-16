import re
import ftfy
import unicodedata
from tqdm import tqdm
from collections import Counter
from datasketch import MinHash, MinHashLSH


class DataCleaning:
    def __init__(self, config: dict):
        self.config = config

    # ----- 1. Normalization -----
    def unicode_fix(self, text: str) -> str:
        if ftfy.is_bad(text):
            return ftfy.fix_text(text)
        return text

    
    def whitespace_normalization(self, text: str) -> str:
        return " ".join(text.split())

    

    # ----- 2. Heuristic Filters -----

    # ── 2a. Length Filter ────────────────────────────────────
    
    def length_filter(self, text: str) -> bool:
        """
            1. Word-count bounds
            2. Character-level bounds + avg word length
            3. Line-level bounds + avg words per line
        """

        words      = text.split()
        word_count = len(words)
        char_count = len(text)
        lines      = [l for l in text.splitlines() if l.strip()]
        line_count = len(lines)

        
        # ── 1. Word count bounds ──────────────────────────
        if word_count < self.config["min_words"]:
            return False

        if word_count > self.config["max_words"]:
            return False 
        
        # ── 2. Character-level checks ─────────────────────
        if char_count < self.config["min_chars"]:
            return False

        if char_count > self.config["max_chars"]:
            return False

        avg_word_len = char_count / word_count
        if avg_word_len < self.config["min_avg_word_len"]:
            return False

        if avg_word_len > self.config["max_avg_word_len"]:
            return False 
        

        # ── 3. Line-level heuristics ──────────────────────
        if line_count > 1:
            if line_count < self.config["min_lines"]:
                return False

            if line_count > self.config["max_lines"]:
                return False 

            avg_words_per_line = word_count / line_count
            if avg_words_per_line < self.config["min_avg_words_per_line"]:
                return False

            if avg_words_per_line > self.config["max_avg_words_per_line"]:
                return False

        return True


    # ── 2a. Repetition Filter ────────────────────────────────────

    def repetition_filter(self, text: str) -> bool:
        """
            1. Line duplicate ratio
            2. Line character duplicate ratio
            3. Paragraph duplicate ratio
            4. N-gram repetition ratio (2, 3, 4-grams)
            5. Word duplicate ratio
        """

        lines           = [l.strip() for l in text.splitlines()]
        non_empty_lines = [l for l in lines if l]
        paragraphs      = [p.strip() for p in text.split("\n\n") if p.strip()]
        words           = text.lower().split()


        # ── 1. Line duplicate ratio ───────────────────────
        # Catches boilerplate lines repeated across page
        if non_empty_lines:
            line_counts      = Counter(non_empty_lines)
            duplicated_lines = sum(c - 1 for c in line_counts.values() if c > 2)
            line_dup_ratio   = duplicated_lines / len(non_empty_lines)

            if line_dup_ratio > self.config["max_line_duplicate_ratio"]:
                return False
            

        # ── 2. Line character duplicate ratio ─────────────
        # Catches long repeated lines dominating char count
        if non_empty_lines:
            total_chars    = sum(len(l) for l in non_empty_lines)
            dup_line_chars = sum(
                len(l) * (c - 1)
                for l, c in Counter(non_empty_lines).items()
                if c > 1
            )
            char_dup_ratio = dup_line_chars / max(total_chars, 1)

            if char_dup_ratio > self.config["max_line_chars_duplicate_ratio"]:
                return False
            

        # ── 3. Paragraph duplicate ratio ──────────────────
        # Catches copy-pasted templated sections
        if paragraphs:
            para_counts     = Counter(paragraphs)
            duplicated_para = sum(c - 1 for c in para_counts.values() if c > 1)
            para_dup_ratio  = duplicated_para / len(paragraphs)

            if para_dup_ratio > self.config["max_para_duplicate_ratio"]:
                return False
            

        # ── 4. N-gram repetition ratio ────────────────────
        # Catches SEO keyword stuffing and phrase repetition
        MIN_WORDS_FOR_NGRAM = 30

        for n, config_key in [
            (2, "max_top_ngram_ratio_2"),
            (3, "max_top_ngram_ratio_3"),
            (4, "max_top_ngram_ratio_4"),
        ]:
            if len(words) < MIN_WORDS_FOR_NGRAM:
                continue

            ngrams       = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            top_count    = ngram_counts.most_common(1)[0][1]
            ngram_ratio  = (top_count * n) / max(len(words), 1)

            if ngram_ratio > self.config[config_key]:
                return False
            

        # ── 5. Word duplicate ratio ───────────────────────
        # Catches single-word spam: "sale sale sale sale..."
        if words:
            word_counts    = Counter(words)
            top_word_count = word_counts.most_common(1)[0][1]
            word_dup_ratio = top_word_count / len(words)

            if word_dup_ratio > self.config["max_word_duplicate_ratio"]:
                return False

        return True
    


    # ----- 3. PII Scrubbing -----


    _PII_PATTERNS = {
        "email": (
            re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'),
            "<EMAIL>"
        ),
        "phone": (
            re.compile(r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'),
            "<PHONE>"
        ),
        "ip": (
            re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            "<IP>"
        ),
        "ssn": (
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "<SSN>"
        ),
        "creditcard": (
            re.compile(r'\b(?:\d[ -]?){13,16}\b'),
            "<CREDITCARD>"
        ),
    }

    def pii_scrubbing(self, text: str) -> str:
        """
            Patterns:
            email →  <EMAIL>
            phone →  <PHONE>
            ip address →  <IP>
            SSN →  <SSN>
            credit  →  <CREDITCARD>
        """

        for pii_type, (pattern, placeholder) in self._PII_PATTERNS.items():
            if self.config.get(f"pii_replace_{pii_type}", True):
                text = pattern.sub(placeholder, text)
        return text


    
    # ----- Subset Deduplication -----

    def _make_minhash(self, text: str) ->  MinHash:
        """
            MinHash signature from character n-gram shingles
        """

        m    = MinHash(num_perm=self.config["minhash_num_perm"])
        n    = self.config["minhash_ngram_size"]
        text = text.lower()

        for i in range(len(text) - n + 1):
            shingle = text[i:i+n]
            m.update(shingle.encode("utf-8"))

        return m


    def deduplicate(self, texts: list[str]) -> list[str]:
        """
            1. Build a MinHash signature for each document
            2. Insert into LSH index — similar docs hash to
               the same bucket (no all-pairs comparison needed)
            3. Query each doc before inserting — if a similar
               doc already exists, skip it as a duplicate

        Threshold:
            0.85 = documents sharing 85%+ of their n-gram
            shingles are considered near-duplicates
        """

        print("\n  [Deduplication] Building MinHash LSH index...")

        if not texts:
            return []

        lsh = MinHashLSH(
            threshold=self.config["minhash_threshold"],
            num_perm=self.config["minhash_num_perm"]
        )

        unique  = []
        skipped = 0

        for i, text in enumerate(tqdm(
            texts,
            desc="  MinHash LSH",
            unit="docs",
            ncols=80
        )):
            m = self._make_minhash(text)

            # Query before inserting — is a near-duplicate already indexed?
            results = lsh.query(m)
            if results:
                skipped += 1
                continue

            # No duplicate found — insert and keep
            lsh.insert(f"doc_{i}", m)
            unique.append(text)


        total = len(texts)
        print(f"\n  {'Input docs':<30} {total:,}")
        print(f"  {'Duplicates removed':<30} {skipped:,} ({skipped/total*100:.1f}%)")
        print(f"  {'Unique docs kept':<30} {len(unique):,} ({len(unique)/total*100:.1f}%)")
        print("  ✓ Deduplication complete\n")

        return unique
    


    # ----- Final Normalization -----
    def final_normalization(self, text: str) -> str:
        """
            1. Collapse repeated punctuation
            2. Collapse excess newlines
            3. Normalize unicode to NFC form
            4. Strip leading/trailing whitespace
        """

        if self.config.get("norm_collapse_puncts", True):
            text = re.sub(r'\.{4,}', '...',  text)
            text = re.sub(r'([!?]){2,}', r'\1', text)

        if self.config.get("norm_collapse_newlines", True):
            text = re.sub(r'\n{3,}', '\n\n', text)

        text = unicodedata.normalize("NFC", text)

        if self.config.get("norm_lowercase", False):
            text = text.lower()

        if self.config.get("norm_remove_urls", False):
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        return text.strip()
    

    def clean(self, text: str) -> str | None:
        """
            1. Unicode fix              (normalization)
            2. Whitespace norm          (normalization)
            3. Length filter            (heuristic — rejects doc)
            4. Repetition filter        (heuristic — rejects doc)
            5. PII scrubbing            (replacement)
            6. Final normalization      (cleanup)
        """

        text = self.unicode_fix(text)
        text = self.whitespace_normalization(text)


        if not self.length_filter(text):
            return None
        if not self.repetition_filter(text):
            return None

        text = self.pii_scrubbing(text)

        text = self.final_normalization(text)

        return text
    

    def apply(self, subset) -> tuple[list[str], dict]:
        print("\n[DataCleaning] Running pipeline...")

        stats = {
            "total":            0,
            "after_unicode":    0,
            "after_whitespace": 0,
            "after_length":     0,
            "after_repetition": 0,
            "after_pii":        0,
            "after_final_norm": 0,
            "after_dedup":      0,
        }

        passed = []

        for row in tqdm(subset, desc="  Cleaning", unit="docs", ncols=80):
            text = row["text"] if isinstance(row, dict) else row
            stats["total"] += 1

            # Stage 1 — Unicode fix
            text = self.unicode_fix(text)
            stats["after_unicode"] += 1

            # Stage 2 — Whitespace normalization
            text = self.whitespace_normalization(text)
            stats["after_whitespace"] += 1

            # Stage 3 — Length filter
            if not self.length_filter(text):
                continue
            stats["after_length"] += 1

            # Stage 4 — Repetition filter
            if not self.repetition_filter(text):
                continue
            stats["after_repetition"] += 1

            # Stage 5 — PII scrubbing
            text = self.pii_scrubbing(text)
            stats["after_pii"] += 1

            # Stage 6 — Final normalization
            text = self.final_normalization(text)
            stats["after_final_norm"] += 1

            passed.append(text)


        results = self.deduplicate(passed)
        stats["after_dedup"] = len(results)

        self._print_stats(stats)
        return results, stats
    

    def _print_stats(self, stats: dict):
        """Print per-stage breakdown table after apply()."""
        total = stats["total"]

        rows = [
            ("Total input",             stats["total"]),
            ("After unicode fix",        stats["after_unicode"]),
            ("After whitespace norm",    stats["after_whitespace"]),
            ("After length filter",      stats["after_length"]),
            ("After repetition filter",  stats["after_repetition"]),
            ("After PII scrubbing",      stats["after_pii"]),
            ("After final normalization",stats["after_final_norm"]),
            ("After deduplication",      stats["after_dedup"]),
        ]

        print(f"\n  {'Stage':<30} {'Docs':>10} {'Retention':>10}")
        print(f"  {'-' * 52}")
        for label, count in rows:
            retention = f"{count / total * 100:.1f}%" if total > 0 else "n/a"
            print(f"  {label:<30} {count:>10,} {retention:>10}")
        print(f"  {'-' * 52}")
        print(f"  {'Final output':<30} {stats['after_dedup']:>10,} "
              f"{stats['after_dedup'] / total * 100:>9.1f}%\n")