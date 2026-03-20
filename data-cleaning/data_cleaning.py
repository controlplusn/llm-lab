import re
import ftfy
import unicodedata
import multiprocessing as mp
from tqdm import tqdm
from collections import Counter


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


    _NOISE_PATTERNS = [
        # Null bytes and ASCII control characters (except \n \t)
        (re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'), ''),

        # Zero-width and invisible unicode characters
        (re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]'), ''),

        # HTML entities that slipped through trafilatura
        (re.compile(r'&amp;'),   '&'),
        (re.compile(r'&nbsp;'),  ' '),
        (re.compile(r'&lt;'),    '<'),
        (re.compile(r'&gt;'),    '>'),
        (re.compile(r'&#\d+;'),  ''),   # numeric HTML entities  &#39; &#160;

        # Blocks of repeated non-alphanumeric symbols (visual spam)
        # e.g. "████████" "————————" "========" "........"
        (re.compile(r'([^\w\s])\1{4,}'), r'\1'),  # 5+ same symbol → keep 1

        (re.compile(r'\|{2,}'), ' '),          # collapse double pipes
        (re.compile(r'^\||\|$', re.M), ' '),   # strip leading/trailing pipes per line
    ]

    def noise_removal(self, text: str) -> str:
        for pattern, replacement in self._NOISE_PATTERNS:
            text = pattern.sub(replacement, text)

        return text
        
    
    def pipe_table_filter(self, text: str) -> bool:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return True

        pipe_lines = sum(1 for l in lines if l.count('|') >= 2)
        pipe_ratio = pipe_lines / len(lines)

        if pipe_ratio > self.config.get("max_pipe_line_ratio", 0.15):
            return False
        return True

    _LOGINWALL_PATTERN = re.compile(
        r'(?i)(you must (be|have)|must be (a |logged|registered|signed)|'
        r'sign in (to|now)|log in (to|now)|create an account|'
        r'register (to|for|now)|already a member|members? only)',
    )

    def loginwall_filter(self, text: str) -> bool:
        if self._LOGINWALL_PATTERN.search(text[:300]):
            if len(text.split()) < 150:
                return False
        return True
    

    _DATESTAMP_LEAD = re.compile(
        r'^(january|february|march|april|may|june|july|august|'
        r'september|october|november|december|\d{1,2}[\/\-\.]\d{1,2})',
        re.IGNORECASE
    )
    _NAV_LEAD = re.compile(
        r'^(category\s*(archives?)?:|news of the week|tags?:|posted (in|by|on))',
        re.IGNORECASE
    )

    def content_lead_filter(self, text: str) -> bool:
        lead = text[:80].strip()
        if self._DATESTAMP_LEAD.match(lead) and len(text.split()) < 120:
            return False
        if self._NAV_LEAD.match(lead):
            return False
        return True
    

    # ----- 2. Heuristic Filters -----

    # ── 2a. Length Filter ────────────────────────────────────
    
    def length_filter(self, text: str) -> bool:
        """
            1. Word-count bounds
            2. Character-level bounds + avg word length
            3. Line-level bounds + avg words per line
        """

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        lines = [l for l in text.splitlines() if l.strip()]
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
            2. Noise removal
            3. Whitespace norm          (normalization)
            4. Length filter            (heuristic — rejects doc)
            5. Repetition filter        (heuristic — rejects doc)
            6. PII scrubbing            (replacement)
            7. Final normalization      (cleanup)
        """

        stats = {
            "after_length": 0, 
            "after_repetition": 0,
            "after_pipe_table": 0,
            "after_loginwall": 0,
            "after_content_lead": 0,
            "after_pii": 0, 
            "after_final_norm": 0,
        }
        return self._clean_one(text, stats)
    

    def apply(self, subset) -> tuple[list[str], dict]:
        print("\n[DataCleaning] Running pipeline...")

        stats = {
            "total": 0,
            "after_length": 0,
            "after_repetition": 0,
            "after_pipe_table": 0,
            "after_loginwall": 0,
            "after_content_lead": 0,
            "after_pii": 0,
            "after_final_norm": 0,
        }

        passed = []

        for row in tqdm(
            subset, 
            desc="Cleaning", 
            unit="docs", 
            ncols=80
        ):
            stats["total"] += 1
            text = self._extract_text(row)
            cleaned = self._clean_one(text, stats)
            if cleaned is not None:
                passed.append(cleaned)

        self._print_stats(stats)
        return passed, stats


    @staticmethod
    def _extract_text(row) -> str:
        return row["text"] if isinstance(row, dict) else row


    # Deduplication stream: clean one document
    def _clean_one(self, text: str, stats: dict) -> str | None:
        text = self.unicode_fix(text)
        text = self.noise_removal(text)
        text = self.whitespace_normalization(text)

        if not self.length_filter(text):
            return None
        stats["after_length"] += 1

        if not self.repetition_filter(text):
            return None
        stats["after_repetition"] += 1

        if not self.pipe_table_filter(text):
            return None
        stats["after_pipe_table"] += 1

        if not self.loginwall_filter(text):
            return None
        stats["after_loginwall"] += 1

        if not self.content_lead_filter(text):
            return None
        stats["after_content_lead"] += 1

        text = self.pii_scrubbing(text)
        stats["after_pii"] += 1

        text = self.final_normalization(text)
        stats["after_final_norm"] += 1

        return text
    

    # Deduplication stream: generate cleaned texts
    def _cleaned_stream(self, source_iterator, stats: dict):
        for row in tqdm(source_iterator, desc="Cleaning", unit="docs", ncols=80):
            stats["total"] += 1
            text = self._extract_text(row)
            cleaned = self._clean_one(text, stats)
            if cleaned is not None:
                yield cleaned

    
    def apply_streaming(
        self,
        source_iterator
    ) -> tuple[list[str], dict]:
        stats = {
            "total": 0,
            "after_length": 0,
            "after_repetition": 0,
            "after_pipe_table": 0,
            "after_loginwall": 0,
            "after_content_lead": 0, 
            "after_pii": 0,
            "after_final_norm": 0,
        }

        cleaned  = self._cleaned_stream(source_iterator, stats)

        results = list(cleaned)
        self._print_stats(stats)

        return results, stats

    

    def _print_stats(self, stats: dict):
        total = stats["total"]

        rows = [
            ("Total input", stats["total"]),
            ("After length filter", stats["after_length"]),
            ("After repetition filter", stats["after_repetition"]),
            ("After pipe-table filter", stats["after_pipe_table"]),
            ("After login-wall filter", stats["after_loginwall"]),
            ("After content-lead filter",stats["after_content_lead"]),
            ("After PII scrubbing", stats["after_pii"]),
            ("After final normalization", stats["after_final_norm"]),
        ]

        print(f"\n  {'Stage':<30} {'Docs':>10} {'Retention':>10}")
        print(f"  {'-' * 52}")

        for label, count in rows:
            retention = f"{count / total * 100:.1f}%" if total > 0 else "n/a"
            print(f"  {label:<30} {count:>10,} {retention:>10}")

        print(f"  {'-' * 52}")
        print(f"  {'Final output':<30} {stats['after_final_norm']:>10,} "
                f"{stats['after_final_norm'] / total * 100:>9.1f}%\n")