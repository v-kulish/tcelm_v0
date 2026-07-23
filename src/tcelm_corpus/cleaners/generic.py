import re
import string
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

@dataclass
class CleaningResult:
    cleaned_text: str
    is_rejected: bool
    rejection_reason: Optional[str]
    metrics: Dict[str, float]

class GenericCleaner:
    def __init__(self):
        self.printable_chars = set(string.printable)

    def clean(
        self,
        text: str,
        source_category: str,
        min_doc_length: int = 128,
        max_doc_length: int = 32768,
        min_english_prob: float = 0.50
    ) -> CleaningResult:
        if not text or not text.strip():
            return CleaningResult("", True, "empty_document", {})

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        total_lines = len(lines)
        if total_lines == 0:
            return CleaningResult("", True, "no_non_empty_lines", {})

        # Length estimate in words (token approximation)
        word_tokens = text.split()
        num_tokens = len(word_tokens)

        if num_tokens < min_doc_length:
            return CleaningResult(text, True, f"below_min_length_{num_tokens}<{min_doc_length}", {})

        # Character Quality Ratios
        char_count = len(text)
        printable_count = sum(1 for c in text if c in self.printable_chars)
        printable_ratio = printable_count / max(char_count, 1)

        alpha_count = sum(1 for c in text if c.isalpha())
        alphabetic_ratio = alpha_count / max(char_count, 1)

        # Thresholds by source category
        is_technical_or_scientific = source_category in ["scientific", "technical", "government_legal"]
        min_alphabetic = 0.25 if is_technical_or_scientific else 0.45

        if printable_ratio < 0.98:
            return CleaningResult(text, True, f"low_printable_ratio_{printable_ratio:.3f}<0.98", {})
        if alphabetic_ratio < min_alphabetic:
            return CleaningResult(text, True, f"low_alphabetic_ratio_{alphabetic_ratio:.3f}<{min_alphabetic}", {})

        # Line Repetition Check
        unique_lines = set(lines)
        unique_line_ratio = len(unique_lines) / total_lines
        duplicate_line_ratio = 1.0 - unique_line_ratio

        if duplicate_line_ratio > 0.20:
            return CleaningResult(text, True, f"high_duplicate_line_ratio_{duplicate_line_ratio:.3f}>0.20", {})
        if unique_line_ratio < 0.60:
            return CleaningResult(text, True, f"low_unique_line_ratio_{unique_line_ratio:.3f}<0.60", {})

        # Median apparent word length check (excluding URLs and equations)
        non_url_words = [w for w in word_tokens if not w.startswith('http') and not w.startswith('<')]
        if non_url_words:
            lengths = sorted([len(w) for w in non_url_words])
            median_word_length = lengths[len(lengths) // 2]
            if median_word_length > 15:
                return CleaningResult(text, True, f"median_word_length_exceeded_{median_word_length}>15", {})

        # Page number / navigation / punctuation line ratio check
        junk_line_count = 0
        for l in lines:
            if l.isdigit() or all(c in string.punctuation for c in l) or len(l.split()) < 2:
                junk_line_count += 1
        junk_ratio = junk_line_count / total_lines
        if junk_ratio > 0.30:
            return CleaningResult(text, True, f"high_junk_line_ratio_{junk_ratio:.3f}>0.30", {})

        metrics = {
            "printable_ratio": printable_ratio,
            "alphabetic_ratio": alphabetic_ratio,
            "unique_line_ratio": unique_line_ratio,
            "approx_tokens": float(num_tokens)
        }

        return CleaningResult(text, False, None, metrics)
