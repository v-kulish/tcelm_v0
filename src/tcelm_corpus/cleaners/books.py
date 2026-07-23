import re
from typing import Optional, List, Dict
from .generic import CleaningResult

GUTENBERG_START_REGEX = re.compile(r'\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.I)
GUTENBERG_END_REGEX = re.compile(r'\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.I)

class BooksCleaner:
    def __init__(self):
        pass

    def clean_gutenberg(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Strip Gutenberg header and footer
        start_match = GUTENBERG_START_REGEX.search(text)
        if start_match:
            text = text[start_match.end():]

        end_match = GUTENBERG_END_REGEX.search(text)
        if end_match:
            text = text[:end_match.start()]

        text = text.strip()
        tokens = text.split()
        if len(tokens) < 1024:
            return CleaningResult(text, True, f"gutenberg_post_strip_below_1024_{len(tokens)}", {})

        return CleaningResult(text, False, None, {"approx_tokens": len(tokens)})

    def clean_pre1929(self, text: str, unigram_log_likelihood: float = -10.0) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        if unigram_log_likelihood <= -20.0:
            return CleaningResult(text, True, f"low_ocr_log_likelihood_{unigram_log_likelihood}<-20", {})

        # Controlled dehyphenation: line ends with letter + hyphen, next line begins with lowercase letter
        dehyphenated = re.sub(r'([a-zA-Z]+)-\n([a-z]+)', r'\1\2', text)

        # Check internal digit words ratio (words like w0rd or th1s)
        words = dehyphenated.split()
        internal_digit_words = sum(1 for w in words if re.search(r'[a-zA-Z]\d+[a-zA-Z]', w))
        if words and (internal_digit_words / len(words)) > 0.15:
            return CleaningResult(dehyphenated, True, f"ocr_internal_digits_ratio_exceeded_{(internal_digit_words/len(words)):.3f}>0.15", {})

        if len(words) < 1024:
            return CleaningResult(dehyphenated, True, f"pre1929_below_1024_{len(words)}", {})

        return CleaningResult(dehyphenated, False, None, {"approx_tokens": len(words)})

    def clean_doab_or_pressbooks(self, text: str) -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Strip repeated TOC / publisher ads / navigation
        cleaned = re.sub(r'(?:Table of Contents|Publisher Advertisement|Downloaded from).*?\n', '', text, flags=re.I)
        words = cleaned.split()

        if len(words) < 512:
            return CleaningResult(cleaned, True, f"book_segment_below_512_{len(words)}", {})

        return CleaningResult(cleaned, False, None, {"approx_tokens": len(words)})
