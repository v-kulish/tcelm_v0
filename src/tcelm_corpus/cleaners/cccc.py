import re
from urllib.parse import urlparse
from typing import Dict, Tuple, Optional
from .generic import CleaningResult

BOILERPLATE_PATTERNS = [
    re.compile(r'cookie\s+policy', re.I),
    re.compile(r'privacy\s+policy', re.I),
    re.compile(r'all\s+rights\s+reserved', re.I),
    re.compile(r'subscribe\s+to\s+newsletter', re.I),
    re.compile(r'accept\s+all\s+cookies', re.I),
    re.compile(r'cart\s+checkout', re.I),
    re.compile(r'search\s+results\s+for', re.I),
]

class CCCCCleaner:
    def __init__(self, max_hostname_quota_tokens: int = 2250000):
        self.max_hostname_quota_tokens = max_hostname_quota_tokens
        self.hostname_token_counts: Dict[str, int] = {}

    def extract_hostname(self, url_or_provenance: str) -> str:
        if not url_or_provenance:
            return "unknown"
        try:
            if not url_or_provenance.startswith(('http://', 'https://')):
                url_or_provenance = 'http://' + url_or_provenance
            parsed = urlparse(url_or_provenance)
            return parsed.netloc.lower() or "unknown"
        except Exception:
            return "unknown"

    def clean(self, text: str, url_or_provenance: str = "") -> CleaningResult:
        if not text:
            return CleaningResult("", True, "empty_text", {})

        # Hostname quota check
        hostname = self.extract_hostname(url_or_provenance)
        current_hostname_tokens = self.hostname_token_counts.get(hostname, 0)
        approx_doc_tokens = len(text.split())

        if current_hostname_tokens >= self.max_hostname_quota_tokens:
            return CleaningResult(text, True, f"hostname_quota_exceeded_{hostname}", {})

        # Paragraph count check (must have at least 3 prose paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) >= 10]
        if len(paragraphs) < 3:
            return CleaningResult(text, True, f"insufficient_prose_paragraphs_{len(paragraphs)}<3", {})

        # Short lines ratio check (> 30% of lines have fewer than 4 words)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        short_lines = sum(1 for line in lines if len(line.split()) < 4)
        if lines and (short_lines / len(lines)) > 0.30:
            return CleaningResult(text, True, f"too_many_short_lines_{(short_lines/len(lines)):.3f}>0.30", {})

        # URL token ratio (> 20% tokens are URLs)
        tokens = text.split()
        url_tokens = sum(1 for t in tokens if t.startswith(('http://', 'https://', 'www.')))
        if tokens and (url_tokens / len(tokens)) > 0.20:
            return CleaningResult(text, True, f"too_many_url_tokens_{(url_tokens/len(tokens)):.3f}>0.20", {})

        # Update hostname tracker
        self.hostname_token_counts[hostname] = current_hostname_tokens + approx_doc_tokens

        return CleaningResult(text, False, None, {"approx_tokens": approx_doc_tokens, "hostname": hostname})
