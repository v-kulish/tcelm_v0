import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

# Regex patterns for PII
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_REGEX = re.compile(r'\+?\d{1,4}[-.\s]?(?:\(\d{1,3}\)|\d{1,3})[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
IPV4_REGEX = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
IPV6_REGEX = re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b')
API_KEY_REGEX = re.compile(r'\b(?:sk-[a-zA-Z0-9]{32,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36})\b')
PRIVATE_KEY_REGEX = re.compile(r'-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP)?\s*PRIVATE\s+KEY-----')

# Control characters to strip (excluding \n and \t)
STRIP_CHARS_REGEX = re.compile(r'[\x00\u200b\u200e\u200f\u202a-\u202e\ue000-\uf8ff]')
REPEATED_SOFT_HYPHEN = re.compile(r'\u00ad{2,}')

@dataclass
class NormalizationResult:
    normalized_text: str
    is_rejected: bool
    rejection_reason: Optional[str]
    pii_counts: Dict[str, int]
    ufffd_ratio: float

class TextNormalizer:
    def __init__(self, ufffd_max_threshold: float = 0.001):
        self.ufffd_max_threshold = ufffd_max_threshold

    def normalize(self, text: str, is_code_or_math: bool = False) -> NormalizationResult:
        if not text:
            return NormalizationResult("", True, "empty_text", {}, 0.0)

        # Check U+FFFD ratio
        char_count = len(text)
        ufffd_count = text.count('\ufffd')
        ufffd_ratio = ufffd_count / max(char_count, 1)
        if ufffd_ratio > self.ufffd_max_threshold:
            return NormalizationResult(text, True, f"ufffd_ratio_exceeded_{ufffd_ratio:.4f}", {}, ufffd_ratio)

        # Unicode NFC normalization
        normalized = unicodedata.normalize('NFC', text)

        # Convert line endings to Unix \n
        normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')

        # Strip unneeded control chars and repeated soft hyphens
        normalized = STRIP_CHARS_REGEX.sub('', normalized)
        normalized = REPEATED_SOFT_HYPHEN.sub('\u00ad', normalized)

        # Whitespace handling outside code/math
        if not is_code_or_math:
            # Collapse runs of horizontal space (excluding \n) to single space
            lines = normalized.split('\n')
            processed_lines = []
            for line in lines:
                # Replace non-tab horizontal whitespace runs with single space
                line_sub = re.sub(r'[^\S\n\t]+', ' ', line).strip()
                processed_lines.append(line_sub)
            normalized = '\n'.join(processed_lines)
            
            # Convert more than 3 consecutive blank lines to 2 blank lines (\n\n\n -> \n\n)
            normalized = re.sub(r'\n{4,}', '\n\n\n', normalized)

        # PII Check & Redaction
        pii_counts = {
            "email": 0,
            "phone": 0,
            "ip": 0,
            "api_key": 0,
            "private_key": 0
        }

        # Check private key credential block first
        if PRIVATE_KEY_REGEX.search(normalized):
            return NormalizationResult(normalized, True, "private_key_block_detected", pii_counts, ufffd_ratio)

        # Redact API keys
        normalized, api_key_count = API_KEY_REGEX.subn('<API_KEY>', normalized)
        pii_counts["api_key"] = api_key_count

        # Redact emails
        normalized, email_count = EMAIL_REGEX.subn('<EMAIL>', normalized)
        pii_counts["email"] = email_count

        # Redact IP addresses
        normalized, ipv4_count = IPV4_REGEX.subn('<IP>', normalized)
        normalized, ipv6_count = IPV6_REGEX.subn('<IP>', normalized)
        pii_counts["ip"] = ipv4_count + ipv6_count

        # Redact phones
        normalized, phone_count = PHONE_REGEX.subn('<PHONE>', normalized)
        pii_counts["phone"] = phone_count

        # Rejection rules based on token density (only enforce density check on documents >= 50 tokens)
        approx_tokens = max(len(normalized.split()), 1)
        if approx_tokens >= 50:
            if (email_count * 1000 / approx_tokens) > 20:
                return NormalizationResult(normalized, True, "excessive_email_pii", pii_counts, ufffd_ratio)
            if (phone_count * 1000 / approx_tokens) > 20:
                return NormalizationResult(normalized, True, "excessive_phone_pii", pii_counts, ufffd_ratio)

        return NormalizationResult(normalized, False, None, pii_counts, ufffd_ratio)
