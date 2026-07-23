import pytest
from src.tcelm_corpus.normalize import TextNormalizer

def test_nfc_and_space_normalization():
    normalizer = TextNormalizer()
    raw_text = "This is a  test\r\nwith   multiple    spaces.\n\n\n\n\nAnd newline."
    res = normalizer.normalize(raw_text)
    assert not res.is_rejected
    assert "\r\n" not in res.normalized_text
    assert "multiple spaces." in res.normalized_text

def test_pii_redaction_email_phone_ip():
    normalizer = TextNormalizer()
    text = "Contact john.doe@example.com or call +1-555-123-4567. Server IP is 192.168.1.1."
    res = normalizer.normalize(text)
    assert not res.is_rejected
    assert "<EMAIL>" in res.normalized_text
    assert "<PHONE>" in res.normalized_text
    assert "<IP>" in res.normalized_text
    assert "john.doe@example.com" not in res.normalized_text

def test_private_key_rejection():
    normalizer = TextNormalizer()
    text = "Here is my secret:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    res = normalizer.normalize(text)
    assert res.is_rejected
    assert res.rejection_reason == "private_key_block_detected"

def test_ufffd_rejection():
    normalizer = TextNormalizer(ufffd_max_threshold=0.001)
    damaged_text = "Damaged " + "\ufffd" * 10
    res = normalizer.normalize(damaged_text)
    assert res.is_rejected
    assert "ufffd_ratio_exceeded" in res.rejection_reason
