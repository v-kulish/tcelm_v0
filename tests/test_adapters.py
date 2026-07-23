import pytest
from src.tcelm_corpus.adapters import get_source_adapter
from src.tcelm_corpus.adapters.stackexchange import StackExchangeAdapter
from src.tcelm_corpus.adapters.technical import TechnicalAdapter

def test_stackexchange_adapter():
    raw_item = {
        "title": "How to sort in Python?",
        "question_body": "I have an unsorted list of integers.",
        "answers": [{"body": "Use sorted()."}],
        "license": "CC BY-SA 4.0"
    }
    adapter = get_source_adapter("common-pile/stackexchange_filtered", {})
    assert isinstance(adapter, StackExchangeAdapter)

    text, lic, meta = adapter.extract_record(raw_item)
    assert "<QUESTION_TITLE>" in text
    assert "<ANSWER>" in text
    assert lic == "CC BY-SA 4.0"

def test_technical_adapter():
    raw_item = {
        "title": "Fix bug in parser",
        "body": "Parser crashes on null byte.",
        "comments": [{"body": "PR looks good."}],
        "is_pull_request": True,
        "license": "MIT"
    }
    adapter = get_source_adapter("common-pile/github_archive_filtered", {})
    assert isinstance(adapter, TechnicalAdapter)

    text, lic, meta = adapter.extract_record(raw_item)
    assert "<PULL_REQUEST_TITLE>" in text
    assert "<REVIEW_COMMENT>" in text
    assert lic == "MIT"

def test_missing_license_fallback():
    raw_item = {"text": "Simple text without license field."}
    adapter = get_source_adapter("common-pile/cccc_filtered", {})
    text, lic, meta = adapter.extract_record(raw_item)
    assert lic == "missing"
