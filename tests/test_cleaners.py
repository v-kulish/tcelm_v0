import pytest
from src.tcelm_corpus.cleaners.generic import GenericCleaner
from src.tcelm_corpus.cleaners.cccc import CCCCCleaner
from src.tcelm_corpus.cleaners.stackexchange import StackExchangeCleaner
from src.tcelm_corpus.cleaners.scientific import ScientificCleaner

def test_generic_cleaner_min_length():
    cleaner = GenericCleaner()
    short_text = "Too short"
    res = cleaner.clean(short_text, source_category="web", min_doc_length=128)
    assert res.is_rejected
    assert "below_min_length" in res.rejection_reason

def test_cccc_hostname_quota_cap():
    cleaner = CCCCCleaner(max_hostname_quota_tokens=100)
    prose = "Word " * 60 + "\n\n" + "Prose " * 60 + "\n\n" + "Paragraph " * 60
    
    # First doc accepted
    res1 = cleaner.clean(prose, url_or_provenance="https://example.com/article1")
    assert not res1.is_rejected

    # Second doc from same hostname rejected due to quota cap
    res2 = cleaner.clean(prose, url_or_provenance="https://example.com/article2")
    assert res2.is_rejected
    assert "hostname_quota_exceeded" in res2.rejection_reason

def test_stackexchange_turns():
    cleaner = StackExchangeCleaner()
    title = "How to sort a list in Python?"
    q_body = "I have an unsorted list of integers and want to sort it efficiently using Python built-in functions. " * 8
    answers = [
        {"body": "Use the built-in sorted() function or list.sort() method to sort lists efficiently in Python. " * 8, "score": 10, "is_accepted": True}
    ]
    res = cleaner.clean_thread(title, q_body, answers, site_name="stackoverflow")
    assert not res.is_rejected
    assert "<QUESTION_TITLE>" in res.cleaned_text
    assert "<ANSWER>" in res.cleaned_text

def test_arxiv_equations():
    cleaner = ScientificCleaner()
    text = ("The Einstein equation is \\begin{equation} E = mc^2 \\end{equation} as shown in physics literature. " * 40)
    res = cleaner.clean_arxiv(text)
    assert not res.is_rejected
    assert "<EQUATION>" in res.cleaned_text
