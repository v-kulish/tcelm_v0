import pytest
from src.tcelm_corpus.views import DerivedViewGenerator
from src.tcelm_corpus.schema import TokenizedDocument

def test_causal_packing_views():
    generator = DerivedViewGenerator(ctx_length=32)
    docs = [
        TokenizedDocument(document_id="d1", parent_document_id="p1", source="s1", split="train", token_ids=list(range(100))),
        TokenizedDocument(document_id="d2", parent_document_id="p2", source="s2", split="train", token_ids=list(range(20))),
    ]
    views = generator.generate_causal_packing_views(docs)
    assert len(views) > 0
    assert len(views[0].input_token_ids) == 31
    assert len(views[0].target_token_ids) == 31
