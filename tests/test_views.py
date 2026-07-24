import pytest
from src.tcelm_corpus.views import DerivedViewGenerator
from src.tcelm_corpus.schema import TokenizedDocument

def test_causal_packing_views():
    generator = DerivedViewGenerator(ctx_length=32)
    docs = [
        TokenizedDocument(document_id="d1", parent_document_id="p1", source="s1", split="train", token_ids=list(range(100))),
        TokenizedDocument(document_id="d2", parent_document_id="p2", source="s2", split="train", token_ids=list(range(20))),
    ]
    views = generator.generate_causal_packing_views(docs, split="train", allow_packing=True)
    assert len(views) > 0
    assert len(views[0].input_token_ids) == 31
    assert len(views[0].target_token_ids) == 31
    assert views[0].split == "train"
    assert views[0].usage == "pretraining"
    assert "d1" in views[0].source_document_ids
    assert "p1" in views[0].source_parent_document_ids

def test_split_isolation_and_no_cross_split_packing():
    generator = DerivedViewGenerator(ctx_length=32)
    train_docs = [
        TokenizedDocument(document_id="d_train", parent_document_id="p_train", source="s1", split="train", token_ids=list(range(50)))
    ]
    holdout_docs = [
        TokenizedDocument(document_id="d_holdout", parent_document_id="p_holdout", source="s1", split="trajectory_holdout", token_ids=list(range(50)))
    ]

    train_views = generator.generate_causal_packing_views(train_docs, split="train", allow_packing=True)
    holdout_views = generator.generate_causal_packing_views(holdout_docs, split="trajectory_holdout", allow_packing=False)

    for v in train_views:
        assert v.split == "train"
        assert v.usage == "pretraining"
        assert "d_holdout" not in v.source_document_ids

    for v in holdout_views:
        assert v.split == "trajectory_holdout"
        assert v.usage == "evaluation"
        assert "d_train" not in v.source_document_ids
