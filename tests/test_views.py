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
    assert len(views[0].input_token_ids) == 32
    assert len(views[0].target_token_ids) == 32
    assert len(views[0].loss_mask) == 32
    assert len(views[0].attention_mask) == 32
    assert views[0].split == "train"
    assert views[0].usage == "pretraining"
    assert "d1" in views[0].source_document_ids
    assert "p1" in views[0].source_parent_document_ids

def test_shifted_target_loss_and_attention_mask_alignment():
    generator = DerivedViewGenerator(ctx_length=8)
    # real sequence length R = 5
    real_seq = [0, 10, 20, 30, 1]
    view = generator.finalize_packed_sequence(
        real_sequence=real_seq,
        source_document_ids=["docA"],
        source_parent_document_ids=["parentA"],
        split="train",
        view_type="causal_single_doc",
        view_counter=0,
        eos_id=1
    )
    assert len(view.input_token_ids) == 8
    assert len(view.target_token_ids) == 8
    assert len(view.loss_mask) == 8
    assert len(view.attention_mask) == 8

    # R = 5, target count R - 1 = 4, input count min(R, C) = 5
    assert sum(view.loss_mask) == 4
    assert sum(view.attention_mask) == 5
    assert view.loss_mask == [1, 1, 1, 1, 0, 0, 0, 0]
    assert view.attention_mask == [1, 1, 1, 1, 1, 0, 0, 0]

def test_final_buffer_flush_and_lineage_coverage():
    generator = DerivedViewGenerator(ctx_length=32)
    short_docs = [
        TokenizedDocument(document_id="short_1", parent_document_id="p1", source="s1", split="train", token_ids=[10, 11, 12]),
        TokenizedDocument(document_id="short_2", parent_document_id="p2", source="s1", split="train", token_ids=[20, 21, 22]),
        TokenizedDocument(document_id="short_3", parent_document_id="p3", source="s1", split="train", token_ids=[30, 31, 32]),
    ]

    views = generator.generate_causal_packing_views(short_docs, split="train", allow_packing=True)
    assert len(views) > 0

    # Lineage coverage invariant test: all short documents must be in causal lineage
    all_causal_doc_ids = set()
    for v in views:
        all_causal_doc_ids.update(v.source_document_ids)

    for doc in short_docs:
        assert doc.document_id in all_causal_doc_ids, f"Document `{doc.document_id}` missing from causal view lineage"

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
