import pytest
from src.tcelm_corpus.splits import SplitAssigner
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition

def test_split_assignment():
    assigner = SplitAssigner(train_share=0.8, val_share=0.1, test_share=0.1, trajectory_holdout_share=0.0)
    docs = []
    for i in range(100):
        d = CanonicalDocument(
            document_id=f"doc_{i}",
            parent_document_id=f"parent_{i}",
            source="test",
            source_revision="",
            source_record_id="",
            source_url_or_provenance="",
            license="",
            authors="",
            title="",
            publication_date="",
            language="en",
            raw_text_hash="",
            normalized_text_hash=f"hash_{i}",
            dedup_cluster_id=f"parent_{i}",
            normalized_text=f"Text for document {i}",
            document_type="",
            domain="web",
            genre="",
            quality=QualityScores(),
            position=SegmentPosition()
        )
        docs.append(d)

    assigned = assigner.assign_splits(docs)
    splits = {d.split for d in assigned}
    assert "train" in splits
    assert len(assigned) == 100
