import pytest
from src.tcelm_corpus.decontaminate import Decontaminator
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition

def test_benchmark_decontamination():
    decontam = Decontaminator(shingle_size=5, min_token_overlap_ratio=0.5)
    decontam.register_benchmark_item("test_benchmark", "item_1", "The capital of France is Paris.")

    clean_doc = CanonicalDocument(
        document_id="d1",
        parent_document_id="p1",
        source="wiki",
        source_revision="",
        source_record_id="",
        source_url_or_provenance="",
        license="",
        authors="",
        title="",
        publication_date="",
        language="en",
        raw_text_hash="",
        normalized_text_hash="h1",
        dedup_cluster_id="p1",
        normalized_text="This is a clean document about physics and astronomy.",
        document_type="",
        domain="wiki",
        genre="",
        quality=QualityScores(),
        position=SegmentPosition()
    )

    dirty_doc = CanonicalDocument(
        document_id="d2",
        parent_document_id="p2",
        source="wiki",
        source_revision="",
        source_record_id="",
        source_url_or_provenance="",
        license="",
        authors="",
        title="",
        publication_date="",
        language="en",
        raw_text_hash="",
        normalized_text_hash="h2",
        dedup_cluster_id="p2",
        normalized_text="Geography facts: The capital of France is Paris. It is a large city.",
        document_type="",
        domain="wiki",
        genre="",
        quality=QualityScores(),
        position=SegmentPosition()
    )

    retained, logs = decontam.decontaminate([clean_doc, dirty_doc])
    assert len(retained) == 1
    assert retained[0].document_id == "d1"
    assert len(logs) == 1
    assert logs[0]["matched_document_id"] == "d2"
