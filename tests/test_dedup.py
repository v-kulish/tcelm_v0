import pytest
from src.tcelm_corpus.dedup import Deduplicator
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition

def test_exact_document_deduplication():
    dedup = Deduplicator()
    doc1 = CanonicalDocument(
        document_id="doc1",
        parent_document_id="p1",
        source="books",
        source_revision="v1",
        source_record_id="r1",
        source_url_or_provenance="",
        license="CC",
        authors="",
        title="Title 1",
        publication_date="",
        language="en",
        raw_text_hash="hash123",
        normalized_text_hash="hash123",
        dedup_cluster_id="p1",
        normalized_text="Identical text content for exact deduplication test.",
        document_type="book",
        domain="books",
        genre="prose",
        quality=QualityScores(final_quality_score=0.9),
        position=SegmentPosition()
    )
    doc2 = CanonicalDocument(
        document_id="doc2",
        parent_document_id="p2",
        source="web",
        source_revision="v1",
        source_record_id="r2",
        source_url_or_provenance="",
        license="CC",
        authors="",
        title="Title 2",
        publication_date="",
        language="en",
        raw_text_hash="hash123",
        normalized_text_hash="hash123",
        dedup_cluster_id="p2",
        normalized_text="Identical text content for exact deduplication test.",
        document_type="article",
        domain="web",
        genre="prose",
        quality=QualityScores(final_quality_score=0.5),
        position=SegmentPosition()
    )

    retained = dedup.deduplicate([doc1, doc2])
    assert len(retained) == 1
    # Book winner should be selected based on quality score
    assert retained[0].document_id == "doc1"
