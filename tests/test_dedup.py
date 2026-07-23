import pytest
from src.tcelm_corpus.dedup import Deduplicator
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition
from src.tcelm_corpus.stages.s05_dedup import Stage05Dedup, winner_key
from src.tcelm_corpus.config import CorpusPipelineConfig
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO

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

def test_winner_key_tie_breaking():
    rec1 = {"domain": "books", "priority": 100, "document_id": "doc1"}
    rec2 = {"domain": "books", "priority": 50, "document_id": "doc2"} # smaller priority preferred
    
    recs = [rec1, rec2]
    winner = min(recs, key=winner_key)
    assert winner["document_id"] == "doc2"

def test_parent_family_graph_union_stage05(tmp_path):
    stage_dir = str(tmp_path / "stage_05_test")
    config = CorpusPipelineConfig()

    # Create Parent A and Parent B with matching duplicate segment A1/B1 (> 64 words each)
    shared_text = "The quick brown fox jumps over the lazy dog " * 10
    unique_text_a = "Unique text for A2 chapter content description " * 10
    unique_text_b = "Unique text for B2 chapter content description " * 10

    recs = [
        {"document_id": "A1", "parent_document_id": "ParentA", "split_group_id": "ParentA", "source": "books", "domain": "books", "priority": 10, "normalized_text": shared_text},
        {"document_id": "A2", "parent_document_id": "ParentA", "split_group_id": "ParentA", "source": "books", "domain": "books", "priority": 20, "normalized_text": unique_text_a},
        {"document_id": "B1", "parent_document_id": "ParentB", "split_group_id": "ParentB", "source": "web", "domain": "web", "priority": 30, "normalized_text": shared_text},
        {"document_id": "B2", "parent_document_id": "ParentB", "split_group_id": "ParentB", "source": "web", "domain": "web", "priority": 40, "normalized_text": unique_text_b}
    ]

    input_io = ParquetShardIO(f"{stage_dir}/stages/04_segment")
    input_io.write_records_to_shards(recs)

    stage = Stage05Dedup(stage_dir, config)
    res = stage.run_stage()

    output_io = ParquetShardIO(f"{stage_dir}/stages/05_dedup")
    out_recs = list(output_io.read_shards())

    assert len(out_recs) == 3 # A1 (winner), A2, B2
    split_groups = {r["split_group_id"] for r in out_recs}
    # Parent A and Parent B must share the EXACT SAME unified split_group_id across ALL retained segments!
    assert len(split_groups) == 1, f"Parent-family graph union failed: split_groups={split_groups}"
