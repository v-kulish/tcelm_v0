import pytest
from src.tcelm_corpus.splits import SplitAssigner
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition
from src.tcelm_corpus.stages.s07_split import Stage07Split
from src.tcelm_corpus.config import CorpusPipelineConfig

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
            split_group_id=f"parent_{i}",
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

def test_multi_segment_zero_split_leakage(tmp_path):
    stage_dir = str(tmp_path / "stage_07_test")
    config = CorpusPipelineConfig()
    
    # Create multi-segment documents sharing the same parent / split_group_id
    recs = [
        {"document_id": "book_chapter1", "parent_document_id": "book_1", "split_group_id": "book_1", "source": "books", "normalized_text": "Chapter 1"},
        {"document_id": "book_chapter2", "parent_document_id": "book_1", "split_group_id": "book_1", "source": "books", "normalized_text": "Chapter 2"},
        {"document_id": "book_chapter3", "parent_document_id": "book_1", "split_group_id": "book_1", "source": "books", "normalized_text": "Chapter 3"}
    ]

    # Initialize input parquet file in stage 06 dir
    from src.tcelm_corpus.storage.parquet_io import ParquetShardIO
    input_io = ParquetShardIO(f"{stage_dir}/stages/06_decontaminate")
    input_io.write_records_to_shards(recs)

    stage = Stage07Split(stage_dir, config)
    res = stage.run_stage()

    output_io = ParquetShardIO(f"{stage_dir}/stages/07_split")
    out_recs = list(output_io.read_shards())

    # Assert ALL 3 segments of book_1 are assigned to the EXACT SAME split
    assigned_splits = {r["split"] for r in out_recs if r["split_group_id"] == "book_1"}
    assert len(assigned_splits) == 1, f"Parent document split leakage detected: {assigned_splits}"
