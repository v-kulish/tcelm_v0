import pytest
from unittest.mock import patch, MagicMock
from src.tcelm_corpus.stages.s01_ingest import Stage01Ingest, SHA_RE
from src.tcelm_corpus.config import CorpusPipelineConfig, SourceConfig
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO

def test_sha_regex():
    assert SHA_RE.fullmatch("a1b2c3d4e5f60718293a4b5c6d7e8f9012345678") is not None
    assert SHA_RE.fullmatch("main") is None
    assert SHA_RE.fullmatch("a1b2c3") is None

def test_ingest_structural_probe_and_oversized_skipping(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
    config.smoke_total_tokens = 1000
    config.sources = [SourceConfig(name="common-pile/gutenberg_filtered", category="books", share=1.0, quota_3b_tokens=1000, revision="main")]

    oversized_text_1 = "word " * 5000  # ~5000 tokens > max_single_doc_prov_tokens (1000)
    oversized_text_2 = "book " * 6000  # ~6000 tokens > max_single_doc_prov_tokens (1000)
    fitting_text = "small text sample " * 20  # ~60 tokens <= max_single_doc_prov_tokens (1000)

    mock_ds = [
        {"id": "doc1", "text": oversized_text_1},
        {"id": "doc2", "text": oversized_text_2},
        {"id": "doc3", "text": fitting_text},
    ]

    with patch("src.tcelm_corpus.stages.s01_ingest.HfApi.dataset_info") as mock_api, \
         patch("src.tcelm_corpus.stages.s01_ingest.load_dataset") as mock_load:
        
        mock_info = MagicMock()
        mock_info.sha = "1111222233334444555566667777888899990000"
        mock_api.return_value = mock_info
        mock_load.return_value = mock_ds

        stage = Stage01Ingest(output_dir, config)
        res = stage.run_stage()

        recs = list(ParquetShardIO(f"{output_dir}/stages/01_ingest").read_shards())
        assert len(recs) == 2  # doc1 (probe) + doc3 (balanced)
        
        probe_rec = next(r for r in recs if r["source_record_id"] == "doc1")
        balanced_rec = next(r for r in recs if r["source_record_id"] == "doc3")

        assert probe_rec["smoke_structural_probe"] is True
        assert probe_rec["eligible_for_balanced_pool"] is False

        assert balanced_rec["smoke_structural_probe"] is False
        assert balanced_rec["eligible_for_balanced_pool"] is True

        assert res["rejection_counts"]["oversized_documents_skipped"]["common-pile/gutenberg_filtered"] == 1

def test_ingest_scan_limit(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
    config.smoke_total_tokens = 1000
    config.max_records_scanned_per_source = 2
    config.sources = [SourceConfig(name="common-pile/gutenberg_filtered", category="books", share=1.0, quota_3b_tokens=1000, revision="main")]

    mock_ds = [{"id": f"doc_{i}", "text": f"sample text {i}"} for i in range(10)]

    with patch("src.tcelm_corpus.stages.s01_ingest.HfApi.dataset_info") as mock_api, \
         patch("src.tcelm_corpus.stages.s01_ingest.load_dataset") as mock_load:
        
        mock_info = MagicMock()
        mock_info.sha = "1111222233334444555566667777888899990000"
        mock_api.return_value = mock_info
        mock_load.return_value = mock_ds

        stage = Stage01Ingest(output_dir, config)
        res = stage.run_stage()

        assert res["rejection_counts"]["records_scanned"]["common-pile/gutenberg_filtered"] == 2

def test_ingest_unresolved_sha_fallback(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
    config.production_mode = False
    config.sources = [SourceConfig(name="common-pile/test_filtered", category="web", share=1.0, quota_3b_tokens=1000, revision="main")]

    mock_ds = [{"id": "doc1", "text": "hello world"}]

    with patch("src.tcelm_corpus.stages.s01_ingest.HfApi.dataset_info") as mock_api, \
         patch("src.tcelm_corpus.stages.s01_ingest.load_dataset") as mock_load:
        
        mock_api.side_effect = Exception("Network offline")
        mock_load.return_value = mock_ds

        stage = Stage01Ingest(output_dir, config)
        res = stage.run_stage()

        mock_load.assert_called_once_with("common-pile/test_filtered", split="train", streaming=True, revision="main")

        recs = list(ParquetShardIO(f"{output_dir}/stages/01_ingest").read_shards())
        assert recs[0]["resolved_source_revision_sha"] is None
        assert recs[0]["revision_resolution_status"] == "unresolved_fallback"
