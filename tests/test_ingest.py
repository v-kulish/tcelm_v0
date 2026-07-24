import pytest
from unittest.mock import patch, MagicMock
from src.tcelm_corpus.stages.s01_ingest import Stage01Ingest, SHA_RE
from src.tcelm_corpus.config import CorpusPipelineConfig, SourceConfig
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO

def test_sha_regex():
    assert SHA_RE.fullmatch("a1b2c3d4e5f60718293a4b5c6d7e8f9012345678") is not None
    assert SHA_RE.fullmatch("main") is None
    assert SHA_RE.fullmatch("a1b2c3") is None

def test_ingest_pinned_revision_bypasses_api_call(tmp_path):
    output_dir = str(tmp_path / "run")
    pinned_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    config = CorpusPipelineConfig()
    config.sources = [SourceConfig(name="common-pile/test_filtered", category="web", share=1.0, quota_3b_tokens=1000, revision=pinned_sha)]

    mock_ds = [{"id": "doc1", "text": "hello world"}]

    with patch("src.tcelm_corpus.stages.s01_ingest.HfApi.dataset_info") as mock_api, \
         patch("src.tcelm_corpus.stages.s01_ingest.load_dataset") as mock_load:
        
        mock_load.return_value = mock_ds

        stage = Stage01Ingest(output_dir, config)
        res = stage.run_stage()

        mock_api.assert_not_called()
        mock_load.assert_called_once_with("common-pile/test_filtered", split="train", streaming=True, revision=pinned_sha)

        recs = list(ParquetShardIO(f"{output_dir}/stages/01_ingest").read_shards())
        assert recs[0]["resolved_source_revision_sha"] == pinned_sha
        assert recs[0]["revision_resolution_status"] == "pinned_sha"

def test_ingest_probe_skips_balanced_recovery(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
    config.smoke_total_tokens = 1000
    config.sources = [SourceConfig(name="common-pile/gutenberg_filtered", category="books", share=1.0, quota_3b_tokens=1000, revision="main")]

    oversized_1 = "word " * 5000  # ~5000 tokens > max_single_doc_prov_tokens (1000)
    oversized_2 = "book " * 6000  # ~6000 tokens > max_single_doc_prov_tokens (1000)
    oversized_3 = "text " * 7000  # ~7000 tokens > max_single_doc_prov_tokens (1000)
    fitting_text = "small text sample " * 20  # ~60 tokens <= max_single_doc_prov_tokens (1000)

    mock_ds = [
        {"id": "doc1", "text": oversized_1},
        {"id": "doc2", "text": oversized_2},
        {"id": "doc3", "text": oversized_3},
        {"id": "doc4", "text": fitting_text},
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
        assert len(recs) == 2  # doc1 (probe) + doc4 (balanced)
        
        probe_rec = next(r for r in recs if r["source_record_id"] == "doc1")
        balanced_rec = next(r for r in recs if r["source_record_id"] == "doc4")

        assert probe_rec["smoke_structural_probe"] is True
        assert probe_rec["eligible_for_balanced_pool"] is False

        assert balanced_rec["smoke_structural_probe"] is False
        assert balanced_rec["eligible_for_balanced_pool"] is True

        assert res["rejection_counts"]["oversized_documents_skipped"]["common-pile/gutenberg_filtered"] == 2

def test_ingest_consecutive_oversized_skips_bound(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
    config.smoke_total_tokens = 1000
    config.max_consecutive_oversized_skips_per_source = 3
    config.sources = [SourceConfig(name="common-pile/gutenberg_filtered", category="books", share=1.0, quota_3b_tokens=1000, revision="main")]

    mock_ds = [{"id": f"doc_{i}", "text": "word " * 5000} for i in range(10)]

    with patch("src.tcelm_corpus.stages.s01_ingest.HfApi.dataset_info") as mock_api, \
         patch("src.tcelm_corpus.stages.s01_ingest.load_dataset") as mock_load:
        
        mock_info = MagicMock()
        mock_info.sha = "1111222233334444555566667777888899990000"
        mock_api.return_value = mock_info
        mock_load.return_value = mock_ds

        stage = Stage01Ingest(output_dir, config)
        res = stage.run_stage()

        assert res["rejection_counts"]["stop_reasons"]["common-pile/gutenberg_filtered"] == "consecutive_oversized_limit"
        assert res["rejection_counts"]["oversized_documents_skipped"]["common-pile/gutenberg_filtered"] == 3

def test_ingest_global_records_scanned_bound(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
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
        assert res["rejection_counts"]["stop_reasons"]["common-pile/gutenberg_filtered"] == "max_records_scanned"

def test_ingest_strict_mode_fails_on_missing_sha(tmp_path):
    output_dir = str(tmp_path / "run")
    config = CorpusPipelineConfig()
    config.production_mode = True
    config.sources = [SourceConfig(name="common-pile/test_filtered", category="web", share=1.0, quota_3b_tokens=1000, revision="main")]

    with patch("src.tcelm_corpus.stages.s01_ingest.HfApi.dataset_info") as mock_api:
        mock_api.side_effect = Exception("API error")

        stage = Stage01Ingest(output_dir, config)
        with pytest.raises(RuntimeError, match="Production Build Failure"):
            stage.run_stage()
