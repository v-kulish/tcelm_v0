import pytest
import os
import shutil
from src.tcelm_corpus.runner import CorpusPipelineRunner

def test_full_pipeline_stream_real_data(tmp_path):
    output_dir = str(tmp_path / "pipeline_run")
    config_path = "config/corpus_v0_3b.json"

    runner = CorpusPipelineRunner(
        config_path=config_path,
        output_dir=output_dir,
        target_scale_tokens=50_000_000 # 50M scale target
    )

    # Run pipeline with a small limit of 5 records per source for real HF streaming test
    summary = runner.run(max_records_per_source=5)

    assert "canonical_document_count" in summary
    assert summary["canonical_document_count"] >= 0
    assert os.path.exists(os.path.join(output_dir, "reports", "01_source_report.md"))
    assert os.path.exists(os.path.join(output_dir, "reports", "02_quality_report.md"))
    assert os.path.exists(os.path.join(output_dir, "reports", "03_deduplication_report.md"))
    assert os.path.exists(os.path.join(output_dir, "reports", "04_structure_report.md"))
    assert os.path.exists(os.path.join(output_dir, "reports", "05_tokenizer_report.md"))
    assert os.path.exists(os.path.join(output_dir, "reports", "06_split_report.md"))
    assert os.path.exists(os.path.join(output_dir, "reports", "07_benchmark_report.md"))
