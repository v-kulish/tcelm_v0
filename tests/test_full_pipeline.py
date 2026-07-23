import pytest
import os
from src.tcelm_corpus.runner import CorpusPipelineRunner

def test_full_pipeline_sharded_stages(tmp_path):
    output_dir = str(tmp_path / "pipeline_run")
    config_path = "config/corpus_v0_3b.json"

    runner = CorpusPipelineRunner(
        config_path=config_path,
        output_dir=output_dir,
        target_scale_tokens=50_000_000,
        max_records_per_source=3 # micro limit for fast test
    )

    summary = runner.run(force_restart=True)

    # Check stage manifest outputs
    stages_dir = os.path.join(output_dir, "stages")
    assert os.path.exists(os.path.join(stages_dir, "01_ingest", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "02_select_pool", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "03_normalize_clean", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "04_segment", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "05_dedup", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "06_decontaminate", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "07_split", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "08_train_tokenizer", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "09_tokenize_select", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "10_freeze", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "11_generate_views", "manifest.json"))
    assert os.path.exists(os.path.join(stages_dir, "12_stats_reports", "manifest.json"))

    # Check mandatory reports exist
    reports_dir = os.path.join(output_dir, "reports")
    assert os.path.exists(os.path.join(reports_dir, "01_source_report.md"))
    assert os.path.exists(os.path.join(reports_dir, "07_benchmark_report.md"))
