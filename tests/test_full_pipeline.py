import pytest
import os
import json
from src.tcelm_corpus.runner import CorpusPipelineRunner
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO

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

    stages_dir = os.path.join(output_dir, "stages")

    # 1. Assert all 12 stage manifests exist and status == SUCCESS
    stage_names = [
        "01_ingest", "02_select_pool", "03_normalize_clean", "04_segment",
        "05_dedup", "06_decontaminate", "07_split", "08_train_tokenizer",
        "09_tokenize_select", "10_freeze", "11_generate_views", "12_stats_reports"
    ]
    for s_name in stage_names:
        manifest_path = os.path.join(stages_dir, s_name, "manifest.json")
        assert os.path.exists(manifest_path), f"Manifest missing for stage {s_name}"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data.get("status") == "SUCCESS", f"Stage {s_name} did not succeed"

    # 2. Assert Stage 01 ingested records > 0 and IDs are namespaced 32-char hex strings
    io_01 = ParquetShardIO(os.path.join(stages_dir, "01_ingest"))
    recs_01 = list(io_01.read_shards())
    assert len(recs_01) > 0, "Stage 01 output 0 records"
    assert len(recs_01[0]["document_id"]) == 32, "Stage 01 document_id is not a namespaced 32-char hex string"
    assert "raw_text_hash" in recs_01[0], "Stage 01 record missing raw_text_hash"

    # 3. Assert Stage 09 output layer_a_selected and layer_b_selected records > 0
    io_09_a = ParquetShardIO(os.path.join(stages_dir, "09_tokenize_select", "layer_a_selected"))
    io_09_b = ParquetShardIO(os.path.join(stages_dir, "09_tokenize_select", "layer_b_selected"))
    recs_09_a = list(io_09_a.read_shards())
    recs_09_b = list(io_09_b.read_shards())
    assert len(recs_09_a) > 0, "Stage 09 Layer A selected output 0 records"
    assert len(recs_09_b) > 0, "Stage 09 Layer B selected output 0 records"

    # 4. Assert Stage 10 Freeze verified 1-to-1 document alignment
    ids_a = [r["document_id"] for r in recs_09_a]
    ids_b = [r["document_id"] for r in recs_09_b]
    assert ids_a == ids_b, "Layer A and Layer B document IDs do not match 1-to-1"

    # 5. Assert Stage 11 generated split-partitioned view recipes containing materialized token sequences
    io_11 = ParquetShardIO(os.path.join(stages_dir, "11_generate_views", "train"))
    recs_11 = list(io_11.read_shards())
    assert len(recs_11) > 0, "Stage 11 generated 0 view recipes in train split"
    assert "input_token_ids_json" in recs_11[0], "Layer C record missing input_token_ids_json"
    assert recs_11[0]["split"] == "train", "Layer C record split mismatch"
    assert "source_document_ids_json" in recs_11[0], "Layer C record missing source_document_ids_json"
    input_ids = json.loads(recs_11[0]["input_token_ids_json"])
    assert isinstance(input_ids, list) and len(input_ids) > 0, "Layer C input_token_ids_json is empty"

    # 6. Assert Stage 12 saved smoothed unigram frequency log probabilities in float32 numpy (.npy & .npz) and JSON formats
    s12_dir = os.path.join(stages_dir, "12_stats_reports")
    assert os.path.exists(os.path.join(s12_dir, "unigram_log_probs.npy")), "Unigram .npy file missing in Stage 12"
    assert os.path.exists(os.path.join(s12_dir, "source_unigram_log_probs.npz")), "Source unigram .npz file missing in Stage 12"
    assert os.path.exists(os.path.join(s12_dir, "unigram_log_probs.json")), "Unigram JSON file missing in Stage 12"

    # 7. Assert mandatory reports exist in output/reports/
    reports_dir = os.path.join(output_dir, "reports")
    assert os.path.exists(os.path.join(reports_dir, "01_source_report.md"))
    assert os.path.exists(os.path.join(reports_dir, "07_benchmark_report.md"))
