import os
import json
import pytest
from src.tcelm_corpus.stages.s11_generate_views import Stage11GenerateViews
from src.tcelm_corpus.config import CorpusPipelineConfig
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO

def test_stage11_requires_freeze_manifest(tmp_path):
    output_dir = str(tmp_path / "run")
    os.makedirs(os.path.join(output_dir, "stages", "09_tokenize_select", "layer_b_selected"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stages", "08_train_tokenizer"), exist_ok=True)
    
    # Write tokenizer file
    tok_path = os.path.join(output_dir, "stages", "08_train_tokenizer", "tokenizer.json")
    with open(tok_path, "w") as f:
        f.write("{}")

    # Write layer B dummy shard
    shard_io = ParquetShardIO(os.path.join(output_dir, "stages", "09_tokenize_select", "layer_b_selected"))
    shard_io.write_records_to_shards([{"doc": 1}], shard_prefix="part")

    config = CorpusPipelineConfig()
    stage = Stage11GenerateViews(output_dir, config)

    # Missing 10_freeze/freeze_manifest.json must raise RuntimeError in get_additional_cache_inputs
    with pytest.raises(RuntimeError, match="freeze_manifest.json missing"):
        stage.get_additional_cache_inputs()

def test_layer_b_shard_modification_invalidates_stage11_cache(tmp_path):
    output_dir = str(tmp_path / "run")
    layer_b_dir = os.path.join(output_dir, "stages", "09_tokenize_select", "layer_b_selected")
    s08_dir = os.path.join(output_dir, "stages", "08_train_tokenizer")
    s10_dir = os.path.join(output_dir, "stages", "10_freeze")
    os.makedirs(layer_b_dir, exist_ok=True)
    os.makedirs(s08_dir, exist_ok=True)
    os.makedirs(s10_dir, exist_ok=True)

    with open(os.path.join(s08_dir, "tokenizer.json"), "w") as f:
        f.write('{"test": 1}')

    shard_io = ParquetShardIO(layer_b_dir)
    shard_io.write_records_to_shards([{"doc": 1}], shard_prefix="part")

    with open(os.path.join(s10_dir, "freeze_manifest.json"), "w") as f:
        json.dump({"tokenizer_sha256": "dummy"}, f)

    config = CorpusPipelineConfig()
    stage = Stage11GenerateViews(output_dir, config)

    inputs1 = stage.get_additional_cache_inputs()
    assert "current_layer_b_artifact_digest" in inputs1

    # Modify Layer B shard
    shard_files = [os.path.join(layer_b_dir, f) for f in os.listdir(layer_b_dir) if f.endswith(".parquet")]
    with open(shard_files[0], "ab") as f:
        f.write(b"\nEXTRA_DATA")

    # Clear internal cache if any and re-check cache inputs
    inputs2 = stage.get_additional_cache_inputs()
    assert inputs1["current_layer_b_artifact_digest"] != inputs2["current_layer_b_artifact_digest"], "Stage 11 cache inputs did not change after Layer B shard modification!"
