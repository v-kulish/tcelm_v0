import os
import pytest
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO
from src.tcelm_corpus.storage.manifest import StageManifest
from src.tcelm_corpus.storage.checkpoint import StageCheckpointManager

def test_parquet_shard_io_read_write(tmp_path):
    stage_dir = str(tmp_path / "stage_test")
    io = ParquetShardIO(stage_dir, shard_size=2)

    records = [
        {"doc_id": "doc1", "text": "Text 1", "val": 10},
        {"doc_id": "doc2", "text": "Text 2", "val": 20},
        {"doc_id": "doc3", "text": "Text 3", "val": 30}
    ]

    shards = io.write_records_to_shards(records, shard_prefix="test_shard")
    assert len(shards) == 2 # 2 shards created (size 2 and size 1)

    read_back = list(io.read_shards(shard_prefix="test_shard"))
    assert len(read_back) == 3
    assert read_back[0]["doc_id"] == "doc1"
    assert read_back[2]["val"] == 30

def test_stage_manifest_and_checkpoint(tmp_path):
    stage_dir = str(tmp_path / "stage_test")
    manifest = StageManifest(stage_dir)
    chk = StageCheckpointManager(stage_dir)

    manifest.save("stage_test", "SUCCESS", record_counts={"docs": 100})
    loaded = manifest.load()
    assert loaded is not None
    assert loaded["status"] == "SUCCESS"
    assert loaded["record_counts"]["docs"] == 100

    chk.mark_shard_completed("shard_00001.parquet")
    assert chk.is_shard_completed("shard_00001.parquet")
    assert not chk.is_shard_completed("shard_00002.parquet")
