import pytest
import os
import json
from src.tcelm_corpus.tokenizer import BPECorpusTokenizer
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition, StructureSpans
from src.tcelm_corpus.stages.s09_tokenize_select import Stage09TokenizeSelect
from src.tcelm_corpus.stages.s10_freeze import Stage10Freeze
from src.tcelm_corpus.config import CorpusPipelineConfig
from src.tcelm_corpus.storage.parquet_io import ParquetShardIO
from src.tcelm_corpus.storage.manifest import StageManifest

def test_tokenizer_training_and_encoding(tmp_path):
    tokenizer = BPECorpusTokenizer(vocab_size=1000)
    texts = ["The quick brown fox jumps over the lazy dog.", "Machine learning LLM pretraining pipeline."]
    tok_file = str(tmp_path / "tokenizer.json")
    tokenizer.train_from_texts(texts, save_path=tok_file)

    assert os.path.exists(tok_file)

    doc = CanonicalDocument(
        document_id="doc1",
        parent_document_id="p1",
        source="web",
        source_revision="",
        source_record_id="",
        source_url_or_provenance="",
        license="",
        authors="",
        title="",
        publication_date="",
        language="en",
        raw_text_hash="",
        normalized_text_hash="",
        dedup_cluster_id="p1",
        normalized_text="The quick brown fox jumps over the lazy dog.",
        document_type="",
        domain="web",
        genre="",
        structure=StructureSpans(paragraph_spans=[[0, 43]]),
        quality=QualityScores(),
        position=SegmentPosition()
    )

    encoded = tokenizer.encode_document(doc)
    assert len(encoded.token_ids) > 0
    assert encoded.document_id == "doc1"

def test_freeze_gate_fails_on_same_length_token_mismatch(tmp_path):
    output_dir = str(tmp_path)
    config = CorpusPipelineConfig()

    tok_dir = os.path.join(output_dir, "stages", "08_train_tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    
    tokenizer = BPECorpusTokenizer(vocab_size=1000)
    tokenizer.train_from_texts(["The quick brown fox jumps over the lazy dog."], save_path=tok_path)
    tok_sha = StageManifest.compute_file_hash(tok_path)

    # Save matching Stage 08 and Stage 09 manifests
    StageManifest(tok_dir).save("08_train_tokenizer", "SUCCESS", output_hashes={"tokenizer_sha256": tok_sha})

    stage_09_dir = os.path.join(output_dir, "stages", "09_tokenize_select")
    layer_a_dir = os.path.join(stage_09_dir, "layer_a_selected")
    layer_b_dir = os.path.join(stage_09_dir, "layer_b_selected")
    StageManifest(stage_09_dir).save("09_tokenize_select", "SUCCESS", output_hashes={"tokenizer_sha256": tok_sha})

    rec_a = {"document_id": "doc1", "normalized_text": "The quick brown fox jumps over the lazy dog."}
    
    encoded_ids = tokenizer.tokenizer.encode("The quick brown fox jumps over the lazy dog.").ids
    mismatched_ids = list(encoded_ids)
    mismatched_ids[0] = (mismatched_ids[0] + 1) % 1000 # Same length, different token ID at position 0!

    rec_b = {
        "document_id": "doc1",
        "parent_document_id": "p1",
        "source": "web",
        "split": "train",
        "token_ids_json": json.dumps(mismatched_ids),
        "token_count": len(mismatched_ids)
    }

    ParquetShardIO(layer_a_dir).write_records_to_shards([rec_a])
    ParquetShardIO(layer_b_dir).write_records_to_shards([rec_b])

    freeze_stage = Stage10Freeze(output_dir, config)
    with pytest.raises(RuntimeError, match="Freeze Gate Failure: Token ID sequence mismatch"):
        freeze_stage.run_stage()

def test_tokenizer_file_modification_invalidates_stage09_cache_inputs(tmp_path):
    output_dir = str(tmp_path)
    config = CorpusPipelineConfig()

    tok_dir = os.path.join(output_dir, "stages", "08_train_tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    with open(tok_path, "w") as f:
        f.write("tokenizer_v1")

    stage09 = Stage09TokenizeSelect(output_dir, config)
    key1 = stage09._compute_stage_cache_key()

    # Modify tokenizer.json content
    with open(tok_path, "w") as f:
        f.write("tokenizer_v2_altered")

    key2 = stage09._compute_stage_cache_key()

    # Cache key MUST change due to get_additional_cache_inputs() hook!
    assert key1 != key2

def test_stage09_rejects_mismatched_tokenizer_hash_from_stage08(tmp_path):
    output_dir = str(tmp_path)
    config = CorpusPipelineConfig()

    tok_dir = os.path.join(output_dir, "stages", "08_train_tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    with open(tok_path, "w") as f:
        f.write("current_tokenizer_content")

    # Stage 08 manifest records a DIFFERENT hash
    StageManifest(tok_dir).save("08_train_tokenizer", "SUCCESS", output_hashes={"tokenizer_sha256": "fake_hash_12345"})

    stage09 = Stage09TokenizeSelect(output_dir, config)
    with pytest.raises(RuntimeError, match="Tokenizer Provenance Failure: Current tokenizer hash"):
        stage09.run_stage()

def test_stage10_requires_stage08_and_stage09_tokenizer_hash_match(tmp_path):
    output_dir = str(tmp_path)
    config = CorpusPipelineConfig()

    tok_dir = os.path.join(output_dir, "stages", "08_train_tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    with open(tok_path, "w") as f:
        f.write("valid_tokenizer")
    actual_sha = StageManifest.compute_file_hash(tok_path)

    # Stage 08 records actual_sha
    StageManifest(tok_dir).save("08_train_tokenizer", "SUCCESS", output_hashes={"tokenizer_sha256": actual_sha})

    # Stage 09 records a DIFFERENT sha
    stage_09_dir = os.path.join(output_dir, "stages", "09_tokenize_select")
    StageManifest(stage_09_dir).save("09_tokenize_select", "SUCCESS", output_hashes={"tokenizer_sha256": "mismatched_s09_sha"})

    freeze_stage = Stage10Freeze(output_dir, config)
    with pytest.raises(RuntimeError, match="Freeze Gate Failure: 3-way tokenizer SHA-256 mismatch"):
        freeze_stage.run_stage()
