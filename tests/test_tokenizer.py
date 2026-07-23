import pytest
import os
from src.tcelm_corpus.tokenizer import BPECorpusTokenizer
from src.tcelm_corpus.schema import CanonicalDocument, QualityScores, SegmentPosition, StructureSpans

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
