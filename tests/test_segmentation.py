import pytest
from src.tcelm_corpus.segmentation import StructuralSegmenter
from src.tcelm_corpus.schema import QualityScores

def test_structural_segmenter():
    segmenter = StructuralSegmenter(target_max_tokens=500, hard_max_tokens=1000)
    text = ("Paragraph one text. " * 30) + "\n\n" + ("Paragraph two text. " * 30)
    docs = segmenter.segment_document(
        doc_id="doc1",
        parent_doc_id="doc1",
        source="common-pile/cccc_filtered",
        normalized_text=text,
        metadata={"title": "Test Doc"},
        quality=QualityScores()
    )
    assert len(docs) >= 1
    assert docs[0].document_id.startswith("doc1_seg")
    assert len(docs[0].structure.paragraph_spans) > 0
