from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

@dataclass
class StructureSpans:
    section_spans: List[List[int]] = field(default_factory=list)      # [[start_char, end_char], ...]
    subsection_spans: List[List[int]] = field(default_factory=list)
    paragraph_spans: List[List[int]] = field(default_factory=list)
    sentence_spans: List[List[int]] = field(default_factory=list)
    turn_spans: List[List[int]] = field(default_factory=list)
    equation_spans: List[List[int]] = field(default_factory=list)
    code_spans: List[List[int]] = field(default_factory=list)
    list_spans: List[List[int]] = field(default_factory=list)

@dataclass
class QualityScores:
    language_probability: float = 1.0
    printable_character_ratio: float = 1.0
    alphabetic_character_ratio: float = 1.0
    repetition_ratio: float = 0.0
    pii_count: int = 0
    ocr_quality: float = 1.0
    source_specific_quality: float = 1.0
    final_quality_score: float = 1.0

@dataclass
class SegmentPosition:
    segment_index: int = 0
    segment_count: int = 1
    previous_segment_id: Optional[str] = None
    next_segment_id: Optional[str] = None

@dataclass
class CanonicalDocument:
    document_id: str
    parent_document_id: str
    source: str
    source_revision: str
    source_record_id: str
    source_url_or_provenance: str
    license: str
    authors: str
    title: str
    publication_date: str
    language: str

    raw_text_hash: str
    normalized_text_hash: str
    dedup_cluster_id: str
    split_group_id: str = ""

    normalized_text: str = ""
    document_type: str = "article"
    domain: str = "web"
    genre: str = "prose"

    structure: StructureSpans = field(default_factory=StructureSpans)
    quality: QualityScores = field(default_factory=QualityScores)
    position: SegmentPosition = field(default_factory=SegmentPosition)
    split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("split_group_id"):
            d["split_group_id"] = d.get("parent_document_id", d.get("document_id", ""))
        return d

@dataclass
class TokenizedDocument:
    document_id: str
    parent_document_id: str
    source: str
    split: str
    token_ids: List[int]
    sentence_token_spans: List[List[int]] = field(default_factory=list)
    paragraph_token_spans: List[List[int]] = field(default_factory=list)
    section_token_spans: List[List[int]] = field(default_factory=list)
    turn_token_spans: List[List[int]] = field(default_factory=list)
    equation_token_spans: List[List[int]] = field(default_factory=list)

@dataclass
class LayerCViewRecord:
    view_id: str
    document_id: str
    view_type: str
    input_token_ids: List[int]
    target_token_ids: List[int]
    horizon: int = 1
    relation: str = "causal"
    sampling_seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)
