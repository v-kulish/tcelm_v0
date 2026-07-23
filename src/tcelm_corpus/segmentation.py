import re
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from .schema import StructureSpans, CanonicalDocument, QualityScores, SegmentPosition

SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9<])')
PARAGRAPH_SPLIT_REGEX = re.compile(r'\n\n+')
SECTION_SPLIT_REGEX = re.compile(r'\n(?=#+|\n==+)')

class StructuralSegmenter:
    def __init__(self, target_min_tokens: int = 8000, target_max_tokens: int = 24000, hard_max_tokens: int = 32768, min_segment_tokens: int = 256):
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens
        self.hard_max_tokens = hard_max_tokens
        self.min_segment_tokens = min_segment_tokens

    def extract_structure_spans(self, text: str) -> StructureSpans:
        spans = StructureSpans()
        if not text:
            return spans

        # Paragraph Spans
        p_start = 0
        p_matches = list(PARAGRAPH_SPLIT_REGEX.finditer(text))
        for m in p_matches:
            p_end = m.start()
            if p_end > p_start:
                spans.paragraph_spans.append([p_start, p_end])
            p_start = m.end()
        if p_start < len(text):
            spans.paragraph_spans.append([p_start, len(text)])

        # Sentence Spans
        s_start = 0
        for m in SENTENCE_SPLIT_REGEX.finditer(text):
            s_end = m.start()
            if s_end > s_start:
                spans.sentence_spans.append([s_start, s_end])
            s_start = m.end()
        if s_start < len(text):
            spans.sentence_spans.append([s_start, len(text)])

        # Turn Spans (<QUESTION...>, <ANSWER...>, <UTTERANCE...>, <COMMENT...>)
        turn_matches = list(re.finditer(r'<(QUESTION_BODY|ANSWER|COMMENT|UTTERANCE|ISSUE_BODY|PULL_REQUEST_BODY|REVIEW_COMMENT)>.*?</\1>', text, re.DOTALL))
        for m in turn_matches:
            spans.turn_spans.append([m.start(), m.end()])

        # Equation Spans
        eq_matches = list(re.finditer(r'<EQUATION>.*?</EQUATION>', text, re.DOTALL))
        for m in eq_matches:
            spans.equation_spans.append([m.start(), m.end()])

        return spans

    def segment_document(
        self,
        doc_id: str,
        parent_doc_id: str,
        source: str,
        normalized_text: str,
        metadata: Dict[str, Any],
        quality: QualityScores,
        split: str = "train"
    ) -> List[CanonicalDocument]:
        if not normalized_text or not normalized_text.strip():
            return []

        words = normalized_text.split()
        total_tokens = len(words)

        raw_text = metadata.get("raw_text", "") or metadata.get("text", "") or normalized_text
        raw_hash = metadata.get("raw_hash") or hashlib.sha256(raw_text.encode('utf-8')).hexdigest()

        # If document fits within hard max, create single canonical segment
        if total_tokens <= self.hard_max_tokens:
            spans = self.extract_structure_spans(normalized_text)
            norm_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
            doc = CanonicalDocument(
                document_id=f"{doc_id}_seg0",
                parent_document_id=parent_doc_id,
                source=source,
                source_revision=metadata.get("revision", "v0.1"),
                source_record_id=doc_id,
                source_url_or_provenance=metadata.get("url", ""),
                license=metadata.get("license", "open"),
                authors=metadata.get("authors", ""),
                title=metadata.get("title", ""),
                publication_date=metadata.get("date", ""),
                language="en",
                raw_text_hash=raw_hash,
                normalized_text_hash=norm_hash,
                dedup_cluster_id=parent_doc_id,
                normalized_text=normalized_text,
                document_type=metadata.get("document_type", "article"),
                domain=metadata.get("domain", "general"),
                genre=metadata.get("genre", "prose"),
                structure=spans,
                quality=quality,
                position=SegmentPosition(segment_index=0, segment_count=1, previous_segment_id=None, next_segment_id=None),
                split=split
            )
            return [doc]

        # Multi-segment splitting at paragraph boundaries
        paragraphs = [p.strip() for p in normalized_text.split('\n\n') if p.strip()]
        segments: List[str] = []
        current_p: List[str] = []
        current_tokens = 0

        for p in paragraphs:
            p_tokens = len(p.split())
            if current_tokens + p_tokens > self.target_max_tokens and current_tokens >= self.min_segment_tokens:
                segments.append("\n\n".join(current_p))
                current_p = [p]
                current_tokens = p_tokens
            else:
                current_p.append(p)
                current_tokens += p_tokens

        if current_p:
            segments.append("\n\n".join(current_p))

        # Build Canonical Document objects with adjacency
        result_docs = []
        num_segments = len(segments)
        for idx, seg_text in enumerate(segments):
            seg_id = f"{doc_id}_seg{idx}"
            prev_id = f"{doc_id}_seg{idx-1}" if idx > 0 else None
            next_id = f"{doc_id}_seg{idx+1}" if idx < num_segments - 1 else None

            spans = self.extract_structure_spans(seg_text)
            norm_hash = hashlib.sha256(seg_text.encode('utf-8')).hexdigest()
            cdoc = CanonicalDocument(
                document_id=seg_id,
                parent_document_id=parent_doc_id,
                source=source,
                source_revision=metadata.get("revision", "v0.1"),
                source_record_id=doc_id,
                source_url_or_provenance=metadata.get("url", ""),
                license=metadata.get("license", "open"),
                authors=metadata.get("authors", ""),
                title=metadata.get("title", ""),
                publication_date=metadata.get("date", ""),
                language="en",
                raw_text_hash=raw_hash,
                normalized_text_hash=norm_hash,
                dedup_cluster_id=parent_doc_id,
                normalized_text=seg_text,
                document_type=metadata.get("document_type", "article"),
                domain=metadata.get("domain", "general"),
                genre=metadata.get("genre", "prose"),
                structure=spans,
                quality=quality,
                position=SegmentPosition(segment_index=idx, segment_count=num_segments, previous_segment_id=prev_id, next_segment_id=next_id),
                split=split
            )
            result_docs.append(cdoc)

        return result_docs
