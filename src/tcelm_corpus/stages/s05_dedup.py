import json
import hashlib
import re
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..schema import CanonicalDocument, StructureSpans, QualityScores, SegmentPosition
from ..segmentation import StructuralSegmenter

SOURCE_QUALITY_RANKS = {
    "books": 0.95,
    "scientific": 0.90,
    "educational": 0.85,
    "wikimedia": 0.80,
    "government_legal": 0.75,
    "technical": 0.70,
    "discussion": 0.65,
    "web": 0.50
}

def compute_quality_score(rec: Dict[str, Any]) -> float:
    category = rec.get("domain", "web").lower()
    q_source = SOURCE_QUALITY_RANKS.get(category, 0.50)
    q_struct = 1.0 if rec.get("structure_json") else 0.50
    return (0.35 * q_source) + (0.25 * q_struct) + 0.40

def get_20grams(text: str) -> Set[str]:
    words = re.findall(r'\w+', text.lower())
    if len(words) < 20:
        return set()
    return {' '.join(words[i:i+20]) for i in range(len(words) - 19)}

def compute_minhash(ngrams: Set[str], num_perm: int = 128) -> List[int]:
    sig = []
    for i in range(num_perm):
        min_val = 0xFFFFFFFF_FFFFFFFF
        for ng in ngrams:
            h = int(hashlib.md5(f"{i}:{ng}".encode('utf-8')).hexdigest()[:16], 16)
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig

def jaccard_sim(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / max(len(set_a.union(set_b)), 1)

class Stage05Dedup(BaseStage):
    def __init__(self, output_dir: str, config):
        super().__init__("05_dedup", output_dir, config)
        self.input_io = ParquetShardIO(f"{output_dir}/stages/04_segment")
        self.segmenter = StructuralSegmenter()

    def run_stage(self) -> Dict[str, Any]:
        all_recs = list(self.input_io.read_shards())
        initial_count = len(all_recs)

        # 1. Exact Document Deduplication
        exact_map = defaultdict(list)
        for rec in all_recs:
            h = rec.get("normalized_text_hash") or hashlib.sha256(rec["normalized_text"].encode('utf-8')).hexdigest()
            rec["normalized_text_hash"] = h
            exact_map[h].append(rec)

        exact_deduped = []
        exact_removed_count = 0
        for h, recs in exact_map.items():
            if len(recs) == 1:
                exact_deduped.append(recs[0])
            else:
                winner = max(recs, key=compute_quality_score)
                exact_deduped.append(winner)
                exact_removed_count += (len(recs) - 1)

        # 2. Exact Paragraph Deduplication & Span Recomputation
        para_counts = defaultdict(int)
        for rec in exact_deduped:
            paras = [p.strip() for p in rec["normalized_text"].split('\n\n') if len(p.strip().split()) >= 50]
            for p in set(paras):
                p_hash = hashlib.md5(p.encode('utf-8')).hexdigest()
                para_counts[p_hash] += 1

        frequent_paras = {p_hash for p_hash, count in para_counts.items() if count >= 10}

        para_deduped = []
        for rec in exact_deduped:
            paras = rec["normalized_text"].split('\n\n')
            filtered = []
            for p in paras:
                if len(p.strip().split()) >= 50:
                    p_hash = hashlib.md5(p.strip().encode('utf-8')).hexdigest()
                    if p_hash in frequent_paras:
                        continue
                filtered.append(p)

            new_text = '\n\n'.join(filtered).strip()
            if new_text and len(new_text.split()) >= 64:
                rec["normalized_text"] = new_text
                # RECOMPUTE text hashes and structural spans post-paragraph removal
                rec["normalized_text_hash"] = hashlib.sha256(new_text.encode('utf-8')).hexdigest()
                new_spans = self.segmenter.extract_structure_spans(new_text)
                rec["structure_json"] = json.dumps(new_spans.__dict__)
                para_deduped.append(rec)

        # 3. LSH MinHash Fuzzy Deduplication
        num_bands = getattr(self.config.deduplication, "num_bands", 16)
        rows_per_band = getattr(self.config.deduplication, "rows_per_band", 8)
        num_perm = num_bands * rows_per_band
        final_jaccard = getattr(self.config.deduplication, "final_jaccard", 0.90)

        # Generate MinHash & LSH Bands
        lsh_buckets = defaultdict(list)
        doc_ngrams = {}

        for idx, rec in enumerate(para_deduped):
            ngrams = get_20grams(rec["normalized_text"])
            doc_ngrams[rec["document_id"]] = ngrams
            if not ngrams:
                continue

            sig = compute_minhash(ngrams, num_perm=num_perm)
            for band_idx in range(num_bands):
                band_sig = tuple(sig[band_idx * rows_per_band : (band_idx + 1) * rows_per_band])
                bucket_key = (band_idx, band_sig)
                lsh_buckets[bucket_key].append(rec["document_id"])

        # Candidate pair generation
        candidate_pairs = set()
        for bucket, doc_ids in lsh_buckets.items():
            if len(doc_ids) > 1:
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        candidate_pairs.add(tuple(sorted([doc_ids[i], doc_ids[j]])))

        # Compare Candidate Pairs
        rejected_ids = set()
        doc_dict = {rec["document_id"]: rec for rec in para_deduped}
        fuzzy_clusters = 0

        for id1, id2 in candidate_pairs:
            if id1 in rejected_ids or id2 in rejected_ids:
                continue
            
            sim = jaccard_sim(doc_ngrams[id1], doc_ngrams[id2])
            if sim >= final_jaccard:
                fuzzy_clusters += 1
                q1 = compute_quality_score(doc_dict[id1])
                q2 = compute_quality_score(doc_dict[id2])

                if q1 >= q2:
                    rejected_ids.add(id2)
                    doc_dict[id1]["dedup_cluster_id"] = doc_dict[id1]["document_id"]
                    doc_dict[id2]["dedup_cluster_id"] = doc_dict[id1]["document_id"]
                else:
                    rejected_ids.add(id1)
                    doc_dict[id1]["dedup_cluster_id"] = doc_dict[id2]["document_id"]
                    doc_dict[id2]["dedup_cluster_id"] = doc_dict[id2]["document_id"]

        final_retained = [rec for rec in para_deduped if rec["document_id"] not in rejected_ids]
        written_shards = self.shard_io.write_records_to_shards(final_retained, shard_prefix="dedup")

        record_counts = {}
        token_counts = {}
        for rec in final_retained:
            src = rec["source"]
            record_counts[src] = record_counts.get(src, 0) + 1
            token_counts[src] = token_counts.get(src, 0) + len(rec["normalized_text"].split())

        rejection_counts = {
            "exact_duplicates_removed": exact_removed_count,
            "frequent_paragraphs_stripped": len(frequent_paras),
            "fuzzy_duplicates_removed": len(rejected_ids)
        }

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "rejection_counts": rejection_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
