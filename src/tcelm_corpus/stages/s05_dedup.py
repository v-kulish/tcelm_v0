import json
import hashlib
import re
import numpy as np
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm
from .base_stage import BaseStage
from ..storage.parquet_io import ParquetShardIO
from ..schema import CanonicalDocument, StructureSpans, QualityScores, SegmentPosition
from ..segmentation import StructuralSegmenter

class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            smaller, larger = sorted((root_i, root_j))
            self.parent[larger] = smaller

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

def winner_key(r: Dict[str, Any]) -> Tuple[float, int, str]:
    return (-compute_quality_score(r), r.get("priority", 0), r.get("document_id", ""))

def get_20grams(text: str) -> Set[str]:
    words = re.findall(r'\w+', text.lower())
    if len(words) < 20:
        return set()
    return {' '.join(words[i:i+20]) for i in range(len(words) - 19)}

_MINHASH_PRIME = 4294967311 # Prime > 2^32
_RNG = np.random.RandomState(42)
_MINHASH_A = _RNG.randint(1, 4294967295, size=128, dtype=np.uint64)
_MINHASH_B = _RNG.randint(0, 4294967295, size=128, dtype=np.uint64)

def compute_minhash(ngrams: Set[str], num_perm: int = 128) -> List[int]:
    if not ngrams:
        return [0] * num_perm
    # Fast vectorized linear MinHash: compute single 32-bit hash per n-gram, then numpy matrix mod
    hashes = np.array([int(hashlib.md5(ng.encode('utf-8')).hexdigest()[:8], 16) for ng in ngrams], dtype=np.uint64)
    a = _MINHASH_A[:num_perm, None]
    b = _MINHASH_B[:num_perm, None]
    perm_hashes = (a * hashes[None, :] + b) % _MINHASH_PRIME
    min_values = np.min(perm_hashes, axis=1)
    return min_values.tolist()

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
        if not all_recs:
            raise RuntimeError("Stage '05_dedup' received 0 input records from Stage 04.")

        for rec in all_recs:
            if "document_id" not in rec and "doc_id" in rec:
                rec["document_id"] = rec["doc_id"]
            if not rec.get("split_group_id"):
                rec["split_group_id"] = rec.get("parent_document_id") or rec["document_id"]

        all_split_groups = {rec["split_group_id"] for rec in all_recs}
        parent_uf = UnionFind(all_split_groups)

        # 1. Exact Document Deduplication
        print(f"Stage 05 Deduplication: Exact hashing across {len(all_recs):,} documents...")
        exact_map = defaultdict(list)
        for rec in tqdm(all_recs, desc="Exact Hash Deduplication", unit="doc"):
            h = rec.get("normalized_text_hash") or hashlib.sha256(rec["normalized_text"].encode('utf-8')).hexdigest()
            rec["normalized_text_hash"] = h
            exact_map[h].append(rec)

        exact_deduped = []
        exact_removed_count = 0
        for h, recs in exact_map.items():
            if len(recs) == 1:
                exact_deduped.append(recs[0])
            else:
                winner = min(recs, key=winner_key)
                exact_deduped.append(winner)
                exact_removed_count += (len(recs) - 1)
                # Union parent split groups for exact duplicate items
                for r in recs:
                    parent_uf.union(winner["split_group_id"], r["split_group_id"])

        print(f"Exact Deduplication complete: Retained {len(exact_deduped):,} / {len(all_recs):,} documents ({exact_removed_count:,} duplicates removed).")

        # 2. Exact Paragraph Deduplication & Span Recomputation
        print("Stage 05 Deduplication: Frequency scanning paragraphs...")
        para_counts = defaultdict(int)
        for rec in tqdm(exact_deduped, desc="Scanning Paragraphs", unit="doc"):
            paras = [p.strip() for p in rec["normalized_text"].split('\n\n') if len(p.strip().split()) >= 50]
            for p in set(paras):
                p_hash = hashlib.md5(p.encode('utf-8')).hexdigest()
                para_counts[p_hash] += 1

        frequent_paras = {p_hash for p_hash, count in para_counts.items() if count >= 10}

        para_deduped = []
        for rec in tqdm(exact_deduped, desc="Stripping Frequent Paragraphs", unit="doc"):
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

        # 3. LSH MinHash Fuzzy Deduplication using Union-Find Clustering
        num_bands = getattr(self.config.deduplication, "num_bands", 16)
        rows_per_band = getattr(self.config.deduplication, "rows_per_band", 8)
        num_perm = num_bands * rows_per_band
        final_jaccard = getattr(self.config.deduplication, "final_jaccard", 0.90)

        lsh_buckets = defaultdict(list)
        doc_ngrams = {}
        doc_ids = [rec["document_id"] for rec in para_deduped]
        doc_dict = {rec["document_id"]: rec for rec in para_deduped}

        print(f"Stage 05 Deduplication: Vectorized MinHash indexing across {len(para_deduped):,} documents...")
        for rec in tqdm(para_deduped, desc="MinHash & LSH Bucket Indexing", unit="doc"):
            d_id = rec["document_id"]
            ngrams = get_20grams(rec["normalized_text"])
            doc_ngrams[d_id] = ngrams
            if not ngrams:
                continue

            sig = compute_minhash(ngrams, num_perm=num_perm)
            for band_idx in range(num_bands):
                band_sig = tuple(sig[band_idx * rows_per_band : (band_idx + 1) * rows_per_band])
                bucket_key = (band_idx, band_sig)
                lsh_buckets[bucket_key].append(d_id)

        # Candidate pair generation
        candidate_pairs = set()
        for bucket, b_doc_ids in lsh_buckets.items():
            if len(b_doc_ids) > 1:
                for i in range(len(b_doc_ids)):
                    for j in range(i + 1, len(b_doc_ids)):
                        candidate_pairs.add(tuple(sorted([b_doc_ids[i], b_doc_ids[j]])))

        print(f"Stage 05 Deduplication: Verifying Jaccard similarity for {len(candidate_pairs):,} candidate pairs...")
        segment_uf = UnionFind(doc_ids)
        for id1, id2 in tqdm(sorted(candidate_pairs), desc="Verifying Candidate Pairs", unit="pair"):
            sim = jaccard_sim(doc_ngrams[id1], doc_ngrams[id2])
            if sim >= final_jaccard:
                segment_uf.union(id1, id2)
                # Union parent split groups whenever segments are fuzzy duplicates
                parent_uf.union(doc_dict[id1]["split_group_id"], doc_dict[id2]["split_group_id"])

        # Group segment components and pick 1 deterministic winner per segment cluster
        components = defaultdict(list)
        for d_id in sorted(doc_ids):
            root = segment_uf.find(d_id)
            components[root].append(doc_dict[d_id])

        final_retained = []
        rejected_count = 0
        fuzzy_clusters = 0

        for root in sorted(components.keys()):
            cluster_recs = components[root]
            if len(cluster_recs) == 1:
                rec = cluster_recs[0]
                rec["dedup_cluster_id"] = rec["document_id"]
                final_retained.append(rec)
            else:
                fuzzy_clusters += 1
                winner = min(cluster_recs, key=winner_key)
                for rec in cluster_recs:
                    rec["dedup_cluster_id"] = winner["document_id"]
                final_retained.append(winner)
                rejected_count += (len(cluster_recs) - 1)

        # Assign parent_uf component ID as the unified split_group_id for all retained items
        for rec in final_retained:
            rec["split_group_id"] = parent_uf.find(rec["split_group_id"])

        written_shards = self.shard_io.write_records_to_shards(final_retained, shard_prefix="part")

        record_counts = {}
        token_counts = {}
        for rec in final_retained:
            src = rec["source"]
            record_counts[src] = record_counts.get(src, 0) + 1
            token_counts[src] = token_counts.get(src, 0) + len(rec["normalized_text"].split())

        rejection_counts = {
            "exact_duplicates_removed": exact_removed_count,
            "frequent_paragraphs_stripped": len(frequent_paras),
            "fuzzy_duplicate_clusters": fuzzy_clusters,
            "fuzzy_duplicates_removed": rejected_count
        }

        print(f"Stage 05 Deduplication complete: Retained {len(final_retained):,} documents ({rejected_count:,} fuzzy duplicates removed).")

        return {
            "record_counts": record_counts,
            "token_counts": token_counts,
            "rejection_counts": rejection_counts,
            "output_hashes": {"shard_count": len(written_shards)}
        }
