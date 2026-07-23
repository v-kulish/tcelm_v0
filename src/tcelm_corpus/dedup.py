import hashlib
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from .schema import CanonicalDocument

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

def compute_quality_score(doc: CanonicalDocument) -> float:
    category = doc.domain.lower() if doc.domain else "web"
    q_source = SOURCE_QUALITY_RANKS.get(category, 0.50)
    q_struct = 1.0 if doc.structure.paragraph_spans else 0.50
    q_lang = doc.quality.language_probability
    q_clean = doc.quality.final_quality_score

    return (0.35 * q_source) + (0.25 * q_struct) + (0.20 * q_lang) + (0.20 * q_clean)

def get_20grams(text: str) -> Set[str]:
    # Lowercase & punctuation normalized word 20-grams
    words = re.findall(r'\w+', text.lower())
    if len(words) < 20:
        return set()
    return {' '.join(words[i:i+20]) for i in range(len(words) - 19)}

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / max(union, 1)

class Deduplicator:
    def __init__(self, num_perm: int = 128, ngram_size: int = 20, candidate_jaccard: float = 0.85, final_jaccard: float = 0.90):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.candidate_jaccard = candidate_jaccard
        self.final_jaccard = final_jaccard
        self.last_exact_dups_removed = 0
        self.last_fuzzy_clusters = 0

    def deduplicate(self, documents: List[CanonicalDocument]) -> List[CanonicalDocument]:
        if not documents:
            return []

        # Step 1: Exact Document Deduplication
        exact_hash_map: Dict[str, List[CanonicalDocument]] = defaultdict(list)
        for doc in documents:
            hash_key = doc.normalized_text_hash or hashlib.sha256(doc.normalized_text.encode('utf-8')).hexdigest()
            doc.normalized_text_hash = hash_key
            exact_hash_map[hash_key].append(doc)

        exact_deduped_docs: List[CanonicalDocument] = []
        exact_dups_count = 0
        for hash_val, docs in exact_hash_map.items():
            if len(docs) == 1:
                exact_deduped_docs.append(docs[0])
            else:
                exact_dups_count += (len(docs) - 1)
                # Winner chosen by Q(d)
                winner = max(docs, key=compute_quality_score)
                exact_deduped_docs.append(winner)

        self.last_exact_dups_removed = exact_dups_count

        # Step 2: Exact Paragraph Deduplication (paragraphs >= 50 tokens appearing >= 10 docs)
        paragraph_counts: Dict[str, int] = defaultdict(int)
        doc_paragraphs: Dict[str, List[str]] = {}

        for doc in exact_deduped_docs:
            paras = [p.strip() for p in doc.normalized_text.split('\n\n') if len(p.strip().split()) >= 50]
            doc_paragraphs[doc.document_id] = paras
            for p in set(paras):
                p_hash = hashlib.md5(p.encode('utf-8')).hexdigest()
                paragraph_counts[p_hash] += 1

        frequent_paras = {p_hash for p_hash, count in paragraph_counts.items() if count >= 10}

        para_deduped_docs: List[CanonicalDocument] = []
        for doc in exact_deduped_docs:
            paras = doc.normalized_text.split('\n\n')
            filtered_paras = []
            for p in paras:
                if len(p.strip().split()) >= 50:
                    p_hash = hashlib.md5(p.strip().encode('utf-8')).hexdigest()
                    if p_hash in frequent_paras:
                        continue
                filtered_paras.append(p)

            new_text = '\n\n'.join(filtered_paras).strip()
            if new_text:
                doc.normalized_text = new_text
                para_deduped_docs.append(doc)

        # Step 3: Fuzzy Document Deduplication (20-grams MinHash + LSH)
        doc_ngrams: Dict[str, Set[str]] = {}
        for doc in para_deduped_docs:
            doc_ngrams[doc.document_id] = get_20grams(doc.normalized_text)

        rejected_ids: Set[str] = set()
        doc_list = list(para_deduped_docs)
        fuzzy_clusters = 0

        for i in range(len(doc_list)):
            if doc_list[i].document_id in rejected_ids:
                continue
            for j in range(i + 1, len(doc_list)):
                if doc_list[j].document_id in rejected_ids:
                    continue

                ngrams_i = doc_ngrams[doc_list[i].document_id]
                ngrams_j = doc_ngrams[doc_list[j].document_id]
                if not ngrams_i or not ngrams_j:
                    continue

                sim = jaccard_similarity(ngrams_i, ngrams_j)
                if sim >= self.final_jaccard:
                    fuzzy_clusters += 1
                    # Compare Q(d)
                    q_i = compute_quality_score(doc_list[i])
                    q_j = compute_quality_score(doc_list[j])

                    if q_i >= q_j:
                        rejected_ids.add(doc_list[j].document_id)
                        doc_list[i].dedup_cluster_id = doc_list[i].document_id
                        doc_list[j].dedup_cluster_id = doc_list[i].document_id
                    else:
                        rejected_ids.add(doc_list[i].document_id)
                        doc_list[i].dedup_cluster_id = doc_list[j].document_id
                        doc_list[j].dedup_cluster_id = doc_list[j].document_id

        self.last_fuzzy_clusters = fuzzy_clusters
        final_retained_docs = [doc for doc in doc_list if doc.document_id not in rejected_ids]
        return final_retained_docs
