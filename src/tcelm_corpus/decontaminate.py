import re
from typing import List, Set, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from .schema import CanonicalDocument

@dataclass
class BenchmarkItem:
    benchmark_name: str
    item_id: str
    text: str
    shingles: Set[str]

class Decontaminator:
    def __init__(self, shingle_size: int = 13, min_token_overlap_ratio: float = 0.70):
        self.shingle_size = shingle_size
        self.min_token_overlap_ratio = min_token_overlap_ratio
        self.benchmark_registry: List[BenchmarkItem] = []

    def get_shingles(self, text: str) -> Set[str]:
        words = re.findall(r'\w+', text.lower())
        if len(words) < self.shingle_size:
            return {' '.join(words)}
        return {' '.join(words[i:i+self.shingle_size]) for i in range(len(words) - self.shingle_size + 1)}

    def register_benchmark_item(self, benchmark_name: str, item_id: str, text: str):
        shingles = self.get_shingles(text)
        item = BenchmarkItem(
            benchmark_name=benchmark_name,
            item_id=item_id,
            text=text.strip().lower(),
            shingles=shingles
        )
        self.benchmark_registry.append(item)

    def decontaminate(self, documents: List[CanonicalDocument]) -> Tuple[List[CanonicalDocument], List[Dict[str, Any]]]:
        if not self.benchmark_registry or not documents:
            return documents, []

        contaminated_parent_ids: Set[str] = set()
        contamination_logs: List[Dict[str, Any]] = []

        for doc in documents:
            doc_shingles = self.get_shingles(doc.normalized_text)
            doc_text_clean = doc.normalized_text.lower()

            for b_item in self.benchmark_registry:
                # 1. Exact full-question match
                if b_item.text in doc_text_clean:
                    contaminated_parent_ids.add(doc.parent_document_id)
                    contamination_logs.append({
                        "benchmark_name": b_item.benchmark_name,
                        "item_id": b_item.item_id,
                        "matched_document_id": doc.document_id,
                        "parent_document_id": doc.parent_document_id,
                        "match_type": "exact_full_question",
                        "action": "removed_parent_family"
                    })
                    break

                # 2. 13-token shingle matching + 70% item token overlap
                shared_shingles = doc_shingles.intersection(b_item.shingles)
                if shared_shingles:
                    overlap_ratio = len(shared_shingles) / max(len(b_item.shingles), 1)
                    if overlap_ratio >= self.min_token_overlap_ratio:
                        contaminated_parent_ids.add(doc.parent_document_id)
                        contamination_logs.append({
                            "benchmark_name": b_item.benchmark_name,
                            "item_id": b_item.item_id,
                            "matched_document_id": doc.document_id,
                            "parent_document_id": doc.parent_document_id,
                            "match_type": f"shingle_overlap_{overlap_ratio:.2f}",
                            "action": "removed_parent_family"
                        })
                        break

        clean_documents = [doc for doc in documents if doc.parent_document_id not in contaminated_parent_ids]
        return clean_documents, contamination_logs
