import hashlib
from typing import List, Dict, Set
from collections import defaultdict
from .schema import CanonicalDocument

class SplitAssigner:
    def __init__(
        self,
        train_share: float = 0.9970,
        val_share: float = 0.0010,
        test_share: float = 0.0010,
        trajectory_holdout_share: float = 0.0010,
        seed: int = 42
    ):
        self.train_share = train_share
        self.val_share = val_share
        self.test_share = test_share
        self.trajectory_holdout_share = trajectory_holdout_share
        self.seed = seed

    def assign_splits(self, documents: List[CanonicalDocument]) -> List[CanonicalDocument]:
        if not documents:
            return []

        # Group by parent document / dedup cluster ID
        cluster_docs: Dict[str, List[CanonicalDocument]] = defaultdict(list)
        for doc in documents:
            cluster_id = doc.dedup_cluster_id or doc.parent_document_id
            cluster_docs[cluster_id].append(doc)

        cluster_split_map: Dict[str, str] = {}
        for cluster_id, docs in cluster_docs.items():
            # Deterministic hash score for cluster assignment
            hash_str = f"{cluster_id}:{self.seed}"
            digest = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
            score = int(digest[:8], 16) / 0xFFFFFFFF

            # Check category for trajectory holdout weighting
            category = docs[0].domain.lower() if docs else "other"

            if score < self.val_share:
                split = "validation"
            elif score < (self.val_share + self.test_share):
                split = "test"
            elif score < (self.val_share + self.test_share + self.trajectory_holdout_share):
                split = "trajectory_holdout"
            else:
                split = "train"

            cluster_split_map[cluster_id] = split

        # Assign split to all canonical document segments in cluster
        for doc in documents:
            cluster_id = doc.dedup_cluster_id or doc.parent_document_id
            doc.split = cluster_split_map.get(cluster_id, "train")

        return documents
