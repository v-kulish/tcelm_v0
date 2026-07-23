import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import numpy as np
from .schema import TokenizedDocument

class TokenFrequencyStats:
    def __init__(self, vocab_size: int = 32768, smoothing_alpha: float = 0.1):
        self.vocab_size = vocab_size
        self.smoothing_alpha = smoothing_alpha

    def compute_frequencies(self, train_docs: List[TokenizedDocument]) -> Dict[str, Any]:
        global_counts = np.zeros(self.vocab_size, dtype=np.int64)
        source_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.vocab_size, dtype=np.int64))
        
        total_tokens = 0

        for doc in train_docs:
            if doc.split != "train":
                continue

            ids = np.array(doc.token_ids, dtype=np.int64)
            valid_ids = ids[(ids >= 0) & (ids < self.vocab_size)]

            np.add.at(global_counts, valid_ids, 1)
            np.add.at(source_counts[doc.source], valid_ids, 1)
            total_tokens += len(valid_ids)

        # Compute smoothed log probabilities in float32
        p_global_denom = total_tokens + (self.smoothing_alpha * self.vocab_size)
        p_global_log = np.log((global_counts + self.smoothing_alpha) / p_global_denom).astype(np.float32)

        p_source_log: Dict[str, np.ndarray] = {}
        for src, counts in source_counts.items():
            src_total = counts.sum()
            src_denom = src_total + (self.smoothing_alpha * self.vocab_size)
            p_source_log[src] = np.log((counts + self.smoothing_alpha) / src_denom).astype(np.float32)

        return {
            "p_global_log": p_global_log,
            "p_source_log": p_source_log,
            "total_train_tokens": total_tokens,
            "vocab_size": self.vocab_size
        }
